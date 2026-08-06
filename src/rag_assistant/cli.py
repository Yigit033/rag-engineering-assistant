"""
Komut satırı arayüzü.

TASARIM KARARLARI:

1. AYNI KOMPOZİSYON KÖKÜ
   CLI de `build_rag_system` kullanır — API ile birebir aynı nesne grafiği.
   Böylece "CLI'da çalışıyor ama API'de farklı sonuç veriyor" sınıfı hatalar
   yapısal olarak imkânsız hale gelir.

2. ÇIKIŞ KODLARI ANLAMLIDIR
   0 = başarılı · 1 = çalışma hatası · 2 = kullanım hatası
   CI/CD ve betikler bu kodlara güvenir. Her durumda 0 döndüren bir CLI,
   otomasyonda sessiz başarısızlık üretir.

3. `--json` SEÇENEĞİ
   İnsan için okunabilir çıktı varsayılan; makine için JSON isteğe bağlı.
   İkisini karıştırmak (ör. renkli çıktıyı parse etmeye çalışmak) kırılgan
   betiklere yol açar.

4. LLM YALNIZCA GEREKTİĞİNDE ISITILIR
   `ingest` komutu LLM kullanmaz; 3.5 GB'ı boşuna ayırmamak için
   `warm_llm=False` geçilir. Bellek kısıtlı makinelerde bu fark eder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_assistant.composition import build_rag_system
from rag_assistant.config import get_settings
from rag_assistant.generation.factory import LLMConfigurationError
from rag_assistant.indexing.store import IndexCompatibilityError
from rag_assistant.observability import configure_logging, get_logger

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _bootstrap(*, verbose: bool = False) -> None:
    settings = get_settings()
    configure_logging(
        level="DEBUG" if verbose else settings.log.level,
        json_format=settings.log.json_format,
    )


def _fail(message: str) -> int:
    print(f"\nHATA: {message}\n", file=sys.stderr)
    return EXIT_ERROR


# ---------------------------------------------------------------------------
# rag-ingest
# ---------------------------------------------------------------------------
def ingest(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Kaynak klasördeki dokümanları indeksle (idempotent).",
    )
    parser.add_argument(
        "--dir", type=Path, default=None, help="Kaynak klasör (varsayılan: data/raw)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Değişmemiş dosyaları da yeniden işle. Chunk ayarları veya metin "
            "normalizasyonu değiştiğinde gerekir: bunlar dosya içeriğine "
            "yansımadığı için hash aynı kalır ve dosya normalde atlanır."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Makine okunabilir çıktı")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _bootstrap(verbose=args.verbose)
    settings = get_settings()

    try:
        # LLM gerekmez → ısıtma yok, bellek boşa ayrılmaz.
        system = build_rag_system(settings, warm_llm=False)
        source = args.dir or settings.paths.raw_dir
        report = system.ingestion.run(source, force=args.force)
        system.store.save(settings.paths.index_dir)
    except IndexCompatibilityError as exc:
        return _fail(f"{exc}\nÖneri: data/index klasörünü silip yeniden indeksleyin.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest.failed")
        return _fail(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "chunks_added": report.total_chunks_added,
                    "index_total": system.store.count,
                    "documents": [
                        {
                            "file": d.file_name,
                            "status": str(d.status),
                            "pages": d.page_count,
                            "chunks": d.chunk_count,
                            "needs_ocr": d.needs_ocr,
                            "error": d.error,
                        }
                        for d in report.documents
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    print(f"\n{len(report.documents)} dosya · {report.duration_seconds:.1f}s")
    for d in report.documents:
        mark = {"ok": "✓", "skipped": "·", "no_text_layer": "!", "failed": "✗"}.get(
            str(d.status), "?"
        )
        detail = f"{d.page_count} sayfa, {d.chunk_count} chunk"
        if d.error:
            detail = d.error
        print(f"  {mark} {str(d.status):<14} {d.file_name:<44} {detail}")

    print(f"\n  eklenen chunk: {report.total_chunks_added} · index: {system.store.count}")

    # Bu iki durum ayrıca vurgulanır: "başarılı" sayısının içinde kaybolmamalı.
    if report.needs_ocr:
        print(
            f"\n  ! {len(report.needs_ocr)} dosyada metin katmanı yok (taranmış PDF). "
            "Bu dosyalar aramaya DAHİL DEĞİL; OCR gerekiyor:"
        )
        for d in report.needs_ocr:
            print(f"      {d.file_name}")
    if report.failed:
        print(f"\n  ✗ {len(report.failed)} dosya işlenemedi:")
        for d in report.failed:
            print(f"      {d.file_name}: {d.error}")

    print()
    return EXIT_OK


# ---------------------------------------------------------------------------
# rag-ask
# ---------------------------------------------------------------------------
def ask(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-ask", description="Dokümanlara soru sor (kaynak gösterir)."
    )
    parser.add_argument("question", help="Sorulacak soru")
    parser.add_argument("--top-k", type=int, default=None, help="Kaç chunk kullanılsın")
    parser.add_argument("--stream", action="store_true", help="Token token yazdır")
    parser.add_argument("--json", action="store_true", help="Makine okunabilir çıktı")
    parser.add_argument(
        "--show-context", action="store_true", help="Kullanılan bağlamı da yazdır"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _bootstrap(verbose=args.verbose)
    settings = get_settings()

    try:
        system = build_rag_system(settings)
    except LLMConfigurationError as exc:
        return _fail(str(exc))
    except IndexCompatibilityError as exc:
        return _fail(f"{exc}\nÖneri: rag-ingest --force ile yeniden indeksleyin.")

    if system.store.count == 0:
        return _fail("Index boş. Önce `rag-ingest` çalıştırın.")

    try:
        if args.stream and not args.json:
            print()
            for piece in system.answerer.stream(args.question, top_k=args.top_k):
                print(piece, end="", flush=True)
            print("\n")
            return EXIT_OK

        answer = system.answerer.answer(args.question, top_k=args.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ask.failed")
        return _fail(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.text,
                    "abstained": answer.abstained,
                    "grounded": answer.is_grounded,
                    "citations": [
                        {"marker": c.marker, "source": c.source.label()}
                        for c in answer.citations
                    ],
                    "model": answer.model,
                    "prompt_version": answer.prompt_version,
                    "latency_ms": answer.latency_ms,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    print(f"\n{answer.text}\n")

    if answer.citations:
        print("  Kaynaklar:")
        for c in answer.citations:
            print(f"    [{c.marker}] {c.source.label()}")
    elif answer.abstained:
        print("  (bilgi dokümanlarda bulunamadı — çekimser kalındı)")
    else:
        # Görünür kılınır: atıfsız cevap doğrulanamaz.
        print("  ! Model kaynak göstermedi — bu cevap doğrulanamaz.")

    if args.show_context:
        print("\n  Kullanılan bağlam:")
        for i, sc in enumerate(answer.used_chunks, start=1):
            print(f"    [{i}] {sc.chunk.source.label()} (hits={sc.retriever_hits})")
            print(f"        {sc.chunk.text[:150].replace(chr(10), ' ')}...")

    print(f"\n  {answer.model} · prompt {answer.prompt_version} · {answer.latency_ms} ms\n")
    return EXIT_OK


# ---------------------------------------------------------------------------
# rag-eval
# ---------------------------------------------------------------------------
def evaluate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-eval", description="Golden set üzerinde sistemi değerlendir."
    )
    parser.add_argument("--k", type=int, default=None, help="top_k (varsayılan: config)")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Golden set tutarsız olsa da koş (önerilmez: bozuk ölçü aleti)",
    )
    parser.add_argument("--json", action="store_true", help="Yalnızca JSON özet")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="ORAN",
        help=(
            "Geçme oranı bu değerin altındaysa çıkış kodu 1 döndür (0-1 arası). "
            "CI'da kalite gerilemesini yakalamak için."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _bootstrap(verbose=args.verbose)
    settings = get_settings()

    from rag_assistant.evaluation.golden_set import GoldenSet
    from rag_assistant.evaluation.report import render_report
    from rag_assistant.evaluation.runner import (
        EvaluationError,
        run_evaluation,
        save_summary,
    )

    try:
        golden_set = GoldenSet.load(settings.paths.eval_dir)
        system = build_rag_system(settings)
        summary = run_evaluation(
            system, golden_set, k=args.k, strict=not args.no_strict
        )
    except (EvaluationError, FileNotFoundError, LLMConfigurationError) as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("eval.failed")
        return _fail(str(exc))

    if args.json:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(render_report(summary))
        path = save_summary(summary, settings.paths.eval_dir)
        print(f"\n  ayrıntılı sonuç: {path}\n")

    # CI kapısı: kalite gerilemesi derlemeyi kırar.
    if args.fail_under is not None and summary.pass_rate < args.fail_under:
        print(
            f"HATA: geçme oranı {summary.pass_rate:.1%} < eşik {args.fail_under:.1%}",
            file=sys.stderr,
        )
        return EXIT_ERROR

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(EXIT_USAGE)


# ---------------------------------------------------------------------------
# rag-docs
# ---------------------------------------------------------------------------
def documents(argv: list[str] | None = None) -> int:
    """
    Doküman kütüphanesini yönet.

    API ile AYNI `DocumentLibrary` örneğini kullanır — güvenlik kontrolleri
    (dosya adı temizleme, uzantı, boyut) tek yerde durur. CLI kendi
    doğrulamasını yazsaydı biri eksik kalırdı.
    """
    parser = argparse.ArgumentParser(
        prog="rag-docs", description="Kütüphanedeki dokümanları listele / sil / ekle."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Dokümanları listele")
    p_list.add_argument("--json", action="store_true")

    p_add = sub.add_parser("add", help="Dosya ekle (kopyalar ve indeksler)")
    p_add.add_argument("path", type=Path, help="Eklenecek PDF")
    p_add.add_argument("--no-index", action="store_true", help="Sadece kopyala, indeksleme")
    p_add.add_argument("--overwrite", action="store_true")

    p_del = sub.add_parser("delete", help="Doküman sil (chunk'ları dahil)")
    p_del.add_argument("file_name")
    p_del.add_argument("-y", "--yes", action="store_true", help="Onay sorma")

    for p in (p_list, p_add, p_del):
        p.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    _bootstrap(verbose=getattr(args, "verbose", False))
    settings = get_settings()

    from rag_assistant.library import LibraryError

    try:
        # LLM gerekmez → ısıtma yok.
        system = build_rag_system(settings, warm_llm=False)
    except IndexCompatibilityError as exc:
        return _fail(str(exc))

    library = system.library

    try:
        if args.command == "list":
            docs = library.list_documents()
            if args.json:
                print(
                    json.dumps(
                        [
                            {
                                "file_name": d.file_name,
                                "status": str(d.status),
                                "size_bytes": d.size_bytes,
                                "pages": d.page_count,
                                "chunks": d.chunk_count,
                                "searchable": d.is_searchable,
                                "needs_ocr": d.needs_ocr,
                            }
                            for d in docs
                        ],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return EXIT_OK

            if not docs:
                print("\n  Kütüphane boş. `rag-docs add <dosya.pdf>` ile ekleyin.\n")
                return EXIT_OK

            print(f"\n  {len(docs)} doküman · index: {system.store.count} vektör\n")
            print(f"  {'':<3}{'DOSYA':<44}{'DURUM':<16}{'SAYFA':>6}{'CHUNK':>7}{'BOYUT':>10}")
            for d in docs:
                mark = "✓" if d.is_searchable else ("!" if d.needs_ocr else "·")
                size = f"{d.size_bytes / 1024:.0f} KB"
                print(
                    f"  {mark:<3}{d.file_name[:42]:<44}{str(d.status):<16}"
                    f"{d.page_count:>6}{d.chunk_count:>7}{size:>10}"
                )

            ocr = [d for d in docs if d.needs_ocr]
            if ocr:
                print(
                    f"\n  ! {len(ocr)} doküman aramaya DAHİL DEĞİL (taranmış PDF, OCR gerekli)"
                )
            print()
            return EXIT_OK

        if args.command == "add":
            if not args.path.is_file():
                return _fail(f"Dosya bulunamadı: {args.path}")

            with args.path.open("rb") as f:
                result = library.save_upload(
                    args.path.name,
                    iter(lambda: f.read(8192), b""),
                    overwrite=args.overwrite,
                )

            print(f"\n  eklendi: {result.file_name} ({result.size_bytes / 1024:.0f} KB)")
            if result.duplicate_of:
                print(
                    f"  ! aynı içerik '{result.duplicate_of}' adıyla zaten var — "
                    "aynı bilgi sonuçlarda iki kez çıkabilir"
                )

            if args.no_index:
                print("  (indekslenmedi — `rag-ingest` ile indeksleyin)\n")
                return EXIT_OK

            report = system.ingestion.run(library.raw_dir)
            system.store.save(settings.paths.index_dir)
            added = next(
                (d for d in report.documents if d.file_name == result.file_name), None
            )
            if added and added.needs_ocr:
                print("  ! metin katmanı yok (taranmış PDF) — aramaya DAHİL DEĞİL\n")
            else:
                print(
                    f"  indekslendi: {added.chunk_count if added else 0} chunk · "
                    f"index toplam {system.store.count}\n"
                )
            return EXIT_OK

        if args.command == "delete":
            doc = library.get(args.file_name)
            if not args.yes:
                print(
                    f"\n  '{doc.file_name}' ve {doc.chunk_count} chunk'ı SİLİNECEK."
                )
                if input("  Onaylıyor musunuz? (e/h): ").strip().lower() not in ("e", "evet"):
                    print("  İptal edildi.\n")
                    return EXIT_OK

            library.delete(doc.file_name)
            print(f"\n  silindi: {doc.file_name} · index: {system.store.count} vektör\n")
            return EXIT_OK

    except LibraryError as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("docs.failed")
        return _fail(str(exc))

    return EXIT_USAGE
