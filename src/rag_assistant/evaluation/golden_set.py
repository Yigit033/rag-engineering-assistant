"""
Golden set — değerlendirmenin temeli.

EN KRİTİK TASARIM KARARI: DOĞRU CEVAP NEYE BAĞLANIR?

  Naif yaklaşım: beklenen `chunk_id`'leri kaydetmek.
  Neden YANLIŞ: chunk kimliği içerikten türüyor. Chunk boyutunu, örtüşmeyi
  veya metin normalizasyonunu değiştirdiğin an TÜM kimlikler değişir ve
  golden set'in tamamı çöp olur. Yani tam olarak ölçmek istediğin şeyi
  (chunking stratejisini) değiştirdiğinde ölçü aletini kaybediyorsun.

  Doğru yaklaşım: cevabın bulunduğu YERE bağla — (dosya, sayfa).
  Sayfa numarası chunking'den bağımsızdır; dokümanın kendi gerçeğidir.
  Chunk boyutunu 400'den 800'e çıkarsan golden set hâlâ geçerli kalır.

NEGATİF ÖRNEKLER ZORUNLU:
  Yalnızca cevaplanabilir sorularla ölçüm yapmak, sistemin en tehlikeli
  hatasını (uydurma) hiç görmemektir. Bu yüzden golden set'te bilinçli
  olarak dokümanda OLMAYAN sorular bulunur ve sistemin çekimser kalması
  beklenir. "Uydurma oranı" ancak böyle ölçülebilir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rag_assistant.observability import get_logger

logger = get_logger(__name__)

GOLDEN_SET_VERSION = 1


class QuestionKind(StrEnum):
    """
    Soru tipi.

    ANSWERABLE   → cevap dokümanlarda var; sistem cevaplamalı ve atıf vermeli
    UNANSWERABLE → cevap dokümanlarda YOK; sistem çekimser kalmalı
                   (model dünya bilgisinden biliyor olsa bile!)
    """

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True, slots=True)
class ExpectedSource:
    """Cevabın bulunduğu yer. Chunking'den BAĞIMSIZ çıpa."""

    file_name: str
    page: int | None = None

    def matches(self, file_name: str, page: int | None) -> bool:
        """
        Bu beklenen kaynak, getirilen bir chunk'ın kaynağıyla eşleşiyor mu?

        `page is None` ise yalnızca dosya adına bakılır — bazı sorular için
        "şu dokümandan gelmesi yeter" demek yeterlidir ve sayfa numarasını
        elle işaretleme maliyetinden kurtarır.
        """
        if self.file_name != file_name:
            return False
        return self.page is None or self.page == page


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """Tek bir değerlendirme sorusu."""

    id: str
    question: str
    kind: QuestionKind

    # Retrieval doğruluğu için: cevap hangi sayfalarda?
    expected_sources: tuple[ExpectedSource, ...] = ()

    # Cevap doğruluğu için: bu ifadeler cevapta GEÇMELİ.
    # Tam metin karşılaştırması yapmıyoruz — LLM aynı bilgiyi farklı
    # cümlelerle ifade eder ve bu bir hata değildir. Kritik olguları
    # (sayı, oran, isim) kontrol etmek hem daha sağlam hem daha anlamlı.
    must_contain: tuple[str, ...] = ()

    # Bu ifadeler cevapta GEÇMEMELİ. Uydurma tespiti için:
    # örneğin "Titanik hangi yıl battı?" sorusunda "1912" geçerse, model
    # bağlamı bırakıp dünya bilgisinden cevap vermiş demektir.
    must_not_contain: tuple[str, ...] = ()

    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "kind": str(self.kind),
            "expected_sources": [
                {"file_name": s.file_name, "page": s.page} for s in self.expected_sources
            ],
            "must_contain": list(self.must_contain),
            "must_not_contain": list(self.must_not_contain),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuestion:
        return cls(
            id=data["id"],
            question=data["question"],
            kind=QuestionKind(data["kind"]),
            expected_sources=tuple(
                ExpectedSource(file_name=s["file_name"], page=s.get("page"))
                for s in data.get("expected_sources", [])
            ),
            must_contain=tuple(data.get("must_contain", [])),
            must_not_contain=tuple(data.get("must_not_contain", [])),
            notes=data.get("notes", ""),
        )


@dataclass(slots=True)
class GoldenSet:
    """Değerlendirme soruları kümesi."""

    questions: list[GoldenQuestion] = field(default_factory=list)
    description: str = ""

    FILE_NAME = "golden_set.json"

    # ------------------------------------------------------------------
    @property
    def answerable(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.kind is QuestionKind.ANSWERABLE]

    @property
    def unanswerable(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.kind is QuestionKind.UNANSWERABLE]

    def validate(self) -> list[str]:
        """
        Golden set'in kendisi tutarlı mı?

        Değerlendirme setindeki bir hata, ölçtüğün her şeyi geçersiz kılar —
        ve bunu fark etmek çok zordur (sistem "kötü" görünür, oysa ölçü
        aleti bozuktur). Bu yüzden set de doğrulanır.
        """
        problems: list[str] = []

        ids = [q.id for q in self.questions]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            problems.append(f"tekrarlanan id: {sorted(duplicates)}")

        for q in self.questions:
            if q.kind is QuestionKind.ANSWERABLE and not q.expected_sources:
                problems.append(
                    f"{q.id}: cevaplanabilir soruda beklenen kaynak yok "
                    "(retrieval doğruluğu ölçülemez)"
                )
            if q.kind is QuestionKind.UNANSWERABLE and q.expected_sources:
                problems.append(
                    f"{q.id}: cevaplanamaz soruda beklenen kaynak tanımlı (çelişki)"
                )
            if q.kind is QuestionKind.UNANSWERABLE and q.must_contain:
                problems.append(
                    f"{q.id}: cevaplanamaz soruda must_contain var — sistem çekimser "
                    "kalmalı, bir şey içermemeli"
                )

        if not self.unanswerable:
            problems.append(
                "hiç negatif örnek yok — uydurma oranı ÖLÇÜLEMEZ. "
                "En az birkaç 'cevabı dokümanlarda olmayan' soru ekleyin."
            )

        return problems

    # ------------------------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.FILE_NAME
        payload = {
            "version": GOLDEN_SET_VERSION,
            "description": self.description,
            "questions": [q.to_dict() for q in self.questions],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("golden_set.saved", path=str(path), questions=len(self.questions))
        return path

    @classmethod
    def load(cls, directory: Path) -> GoldenSet:
        path = directory / cls.FILE_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"Golden set bulunamadı: {path}\n"
                "Örnek bir set oluşturmak için: rag-eval --init"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != GOLDEN_SET_VERSION:
            raise ValueError(
                f"Golden set sürümü {data.get('version')}, beklenen {GOLDEN_SET_VERSION}"
            )
        gs = cls(
            questions=[GoldenQuestion.from_dict(q) for q in data.get("questions", [])],
            description=data.get("description", ""),
        )
        logger.info("golden_set.loaded", questions=len(gs.questions))
        return gs
