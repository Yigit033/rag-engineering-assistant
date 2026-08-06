"""
Faz 5 testleri: HTTP sözleşmesi.

Model yüklenmiyor. `get_system` bağımlılığı sahte bir sistemle değiştiriliyor —
FastAPI'nin `dependency_overrides` mekanizması tam olarak bunun için var.

Böylece test ettiğimiz şey NET: HTTP katmanının davranışı. Doğru durum kodu
mu dönüyor, doğru alanları mı sızdırıyor, doğrulama çalışıyor mu? Modelin
cevap kalitesi burada test edilmez (o Faz 7'nin işi).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api import app as app_module
from rag_assistant.api.app import create_app, get_system
from rag_assistant.domain.models import (
    Answer,
    Chunk,
    Citation,
    DocumentReport,
    DocumentStatus,
    IngestReport,
    RetrievalStage,
    ScoredChunk,
    SourceRef,
)


# ---------------------------------------------------------------------------
# Sahte bileşenler
# ---------------------------------------------------------------------------
def make_answer(*, abstained: bool = False, citations: int = 1) -> Answer:
    src = SourceRef("rapor.pdf", 3)
    chunk = Chunk(text="Bu chunk'ın TAM metni dışarı sızmamalı." * 20, source=src, token_count=50)
    scored = ScoredChunk(chunk, 0.87, RetrievalStage.RERANKED, 1, retriever_hits=2)
    return Answer(
        question="soru?",
        text="Bu bilgi verilen dokümanlarda yok." if abstained else "Cevap [1].",
        citations=tuple(
            Citation(marker=i, source=src, chunk_id=chunk.id) for i in range(1, citations + 1)
        ),
        used_chunks=(scored,),
        abstained=abstained,
        model="fake-llm",
        prompt_version="v1",
        latency_ms=42,
    )


class FakeStore:
    def __init__(self, count: int = 6) -> None:
        self.count = count
        self.saved = False

    def save(self, directory: Any) -> None:
        self.saved = True


class FakeAnswerer:
    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.last_top_k: int | None = None

    def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        self.last_top_k = top_k
        return self._answer

    def stream(self, question: str, *, top_k: int | None = None) -> Iterator[str]:
        yield "Cevabın "
        yield "ilk parçası\nikinci satır"


class FakeRetriever:
    name = "hybrid(dense+sparse)+rerank"
    reranker = None

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return []


class FakeLLM:
    model_id = "fake-llm"

    def health(self) -> bool:
        return True


class FakeEmbedder:
    model_id = "fake-embedder"


class FakeIngestion:
    def __init__(self) -> None:
        self.last_force: bool | None = None

    def run(self, source_dir: Any, *, force: bool = False) -> IngestReport:
        from datetime import UTC, datetime

        self.last_force = force
        now = datetime.now(UTC)
        return IngestReport(
            documents=(
                DocumentReport("ok.pdf", DocumentStatus.OK, 5, 6),
                DocumentReport("tarama.pdf", DocumentStatus.NO_TEXT_LAYER, 7, 0),
                DocumentReport("bozuk.pdf", DocumentStatus.FAILED, 0, 0, error="açılamadı"),
            ),
            total_chunks_added=6,
            started_at=now,
            finished_at=now,
        )


class FakeLibrary:
    """Sahte doküman kütüphanesi."""

    def __init__(self) -> None:
        from rag_assistant.domain.models import StoredDocument

        self.docs = [
            StoredDocument("ok.pdf", DocumentStatus.OK, 1024, 5, 6, "h1"),
            StoredDocument("tarama.pdf", DocumentStatus.NO_TEXT_LAYER, 2048, 7, 0, "h2"),
        ]
        self.uploaded: list[str] = []
        self.deleted: list[str] = []
        self.raw_dir = "raw"

    def list_documents(self):  # type: ignore[no-untyped-def]
        return list(self.docs)

    def disk_usage_bytes(self) -> int:
        return 3072

    def save_upload(self, file_name, chunks, *, overwrite=False):  # type: ignore[no-untyped-def]
        from rag_assistant.library import UploadResult, validate_upload_name

        # Sahte, GERÇEĞİN kullandığı doğrulamayı çağırır — kendi kopyasını
        # yazsaydı gerçekten sapar ve testler yanlış güven verirdi.
        safe = validate_upload_name(file_name, allowed_extensions=(".pdf",))
        size = sum(len(b) for b in chunks)
        self.uploaded.append(safe)
        return UploadResult(file_name=safe, size_bytes=size, content_hash="h")

    def delete(self, file_name):  # type: ignore[no-untyped-def]
        from rag_assistant.library import DocumentNotFoundError, sanitize_file_name

        safe = sanitize_file_name(file_name)
        for d in self.docs:
            if d.file_name == safe:
                self.deleted.append(safe)
                return d
        raise DocumentNotFoundError(f"Doküman bulunamadı: {safe}")


@dataclass
class FakeSystem:
    settings: Any
    embedder: Any
    store: Any
    retriever: Any
    llm: Any
    answerer: Any
    ingestion: Any
    library: Any


@pytest.fixture
def client_factory():  # type: ignore[no-untyped-def]
    """Sahte sistemle bir TestClient üret."""
    created: list[TestClient] = []

    def build(*, answer: Answer | None = None, index_count: int = 6) -> tuple[TestClient, FakeSystem]:
        from rag_assistant.config import get_settings

        system = FakeSystem(
            settings=get_settings(),
            embedder=FakeEmbedder(),
            store=FakeStore(index_count),
            retriever=FakeRetriever(),
            llm=FakeLLM(),
            answerer=FakeAnswerer(answer or make_answer()),
            ingestion=FakeIngestion(),
            library=FakeLibrary(),
        )
        # lifespan'i atlıyoruz: gerçek modelleri yüklemesin.
        app_module._system = system  # type: ignore[assignment]
        app = create_app()
        app.dependency_overrides[get_system] = lambda: system
        client = TestClient(app)
        created.append(client)
        return client, system

    yield build

    for c in created:
        c.close()
    app_module._system = None


# ---------------------------------------------------------------------------
# Sağlık uçları
# ---------------------------------------------------------------------------
class TestHealth:
    def test_liveness_bagimliliklara_bakmaz(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        LIVENESS yalnızca sürecin yaşadığını söyler. Bağımlılık kontrol
        etseydi, LLM geçici düştüğünde orkestratör sağlıklı süreci
        gereksizce yeniden başlatırdı.
        """
        client, _ = client_factory()
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "alive"}

    def test_readiness_bilesenleri_raporlar(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        d = client.get("/ready").json()
        assert d["ready"] is True
        names = {c["name"] for c in d["components"]}
        assert {"index", "embedder", "llm"} <= names
        assert d["index_vectors"] == 6

    def test_bos_index_hazir_degil(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """Index boşsa sistem trafik almaya hazır değildir."""
        client, _ = client_factory(index_count=0)
        assert client.get("/ready").json()["ready"] is False


# ---------------------------------------------------------------------------
# /ask sözleşmesi
# ---------------------------------------------------------------------------
class TestAsk:
    def test_basarili_cevap(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        r = client.post("/ask", json={"question": "SmartSafe nedir?"})
        assert r.status_code == 200
        d = r.json()
        assert d["answer"] == "Cevap [1]."
        assert d["grounded"] is True
        assert d["citations"][0]["marker"] == 1
        assert d["model"] == "fake-llm"
        assert d["prompt_version"] == "v1"

    def test_chunkin_tam_metni_sizdirilmaz(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        API şemasının domain modelinden ayrı olmasının somut faydası:
        chunk'ın tam metni yerine yalnızca önizleme gönderiyoruz.
        """
        client, _ = client_factory()
        preview = client.post("/ask", json={"question": "soru mu bu"}).json()["sources"][0][
            "preview"
        ]
        assert len(preview) <= 300, "tam metin sızdırıldı"

    def test_uzlasma_bilgisi_disari_verilir(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """retriever_hits, hybrid aramanın çalıştığını gösteren sinyaldir."""
        client, _ = client_factory()
        assert client.post("/ask", json={"question": "soru mu bu"}).json()["sources"][0][
            "retriever_hits"
        ] == 2

    def test_cekimser_cevap(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory(answer=make_answer(abstained=True, citations=0))
        d = client.post("/ask", json={"question": "Titanik ne zaman battı?"}).json()
        assert d["abstained"] is True
        assert d["citations"] == []
        assert d["grounded"] is True, "çekimserlik de geçerli bir grounded durumdur"

    def test_bos_index_409(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        Boş index'te soru sormak sunucu hatası DEĞİL, önkoşul eksikliğidir.
        500 döndürmek hatayı bize aitmiş gibi gösterir ve istemci ne
        yapacağını bilemez.
        """
        client, _ = client_factory(index_count=0)
        r = client.post("/ask", json={"question": "herhangi bir soru"})
        assert r.status_code == 409
        assert "ingest" in r.json()["detail"].lower()

    def test_top_k_paylasilan_durumu_degistirmez(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        top_k istek başına geçilir. Paylaşılan nesnenin alanı güncellenirse
        eşzamanlı isteklerde yarış koşulu oluşur.
        """
        client, system = client_factory()
        client.post("/ask", json={"question": "soru mu bu", "top_k": 3})
        assert system.answerer.last_top_k == 3

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": "ab"},  # çok kısa
            {"question": "x" * 1001},  # çok uzun
            {"question": "gecerli soru", "top_k": 999},  # sunucu üst sınırı
            {"question": "gecerli soru", "top_k": 0},  # alt sınır
            {},  # eksik alan
        ],
    )
    def test_dogrulama_reddeder(self, client_factory, payload) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        assert client.post("/ask", json=payload).status_code == 422

    def test_request_id_dondurulur(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """Korelasyon kimliği olmadan 'bu cevap neden böyle geldi' sorusu cevaplanamaz."""
        client, _ = client_factory()
        r = client.post("/ask", json={"question": "soru mu bu"})
        assert r.headers.get("x-request-id")

    def test_verilen_request_id_korunur(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        r = client.post("/ask", json={"question": "soru mu bu"}, headers={"X-Request-ID": "abc123"})
        assert r.headers["x-request-id"] == "abc123"


# ---------------------------------------------------------------------------
# Akış
# ---------------------------------------------------------------------------
class TestStream:
    def test_sse_bicimi(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        r = client.post("/ask/stream", json={"question": "soru mu bu"})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        assert r.text.rstrip().endswith("data: [DONE]")

    def test_satir_sonlari_kacirilir(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        SSE'de ham satır sonu OLAY SINIRI demektir. Kaçırılmazsa tek bir
        cevap birden fazla olaya bölünür ve istemci bozuk metin gösterir.
        """
        client, _ = client_factory()
        body = client.post("/ask/stream", json={"question": "soru mu bu"}).text
        # Üretilen parçada gerçek "\n" vardı; çıktıda kaçırılmış olmalı.
        assert "\\n" in body
        for line in body.splitlines():
            assert line == "" or line.startswith("data: ")

    def test_bos_index_409(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory(index_count=0)
        assert client.post("/ask/stream", json={"question": "herhangi soru"}).status_code == 409


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------
class TestIngest:
    def test_rapor_donulur_ve_index_kaydedilir(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, system = client_factory()
        d = client.post("/ingest", json={"force": False}).json()
        assert d["chunks_added"] == 6
        assert d["index_total"] == 6
        assert system.store.saved, "index diske kaydedilmedi"

    def test_basarisiz_ve_ocr_ayri_raporlanir(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        Bu iki sayı "başarılı" sayısının içinde kaybolmamalı: taranmış PDF
        sessizce yok sayılırsa doküman aramaya hiç girmez.
        """
        client, _ = client_factory()
        d = client.post("/ingest", json={"force": False}).json()
        assert d["failed_count"] == 1
        assert d["needs_ocr_count"] == 1
        ocr = [doc for doc in d["documents"] if doc["needs_ocr"]]
        assert ocr and ocr[0]["status"] == "no_text_layer"

    def test_force_iletiliyor(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, system = client_factory()
        client.post("/ingest", json={"force": True})
        assert system.ingestion.last_force is True

    def test_force_varsayilani_false(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, system = client_factory()
        client.post("/ingest", json={})
        assert system.ingestion.last_force is False


# ---------------------------------------------------------------------------
# OpenAPI sözleşmesi
# ---------------------------------------------------------------------------
class TestContract:
    def test_openapi_uretilir(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        Makine okunabilir sözleşme: ileride React/Next istemci bundan
        tip üretebilir.
        """
        client, _ = client_factory()
        spec = client.get("/openapi.json").json()
        for path in ("/ask", "/ask/stream", "/ingest", "/health", "/ready"):
            assert path in spec["paths"], f"{path} sözleşmede yok"


# ---------------------------------------------------------------------------
# Doküman yönetimi uç noktaları
# ---------------------------------------------------------------------------
class TestDocumentEndpoints:
    def test_listeleme(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        d = client.get("/documents").json()

        assert d["total"] == 2
        assert d["searchable"] == 1, "0 chunk'lı doküman aranabilir sayılmamalı"
        assert d["needs_ocr"] == 1
        assert d["disk_usage_bytes"] == 3072

    def test_listede_turetilmis_alanlar_var(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        `is_searchable` / `needs_ocr` sunucuda hesaplanır. İstemcinin
        `status` string'ini yorumlayıp aynı mantığı yeniden yazmasını
        istemiyoruz — kural tek yerde kalmalı.
        """
        client, _ = client_factory()
        docs = {x["file_name"]: x for x in client.get("/documents").json()["documents"]}

        assert docs["ok.pdf"]["is_searchable"] is True
        assert docs["tarama.pdf"]["is_searchable"] is False
        assert docs["tarama.pdf"]["needs_ocr"] is True

    def test_yukleme_indekslemeden(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, system = client_factory()
        r = client.post(
            "/documents",
            files={"file": ("yeni.pdf", b"%PDF-1.4 icerik", "application/pdf")},
            data={"index": "false"},
        )
        assert r.status_code == 201
        d = r.json()
        assert d["file_name"] == "yeni.pdf"
        assert d["indexed"] is False
        assert system.library.uploaded == ["yeni.pdf"]

    def test_yukleme_ve_indeksleme(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """Varsayılan: yükle + indeksle. Kullanıcı sonucu ANINDA öğrenmeli."""
        client, _ = client_factory()
        d = client.post(
            "/documents",
            files={"file": ("ok.pdf", b"%PDF veri", "application/pdf")},
        ).json()
        assert d["indexed"] is True
        assert d["chunk_count"] == 6
        assert d["status"] == "ok"

    def test_yuklemede_dizin_asimi_temizlenir(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, system = client_factory()
        d = client.post(
            "/documents",
            files={"file": ("../../../kacti.pdf", b"veri", "application/pdf")},
            data={"index": "false"},
        ).json()
        assert d["file_name"] == "kacti.pdf"
        assert system.library.uploaded == ["kacti.pdf"]

    @pytest.mark.parametrize(
        ("bad_name", "expected_status", "expected_error"),
        [
            ("zararli.exe", 415, "unsupported_file_type"),
            ("CON.pdf", 400, "invalid_file_name"),
            ("rapor<>.pdf", 400, "invalid_file_name"),
        ],
    )
    def test_gecersiz_dosyalar_dogru_koda_eslenir(
        self, client_factory, bad_name, expected_status, expected_error
    ) -> None:  # type: ignore[no-untyped-def]
        """
        Hepsini 500 yapmak "sunucu bozuk" demek olurdu. Bunlar İSTEMCİ
        hatası — istemci ne yapacağını bilebilmeli.
        """
        client, _ = client_factory()
        r = client.post(
            "/documents",
            files={"file": (bad_name, b"veri", "application/octet-stream")},
            data={"index": "false"},
        )
        assert r.status_code == expected_status
        assert r.json()["error"] == expected_error

    def test_silme(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, system = client_factory()
        r = client.delete("/documents/ok.pdf")
        assert r.status_code == 200
        assert system.library.deleted == ["ok.pdf"]

    def test_olmayan_dokuman_404(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        client, _ = client_factory()
        r = client.delete("/documents/yok.pdf")
        assert r.status_code == 404
        assert r.json()["error"] == "document_not_found"

    def test_cors_delete_izinli(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        """
        DELETE, CORS izin listesinde olmazsa tarayıcı ön kontrolde reddeder:
        uç nokta çalışır ama frontend'den ERİŞİLEMEZ.
        """
        client, _ = client_factory()
        r = client.options(
            "/documents/ok.pdf",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        assert "DELETE" in r.headers.get("access-control-allow-methods", "")
