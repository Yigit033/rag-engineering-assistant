"""
FastAPI uygulaması.

SENIOR AYRINTILAR — HER BİRİ BİLİNÇLİ BİR KARAR:

1. `lifespan` İLE TEK SEFERLİK YÜKLEME
   Modeller uygulama açılışında bir kez yüklenir, her istekte değil.
   Bu olmadan her soru 15+ saniye model yükleme bekler.

2. UÇ NOKTALAR `async def` DEĞİL, `def`
   Bu kasıtlı ve en sık yapılan FastAPI hatasının çözümü:
   `async def` içinde BLOKLAYAN iş (model çıkarımı, FAISS aramasi) yapmak
   olay döngüsünü (event loop) kilitler ve TÜM sunucuyu durdurur — tek bir
   soru işlenirken sağlık kontrolü bile cevap vermez.
   Senkron `def` tanımlandığında FastAPI fonksiyonu otomatik olarak bir iş
   parçacığı havuzunda çalıştırır; olay döngüsü serbest kalır.
   Kural: işin kendisi async değilse, uç noktayı da async yapma.

3. HER İSTEĞE KORELASYON KİMLİĞİ (request_id)
   Log satırları isteğe bağlanabilir olmalı. "Şu cevap neden böyle geldi?"
   sorusu ancak o isteğin tüm log satırlarını bir arada görebiliyorsan
   cevaplanabilir.

4. LIVENESS ≠ READINESS
   `/health` süreç ayakta mı der (yeniden başlatma kararı için),
   `/ready` trafik alabilir mi der (modeller yüklü, index dolu mu).

5. HATALAR SIZDIRMAZ
   Beklenmeyen hatalar tek biçimli bir cevaba dönüştürülür; yığın izi
   istemciye gitmez (dosya yolları ve iç yapı sızdırır), loga gider.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from rag_assistant.composition import RagSystem, build_rag_system
from rag_assistant.config import Settings, get_settings
from rag_assistant.generation.factory import LLMConfigurationError
from rag_assistant.generation.llm import LLMError, LLMUnavailableError
from rag_assistant.indexing.store import IndexCompatibilityError
from rag_assistant.library import (
    DocumentExistsError,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidFileNameError,
    UnsupportedFileTypeError,
)
from rag_assistant.observability import configure_logging, get_logger

from .schemas import (
    AskRequest,
    AskResponse,
    ComponentHealth,
    DeleteResponse,
    DocumentListResponse,
    DocumentOut,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    UploadResponse,
)

logger = get_logger(__name__)

# Uygulama durumu. Modül düzeyinde tutuluyor çünkü `lifespan` tarafından
# doldurulup bağımlılıklar tarafından okunuyor.
_system: RagSystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Açılış ve kapanış.

    Modeller BURADA yüklenir — ilk istekte değil. Yükleme sırası
    `composition.build_rag_system` içinde sabitlenmiştir (LLM önce ısıtılır).
    """
    global _system
    settings = get_settings()
    configure_logging(level=settings.log.level, json_format=settings.log.json_format)

    logger.info("api.starting", host=settings.api.host, port=settings.api.port)
    try:
        _system = build_rag_system(settings, warm_llm=True)
    except LLMConfigurationError as exc:
        # Yapılandırma hatasıyla açılmak, ilk istekte 500 vermekten iyidir:
        # sorun anında ve eyleme dönüştürülebilir biçimde görünür.
        logger.error("api.startup_failed", reason=str(exc))
        raise

    logger.info("api.ready", index_vectors=_system.store.count)
    try:
        yield
    finally:
        logger.info("api.stopping")
        _system = None


def get_system() -> RagSystem:
    """Kurulmuş sistemi ver. Hazır değilse 503 döndür."""
    if _system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sistem henüz hazır değil (modeller yükleniyor).",
        )
    return _system


SystemDep = Annotated[RagSystem, Depends(get_system)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def create_app() -> FastAPI:
    """Uygulama fabrikası — testte ayrı bir örnek kurmayı mümkün kılar."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Mühendislik Asistanı",
        description=(
            "Türkçe teknik dokümanlar üzerinde kaynak gösteren soru-cevap. "
            "Cevaplar yalnızca verilen dokümanlara dayanır; bilgi yoksa sistem "
            "çekimser kalır."
        ),
        version="0.1.0",
        lifespan=lifespan,
        responses={500: {"model": ErrorResponse}},
    )

    # CORS: yalnızca bilinen kaynaklar. `*` ile kimlik bilgisi birlikte
    # kullanılamaz (tarayıcı reddeder) ve zaten güvensizdir.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api.cors_origins),
        allow_credentials=True,
        # DELETE eklenmezse tarayıcı /documents/{ad} silme isteğini
        # ön kontrolde (preflight) reddeder — uç nokta çalışır ama
        # frontend'den ERİŞİLEMEZ. Yöntem listesi uç noktalarla birlikte
        # güncellenmek zorunda.
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def correlate_requests(request: Request, call_next: Callable) -> JSONResponse:
        """
        Her isteğe bir kimlik ata ve log bağlamına yaz.

        Bu sayede o isteğe ait TÜM log satırları (retrieval, llm, answer)
        aynı `request_id` ile işaretlenir ve bir arada filtrelenebilir.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response

    _register_error_handlers(app)
    _register_routes(app)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Alan hatalarını doğru HTTP durum koduna eşle."""

    # ---- Doküman kütüphanesi hataları ----
    # Her biri DOĞRU HTTP koduna eşlenir. Hepsini 500 yapmak, istemciye
    # "sunucu bozuk" demek olurdu; oysa bunlar İSTEMCİ hatalarıdır ve
    # istemci ne yapacağını bilebilir.
    _LIBRARY_STATUS: dict[type[Exception], tuple[int, str]] = {
        InvalidFileNameError: (status.HTTP_400_BAD_REQUEST, "invalid_file_name"),
        UnsupportedFileTypeError: (status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported_file_type"),
        FileTooLargeError: (413, "file_too_large"),  # CONTENT_TOO_LARGE
        DocumentExistsError: (status.HTTP_409_CONFLICT, "document_exists"),
        DocumentNotFoundError: (status.HTTP_404_NOT_FOUND, "document_not_found"),
    }

    for _exc_type, (_code, _slug) in _LIBRARY_STATUS.items():

        def _handler(request: Request, exc: Exception, _code: int = _code, _slug: str = _slug):  # noqa: ANN202
            logger.warning("api.library_error", error=_slug, detail=str(exc)[:200])
            return JSONResponse(
                status_code=_code,
                content=ErrorResponse(
                    error=_slug,
                    detail=str(exc),
                    request_id=request.headers.get("X-Request-ID"),
                ).model_dump(),
            )

        app.add_exception_handler(_exc_type, _handler)

    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(request: Request, exc: LLMUnavailableError) -> JSONResponse:
        # 503: geçici — istemci sonra tekrar denemeli.
        logger.error("api.llm_unavailable", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="llm_unavailable",
                detail="Dil modeli servisine ulaşılamıyor. Lütfen sonra tekrar deneyin.",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )

    @app.exception_handler(LLMError)
    async def _llm_error(request: Request, exc: LLMError) -> JSONResponse:
        # 502: yukarı akış (upstream) servis hatası.
        logger.error("api.llm_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error="llm_error",
                detail="Cevap üretilirken hata oluştu.",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )

    @app.exception_handler(IndexCompatibilityError)
    async def _index_error(request: Request, exc: IndexCompatibilityError) -> JSONResponse:
        logger.error("api.index_incompatible", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="index_incompatible",
                detail="Index mevcut yapılandırmayla uyumsuz; yeniden oluşturulmalı.",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Yığın izi LOGA gider, istemciye GİTMEZ: dosya yolları ve iç yapı
        # sızdırır. İstemciye korelasyon kimliği verilir ki destek isteğinde
        # ilgili loglar bulunabilsin.
        logger.exception("api.unexpected_error", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                detail="Beklenmeyen bir hata oluştu.",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )


def _register_routes(app: FastAPI) -> None:
    # ------------------------------------------------------------------
    @app.get("/health", tags=["sistem"], summary="Süreç ayakta mı? (liveness)")
    def health() -> dict[str, str]:
        """
        LIVENESS: yalnızca sürecin yaşadığını söyler.

        Bilinçli olarak hiçbir bağımlılığı kontrol etmez. Aksi halde LLM
        geçici olarak düştüğünde orkestratör (Docker/Kubernetes) sağlıklı
        bir süreci gereksizce yeniden başlatır.
        """
        return {"status": "alive"}

    @app.get(
        "/ready",
        tags=["sistem"],
        response_model=HealthResponse,
        summary="Trafik alabilir mi? (readiness)",
    )
    def ready(system: SystemDep) -> HealthResponse:
        """READINESS: modeller yüklü ve index dolu mu?"""
        llm_ok = bool(getattr(system.llm, "health", lambda: True)())
        index_ok = system.store.count > 0

        components = [
            ComponentHealth(
                name="index",
                ok=index_ok,
                detail=(
                    f"{system.store.count} vektör"
                    if index_ok
                    else "boş — önce POST /ingest çağırın"
                ),
            ),
            ComponentHealth(
                name="embedder", ok=True, detail=system.embedder.model_id
            ),
            ComponentHealth(
                name="llm",
                ok=llm_ok,
                detail=system.llm.model_id if llm_ok else "ulaşılamıyor",
            ),
        ]

        # Reranker İSTEĞE BAĞLI: yüklenemese bile sistem hazır sayılır.
        # Yine de durumu görünür kılıyoruz — sessizce kaybolmuş bir kalite
        # katmanı, en zor fark edilen gerileme (regression) biçimidir.
        reranker = getattr(system.retriever, "reranker", None)
        if reranker is not None:
            active = bool(getattr(reranker, "is_active", True))
            components.append(
                ComponentHealth(
                    name="reranker (isteğe bağlı)",
                    ok=True,  # hazırlık durumunu ETKİLEMEZ
                    detail=(
                        reranker.model_id
                        if active
                        else f"devre dışı — {reranker.model_id} belleğe sığmadı, "
                        "RRF sıralaması kullanılıyor"
                    ),
                )
            )

        return HealthResponse(
            ready=all(c.ok for c in components),
            components=components,
            index_vectors=system.store.count,
            embedder_model=system.embedder.model_id,
            llm_model=system.llm.model_id,
            retrieval_strategy=system.retriever.name,
        )

    # ------------------------------------------------------------------
    @app.post(
        "/ask",
        tags=["soru-cevap"],
        response_model=AskResponse,
        summary="Soru sor (kaynak gösteren cevap)",
    )
    def ask(payload: AskRequest, system: SystemDep) -> AskResponse:
        """
        NOT `async def` — bilinçli.
        Model çıkarımı bloklayan bir iştir; `async def` içinde yapılırsa olay
        döngüsünü kilitler ve tüm sunucu tek soru boyunca durur. Senkron
        tanımlandığında FastAPI bunu iş parçacığı havuzunda çalıştırır.
        """
        if system.store.count == 0:
            # Boş index'te soru sormak bir sunucu hatası değil, bir önkoşul
            # eksikliğidir → 409, 500 değil.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Index boş. Önce POST /ingest ile doküman ekleyin.",
            )

        # top_k istek başına geçilir; paylaşılan answerer'ın durumu
        # DEĞİŞTİRİLMEZ (eşzamanlı isteklerde yarış koşulu olurdu).
        answer = system.answerer.answer(payload.question, top_k=payload.top_k)
        return AskResponse.from_domain(answer)

    @app.post(
        "/ask/stream",
        tags=["soru-cevap"],
        summary="Soru sor (token token akış)",
        response_class=StreamingResponse,
    )
    def ask_stream(payload: AskRequest, system: SystemDep) -> StreamingResponse:
        """
        Server-Sent Events ile akış.

        Toplam süre aynı olsa bile ilk token'ın erken gelmesi algılanan hızı
        belirgin değiştirir. Atıf doğrulama akış sonrası yapılabilir; akış
        sırasında metnin tamamı henüz yoktur.
        """
        if system.store.count == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Index boş. Önce POST /ingest ile doküman ekleyin.",
            )

        def event_stream() -> Iterator[str]:
            try:
                for piece in system.answerer.stream(
                    payload.question, top_k=payload.top_k
                ):
                    # SSE biçimi: satır sonları kaçırılmalı, yoksa olay
                            # sınırı bozulur.
                    yield f"data: {piece.replace(chr(10), '\\n')}\n\n"
            except (LLMError, LLMUnavailableError) as exc:
                logger.error("api.stream_failed", error=str(exc))
                yield 'data: [HATA] Cevap üretilemedi.\n\n'
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Ara sunucuların (nginx) akışı tamponlamasını engelle;
                # aksi halde streaming'in tüm faydası kaybolur.
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    @app.get(
        "/documents",
        tags=["doküman"],
        response_model=DocumentListResponse,
        summary="Kütüphanedeki dokümanları listele",
    )
    def list_documents(system: SystemDep) -> DocumentListResponse:
        documents = system.library.list_documents()
        return DocumentListResponse(
            documents=[DocumentOut.from_domain(d) for d in documents],
            total=len(documents),
            searchable=sum(1 for d in documents if d.is_searchable),
            needs_ocr=sum(1 for d in documents if d.needs_ocr),
            index_vectors=system.store.count,
            disk_usage_bytes=system.library.disk_usage_bytes(),
        )

    @app.post(
        "/documents",
        tags=["doküman"],
        response_model=UploadResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Doküman yükle (ve isteğe bağlı indeksle)",
    )
    def upload_document(
        system: SystemDep,
        file: Annotated[UploadFile, File(description="PDF dosyası")],
        index: Annotated[bool, Form()] = True,
        overwrite: Annotated[bool, Form()] = False,
    ) -> UploadResponse:
        """
        Dosyayı yükle, isteğe bağlı olarak hemen indeksle.

        AKIŞ HALİNDE OKUMA: dosya belleğe TOPTAN alınmıyor. 8 KB'lık
        bloklar hâlinde okunup diske yazılıyor ve bayt sayısı yazarken
        sayılıyor. `Content-Length` başlığına güvenilmez — istemci yalan
        söyleyebilir; sınır gerçek yazılan bayta göre uygulanır.

        `index=True` (varsayılan): yükleme sonrası indeksleme senkron
        çalışır. Kullanıcı sonucu ANINDA öğrenir — dosya aranabilir mi,
        yoksa taranmış olduğu için OCR mı gerekiyor? Arka plan görevine
        atsaydık istemci "yüklendi" cevabını alır ama dosyanın işe yarayıp
        yaramadığını bilemezdi.
        """

        def read_chunks() -> Iterator[bytes]:
            while block := file.file.read(8192):
                yield block

        result = system.library.save_upload(
            file.filename or "", read_chunks(), overwrite=overwrite
        )

        warning = (
            f"Aynı içerik '{result.duplicate_of}' adıyla zaten var — "
            "aynı bilgi arama sonuçlarında iki kez çıkabilir."
            if result.duplicate_of
            else None
        )

        if not index:
            return UploadResponse(
                file_name=result.file_name,
                size_bytes=result.size_bytes,
                indexed=False,
                status="uploaded",
                duplicate_of=result.duplicate_of,
                warning=warning,
            )

        report = system.ingestion.run(system.library.raw_dir)
        system.store.save(system.settings.paths.index_dir)

        uploaded = next(
            (d for d in report.documents if d.file_name == result.file_name), None
        )
        return UploadResponse(
            file_name=result.file_name,
            size_bytes=result.size_bytes,
            indexed=True,
            chunk_count=uploaded.chunk_count if uploaded else 0,
            status=str(uploaded.status) if uploaded else "unknown",
            needs_ocr=bool(uploaded and uploaded.needs_ocr),
            duplicate_of=result.duplicate_of,
            warning=warning,
        )

    @app.delete(
        "/documents/{file_name}",
        tags=["doküman"],
        response_model=DeleteResponse,
        summary="Dokümanı ve chunk'larını sil",
    )
    def delete_document(file_name: str, system: SystemDep) -> DeleteResponse:
        """
        Dokümanı index'ten, manifest'ten ve diskten siler.

        Üçü birden yapılmazsa sistem tutarsız kalır: silinmiş bir
        dokümandan alıntı yapan cevaplar üretilir ve kullanıcı kaynağı
        bulamaz.
        """
        before = system.store.count
        doc = system.library.delete(file_name)
        return DeleteResponse(
            file_name=doc.file_name,
            removed_chunks=before - system.store.count,
            index_vectors=system.store.count,
        )

    @app.post(
        "/ingest",
        tags=["doküman"],
        response_model=IngestResponse,
        summary="data/raw klasörünü indeksle",
    )
    def ingest(payload: IngestRequest, system: SystemDep) -> IngestResponse:
        """
        Kaynak klasörü indeksle (idempotent).

        Değişmemiş dosyalar atlanır. `force=true` ile tümü yeniden işlenir —
        chunk ayarları değiştiğinde gerekir, çünkü bu değişiklik dosya
        içeriğine yansımaz ve hash aynı kalır.
        """
        report = system.ingestion.run(
            system.settings.paths.raw_dir, force=payload.force
        )
        system.store.save(system.settings.paths.index_dir)
        return IngestResponse.from_domain(report, index_total=system.store.count)


app = create_app()
