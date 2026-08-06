"""
Kompozisyon kökü (composition root).

BU DOSYA NE İŞE YARAR:
  Tüm gerçek bileşenler (embedder, store, retriever'lar, reranker, LLM)
  TEK BİR YERDE kurulur ve birbirine bağlanır. Başka hiçbir modül "hangi
  implementasyonu kullanacağım?" sorusunu sormaz; herkes protokolle çalışır.

  Bu desenin adı Dependency Injection ve faydası şu: uygulamanın bağımlılık
  grafiği tek bir dosyadan okunabilir. Bir bileşeni değiştirmek istediğinde
  aramak zorunda kalmazsın — burada olduğunu bilirsin.

  API, CLI ve değerlendirme betikleri aynı fabrikayı kullanır. Böylece
  "Streamlit'te çalışıyor ama API'de farklı davranıyor" sınıfı hatalar
  ortadan kalkar: üçü de birebir aynı nesne grafiğini kurar.

YÜKLEME SIRASI TESADÜF DEĞİL (ölçülmüş kısıt):
  1. LLM ısıtılır  → Ollama modeli belleğe/VRAM'e alır ve keep_alive ile tutar
  2. Embedder yüklenir (2.3 GB)
  3. Reranker (tembel — ilk kullanımda)

  Ters sırada, bellek kısıtlı bir makinede LLM'e yer kalmıyor:
      "model requires more system memory (3.8 GiB) than is available (1.1 GiB)"
  Sıra düzeltilince ikisi bir arada çalışıyor. Bu yüzden sıra bir yorum
  satırı değil, kodun sözleşmesidir.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.config import Settings
from rag_assistant.domain.protocols import LLM, Embedder, Retriever
from rag_assistant.generation.answerer import GroundedAnswerer
from rag_assistant.generation.factory import build_llm, preflight
from rag_assistant.generation.prompt import PromptLibrary
from rag_assistant.indexing.embedder import SentenceTransformerEmbedder
from rag_assistant.indexing.store import FaissVectorStore
from rag_assistant.ingestion.chunker import TokenAwareChunker
from rag_assistant.ingestion.manifest import IngestManifest
from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.library import DocumentLibrary
from rag_assistant.observability import get_logger
from rag_assistant.retrieval.pipeline import HybridRetriever
from rag_assistant.retrieval.rerank import CrossEncoderReranker
from rag_assistant.retrieval.retrievers import BM25Retriever, DenseRetriever

logger = get_logger(__name__)


@dataclass(slots=True)
class RagSystem:
    """
    Kurulmuş RAG sistemi — tüm bileşenler hazır.

    Bu nesne uygulama ömrü boyunca TEK BİR KEZ oluşturulur. Modeller pahalı
    olduğu için her istekte yeniden kurulması söz konusu değildir; API
    tarafında `lifespan` bunu garanti eder.
    """

    settings: Settings
    embedder: Embedder
    store: FaissVectorStore
    retriever: Retriever
    llm: LLM
    answerer: GroundedAnswerer
    ingestion: IngestionPipeline
    library: DocumentLibrary

    @property
    def is_ready(self) -> bool:
        """Sistem soru cevaplayabilir durumda mı?"""
        return self.store.count > 0


def build_rag_system(settings: Settings, *, warm_llm: bool = True) -> RagSystem:
    """
    Tüm bileşenleri kur ve bağla.

    Args:
        warm_llm: LLM'i açılışta ısıt. Yalnızca ingestion yapılacaksa
            (LLM gerekmez) kapatılabilir — 3.5 GB gereksiz bellek ayırmamak
            için.
    """
    settings.paths.ensure()

    # ---- 1. LLM: ÖNCE ısıtılır (bellek sırası kritik, bkz. modül başlığı)
    llm = build_llm(settings.llm)
    if warm_llm:
        preflight(settings.llm).raise_if_failed()

    # ---- 2. Embedder (pahalı: ~2.3 GB)
    embedder = SentenceTransformerEmbedder(
        settings.embedding.model,
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
        max_tokens=settings.embedding.max_tokens,
        query_prefix=settings.embedding.query_prefix,
        document_prefix=settings.embedding.document_prefix,
        normalize=settings.embedding.normalize,
    )

    # ---- 3. Vektör deposu (varsa yüklenir; model uyumsuzsa açıkça hata verir)
    store = FaissVectorStore.open_or_create(
        settings.paths.index_dir,
        dimension=embedder.dimension,
        embedder_model_id=embedder.model_id,
    )

    # ---- 4. Retriever'lar + birleştirme + (tembel) reranking
    retrievers: list[Retriever] = []
    if settings.retrieval.use_dense:
        retrievers.append(DenseRetriever(embedder, store))
    if settings.retrieval.use_sparse:
        retrievers.append(BM25Retriever(store))

    reranker = (
        CrossEncoderReranker(
            settings.retrieval.reranker_model,
            device=settings.retrieval.reranker_device,
            min_score=settings.retrieval.reranker_min_score,
        )
        if settings.retrieval.use_reranker
        else None
    )

    retriever = HybridRetriever(
        retrievers,
        fetch_k=settings.retrieval.fetch_k,
        rrf_k=settings.retrieval.rrf_k,
        reranker=reranker,
    )

    # ---- 5. Cevap üretimi
    prompts = PromptLibrary(settings.generation.prompt_version)
    answerer = GroundedAnswerer(
        retriever=retriever,
        llm=llm,
        prompts=prompts,
        top_k=settings.retrieval.top_k,
        abstain_phrase=settings.generation.abstain_phrase,
        abstain_when_no_context=settings.generation.abstain_when_no_context,
        require_citations=settings.generation.require_citations,
        check_groundedness=settings.generation.check_groundedness,
        # Bağlam bütçesi, modelin penceresinin yarısını geçmez: geri kalanı
        # prompt talimatları ve cevabın kendisi için gerekli.
        max_context_tokens=settings.llm.context_window // 2,
    )

    # ---- 6. Ingestion hattı
    chunker = TokenAwareChunker(
        embedder.count_tokens,
        target_tokens=settings.chunking.target_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
        min_tokens=settings.chunking.min_tokens,
        hard_limit_tokens=settings.embedding.max_tokens,
    )
    # Manifest TEK örnek: hem ingestion hem library aynı defteri kullanır.
    # İki ayrı örnek olsaydı biri kayıt eklerken diğeri eski hâli tutar ve
    # silme işlemi manifest'te iz bırakırdı.
    manifest = IngestManifest(settings.paths.index_dir)

    ingestion = IngestionPipeline(
        embedder=embedder,
        chunker=chunker,
        store=store,
        manifest=manifest,
    )

    library = DocumentLibrary(settings, store=store, manifest=manifest)

    logger.info(
        "system.built",
        embedder=embedder.model_id,
        llm=llm.model_id,
        strategy=retriever.name,
        index_vectors=store.count,
        prompt_version=prompts.version,
    )

    return RagSystem(
        settings=settings,
        embedder=embedder,
        store=store,
        retriever=retriever,
        llm=llm,
        answerer=answerer,
        ingestion=ingestion,
        library=library,
    )
