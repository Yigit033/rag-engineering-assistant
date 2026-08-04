"""
Merkezi yapılandırma.

TASARIM KURALI: Kodun hiçbir yerinde "sihirli sayı" olmaz. Chunk boyutu,
model adı, top_k, temperature — hepsi burada, tek yerde, tipli ve
doğrulanmış olarak durur.

NEDEN BU KADAR ÖNEMLİ:
  Bir RAG sisteminin kalitesi bir avuç sayıya bağlıdır (chunk boyutu,
  fetch_k, top_k, temperature). Bunlar kodun içine dağılmışsa:
    * Neyi değiştirdiğini bilemezsin  → ölçüm anlamsızlaşır
    * Deney yapamazsın                → iyileştirme tesadüfe kalır
    * Ortam bazlı ayar yapamazsın     → prod ile lokal ayrışır
  Tek yerde toplandığında ise her ayar bir DENEY DEĞİŞKENİ olur.

Ortam değişkeniyle geçersiz kılma (nokta ayırıcı `__`):
    RAG_RETRIEVAL__TOP_K=5
    RAG_LLM__MODEL=qwen3:8b
    RAG_EMBEDDING__DEVICE=cuda
veya proje kökündeki `.env` dosyasıyla.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Proje kökü: config.py → rag_assistant → src → <kök>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PathSettings(BaseModel):
    """Dosya sistemi düzeni. Tüm yollar tek yerden türer."""

    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        """Kaynak dokümanlar (PDF)."""
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        """FAISS index + chunk deposu."""
        return self.data_dir / "index"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def eval_dir(self) -> Path:
        """Golden set ve değerlendirme çıktıları."""
        return self.data_dir / "eval"

    def ensure(self) -> None:
        for d in (self.raw_dir, self.index_dir, self.logs_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)


class EmbeddingSettings(BaseModel):
    """
    Embedding modeli ayarları.

    `model`: BAAI/bge-m3 — çok dilli (100+ dil), 1024 boyut, 8192 token bağlam.
      Türkçe için seçildi: İngilizce-only modeller Türkçe diyakritikleri
      (ç, ş, ğ, ı, ö, ü) tokenizasyon aşamasında yok eder ve anlamsal uzay
      çöker.

    `device`: "cpu" — GPU bilinçli olarak LLM'e (Ollama) bırakıldı.
      8 GB VRAM'de embedder + reranker + LLM birlikte sıkışır. Korpus küçük
      olduğu için CPU tarafı darboğaz değil.
    """

    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 8
    # Modelin kendi limiti 8192; bilinçli olarak daha düşük tutuyoruz çünkü
    # çok uzun chunk arama hassasiyetini düşürür (bkz. chunking ödünleşimi).
    max_tokens: int = 1024
    normalize: bool = True

    # Asimetrik retrieval önekleri. bge-m3 önek gerektirmez (boş bırakıldı),
    # ama E5 ailesine geçilirse "query: " / "passage: " buraya yazılır.
    # Önek mantığının yapılandırmada olması, model değişiminde kod
    # değiştirmeyi gereksiz kılar.
    query_prefix: str = ""
    document_prefix: str = ""


class ChunkingSettings(BaseModel):
    """
    Chunk ayarları — TOKEN cinsinden.

    Karakter değil token: embedding modelinin limiti token cinsindendir.
    Karakterle ölçmek, özellikle Türkçe gibi token verimliliği düşük
    dillerde (karakter/token ≈ 2.5, İngilizce'de ≈ 6) chunk'ın sessizce
    kesilmesine yol açar.
    """

    target_tokens: int = 400
    overlap_tokens: int = 80  # ~%20
    min_tokens: int = 32  # bundan küçük parçalar gürültüdür, atılır

    @model_validator(mode="after")
    def _validate(self) -> ChunkingSettings:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens, target_tokens'dan küçük olmalı")
        return self


class RetrievalSettings(BaseModel):
    """
    Retrieval stratejisi.

    `fetch_k` > `top_k` olması TASARIM GEREĞİ:
      fetch_k → recall (doğru cevabı adaylar arasına al)
      top_k   → precision (LLM'e az ve isabetli chunk ver)
      Doğrudan top_k kadar aday çekmek geri alınamaz bir recall kaybıdır;
      hiç aday olmayan bir chunk'ı reranker kurtaramaz.
    """

    fetch_k: int = 20  # her retriever'dan çekilecek aday
    top_k: int = 5  # LLM'e gidecek nihai chunk
    rrf_k: int = 60  # RRF yumuşatma sabiti (literatürdeki ampirik değer)

    use_dense: bool = True
    use_sparse: bool = True

    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    # Reranker skoru bu eşiğin altındaysa chunk ilgisiz sayılır ve atılır.
    # None = filtreleme yok. Eşik ampirik olarak golden set üzerinde bulunur.
    reranker_min_score: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> RetrievalSettings:
        if not (self.use_dense or self.use_sparse):
            raise ValueError("en az bir retriever açık olmalı")
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k >= top_k olmalı (geniş tara, sonra daralt)")
        return self


class LLMSettings(BaseModel):
    """
    LLM ayarları.

    `temperature = 0.0`: bilgi çıkarma işlerinde varsayılan budur.
      Yüksek temperature aynı soruya her seferinde farklı cevap üretir →
      hata ayıklanamaz ve değerlendirilemez bir sistem.
    """

    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = 1024
    timeout_seconds: float = 120.0
    # Modelin bağlam penceresi. Ollama'ya num_ctx olarak geçer.
    context_window: int = 8192


class GenerationSettings(BaseModel):
    """
    Cevap üretimi ve halüsinasyon kontrolü.

    `prompt_version`: prompt'lar koda gömülü değil, `prompts/` altında
      versiyonlu dosyalar. Neden: prompt bir MODEL PARAMETRESİDİR. Onu
      değiştirmek sistemin davranışını değiştirir, dolayısıyla sürümlenmesi
      ve değerlendirmede kaydedilmesi gerekir. Koda gömülü prompt, hangi
      sürümün hangi skoru ürettiğini bilmeni imkânsız kılar.
    """

    prompt_version: str = "v1"
    # Model çekimser kaldığında kullandığı ifade. Cevabın "abstain" olup
    # olmadığını tespit etmek için de bu kullanılır.
    abstain_phrase: str = "Bu bilgi verilen dokümanlarda yok"
    require_citations: bool = True
    # Hiç aday bulunamazsa LLM'i hiç çağırmadan çekimser cevap dön.
    # Boş bağlamla model çağırmak, uydurma için davetiyedir.
    abstain_when_no_context: bool = True
    # Cevabın bağlama dayanma oranını ölç (ek LLM çağrısı gerektirir).
    check_groundedness: bool = False


class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:8501",)


class LogSettings(BaseModel):
    level: str = "INFO"
    # Prod'da JSON (makine okunur), lokalde renkli konsol.
    json_format: bool = False


class Settings(BaseSettings):
    """Tüm uygulama yapılandırması."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathSettings = PathSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()
    generation: GenerationSettings = GenerationSettings()
    api: APISettings = APISettings()
    log: LogSettings = LogSettings()

    @model_validator(mode="after")
    def _cross_validate(self) -> Settings:
        """
        Katmanlar arası tutarlılık.

        Chunk hedefi embedding limitini aşamaz — bu, RAG'de en sık yapılan
        ve en zor fark edilen hatadır. Burada bir kez kontrol edilir ve
        uygulama açılışta patlar (sessizce yanlış çalışmak yerine).
        """
        if self.chunking.target_tokens > self.embedding.max_tokens:
            raise ValueError(
                f"chunking.target_tokens ({self.chunking.target_tokens}) > "
                f"embedding.max_tokens ({self.embedding.max_tokens}): "
                "chunk'lar sessizce kesilir."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Uygulama genelinde tek ayar örneği.

    `lru_cache`: ayarlar bir kez okunur. Ayrıca FastAPI'de doğrudan
    `Depends(get_settings)` olarak kullanılabilir ve testte
    `get_settings.cache_clear()` ile sıfırlanabilir.
    """
    return Settings()
