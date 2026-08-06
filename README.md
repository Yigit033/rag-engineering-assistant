# RAG Mühendislik Asistanı

Türkçe teknik dokümanlar üzerinde **kaynak gösteren** soru-cevap sistemi.
Cevaplar yalnızca verilen dokümanlara dayanır; bilgi yoksa sistem **çekimser
kalır** — uydurmaz.

Orkestrasyon framework'ü (LangChain vb.) **kullanılmaz**. Her katman kendi
`Protocol` sözleşmesinin arkasındadır; bileşen değiştirmek tek satır
yapılandırmadır.

---

## Ölçülmüş kalite

Sistemin kalitesi tahmin edilmez, **ölçülür**. 17 soruluk golden set
(12 cevaplanabilir + 5 kasıtlı cevaplanamaz):

| | |
|---|---|
| Hit@k / Recall@k | %100 / %100 |
| MRR | %83,3 |
| Çekimserlik doğruluğu | %94,1 |
| **Uydurma oranı** | **%5,9** |
| Atıf oranı / doğruluğu | %100 / %91,7 |
| Olgu doğruluğu | %100 |
| **Geçme oranı** | **%88,2** |

Koşu: `bge-m3` · hybrid (dense+BM25) · `top_k=3` · `gemma3:4b` · prompt v1 ·
reranker kapalı. Ayarlar değişirse sonuçlar karşılaştırılamaz — bu yüzden her
koşu kendi ayar anlık görüntüsüyle kaydedilir.

Ölçüm günlüğü, başarısız deneyler ve açık sorunlar:
[`src/rag_assistant/generation/prompts/README.md`](src/rag_assistant/generation/prompts/README.md)

---

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

LLM için [Ollama](https://ollama.com) ve bir model:

```bash
ollama pull gemma3:4b
```

Uzak/bulut bir uç de kullanılabilir (kod değişmez):

```bash
RAG_LLM__PROVIDER=openai_compat
RAG_LLM__BASE_URL=https://<saglayici>/v1
RAG_LLM__MODEL=<model-adi>
RAG_LLM_API_KEY=<anahtar>
```

---

## Kullanım

PDF'leri `data/raw/` içine koy, sonra:

```bash
rag-ingest                              # indeksle (idempotent)
rag-ingest --force                      # chunk ayarları değiştiyse
rag-docs list                           # kütüphaneyi listele
rag-docs add rapor.pdf                  # dosya ekle + indeksle
rag-docs delete rapor.pdf               # dosyayı ve chunk'larını sil
rag-ask "SmartSafe kaç PPE sınıfı tespit ediyor?"
rag-ask "..." --show-context --stream
rag-eval                                # golden set üzerinde ölç
rag-eval --fail-under 0.85              # CI kapısı
```

HTTP API:

```bash
uvicorn rag_assistant.api.app:app --reload
```

| Uç nokta | İş |
|---|---|
| `POST /ask` | Kaynak gösteren cevap |
| `POST /ask/stream` | Token token akış (SSE) |
| `POST /documents` | Doküman yükle (+ indeksle) |
| `GET /documents` | Kütüphaneyi listele |
| `DELETE /documents/{ad}` | Dokümanı ve chunk'larını sil |
| `POST /ingest` | `data/raw/` klasörünü indeksle |
| `GET /health` | Süreç ayakta mı (liveness) |
| `GET /ready` | Trafik alabilir mi (readiness) |
| `GET /docs` | OpenAPI arayüzü |

---

## Mimari

```
src/rag_assistant/
├── domain/          Saf veri modelleri + Protocol sözleşmeleri
│                    (SIFIR üçüncü parti bağımlılık — en içteki katman)
├── config.py        Tüm ayarlar tek yerde, katmanlar arası doğrulamayla
├── composition.py   Kompozisyon kökü: bağımlılık grafiği tek dosyada
├── ingestion/       PDF → yapısı korunmuş metin → token-bazlı chunk
├── indexing/        bge-m3 embedder · FAISS IndexFlatIP store
├── retrieval/       TR tokenizer+stemmer · dense · BM25 · RRF · rerank
├── generation/      Versiyonlu promptlar · LLM · kaynak gösteren cevap
├── library.py       Doküman yaşam döngüsü (yükleme güvenliği, silme tutarlılığı)
├── evaluation/      Golden set · metrikler · teşhis raporu
├── api/             FastAPI (HTTP şeması domain'den ayrı)
└── cli.py           rag-ingest · rag-ask · rag-eval
```

Bileşen değiştirmek tek satır: `Embedder`, `VectorStore`, `Retriever`,
`Reranker`, `LLM` birer protokoldür. FAISS'ten Qdrant'a geçmek yeni bir
implementasyon yazmaktır; retrieval katmanı değişmez.

### Halüsinasyon kontrolü — beş katman

1. Bağlam boşsa LLM **hiç çağrılmaz** (boş bağlam = uydurma davetiyesi)
2. Prompt'ta **açık çekimser kalma izni**
3. Numaralı bağlam + **zorunlu atıf**
4. **Atıf doğrulama** — var olmayan numaraya yapılan atıf ayıklanır
5. İsteğe bağlı **groundedness** denetimi

### Ölçülmüş tasarım kararları

| Karar | Gerekçe |
|---|---|
| Chunk kimliği **içerikten** türer | Nesne kimliği kullanmak RRF birleştirmesini sessizce bozar |
| Chunk boyutu **token** cinsinden | Karakterle ölçmek model limitini aşar, metnin sonu sessizce silinir |
| `IndexFlatIP` + normalize vektör | Skor doğrudan cosine; uydurma dönüşüm yok |
| Kalıcılık **JSON**, pickle değil | Pickle açmak kod çalıştırmaktır |
| Index'e **model kimliği** yazılır | Model değişirse açılışta hata; sessiz çöp sonuç yok |
| Türkçe stemmer **yinelemeli** | Tek geçiş asimetrik gövde üretir (`poliçe` ≠ `poliçenin`) |
| BM25 elemesi **token örtüşmesi** | Küçük korpusta IDF=0 olan gerçek eşleşmeler kaybolur |
| LLM **embedder'dan önce** ısıtılır | Ters sırada bellek yetmiyor (ölçüldü) |
| Reranker **zarif düşer** | İsteğe bağlı bir bileşen zorunlu işlevi düşürmemeli |
| Uç noktalar `async def` **değil** | Bloklayan çıkarım olay döngüsünü kilitler |
| `top_k=3` | Ölçüldü: atıf doğruluğu %66,7 → %91,7, recall kaybı yok |
| Yükleme **akış halinde**, boyut yazarken sayılır | `Content-Length` istemciden gelir, güvenilmez |
| Dosya adı **beyaz listeyle** temizlenir | Dizin aşımı (`../../etc/passwd`) hedef klasör dışına yazardı |
| Silme **üç yerden birden** | Biri atlanırsa silinmiş dokümandan alıntı yapan cevaplar üretilir |

---

## Geliştirme

```bash
pytest                    # 200 test, model yüklemeden ~2 sn
ruff check src tests
mypy src
```

Testler protokoller sayesinde gerçek model yüklemez: `Embedder`, `LLM`,
`VectorStore` yerine 20 satırlık sahte implementasyonlar geçilir.

### Bellek notu

Bu makinede ölçülen: `bge-m3` (2,3 GB) + reranker (2,2 GB) + LLM aynı anda
sığmıyor. Kısıtlı ortamda:

```bash
RAG_RETRIEVAL__USE_RERANKER=false
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4    # yüksek çekirdek sayısında
```

Reranker yer bulamazsa kendini devre dışı bırakır, sistem RRF sıralamasıyla
çalışmaya devam eder ve `/ready` bunu bildirir.

---

## Bilinen sınırlar

- **Taranmış PDF desteği yok.** Metin katmanı olmayan dosyalar
  `no_text_layer` olarak raporlanır ve aramaya dahil edilmez — sessizce
  yok sayılmaz. OCR entegrasyonu yapılmadı.
- **Golden set küçük** (17 soru, tek doküman). Metrikler yön gösterir,
  mutlak kalite iddiası değildir.
- `u05` ve `q06` açık sorun (bkz. deney günlüğü).
- Arayüz katmanı yok; API hazır, istemci ayrı bir adım.
