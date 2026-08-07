# Prompt sürümleri ve deney kaydı

Prompt bir **model parametresidir**: değiştirmek sistemin davranışını değiştirir.
Bu yüzden sürümlenir, ve her sürüm golden set üzerinde **ölçülür**.

Kural: bir prompt sürümü, ölçülmeden varsayılan yapılmaz.

---

## v1 — üretimde (varsayılan)

Altı kural: yalnızca bağlamı kullan · bilmiyorsan çekimser kal · atıf ver ·
sayıları değiştirme · Türkçe ve kısa.

**Ölçüm** (17 soruluk golden set, bge-m3, hybrid dense+sparse, reranker kapalı,
gemma3:4b, top_k=5):

| Metrik | Değer |
|---|---|
| Geçme oranı | **%94,1** |
| Hit@k / Recall@k | %100 / %100 |
| MRR | %83,3 |
| Çekimserlik doğruluğu | **%100** |
| **Uydurma oranı** | **%0** |
| Atıf oranı | %100 |
| Olgu doğruluğu | %91,7 |

Bilinen tek başarısızlık: `q10` — "18-36 ay yatırım miktarı" sorusunda model
`150K USD` (0-6 ay figürü) veriyor. **Retrieval kusursuz** (doğru sayfa 1.
sırada); hata tamamen generation tarafında, iki zaman aralığının karıştırılması.

---

## v2 — BAŞARISIZ DENEY (varsayılan yapılmadı)

**Hipotez:** q10'daki karışıklık, modele "sorudaki niteleyiciyi (zaman aralığı,
dönem, sektör) birebir eşleştir, yakın duran benzer satırı kullanma" talimatı
eklenerek düzeltilebilir.

**Sonuç: HİPOTEZ YANLIŞ. Sistem geriledi.**

| Metrik | v1 | v2 | Fark |
|---|---|---|---|
| Geçme oranı | %94,1 | %88,2 | ▼ |
| Çekimserlik doğruluğu | %100 | %94,1 | ▼ |
| **Uydurma oranı** | **%0** | **%5,9** | ▼▼ |
| Olgu doğruluğu | %91,7 | %91,7 | = |

İki bulgu:

1. **q10 düzelmedi.** Niteleyici talimatı işe yaramadı — sorun talimatın
   yokluğu değil, 4B modelin bağlamdaki benzer satırları ayırt etme
   kapasitesi.
2. **Yeni bir uydurma ortaya çıktı.** `u05` ("hangi programlama dili ile
   geliştirildi?") sorusunda model artık cevap veriyor; v1'de çekimser
   kalıyordu.

**Çıkarılan ders:** Prompt'a kural eklemek, mevcut kuralları **zayıflatır**.
Talimat sayısı arttıkça her birinin ağırlığı düşer; küçük modellerde bu etki
belirgindir. Çekimserlik kuralı bu projede en kritik kural olduğu için onun
baskınlığını azaltan hiçbir değişiklik kabul edilemez.

**Neden silinmedi:** Başarısız deney de bilgidir. Bu dosya olmadan biri
aynı "makul" fikri tekrar dener ve aynı gerilemeyi yeniden üretir.

---

## Sıradaki denenecekler (q10 için)

Ölçmeden hiçbiri varsayılan yapılmayacak:

1. **Daha büyük model** — hipotez: bu bir kapasite sınırı, prompt sorunu değil.
   En düşük riskli deney, çünkü prompt'a dokunmuyor.
2. **Reranker açık** — hipotez: doğru satırı içeren chunk tek başına üste
   çıkarsa karışacak alternatif bağlamda olmaz. (Bu makinede bellek nedeniyle
   ölçülemedi.)
3. **top_k düşürme (5 → 3)** — hipotez: daha az gürültü, daha az karışma.
   Riski: recall düşebilir; her iki metrik birlikte izlenmeli.
4. **Yapı-farkında chunking** — "9. Yatırım ve Harcamalar" bölümü s.4 ve s.5
   arasında bölünmüş durumda; bölüm sınırına göre bölme bu satırları bir arada
   tutar.

**Ölçüm disiplini:** her seferinde TEK değişken. İki şeyi birlikte değiştirip
iyileşmeyi birine atfetmek, ölçüm yapmamakla aynı şeydir.

---

## Deney 3 — `top_k` 5 → 3 (BAŞARILI, varsayılan yapıldı)

**Hipotez:** Atıf doğruluğu %66,7'de kalıyor (4/12 soruda doğru cevap ama
yanlış sayfaya atıf). Sebep talimat eksikliği değil, 4B modelin bağlamdaki
5 kaynak numarasını güvenilir biçimde takip edememesi olabilir. Daha az
kaynak → daha az karışma.

**Tek değişken:** `top_k` 5 → 3. Prompt, model, embedder, chunking sabit.

| Metrik | top_k=5 | top_k=3 | |
|---|---|---|---|
| Geçme oranı | %70,6 | **%88,2** | ▲ |
| Atıf doğruluğu | %66,7 | **%91,7** | ▲▲ |
| Olgu doğruluğu | %91,7 | **%100** | ▲ |
| Precision@k | %25 | **%38,9** | ▲ |
| Recall@k | %100 | %100 | = |
| MRR | %83,3 | %83,3 | = |
| Uydurma oranı | **%0** | %5,9 | ▼ |

**Karar: benimsendi.** Dört metrikte belirgin kazanç, recall'da hiç kayıp yok.
Bu, "daha çok bağlam daha iyi cevap" sezgisinin yanlış olduğunun ölçülmüş
kanıtı: fazla bağlam modeli kaynak eşlemesinde yanıltıyor.

**Ödünleşim dürüstçe kaydedilir:** uydurma oranı %0'dan %5,9'a çıktı (`u05`).
Bu en kritik metrik olduğu için kazanç "bedava" değil.

---

## Açık sorunlar

### `u05` — "SmartSafe AI hangi programlama dili ile geliştirildi?"
Model bazı yapılandırmalarda çekimser kalmak yerine cevap veriyor. Alan içi
ama dokümanda geçmeyen, makul görünen teknik detay — uydurmaya en yatkın
soru tipi. Hem prompt v2'de hem `top_k=3`'te ortaya çıktı, `top_k=5` + v1'de
çıkmadı. Sınırda bir vaka.

Denenecekler (tek tek, ölçerek):
1. Prompt'ta çekimserlik kuralını **ilk** kural yapmak (sıra etkisi).
2. Reranker eşiği (`reranker_min_score`): alakasız chunk hiç bağlama girmezse
   model uydurmak için malzeme bulamaz.
3. Daha büyük model.

### `q06` — "Beşinci yıl ARR" atıfı
`top_k=3` ile hâlâ yanlış sayfaya atıf yapıyor. Kalan tek atıf hatası.

### `q10` — "18-36 ay yatırım" (`top_k=5` ile başarısızdı)
`top_k=3` ile düzeldi. İlginç: prompt'la (v2) çözülemeyen problem, bağlamı
daraltarak çözüldü — sorunun talimat değil **dikkat dağınıklığı** olduğunu
gösteriyor.

---

## Deney 4 — Reranker açık/kapalı (ÖLÇÜLDÜ, kapalı bırakıldı)

**Hipotez:** Cross-encoder reranker, RRF sıralamasını iyileştirir ve cevap
kalitesini artırır.

**Kurulum:** Tek sistem, embedder ve store paylaşımlı; tek değişken reranker.
LLM bilinçli olarak yüklenmedi — reranker yalnızca retrieval'ı etkiler, ve
üçü birden bu makinenin belleğine sığmıyor (ölçüldü: LLM sıcakken reranker
kontrolünde yalnızca 0.18 GB boş kalıyor).

| Metrik | KAPALI | AÇIK | Fark |
|---|---|---|---|
| Hit@k | %100 | %100 | = |
| Recall@k | %100 | %100 | = |
| **MRR** | %83,3 | **%95,8** | **+12,5 puan** |
| Precision@k | %38,9 | %38,9 | = |
| **Gecikme/sorgu** | **71 ms** | **17.474 ms** | **245×** |

**Sonuç: hipotez KISMEN doğru, ama takas kabul edilemez.**

1. **Sıralama gerçekten iyileşiyor.** MRR 0,833 → 0,958: doğru kaynak
   neredeyse her zaman 1. sıraya çıkıyor.
2. **Ama LLM'e giden chunk'lar AYNI.** Hit, Recall ve Precision değişmedi —
   reranker yeni aday bulmuyor, mevcut 3 adayı yeniden sıralıyor. Yani
   kazanç yalnızca sıra kalitesinde.
3. **Bedel 245×.** Cross-encoder her aday için modeli ayrı çalıştırır:
   20 aday × ~870 ms = 17,5 sn. Bi-encoder'da vektörler önceden hazır (71 ms).

**Karar: CPU'da KAPALI.** Kullanıcının cevap süresi 2 sn'den 19,5 sn'ye
çıkıyor; karşılığında modele giden bağlam değişmiyor.

**Ne zaman açılmalı:** embedder + reranker GPU'ya taşınırsa süre ~1 sn'ye
iner ve +12,5 puan MRR pratikte bedavaya gelir. Gereken: CUDA derlemeli
torch + LLM'in uzak/bulut uca taşınması (adaptör hazır, kod değişmez).

**Yan bulgu — bellek eşiği yapılandırılabilir yapıldı.**
`reranker_required_ram_gb` sabit 2,3 GB idi ve 1,4× güvenlik payıyla
3,22 GB arıyordu. Bu eşik, sürecin bir kez işletim sistemi tarafından
öldürüldüğü koşullarda (0,65 GB boşken) belirlenmişti; farklı makinelerde
sığabilecek bir modeli gereksizce reddediyor. Artık ayarlanabilir.

**Yan bulgu — gürültüye dayanıklılık.** Index'e 18 chunk'lık iki alakasız
doküman eklendi (6 → 24 vektör). Tüm metrikler AYNI kaldı. Hybrid retrieval
alakasız içerikten etkilenmedi.
