# EASS

EASS, dosyaları çalıştırmadan inceleyen bir tarama servisidir. REST API dosyayı kabul edip taramayı Celery worker'a bırakır; ICAP servisi aynı analiz hattını HTTP trafiği üzerinde senkron çalıştırır. Sonuçlar PostgreSQL'de tutulur.

Bu depo bir yerel teknik demodur. Zararlı yazılımı kesin olarak tespit etme veya güvenli olduğunu garanti etme iddiası taşımaz.

## Mimari

| Bileşen | Görevi |
| --- | --- |
| FastAPI | Dosya yükleme, tarama durumu ve rapor uçları |
| Redis | Celery görev kuyruğu ve yükleme hız sınırı |
| Celery worker | Hash, MalwareBazaar, YARA, PE ve model kontrollerini çalıştırır |
| PostgreSQL | Tarama durumu, katman sonuçları ve raporlar |
| ICAP | REQMOD/RESPMOD trafiğinde geçiş veya engelleme kararı |
| EMBER extractor | PE dosyasından tam model için 2381 özellik çıkarır |

REST akışı: `POST /scans` → PostgreSQL'de `queued` kayıt → Redis üzerinden Celery görevi → worker'da `run_scan` → PostgreSQL'de sonuç → `GET /scans/{scan_id}`. ICAP aynı `run_scan` fonksiyonunu senkron çağırır. Rapor açıksa tarama sonrasında ayrı bir Celery göreviyle hazırlanır.

## Yerel çalıştırma

Docker Desktop (Linux container desteği) gerekir. İlk kurulumda imajları oluşturmak ve EMBER özellik çıkarıcısını almak için internet erişimi gerekir. `models/` altındaki model dosyaları depoda bulunur.

PowerShell'de:

```powershell
cd C:\Users\aysel\eass
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# .env içine kendi MalwareBazaar Auth-Key değerinizi ekleyin; anahtarı Git'e koymayın.
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

Başka bir dizine klonladıysanız ilk satırda o dizini kullanın. API belgesi: <http://127.0.0.1:8000/docs>. API ve ICAP varsayılan olarak yalnızca bu bilgisayarın `127.0.0.1` adresine bağlanır. `migrate` servisinin `Exited (0)` durumuyla bitmesi normaldir.

`MALWAREBAZAAR_AUTH_KEY` olmadan çevrimiçi hash sorgusu ve döküm güncellemesi çalışmaz; yerel EICAR hash kontrolü ve diğer katmanlar çalışır. `REPORTS_ENABLED=true` rapor uçlarını açar. Dış tehdit istihbaratı sorgusu için ayrıca `OFFLINE_MODE=false` ve Auth-Key gerekir. Raporun isteğe bağlı dil modeli bölümleri için uyumlu bir `LLM_BASE_URL` ve `LLM_MODEL` sağlanmalıdır; model servisi yoksa bu bölümler kullanılamayabilir.

## Demo

1. Swagger'da `POST /scans` ile `C:\Windows\System32\notepad.exe` dosyasını yükleyin.
2. Dönen `scan_id` ile `GET /scans/{scan_id}` ve rapor açıksa `GET /scans/{scan_id}/report` çağırın.
3. Ayrı bir terminalde `docker compose logs -f --tail=20 worker` komutuyla Celery görevinin alınıp bittiğini gösterin.
4. EICAR test imzasını REST ve ICAP üzerinden sınamak için `python scripts/smoke_live.py` çalıştırın. Bu betik iki taramada `malicious` sonucunu ve ICAP engelleme yanıtını kontrol eder. EICAR gerçek zararlı yazılım değildir.

## Model eğitimi ve ölçüm

Üretimde kullanılan tam PE modeli, [EMBER 2018 v2](https://github.com/elastic/ember) ve [BODMAS](https://whyisyoung.github.io/BODMAS/) özellikleriyle eğitilen iki LightGBM modelinin çıktı ortalamasıdır. EMBER'den 100.000 temiz ve 100.000 zararlı kayıt seçildi; bunların 160.000'i eğitim, 40.000'i doğrulama için kullanıldı. BODMAS 2019'dan Ağustos–Ekim dönemindeki 20.049 kayıt eğitime, Kasım–Aralık dönemindeki 16.371 kayıt doğrulamaya ayrıldı. Ham EXE dosyaları modele doğrudan verilmedi.

Her iki modelde 700 ağaç, 127 yaprak ve 0,05 öğrenme hızı kullanıldı. İkinci model iki özelliği dışarıda bırakarak eğitildi. Çalışma zamanında iki çıktı ortalanır; `models/ember_full_lgbm.json` içindeki karar eşiği `0.7376758677690286` değeridir. Eşik, EMBER doğrulamasında en fazla %1 ve geç 2019 BODMAS doğrulamasında en fazla %0,5 yanlış alarm sınırları dikkate alınarak seçildi. Model çıktısı EASS'ın 0–100 risk puanı veya nihai kararıyla aynı şey değildir.

| Ölçüm | Örnek | Recall | Yanlış alarm oranı |
| --- | ---: | ---: | ---: |
| EMBER doğrulama | 40.000 | %95,725 | %1,000 |
| BODMAS 2020 | 87.284 | %99,279 | %0,177 |

Bu sayılar `models/ember_full_lgbm.json` kaydından alınmıştır. Resmî EMBER testine ve BODMAS 2020 verisine önceki denemelerde bakıldığı için sonuçlar tamamen kör bir nihai test değil, keşifsel değerlendirmedir. Bu ölçümler güncel saha performansını garanti etmez. Eğitim ve değerlendirme kodu `experiments/` altındadır; eğitim veri arşivleri depoya dahil değildir.

## Sınırlar

- ZIP içeriği açılıp taranmıyor; PE dışındaki türlerde derin analiz sınırlı.
- Yerel YARA kuralları sınırlı kapsamlıdır. Hash eşleşmesi olmaması dosyanın güvenli olduğunu kanıtlamaz.
- Rapor önerileri otomatik karantina veya dosya serbest bırakma işlemi başlatmaz.
- Bu Compose yapılandırması yerel demo içindir. `compose.vps.yaml` dağıtım hazırlığıdır; VPS dağıtımı doğrulanmış değildir.

## Kontroller

```powershell
python -m pip install '.[dev,train]'
ruff check .
ruff format --check .
pytest -q
python -c "import eass.api, eass.worker, eass.icap.server"
```

Son yerel kontrolde 70 test geçti ve Ruff kontrolleri temizdi. GitHub Actions iş akışı aynı lint ve test komutlarını çalıştıracak şekilde tanımlıdır; uzak CI sonucu ayrıca doğrulanmalıdır.
