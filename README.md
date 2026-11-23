# EASS

EASS is a static file scanning service. Its REST API accepts uploads and queues scans for a Celery worker. An ICAP service applies the same analysis pipeline to HTTP traffic. Scan records and reports are stored in PostgreSQL.

This repository is a local technical demo. A `clean` verdict does not guarantee that a file is safe.

## Architecture

| Component | Responsibility |
| --- | --- |
| FastAPI | Uploads, scan status, and report endpoints |
| Redis | Celery task queue and upload rate limiting |
| Celery worker | Hash reputation, MalwareBazaar, YARA, PE analysis, and model checks |
| PostgreSQL | Scan status, layer results, and reports |
| ICAP | Allow or block decisions for REQMOD and RESPMOD traffic |
| EMBER extractor | Extracts 2,381 PE features for the full model |

The REST path is `POST /scans` → a `queued` PostgreSQL record → a Celery task through Redis → `run_scan` in the worker → a stored result → `GET /scans/{scan_id}`. ICAP calls the same `run_scan` function synchronously. When reports are enabled, a separate Celery task builds the report after the scan.

## Run locally

Docker Desktop with Linux containers is required. The first build needs internet access to build images and fetch the EMBER feature extractor. Model files are included under `models/`.

From PowerShell:

```powershell
git clone https://github.com/elifabanoz/eass.git
cd eass
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Add your MalwareBazaar Auth-Key to .env if you have one. Never commit it.
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

If you already have the repository, run the commands from its root and skip `git clone`. API documentation is at <http://127.0.0.1:8000/docs>. API and ICAP bind to `127.0.0.1` by default. It is normal for the `migrate` service to finish with `Exited (0)`.

Without `MALWAREBAZAAR_AUTH_KEY`, online hash lookup and dump refresh are unavailable; the local EICAR hash check and other scanning layers still work. `REPORTS_ENABLED=true` enables report endpoints. External threat intelligence also requires `OFFLINE_MODE=false` and an Auth-Key. Optional language-model report sections require a compatible `LLM_BASE_URL` and `LLM_MODEL`; those sections may be unavailable without a model service.

## Demo

1. Upload `C:\Windows\System32\notepad.exe` with `POST /scans` in Swagger.
2. Use the returned `scan_id` with `GET /scans/{scan_id}` and, when reports are enabled, `GET /scans/{scan_id}/report`.
3. In another terminal, run `docker compose logs -f --tail=20 worker` to show Celery receiving and completing the task.
4. Run `python scripts/smoke_live.py` to test the EICAR signature through REST and ICAP. The script checks for `malicious` verdicts and an ICAP block response. EICAR is a test file, not malware.

## Model training and results

The full PE model used at runtime averages the outputs of two LightGBM models trained on features from [EMBER 2018 v2](https://github.com/elastic/ember) and [BODMAS](https://whyisyoung.github.io/BODMAS/). The EMBER selection contained 100,000 benign and 100,000 malicious records: 160,000 for fitting and 40,000 for validation. Another 20,049 BODMAS records from August–October 2019 were used for fitting, and 16,371 from November–December 2019 for validation. The models were trained on extracted features, not raw EXE files.

Both models used 700 trees, 127 leaves, and a 0.05 learning rate. The second model excluded two features. Runtime predictions are averaged and compared with the `0.7376758677690286` threshold recorded in `models/ember_full_lgbm.json`. The threshold was selected against limits of 1% false positives on EMBER validation and 0.5% on late-2019 BODMAS validation. Model probability is distinct from EASS's 0–100 risk score and final verdict.

| Evaluation | Samples | Recall | False positive rate |
| --- | ---: | ---: | ---: |
| EMBER validation | 40,000 | 95.725% | 1.000% |
| BODMAS 2020 | 87,284 | 99.279% | 0.177% |

These figures come from `models/ember_full_lgbm.json`. The official EMBER test set and BODMAS 2020 data were viewed during earlier experiments, so these are exploratory results rather than a fully blind final test. They do not guarantee performance on current real-world files. Training and evaluation scripts are in `experiments/`; the training datasets are not included in this repository.

## Limitations

- ZIP contents are not unpacked and scanned. Analysis of non-PE files is limited.
- The local YARA rules have limited coverage. No hash match does not establish that a file is safe.
- Report recommendations do not automatically quarantine or release files.
- This Compose setup is for a local demo. `compose.vps.yaml` is deployment preparation; a VPS deployment has not been verified.

## Checks

```powershell
python -m pip install '.[dev,train]'
ruff check .
ruff format --check .
pytest -q
python -c "import eass.api, eass.worker, eass.icap.server"
```

The latest local run passed all 70 tests and Ruff checks. GitHub Actions runs the same lint and test commands; check the workflow result for the current commit.
