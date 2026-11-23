import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    broker_url: str
    quarantine_dir: Path
    max_upload_bytes: int
    api_key: str | None
    malwarebazaar_key: str | None
    hash_list_path: Path
    model_dir: Path
    rate_limit_per_minute: int
    icap_port: int
    icap_scan_timeout_seconds: float
    icap_suspicious_action: str
    icap_fail_mode: str
    reports_enabled: bool
    offline_mode: bool
    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    llm_timeout_seconds: float


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://eass:eass@localhost:5432/eass"
        ),
        broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        quarantine_dir=Path(os.getenv("QUARANTINE_DIR", "quarantine")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        api_key=os.getenv("EASS_API_KEY") or None,
        malwarebazaar_key=os.getenv("MALWAREBAZAAR_AUTH_KEY") or None,
        hash_list_path=Path(os.getenv("HASH_LIST_PATH", "rules/hashes.txt")),
        model_dir=Path(os.getenv("MODEL_DIR", "models")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "20")),
        icap_port=int(os.getenv("ICAP_PORT", "1344")),
        icap_scan_timeout_seconds=float(os.getenv("ICAP_SCAN_TIMEOUT_SECONDS", "30")),
        icap_suspicious_action=os.getenv("ICAP_SUSPICIOUS_ACTION", "block"),
        icap_fail_mode=os.getenv("ICAP_FAIL_MODE", "closed"),
        reports_enabled=os.getenv("REPORTS_ENABLED", "false").lower() == "true",
        offline_mode=os.getenv("OFFLINE_MODE", "true").lower() == "true",
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
    )
