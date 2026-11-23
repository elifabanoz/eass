import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "deliverables" / "EASS-local-demo-20261120.zip"
ROOT_FILES = (
    ".dockerignore",
    ".gitignore",
    "Caddyfile",
    "Dockerfile",
    "Dockerfile.ember",
    "alembic.ini",
    "compose.yaml",
    "compose.vps.yaml",
    "pyproject.toml",
    "vps.env.example",
)
SOURCE_DIRS = (
    "eass",
    "experiments",
    "ember_extractor",
    "migrations",
    "scripts",
    "tests",
    ".github",
)
MODEL_FILES = (
    "models/ember_full_lgbm.txt",
    "models/ember_full_lgbm.json",
    "models/ember_full_ablation_lgbm.txt",
)


def files_to_package():
    files = {ROOT / name for name in ROOT_FILES + MODEL_FILES}
    files.add(ROOT / "rules" / "eass.yar")
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.add(path)
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("release source file missing")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def package():
    metadata = json.loads((ROOT / "models/ember_full_lgbm.json").read_text(encoding="utf-8"))
    primary = ROOT / "models/ember_full_lgbm.txt"
    secondary = ROOT / "models/ember_full_ablation_lgbm.txt"
    if hashlib.sha256(primary.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("primary model does not match metadata")
    if (
        hashlib.sha256(secondary.read_bytes()).hexdigest()
        != metadata["ensemble"]["secondary_model_sha256"]
    ):
        raise ValueError("secondary model does not match metadata")
    files = files_to_package()
    manifest = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        archive.writestr("package-manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError("damaged release archive")
        names = set(archive.namelist())
        if names != set(manifest) | {"package-manifest.json"}:
            raise ValueError("unexpected release archive members")
        if any(
            name.startswith("data/") or name.endswith(".env") or "hashes.txt" in name
            for name in names
        ):
            raise ValueError("private or fetched data found in release archive")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"release file hash mismatch: {name}")
    return {
        "path": str(OUTPUT),
        "files": len(files),
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(package(), indent=2))
