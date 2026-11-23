import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = "d97a0b523de02f3fe5ea6089d080abacab6ee931"
BASE = f"https://raw.githubusercontent.com/elastic/ember/{COMMIT}/"
FILES = {
    "ember_features.py": (
        "ember/features.py",
        "db0d93bb1e1d4b28558e373c77194a82c3cae09d7a1aeb44eea5dd68e84f3ffd",
    ),
    "EMBER_LICENSE.txt": (
        "LICENSE.txt",
        "330a9511c4e2a5fc79204ca3c103a7c8281a13a19ee9975d135e51aa0d969dc4",
    ),
}


for destination, (source, expected_hash) in FILES.items():
    with urlopen(BASE + source, timeout=30) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("EMBER source hash mismatch: " + source)
    Path(destination).write_bytes(content)
