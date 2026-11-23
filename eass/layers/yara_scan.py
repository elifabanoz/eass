from functools import lru_cache
from pathlib import Path

import yara


@lru_cache(maxsize=4)
def compile_rules(path: Path) -> yara.Rules:
    return yara.compile(filepath=str(path))


def scan(path: Path, rule_path: Path) -> dict:
    matches = compile_rules(rule_path).match(filepath=str(path), timeout=10)
    return {
        "status": "complete",
        "matches": [{"rule": match.rule, "meta": match.meta} for match in matches],
    }
