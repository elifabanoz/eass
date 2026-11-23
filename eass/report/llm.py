import json
import urllib.request
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from eass.config import Settings

Output = TypeVar("Output", bound=BaseModel)
PROMPTS = Path(__file__).resolve().parent / "prompts"


class ModelClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, prompt_name: str, payload: dict, schema: type[Output]) -> Output:
        prompt = (PROMPTS / f"{prompt_name}.txt").read_text(encoding="utf-8")
        body = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
            ],
            "temperature": 0,
            "max_tokens": 192,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        request = urllib.request.Request(
            self.settings.llm_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.llm_timeout_seconds) as response:
            if int(response.headers.get("Content-Length", "0")) > 16384:
                raise ValueError("model response too large")
            raw = response.read(16385)
        if len(raw) > 16384:
            raise ValueError("model response too large")
        content = json.loads(raw)["choices"][0]["message"]["content"]
        return schema.model_validate_json(content)
