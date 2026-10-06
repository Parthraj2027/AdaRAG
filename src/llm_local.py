"""Local LLM wrapper using the Ollama HTTP API."""

import requests

from src.utils import logger


class LocalLLM:
    def __init__(self, model: str = "qwen2.5:0.5b", url: str = "http://localhost:11434"):
        self.model = model
        self.generate_url = f"{url}/api/generate"
        self._health_check(url)

    def _health_check(self, url: str) -> None:
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [item["name"] for item in response.json().get("models", [])]
            if self.model not in models:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {self.model}"
                )
        except requests.exceptions.ConnectionError as error:
            raise RuntimeError(
                f"Ollama server not running at {url}. Start it with: ollama serve"
            ) from error

    def generate(self, prompt: str, model: str = None, max_retries: int = 0) -> tuple[str, int, int]:
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": model or self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 128},
                },
                timeout=120,
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            return answer, len(prompt) // 4, len(answer) // 4
        except Exception as error:
            logger.error(f"Local LLM generation failed: {error}")
            raise
