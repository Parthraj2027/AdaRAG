import time
import re
from openai import APIError, OpenAI, RateLimitError
from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL
from src.utils import logger

class LLM:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not defined in environment.")
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            timeout=30.0,
            max_retries=0,
        )

    @staticmethod
    def _clean_answer(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def generate(self, prompt: str, model: str = LLM_MODEL, max_retries: int = 6) -> tuple[str, int, int]:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful AI assistant. "
                                "Answer ONLY using the provided context. "
                                "Return only the final answer in one concise sentence. "
                                "Do not include reasoning, chain-of-thought, or think tags."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                raw_answer = response.choices[0].message.content or ""
                answer = self._clean_answer(raw_answer)
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                return answer, prompt_tokens, completion_tokens
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e
                msg = str(e)
                wait = (2 ** attempt) * 2 + 2
                match = re.search(r"try again in (\d+(\.\d+)?)s", msg)
                if match:
                    wait = float(match.group(1)) + 1.5
                logger.warning(f"Rate limited. Waiting {wait:.1f}s before retrying...")
                time.sleep(wait)
            except APIError as e:
                if attempt == max_retries - 1:
                    raise e
                msg = str(e)
                wait = (2 ** attempt) * 2 + 2
                match = re.search(r"try again in (\d+(\.\d+)?)s", msg)
                if match:
                    wait = float(match.group(1)) + 1.5
                logger.warning(f"API error: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait = (2 ** attempt) * 2 + 2
                logger.warning(f"Transient error: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        raise RuntimeError("Failed to generate answer after retries.")


class MockLLM(LLM):
    """Return a deterministic context-based answer without an API call."""

    def __init__(self):
        self.client = None

    def generate(self, prompt: str, model: str = None, max_retries: int = 0) -> tuple[str, int, int]:
        context = prompt.split("\n\nQuestion:", 1)[0]
        lines = context.splitlines()
        answer = lines[1] if len(lines) > 1 else "mock answer"
        return answer[:500], len(prompt) // 4, 5
