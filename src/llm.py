from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL
from src.utils import logger

class LLM:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not defined in environment.")
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL
        )

    def generate(self, prompt: str, model: str = LLM_MODEL) -> tuple[str, int, int]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer ONLY using the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            answer = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            return answer, prompt_tokens, completion_tokens
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise e
