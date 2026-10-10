
import os

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError


MODEL_NAME = "gemini-3.8-flash"

RESOLVER_PROMPT = """
You rewrite follow-up questions into standalone questions using
the conversation history.

Rules:
- Preserve the user's intended meaning.
- Resolve references such as "it", "that", "the first method",
  and "you mentioned" using the conversation.
- Do not answer the question.
- Do not invent facts that are absent from the conversation.
- If the question is already standalone, keep its meaning unchanged.
- Return only the rewritten question, without explanations.
"""


class FollowUpResolver:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)

    def resolve(
        self,
        question: str,
        conversation_context: list[dict] | None = None,
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not conversation_context:
            return question

        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_context
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=(
                    f"{RESOLVER_PROMPT}\n\n"
                    f"Conversation history:\n{history}\n\n"
                    f"Latest question:\n{question}"
                ),
            )
        except (APIError, httpx.HTTPError) as error:
            raise RuntimeError(
                "Gemini follow-up resolution failed. Check the API "
                "quota, model access, and network connection."
            ) from error

        resolved = (response.text or "").strip()

        if not resolved:
            raise RuntimeError(
                "Gemini returned an empty rewritten question."
            )

        return resolved
