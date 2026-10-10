
import json
import os

from dotenv import load_dotenv
from google import genai


MODEL_NAME = "gemini-3.8-flash"

SYSTEM_PROMPT = """
You are a query transformation component for a document question-answering
system.

Given a user's question:
1. Create up to three concise search queries that help retrieve evidence.
2. Preserve the user's original intent.
3. For a multi-part question, create queries that cover its distinct parts.
4. Do not answer the question.
5. Do not introduce facts that are not present in the question.

Return only valid JSON in this exact shape:
{"queries": ["query one", "query two"]}
"""


class QueryTransformer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your local .env file."
            )

        self.client = genai.Client(api_key=api_key)

    def transform(self, question: str) -> list[str]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=(
                f"{SYSTEM_PROMPT}\n\n"
                f"User question: {question}\n"
                "Return JSON only."
            ),
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "queries": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        }
                    },
                    "required": ["queries"],
                },
            },
        )

        if not response.text:
            raise ValueError("Gemini returned an empty query transformation.")

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Gemini returned invalid JSON for query transformation."
            ) from error

        queries = data.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("The transformed queries must be a list.")

        cleaned_queries = []
        for query in queries:
            if isinstance(query, str) and query.strip():
                if query.strip() not in cleaned_queries:
                    cleaned_queries.append(query.strip())

        # Keep retrieval usable even if the model returns no valid queries.
        if not cleaned_queries:
            return [question.strip()]

        return cleaned_queries[:3]


if __name__ == "__main__":
    transformer = QueryTransformer()
    question = (
        "What is SQL injection, and how can developers prevent it "
        "in a REST API?"
    )

    print("Original question:", question)
    print("Transformed queries:")
    for query in transformer.transform(question):
        print("-", query)
