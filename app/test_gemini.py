
from dotenv import load_dotenv
from google import genai
import os


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY was not found in .env.")
        return

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with exactly: Gemini connection successful",
    )

    print("Gemini response:", response.text)


if __name__ == "__main__":
    main()
