"""
Stage 0 check: run this first.

It confirms your environment is set up correctly BEFORE we build anything
real. It checks: the packages import, the data file loads, and your API key
is present and works. If all four lines print OK, you are ready.

Run from the project root:  python check_setup.py
"""

import json


def main():
    # 1. Can we import the libraries?
    try:
        import openai            # noqa: F401
        import chromadb          # noqa: F401
        import sentence_transformers  # noqa: F401
        import pydantic          # noqa: F401
        print("OK  1/4  libraries import")
    except Exception as e:
        print("FAIL 1/4  a library did not import:", e)
        return

    # 2. Does the data load?
    try:
        with open("data/incidents.json") as f:
            data = json.load(f)
        print(f"OK  2/4  data loads ({len(data)} incidents)")
    except Exception as e:
        print("FAIL 2/4  could not load data/incidents.json:", e)
        return

    # 3. Is the API key present?
    from rag.config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("FAIL 3/4  no API key. Copy .env.example to .env and paste your key.")
        return
    print("OK  3/4  API key found")

    # 4. Does the key actually work? (one tiny, ~free call)
    try:
        from openai import OpenAI
        from rag.config import LLM_MODEL
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=5,
        )
        print("OK  4/4  API call works, model said:", resp.choices[0].message.content.strip())
    except Exception as e:
        print("FAIL 4/4  API call failed:", e)
        return

    print("\nAll good. You are ready for the next stage.")


if __name__ == "__main__":
    main()
