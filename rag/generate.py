"""
Stage 3: generation.

This is where the app becomes a real RAG system. It takes a new incident,
retrieves similar past incidents (Stage 2), hands both to gpt-4o-mini, and
gets back a structured result: a short summary, a severity label and a
root-cause label.

"Structured" matters. Left alone an LLM replies in free prose. We use
Pydantic to define the exact shape we want, and the OpenAI "structured
outputs" feature to force the model's reply into that shape. So the result
is always clean fields you can use in code, never a paragraph to parse.

Run a quick demo from the project root:  python -m rag.generate
"""

from pydantic import BaseModel, Field
from openai import OpenAI

from rag.config import OPENAI_API_KEY, LLM_MODEL, check_key
from rag.labels import SEVERITY_LEVELS, ROOT_CAUSE_CATEGORIES
from rag.retrieve import Retriever


# ----- 1. The shape we want back -----
# Each field has a description; the model sees these and fills accordingly.
class IncidentAssessment(BaseModel):
    summary: str = Field(
        description="A two to three sentence plain-English summary of the incident."
    )
    severity: str = Field(
        description=f"The severity. Must be exactly one of: {SEVERITY_LEVELS}."
    )
    root_cause: str = Field(
        description=(
            "The most likely root-cause category. Must be exactly one of: "
            f"{ROOT_CAUSE_CATEGORIES}."
        )
    )
    reasoning: str = Field(
        description="One sentence on why this severity and root cause were chosen."
    )


# ----- 2. Build the prompt from the new incident + retrieved context -----
def build_prompt(new_incident_text, retrieved):
    """Assemble the text we send to the model."""
    # Format the retrieved similar incidents as reference context.
    context_blocks = []
    for i, m in enumerate(retrieved, 1):
        meta = m["metadata"]
        context_blocks.append(
            f"Past incident {i} "
            f"(severity: {meta['true_severity']}, "
            f"root cause: {meta['true_root_cause']}):\n{m['text']}"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are a rail safety analyst. You assess incident reports and assign "
        "a severity and a root cause. Use the similar past incidents provided as "
        "reference, but base your assessment on the new incident itself. "
        f"Severity must be one of {SEVERITY_LEVELS}. "
        f"Root cause must be one of {ROOT_CAUSE_CATEGORIES}."
    )
    user = (
        f"SIMILAR PAST INCIDENTS (for reference):\n\n{context}\n\n"
        f"NEW INCIDENT TO ASSESS:\n{new_incident_text}\n\n"
        "Assess the new incident."
    )
    return system, user


# ----- 3. The assessor: ties retrieval + generation together -----
class Assessor:
    def __init__(self):
        check_key()                       # fail early if no key
        self.retriever = Retriever()      # loads embedding model once
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def assess(self, new_incident_text):
        # Retrieval (the R in RAG)
        retrieved = self.retriever.retrieve(new_incident_text)
        # Build the augmented prompt (the A)
        system, user = build_prompt(new_incident_text, retrieved)
        # Generation (the G), forced into our schema
        completion = self.client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=IncidentAssessment,
        )
        assessment = completion.choices[0].message.parsed
        return assessment, retrieved


if __name__ == "__main__":
    assessor = Assessor()
    test = (
        "During the evening peak in freezing fog, a train failed to stop at a "
        "signal showing red just outside a major station and continued a short "
        "distance before halting. No collision occurred."
    )
    print("NEW INCIDENT:\n", test, "\n")
    result, used = assessor.assess(test)
    print("ASSESSMENT")
    print("  summary:   ", result.summary)
    print("  severity:  ", result.severity)
    print("  root cause:", result.root_cause)
    print("  reasoning: ", result.reasoning)
    print("\nBased on retrieved incidents of type:",
          [m["metadata"]["incident_type"] for m in used])
