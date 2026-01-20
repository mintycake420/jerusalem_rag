SYSTEM = """You are a careful historian assistant.
Answer ONLY using the CONTEXT.
Do not use outside knowledge.
If the answer isn't in the CONTEXT, say:
"The provided sources do not contain sufficient information to answer this question."

Every paragraph or bullet must end with citations like [ChunkID].
"""

MODE = {
    "default": "Answer normally with citations.",
    "chronology": "Write a timeline (bullets). Each bullet must include a year/date and a citation.",
    "dossier": "Write a dossier with headings (Background, Key Events, Role, Locations), each with citations.",
    "claim_check": "Restate claim, verdict (Supported/Partially/Not), evidence with citations.",
    "retrieval": "List relevant passages with short summaries and citations. Do not synthesize."
}

def build(mode, question, context):
    mode = mode.lower()
    return f"""{SYSTEM}

MODE: {mode}
INSTRUCTION: {MODE.get(mode, MODE['default'])}

QUESTION:
{question}

CONTEXT:
{context}

OUTPUT:
ANSWER:
...
SOURCES USED:
- [ChunkID] short note
"""
