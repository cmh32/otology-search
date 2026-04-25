#!/usr/bin/env python3
"""
Verify gemma-4-31b-it tool calling and the text-stripping agentic loop.

Turn 1: model should emit function_call(s).
Turn 2: after receiving mock tool results (text stripped from Turn 1 content),
        model should emit a final text answer with no further tool calls.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_papers",
        description="Search the PubMed otology literature database.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Focused keyword search query"),
            },
            required=["query"],
        ),
    )
])

AGENT_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You are a clinical literature assistant. Before answering, call search_papers. "
        "When making a tool call, output only the function call — no surrounding text or explanation."
    ),
    tools=[TOOL],
)
FINAL_CONFIG = types.GenerateContentConfig(
    system_instruction="You are a clinical literature assistant. Synthesize the retrieved results into a concise answer."
)

MOCK_RESULT = {
    "result": {
        "query": "indications for tympanostomy tubes in children guidelines",
        "count": 2,
        "papers": [
            {
                "pmid": "12345678",
                "title": "AAO-HNS Clinical Practice Guideline: Tympanostomy Tubes in Children (2022 Update)",
                "authors": ["Rosenfeld RM", "Tunkel DE"],
                "journal": "Otolaryngology–Head and Neck Surgery",
                "year": 2022,
                "abstract": "This guideline provides evidence-based recommendations for tympanostomy tube insertion in children...",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            },
            {
                "pmid": "87654321",
                "title": "Outcomes of tympanostomy tube insertion for recurrent acute otitis media",
                "authors": ["Smith J"],
                "journal": "International Journal of Pediatric Otorhinolaryngology",
                "year": 2020,
                "abstract": "Retrospective cohort of 200 children with recurrent AOM treated with tympanostomy tubes...",
                "url": "https://pubmed.ncbi.nlm.nih.gov/87654321/",
            },
        ],
    }
}

MAX_TURNS = 5

print("=" * 60)
print("TEST: full agentic loop with text stripping")
print("=" * 60)

contents = [types.Content(role="user", parts=[types.Part(
    text="What are the current indications for tympanostomy tubes in children?"
)])]

passed = True

for turn in range(MAX_TURNS):
    print(f"\n--- Turn {turn + 1} ---")
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=contents,
        config=AGENT_CONFIG,
    )

    candidate = response.candidates[0]
    all_parts = candidate.content.parts
    function_calls = [p for p in all_parts if p.function_call]
    text_parts = [p for p in all_parts if p.text]

    print(f"Parts: {len(all_parts)} total  |  {len(function_calls)} function_call  |  {len(text_parts)} text")

    if not function_calls:
        print("\n✓ Model returned final text answer (no tool calls):")
        print(response.text[:400])
        break

    # Mirror the server's text-stripping fix
    tool_only_content = types.Content(role="model", parts=function_calls)
    contents.append(tool_only_content)

    # Verify stripped content has no text parts
    stripped_text = [p for p in tool_only_content.parts if p.text]
    if stripped_text:
        print("✗ FAIL: stripped content still contains text parts")
        passed = False
    else:
        print("✓ Text parts stripped from model turn before appending to contents")

    # Feed mock tool results back
    tool_response_parts = []
    for fc_part in function_calls:
        fc = fc_part.function_call
        print(f"  Tool called: {fc.name}(query={fc.args.get('query')!r})")
        tool_response_parts.append(types.Part(
            function_response=types.FunctionResponse(
                name=fc.name,
                id=fc.id,
                response=MOCK_RESULT,
            )
        ))
    contents.append(types.Content(role="user", parts=tool_response_parts))

else:
    print("\n✗ FAIL: hit turn cap without receiving a final answer")
    passed = False

print("\n" + "=" * 60)
print("RESULT:", "PASS ✓" if passed else "FAIL ✗")
print("=" * 60)
