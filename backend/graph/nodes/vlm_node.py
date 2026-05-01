"""
graph/nodes/vlm_node.py

Multimodal entry point. Fires when state["image_pending"] is True —
i.e. the student uploaded an anatomical diagram and the API set
state.image_b64 in _initial_state.

Behavior (one Sonnet vision call, JSON output):
  1. Identify the primary anatomical structure in the image
     (1-5 words, lowercase).
  2. Generate a single Socratic opener that asks the student about
     the structure's function, innervation, or clinical impact —
     NOT what it's named (since the system already identified it).

Output:
  - current_concept = identified structure (or "" if "unclear")
  - draft_response  = Socratic opener
  - draft_source_node = "vlm_node"
  - student_phase = "learning" (so the next turn flows through the
    normal manager_and_retrieval → classifier → teacher_socratic loop)
  - image_pending = False, image_b64 = "" (consume the upload)
  - messages: append AIMessage with the opener so the API streams it
    back without needing deliver_response (vlm_node terminates at END
    in graph_builder).
  - turn_count += 1 (the opener counts as the first tutor turn of
    this Socratic loop)

Model: PRIMARY_MODEL (claude-sonnet-4-5) with native vision content.
The Anthropic SDK supports image inputs as a content block in messages;
the existing graph._llm_client.Anthropic() wrapper handles both Bedrock
and direct API transparently.
"""
import json
import re
import sys

from langchain_core.messages import AIMessage

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import load_prompt


_client = Anthropic()


# Image source media-type sniffing — Anthropic requires the type field on
# the content block. We accept the most common formats; default to JPEG
# since that's what most camera / browser uploads end up as.
def _sniff_media_type(b64: str) -> str:
    """Best-effort media-type from a base64 string's first few bytes."""
    if not b64:
        return "image/jpeg"
    sample = b64[:24]
    if sample.startswith("/9j/"):
        return "image/jpeg"
    if sample.startswith("iVBORw0K"):
        return "image/png"
    if sample.startswith("R0lGOD"):
        return "image/gif"
    if sample.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _strip_data_url_prefix(b64: str) -> str:
    """Browsers often send 'data:image/png;base64,iVBOR...'. Strip the
    'data:' prefix so the Anthropic SDK gets pure base64."""
    if "," in b64 and b64.startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


# Fallback message used when the LLM call fails (Bedrock 503, malformed
# response, etc.). Better than crashing the turn on a flaky upload.
_FALLBACK_OPENER = (
    "I'm having trouble identifying that image — could you describe what "
    "you uploaded, or tell me which anatomical structure you'd like to "
    "explore?"
)


def vlm_node(state: GraphState) -> dict:
    image_b64 = _strip_data_url_prefix(state.get("image_b64", "") or "")
    if not image_b64:
        # Defensive — image_pending was set but no payload reached the node.
        # Treat as a chitchat turn rather than crashing the request.
        return {
            "messages": [AIMessage(content=_FALLBACK_OPENER)],
            "draft_response": _FALLBACK_OPENER,
            "draft_source_node": "vlm_node",
            "image_pending": False,
            "image_b64": "",
            "turn_count": state.get("turn_count", 0) + 1,
            "student_phase": "learning",
        }

    media_type = _sniff_media_type(image_b64)
    prompt = load_prompt("vlm_identify.txt")

    try:
        response = _client.messages.create(
            model=config.model_for("vlm"),
            max_tokens=getattr(config, "VLM_MAX_TOKENS", 400),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = (response.content[0].text or "").strip()
    except Exception as exc:
        print(
            f"[vlm_node] Sonnet vision call failed ({exc!r}); using fallback",
            file=sys.stderr,
        )
        raw = ""

    # Parse the JSON envelope. Sonnet sometimes wraps in ```json fences;
    # strip those before json.loads.
    concept = ""
    opener = ""
    if raw:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        # Find the first {...} block — Sonnet occasionally pads with text.
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
                concept = (obj.get("concept") or "").strip().lower()
                opener = (obj.get("opener") or "").strip()
            except json.JSONDecodeError:
                print(
                    f"[vlm_node] JSON parse failed on raw={raw[:200]!r}",
                    file=sys.stderr,
                )

    # If the model couldn't identify (or returned "unclear"), keep concept
    # empty so the next turn doesn't lock onto a bogus topic. Manager will
    # re-extract from the student's follow-up message.
    if concept == "unclear":
        concept = ""

    if not opener:
        opener = _FALLBACK_OPENER

    print(
        f"[vlm_node] identified={concept!r} | opener={opener[:120]!r}",
        file=sys.stderr,
    )

    return {
        "messages": [AIMessage(content=opener)],
        "draft_response": opener,
        "draft_source_node": "vlm_node",
        "current_concept": concept,
        "image_pending": False,
        "image_b64": "",
        "turn_count": state.get("turn_count", 0) + 1,
        "student_phase": "learning",
        # Reset per-loop counters for the fresh Socratic loop on this concept.
        "student_attempted": False,
        "idk_count": 0,
        "dean_revisions": 0,
        "dean_revision_instruction": "",
    }
