# Canonical demo traces

Pre-recorded `/chat/trace` outputs that the architecture visualizer
replays when there's no live API access. See `docs/website.md` §9.

To regenerate, run the backend locally with a valid Anthropic / Bedrock
key and use:

```bash
PYTHONPATH=. uvicorn api.main:app --port 8000  # in one terminal

# in another:
PYTHONPATH=. python3 scripts/record_demo_trace.py \
    --id funny_bone \
    --label "Funny bone — clean Socratic flow" \
    --message "What nerve causes the funny bone sensation?"
```

Ten canonical traces:

| id | label | message / setup |
|---|---|---|
| `funny_bone`        | Funny bone — clean Socratic flow              | "What nerve causes the funny bone sensation?" |
| `idk_x3`            | Three IDKs → reveal via teach_node            | 3 prime turns of "I don't know" then final IDK |
| `off_topic`         | Out-of-scope (CRAG INCORRECT) → redirect      | "What's the best restaurant in Buffalo?" |
| `ambiguous_refine`  | CRAG AMBIGUOUS → refinement                   | "Tell me about nerves" |
| `clinical_synth`    | Clinical synthesis + Dean grounding bypass    | mastery → Choice A |
| `function_mode`     | Path B — function-discovery opener            | "I'd like to learn about the cerebellum" |
| `function_reveal`   | Path B — 3 wrong → function-centered reveal   | name + 3 wrong-function attempts |
| `rapport_open`      | Rapport node — casual greeting without topic  | "hey there" |
| `hint_wrong`        | Wrong attempt → hint_error_node scaffold      | wrong-noun guess after Socratic question |
| `mastery_b_topic`   | Mastery → Choice B → topic_choice_node        | mastery → "B" |

The directory is populated; replay them via GET /demo/traces and
GET /demo/traces/<id>.
