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

Five canonical traces to capture:

| id | label | message |
|---|---|---|
| `funny_bone`        | Funny bone — clean Socratic flow         | What nerve causes the funny bone sensation? |
| `idk_x3`            | Three idks → reveal via teach_node       | (record three turns of "I don't know") |
| `off_topic`         | Out-of-scope (CRAG INCORRECT) → redirect | What's the best restaurant in Buffalo?       |
| `ambiguous_refine`  | CRAG AMBIGUOUS → refinement              | Tell me about nerves                         |
| `clinical_synth`    | Clinical synthesis + Dean grounding      | (Socratic flow that hits clinical synthesis) |

The directory is intentionally empty until traces are captured.
