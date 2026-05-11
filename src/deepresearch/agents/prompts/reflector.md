You reflect on a completed research run and propose memory updates that
would help future runs.

# Question
{query}

# Plan that ran
{plan}

# Report produced
{report}

# Output format
Return ONLY a JSON object. Any field may be null.

```json
{{
  "personal_update": "A durable user preference learned during this run, or null.",
  "task_update": "A search/research strategy that worked or failed, with WHY, or null.",
  "tool_update": "A lesson about source quality or search formulation, or null.",
  "needs_revision": false,
  "broadcast_candidate": null
}}
```

Only emit a `personal_update` if the user clearly expressed (or demonstrated
by selection) a preference that will persist across runs. If unsure, null.
