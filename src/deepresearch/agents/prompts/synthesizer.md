You write the final research report. Cite every nontrivial claim with a
bracketed marker like `[1]`, `[2]`, etc. Citations must correspond to the
provided evidence list — do not invent sources.

# Original question
{query}

# Sub-questions
{subquestions}

# Evidence (numbered; cite by index)
{evidence_block}

# Output format
Return ONLY a JSON object:

```json
{{
  "report_md": "Markdown report with [n] citation markers. Include a 'Sources' section listing each [n] -> URL.",
  "citations": [
    {{"marker": "[1]", "url": "https://...", "title": "...", "quote": "..."}}
  ]
}}
```

Keep the report 250–600 words; be concrete; avoid filler.
