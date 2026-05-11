You extract evidence from a fetched web page for a specific sub-question.

# Sub-question
{subquestion}

# Page URL
{url}

# Page content (may be truncated)
{content}

# Output format
Return ONLY a JSON object. Pick at most 3 short quotes (each ≤ 240 chars)
that directly answer the sub-question. If nothing relevant is on the page,
return an empty list.

```json
{{
  "evidence": [
    {{"quote": "...", "relevance": 0.0_to_1.0}}
  ]
}}
```
