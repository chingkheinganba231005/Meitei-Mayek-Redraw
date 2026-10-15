# AI use log

Dated record of what AI tools did in this project, kept so the journal's disclosure is accurate.

| Date | Tool | What it did |
|---|---|---|
| 2026-10-15 | Claude Code (Anthropic) | Literature search for related work; designed and implemented the stroke renderer, stroke decoder, stroke head, hypothesis-driven redrawing and training code; wrote the tests and the round-1 Colab notebook; ran CPU sanity checks (per-image stroke fitting on 55 real samples, synthetic parsing, toy look-alike experiment). |
| 2026-10-15 | Claude Code (Anthropic) | Implemented StrokeWarp (first version with rigid per-stroke transforms replaced by bends and tails after a visual check on 55 real samples), the stroke-head switch, the end-to-end ablation switch, `verify_eval`, tests and the round-2 notebook; checked that round-1 training is unchanged. |
