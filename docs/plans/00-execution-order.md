# 00 — Execution Order & Status

> Strategy/context: `journal_roadmap.md`. Links: `docs/LINKS.md`. Meeting decisions
> (11 Aug 2026): all work is in scope (no MUST/NICE tiering — Hamza). This file is
> the operational index. Update the Status column as plans complete.

**Deadline: Sunday 30 Aug 2026** (~19 days). APC 1800 CHF (funding TBD). Paper 10–12 pp.
Draft + results summary to Phil by end of this week.

## Operating protocol (per Hamza) — applies to every plan
Subagent loop: **implementer** (only committer; brief+report as files) → **review** gate
(spec + quality) → **verification** (independently re-check numbers; anomalies investigated
before recorded) → support agents in parallel (visual/interpret/latex/audit/teach; never
commit). Ledger: `.superpowers/sdd/progress.md`. **No `Co-Authored-By` trailer** in commits
(`.claude/settings.json`). GP config rule (unified, Plan 10, FINAL): `WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e5))` + `n_restarts_optimizer=15` + subsample GP training folds to <=2000 rows (seed 42).

## Plans

| Plan | What | Status |
|---|---|---|
| [01](01-gp-verification.md) | GP headline verification (grouped CV + noise floor) | DONE |
| [02](02-ifra-ingestion.md) | IFRA newsprint ingestion (wb only) | DONE |
| [03](03-ifra-generalization.md) | IFRA within/cross/leave-one-out | DONE |
| [04](04-llm-predictor.md) | LLM-as-predictor, GPT-4o/mini (preliminary) | DONE (superseded by 08) |
| [05](05-direct-de00-loss.md) | Direct ΔE00 minimization (Powell) | DONE |
| [06](06-ncolor-ladder.md) | **n>4 ladder: n=5 + two n=7, full 3→4→5→7 comparison + figure** | DONE (df8caf8, 2409e23, e3de56d — all gated) |
| [07](07-paper.md) | Paper: fill placeholders, positioning, trim, submit | IN PROGRESS |
| [08](08-multi-llm.md) | Multi-LLM predictor (Claude Fable/Opus, GPT, DeepSeek via OpenRouter) | TODO |
| [09](09-llm-equation.md) | LLM-as-equation-generator (Phil's prompt → portable ≤cubic eqn) | TODO |
| [10](10-gp-consistency.md) | Unify GP config (WhiteKernel 1e-3 + n_restarts=15), IFRA anomaly resolved, all GP re-run | DONE |
| [11](11-colourbill-benchmark.md) | External benchmark vs colourbill tool | DONE (974fbe3 + GP-final figure refresh) |

## Decisions locked (11 Aug grilling)
- All experiments in scope; full plans for all.
- "Dataset A→B" (Phil's LLM prompt) = the ink→colour characterization task, phrased for an LLM.
- LLM work = both flavours: (A) predict colours directly, (B) emit a portable equation. Final paper inclusion decided after seeing results.
- LLM models: Claude Fable, Claude Opus, GPT (latest), DeepSeek (latest), all via **OpenRouter** (needs Hamza's key at execution).
- n-ladder datasets reported **per-dataset** with a measurement-conditions column (different sources/conditions — not one controlled sweep; robustness across independent systems is the framing).
- Kiran/PhD positioning → paper related-work/discussion (Plan 07). **Ref resolved: Deshpande, Green & Pointer, Optics Express 22(26):31786 (2014)** — no longer gated on Phil.

## Owed to / from Phil
- **NOT blocked on Phil for any experiment or writing** — all datasets in hand (n=3/4/5/7); the key reference is resolved (Deshpande & Green, Optics Express 2014).
- Hamza→Phil: draft + results summary (this week).
- Phil→Hamza (admin only, non-blocking): APC funding decision (1800 CHF), author order. Dropped/not needed: Marty's extra sets; IFRA bb chart layout (bb is out of scope).

Done before this plan set (git): AIC correction + `aic2025-published` tag; journal pipeline
(5-fold CV, per-fold scalers, ΔE00 on denorm XYZ, tripwire); results for n=3/n=4 + IFRA + ΔE00-loss;
MDPI template port (`../MDPI-Phil-Journal`, `main.tex`, 14pp, honest placeholders).
