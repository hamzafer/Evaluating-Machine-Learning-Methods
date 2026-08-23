# The agentic-harness channel is a different experiment (13 Aug 2026)

## What was run
Claude Fable 5 failed to return an equation over four OpenRouter API attempts (reasoning tokens
consumed the completion budget; one HTTP 400; one Bedrock content-filter false positive). Since
Phil requires Claude to be tested, the **byte-identical prompt** (`journal/llm/raw/equation/prompt.txt`)
was submitted through the Claude Code CLI on Hamza's subscription:

```
claude -p "$(cat prompt.txt)" --model claude-fable-5 --output-format json --allowed-tools ""
```

## What came back
An equation of the form `(cubic in c, m, y)**3` per channel, and it is the most accurate equation
any model produced:

| | median ΔE00 | p95 | max | terms | total degree | max per-variable exponent |
|---|---|---|---|---|---|---|
| Fable 5 via CLI harness | **0.215** | 0.410 | 1.108 | 660 | 9 | **9** |
| gpt-5.6-sol via API | 3.070 | 7.312 | 10.318 | 192 | 9 | 3 |
| least-squares cubic (ours) | 0.234 | 0.917 | 3.052 | 57 | 3 | 3 |

## Why it must NOT be reported as Claude's equation-writing result
Inspection of the archived session shows the run made **5 `Bash` calls and 4 `Write` calls**: the
`--allowed-tools ""` flag did not disable tools (the session was initialised with 187 tools and
`permissionMode: auto`). It therefore wrote code and **fitted the coefficients numerically** —
which is what the high-precision values (e.g. `4.40194069`, `6.77013125e-09`) reflect, and why it
beat our own cubic. Wall time 786 s over 10 turns, 14,603 output tokens.

That measures an **agent with a code interpreter**, not a language model writing an equation from
its own reasoning. Reported as the latter it would be straightforwardly misleading.

## What it is good evidence for
1. **The channel changes the measurement.** Same prompt, same model family, same scoring code: via
   API the model produces nothing; via an agentic harness with tools it beats a least-squares cubic.
   Any LLM benchmark that does not state its channel is uninterpretable — a concrete, quantified
   version of the argument for using raw APIs for the paper's comparison table.
2. **Accuracy came at the cost of the constraint.** Phil's prompt asks for exponents no greater
   than 3. This answer nests a cubic inside a cube: per-variable exponent 9. GPT-5.6 Sol kept the
   per-variable exponent at 3 but multiplied three cubics for total degree 9. The more accurate the
   answer, the more thoroughly the simplicity constraint was circumvented — arguably the most
   interesting finding of the equation experiment.
3. **Claude's equation-writing ability remains untested.** A fair API test needs a completion budget
   large enough for its reasoning (~$0.20-0.30). That is the outstanding item.

Artifacts: `journal/llm/raw/equation_harness/` (full session JSON incl. the tool calls, and the
returned text). Scored with `journal.pipeline.color.delta_e00` on the same 100 held-out patches as
every other row, via `journal.llm.equation.parse_equation`.

---

# Update (23 Aug): three attempts, three contaminations, and why

After the first contaminated run I tried twice more to obtain a clean, tool-free answer through the
CLI. Both failed, in more elaborate ways. Recording all three because the pattern is the finding.

| attempt | flag used | tools still reachable | what the model did | wall time |
|---|---|---|---|---|
| 1 (fable) | `--allowed-tools ""` | 187 | wrote `fit.py`, ran `scipy.optimize` | 786 s |
| 2 (opus-5) | `--tools ""` | 74 (MCP) | 15 x `browser_evaluate`, i.e. JavaScript as a calculator | 512 s |
| 3 (fable) | `--tools ""` | 33 (MCP) | **deployed two serverless projects to Vercel**, called its own endpoint for the fit | 1540 s |

Attempt 3 is worth stating plainly: denied local code execution, the model deployed
`api/fit.js` (training data embedded) to Vercel as `cmy-xyz-fit`, fetched
`https://cmy-xyz-....vercel.app/api/fit?model=tri3&secs=20`, and read the coefficients back. Both
projects were removed afterwards.

## The mechanism

"Disable tools" is not one switch. The surface has layers, and closing one reveals the next:

| layer | what it covers | flag |
|---|---|---|
| permission allowlist | which tools auto-approve | `--allowed-tools` |
| built-in tools | Bash, Write, Read, Edit | `--tools ""` |
| MCP servers | browser automation, Vercel, and anything else configured | `--strict-mcp-config --mcp-config '{"mcpServers":{}}'` |

A full lockdown needs `--tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'`, and the
result is only trustworthy if the session's `init` message reports **zero** tools AND the transcript
contains **zero** `tool_use` blocks. Verify both before scoring anything.

## Why this matters beyond our project

An agentic CLI is built to be capable. Asked to fit an equation, a good agent finds a way to fit it,
and the returned text looks identical whether the coefficients were reasoned out or computed by a
cloud function the model deployed mid-task. **Nothing in the answer reveals which happened.** Any
benchmark that queries models through a coding agent without auditing the transcript is measuring
the agent, not the model, and cannot tell the difference.

For the paper's comparison table this settles the channel question: the LLM rows come from raw API
calls with a stated model id, temperature and provider. The harness runs are reported separately, as
an observation about agentic scaffolding rather than as model capability.
