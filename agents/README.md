# Review Agents — Reusable Prompts

This folder contains the prompts for four specialist review agents.
Call them any time the plan or code changes significantly to get a fresh review.

## Agents

| File | Role | Focus |
|---|---|---|
| `architect.md` | 🏗️ Senior ML Systems Architect | Design correctness, component integration, dimension consistency |
| `developer.md` | 👨‍💻 Senior PyTorch Developer | API correctness, silent bugs, underspecified details |
| `academic_reviewer.md` | 🎓 NLP Research Reviewer | Methodology, evaluation validity, academic standards |
| `qa_engineer.md` | 🧪 QA / ML Reliability Engineer | Failure modes, edge cases, missing safeguards |
| `diagram_artist.md` | 🎨 Technical Diagram Artist | Publication-quality figures using Graphviz + matplotlib |

## How to Call an Agent

Each `.md` file contains a ready-to-paste prompt. To re-run a review:
1. Open the agent's `.md` file
2. Copy the prompt under `## Prompt`
3. Paste it as a task agent call

## When to Re-Run

| Event | Which agents |
|---|---|
| After implementing any placeholder in `new/` | Developer + QA |
| After changing architecture in `models.py` | Architect + Developer |
| After changing training schedule or evaluation | Academic + QA |
| Before final submission | All four |
| After Phase 1 pipeline is complete | QA + Developer |

## Findings Log

Each agent's `.md` file has a `## Findings` section appended after each run.
The session SQL database (`review_findings` table) is the queryable version.

Query open critical issues:
```sql
SELECT id, reviewer, finding FROM review_findings
WHERE severity = 'critical' AND status = 'open'
ORDER BY reviewer;
```

Mark an issue resolved:
```sql
UPDATE review_findings SET status = 'resolved' WHERE id = 'B1';
```
