# AI Agent Instructions

This repository uses shared handoff logs so multiple AI agents can continue work without losing context.

## Before Starting Work

Read these files first:

1. `AGENT_HANDOFF.md`
2. `AGENT_LOG.md`
3. `NEXT_STEPS.md`

Then inspect the files relevant to the user's request. Do not assume chat history is available.

## While Working

Track:

- current task objective;
- files inspected;
- files changed;
- commands or tests run;
- decisions made;
- blockers, risks, or unfinished work.

Do not log secrets, credentials, API keys, or long command outputs.

## Before Finishing

Update the shared memory files:

1. Append an entry to `AGENT_LOG.md`.
2. Update `AGENT_HANDOFF.md`.
3. Update `NEXT_STEPS.md` if priorities changed.
4. Append to the matching per-agent file in `.agent-logs/`.

Use:

- `.agent-logs/codex.md` for Codex.
- `.agent-logs/gemini.md` for Gemini.
- `.agent-logs/claude.md` for Claude.

## Handoff Rule

A fresh agent should be able to continue from the repository files alone, without needing the previous chat.
