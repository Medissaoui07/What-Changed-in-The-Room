
# Codex Quick Start

This repository implements What Changed in the Room?

## Before making changes

1. Read `AGENTS.md`.
2. Read `README.md`.
3. Inspect `docs/ARCHITECTURE.md`.
4. Check the current phase in `docs/ROADMAP.md`.
5. Inspect existing code and tests.

## How to work

- Work on one phase at a time.
- Do not implement future phases prematurely.
- Preserve working functionality.
- Test every meaningful change.
- Report exact commands and results.
- Keep neural models optional until the geometry baseline works.
- Ask before downloading large model weights.
- Never claim functionality that is not implemented and tested.

## Initial task

Start with Phase 0 in `docs/ROADMAP.md`.

Inspect the repository and environment. Propose a minimal implementation plan and dependency list.

Do not start 3D reconstruction, semantic model integration, or UI development before the preprocessing stage has been implemented and tested.

Use `docs/CODEX_PROMPTS.md` for the prompts associated with each phase.
