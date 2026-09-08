#!/usr/bin/env bash
# PreToolUse guard — protects the one asset in this repo that cannot be rebuilt.
#
# Every rule carries its reason inline. A rule whose reason is forgotten gets
# deleted by the next person who finds it inconvenient.
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

[[ -z "$FILE_PATH" ]] && exit 0

deny() {
  jq -n --arg reason "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
  exit 2
}

# ── 1. The point-in-time surface archive ──────────────────────────────────
# CLAUDE.md: written "one file per run so nothing already recorded is ever
# rewritten". Each file holds the fitted surface AND the two-sided book it was
# fitted to, at one instant on a venue. The book is gone the moment the market
# moves — no amount of money or compute reconstructs it later.
#
# This is deliberately stricter than the parquet rule in tirramind-universe:
# there, a bad row can be regenerated from snapshots. Here there is no source
# to regenerate from. Overwriting one file is a permanent hole in the record.
#
# Blocks overwrites only. scripts/live_page.py creating a NEW run file is the
# entire point and stays allowed.
if [[ "$FILE_PATH" == *docs/archive/* ]]; then
  if [[ -e "$FILE_PATH" ]]; then
    deny "docs/archive/ is an immutable point-in-time record and $(basename "$FILE_PATH") already exists. The book it holds cannot be re-fetched once the market moves. Write a new run file instead; if a past run was genuinely wrong, keep it and record the correction alongside it."
  fi
fi

# ── 2. .env ───────────────────────────────────────────────────────────────
case "$(basename "$FILE_PATH")" in
  .env|.env.*|*.env)
    [[ "$FILE_PATH" != *.example ]] && \
      deny ".env holds live values and is human-edit-only. Templates ending in .example are editable."
    ;;
esac

exit 0
