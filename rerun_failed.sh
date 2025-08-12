#!/usr/bin/env bash
set -euo pipefail

REPO="kengsengwang/Hotel-No-Show-Prediction"  # change if needed
BRANCH="main"

# Requires GitHub CLI: gh auth login  (or GH_TOKEN with repo scope)
echo "Listing failed runs on $REPO ($BRANCH)…"
RUN_IDS=$(gh run list -R "$REPO" --branch "$BRANCH" --status failure --limit 50 \
  --json databaseId -q '.[].databaseId')

if [ -z "$RUN_IDS" ]; then
  echo "No failed runs found."
  exit 0
fi

for id in $RUN_IDS; do
  echo "Re-running failed jobs in run $id…"
  gh run rerun -R "$REPO" "$id" --failed
done

echo "Done. Watch progress in the Actions tab."
