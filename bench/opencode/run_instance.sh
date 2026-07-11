#!/usr/bin/env bash
# per-instance OpenCode 실행 래퍼 (Phase 2, GPU 호스트에서 사용)
# 사용: run_instance.sh <instance_id> <repo> <prompt-file> [session_id]
# scope 전달: tenant/repo/instance는 env→옵션 헤더, session은 OpenCode 자동 헤더(D1).
set -euo pipefail

INSTANCE_ID=${1:?instance_id}
REPO=${2:?repo}
PROMPT_FILE=${3:?prompt file}
SESSION_ID=${4:-}

export SPECLEDGER_BASE_URL=${SPECLEDGER_BASE_URL:-http://localhost:8000/v1}
export SPECLEDGER_API_KEY=${SPECLEDGER_API_KEY:-dummy}
export SPECLEDGER_TENANT=${SPECLEDGER_TENANT:-bench}
export SPECLEDGER_REPO="$REPO"
export SPECLEDGER_INSTANCE="$INSTANCE_ID"
export OPENCODE_CONFIG="$(dirname "$0")/opencode.template.jsonc"

ARGS=(run --format json -m specledger/target)
if [[ -n "$SESSION_ID" ]]; then
  ARGS+=(--session "$SESSION_ID")
fi

# --format json: 모든 이벤트 라인에 sessionID 포함 → trace의 scope.session과 대조 가능
exec opencode "${ARGS[@]}" "$(cat "$PROMPT_FILE")"
