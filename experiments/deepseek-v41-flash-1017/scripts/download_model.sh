#!/usr/bin/env bash
# Run on lich-6000; completed files are reused. HF 2.0 restarts incomplete files.
set -euo pipefail
ROOT="${DSV41_ROOT:-$HOME/src/dsv41-1017-rtx6000}"
MODEL="${PACK:-$HOME/models/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-mcg-hq}"
REV=a8006e99931e7d67c4d218c66543f478b521b857
mkdir -p "$ROOT/logs" "$MODEL"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
printf '%s\n' "$(date -Is) download_start revision=$REV"
ok=0
for attempt in 1 2 3; do
  if "${HF_BIN:-$HOME/.local/bin/hf}" download sfxnz/DeepSeek-V4.1-Flash-1017-EXL3 \
    --revision "$REV" --local-dir "$MODEL" --max-workers 4; then
    ok=1; break
  fi
  printf '%s\n' "$(date -Is) download_attempt_failed attempt=$attempt"
  sleep "$((attempt * 30))"
done
if [[ "$ok" != 1 ]]; then
  printf '%s\n' "$(date -Is) download_failed_after_retries"
  exit 1
fi
python3 "$ROOT/scripts/verify_weights.py" "$ROOT/evidence/sfxnz-model-info.json" \
  "$MODEL" "$ROOT/logs/weight-verification.json"
printf '%s\n' "$(date -Is) download_and_sha256_verification_complete"
