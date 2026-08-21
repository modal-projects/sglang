#!/bin/bash
# Probe series for Claude Code CLI vs the mock Anthropic server.
# Usage: bash probe.sh 'prompt1' 'prompt2' ...   (each prompt = one isolated run)
HARNESS=/home/ec2-user/sglang/docs_new/harness
export HOME=/tmp/claude-home
export ANTHROPIC_BASE_URL=http://127.0.0.1:8077
export ANTHROPIC_AUTH_TOKEN=test
export ANTHROPIC_MODEL=mock-claude
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
mkdir -p "$HOME"
run() {
  local prompt="$1"
  echo "###### PROMPT: $prompt ######"
  out=$(/home/ec2-user/.local/bin/claude -p "$prompt" --output-format json 2>&1)
  rc=$?
  echo "rc=$rc"
  echo "$out"
  echo "$out" | tail -1 | "$HARNESS/.venv312/bin/python" -c "
import json,sys
line = sys.stdin.read().strip()
try:
    d = json.loads(line)
    print('  => is_error:', d.get('is_error'), '| stop:', d.get('stop_reason'), '| turns:', d.get('num_turns'), '| api_error_status:', d.get('api_error_status'))
    print('  => result:', repr(d.get('result'))[:300])
    print('  => output_tokens_details:', d.get('usage', {}).get('output_tokens_details'), '| permission_denials:', d.get('permission_denials'))
except Exception:
    print('  => RAW last line:', line[:400])
"
}
for p in "$@"; do run "$p"; done
