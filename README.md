# self-healing-mlops-bot

GitHub App that automatically detects and repairs ML pipeline failures.

## Features
- **Failure Detection**: Scans workflow logs for OOM, CUDA errors, missing data, import errors, NaN loss
- **Repair Generation**: Creates fix suggestions with code snippets
- **GitHub Webhooks**: Handles `workflow_run` and `check_run` events
- **Auto PR**: Generates repair PRs with detailed fix guidance

## Detected Patterns
| Pattern | Severity | Description |
|---------|----------|-------------|
| `out_of_memory` | critical | CUDA OOM errors |
| `cuda_error` | critical | CUDA runtime errors |
| `data_not_found` | critical | Missing data files |
| `dependency_error` | critical | Missing Python packages |
| `nan_loss` | warning | NaN loss during training |
| `timeout` | warning | Job timeout |

## Setup
```bash
pip install -r requirements.txt
export GITHUB_WEBHOOK_SECRET=your_secret
export GITHUB_TOKEN=your_token
uvicorn self_healing_bot.app:app --reload
```

## Webhook Events
Point GitHub App webhooks to: `POST /webhook`

## Tests
```bash
pytest tests/
```
