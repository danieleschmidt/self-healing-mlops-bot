"""FastAPI webhook handler for GitHub events."""
import hashlib
import hmac
import os
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Header

from .detector import detect_failures, has_critical_failures
from .repair import generate_repair_suggestions, format_pr_body

app = FastAPI(title="Self-Healing MLOps Bot", version="0.1.0")

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "test_secret")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature or not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.get("/health")
def health():
    return {"status": "ok", "bot": "self-healing-mlops-bot"}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None),
):
    """Handle incoming GitHub webhook events."""
    payload_bytes = await request.body()

    # Verify signature if secret is set and non-test
    if GITHUB_WEBHOOK_SECRET and GITHUB_WEBHOOK_SECRET != "test_secret":
        if not verify_webhook_signature(payload_bytes, x_hub_signature_256 or "", GITHUB_WEBHOOK_SECRET):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = x_github_event or payload.get("type", "unknown")

    if event == "workflow_run":
        return await handle_workflow_run(payload)
    elif event == "check_run":
        return await handle_check_run(payload)
    else:
        return {"status": "ignored", "event": event}


async def handle_workflow_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process workflow_run events."""
    action = payload.get("action", "")
    workflow_run = payload.get("workflow_run", {})

    if action != "completed":
        return {"status": "ignored", "reason": f"action={action}"}

    conclusion = workflow_run.get("conclusion", "")
    if conclusion not in ("failure", "timed_out"):
        return {"status": "ignored", "reason": f"conclusion={conclusion}"}

    # Extract details
    workflow_name = workflow_run.get("name", "unknown")
    run_id = workflow_run.get("id", 0)
    repo_full_name = payload.get("repository", {}).get("full_name", "")

    # Get logs URL (in real app would fetch via GitHub API)
    logs_url = workflow_run.get("logs_url", "")

    # For demo: use head_commit message as a stand-in for logs if no real logs
    head_commit = workflow_run.get("head_commit", {})
    simulated_log = head_commit.get("message", "")

    failures = detect_failures(simulated_log) if simulated_log else []

    return {
        "status": "processed",
        "event": "workflow_run",
        "workflow": workflow_name,
        "run_id": run_id,
        "conclusion": conclusion,
        "failures_detected": len(failures),
        "failures": [{"name": f.pattern_name, "severity": f.severity} for f in failures],
        "repair_triggered": len(failures) > 0 and has_critical_failures(failures),
    }


async def handle_check_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process check_run events."""
    action = payload.get("action", "")
    check_run = payload.get("check_run", {})

    if action != "completed":
        return {"status": "ignored", "reason": f"action={action}"}

    conclusion = check_run.get("conclusion", "")
    if conclusion != "failure":
        return {"status": "ignored", "reason": f"conclusion={conclusion}"}

    # Extract output text
    output = check_run.get("output", {})
    log_text = f"{output.get('title', '')} {output.get('summary', '')} {output.get('text', '')}"

    failures = detect_failures(log_text)

    return {
        "status": "processed",
        "event": "check_run",
        "check_name": check_run.get("name", "unknown"),
        "failures_detected": len(failures),
        "failures": [{"name": f.pattern_name, "severity": f.severity} for f in failures],
    }


@app.post("/analyze-logs")
async def analyze_logs(request: Request) -> Dict[str, Any]:
    """Directly analyze log text for failures."""
    body = await request.json()
    log_text = body.get("log_text", "")

    if not log_text:
        raise HTTPException(status_code=400, detail="log_text required")

    failures = detect_failures(log_text)
    suggestions = generate_repair_suggestions(failures)

    return {
        "failures": [
            {"name": f.pattern_name, "severity": f.severity, "hint": f.repair_hint}
            for f in failures
        ],
        "suggestions": [
            {"title": s.title, "files": s.files_to_modify}
            for s in suggestions
        ],
        "has_critical": has_critical_failures(failures),
    }
