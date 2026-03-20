import pytest
import json
from fastapi.testclient import TestClient
from self_healing_bot.app import app, verify_webhook_signature
from self_healing_bot.detector import detect_failures, has_critical_failures, FailureMatch
from self_healing_bot.repair import generate_repair_suggestions, format_pr_body

client = TestClient(app)


# --- Detector tests ---

def test_detect_oom():
    log = "Error: CUDA out of memory. Tried to allocate 2.00 GiB"
    failures = detect_failures(log)
    names = [f.pattern_name for f in failures]
    assert "out_of_memory" in names

def test_detect_cuda_error():
    log = "RuntimeError: CUDA error: device-side assert triggered"
    failures = detect_failures(log)
    names = [f.pattern_name for f in failures]
    assert "cuda_error" in names

def test_detect_data_not_found():
    log = "FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'"
    failures = detect_failures(log)
    names = [f.pattern_name for f in failures]
    assert "data_not_found" in names

def test_detect_import_error():
    log = "ModuleNotFoundError: No module named 'transformers'"
    failures = detect_failures(log)
    names = [f.pattern_name for f in failures]
    assert "dependency_error" in names

def test_detect_nan_loss():
    log = "Training step 100: loss is nan, stopping early"
    failures = detect_failures(log)
    names = [f.pattern_name for f in failures]
    assert "nan_loss" in names

def test_detect_no_failures():
    log = "Training complete. Loss: 0.234. Accuracy: 95.2%"
    failures = detect_failures(log)
    assert len(failures) == 0

def test_detect_multiple_failures():
    log = """
CUDA out of memory at step 50
FileNotFoundError: data/val.json not found
    """
    failures = detect_failures(log)
    assert len(failures) >= 2

def test_has_critical_failures():
    failures = [FailureMatch("out_of_memory", "OOM", 1, "critical", "reduce batch")]
    assert has_critical_failures(failures)

def test_no_critical_failures():
    failures = [FailureMatch("nan_loss", "nan", 1, "warning", "clip grads")]
    assert not has_critical_failures(failures)

def test_deduplicate_failures():
    log = "CUDA out of memory\nCUDA out of memory again\n"
    failures = detect_failures(log)
    oom_count = sum(1 for f in failures if f.pattern_name == "out_of_memory")
    assert oom_count == 1


# --- Repair tests ---

def test_generate_oom_repair():
    failures = [FailureMatch("out_of_memory", "CUDA out of memory", 5, "critical", "reduce batch size")]
    suggestions = generate_repair_suggestions(failures)
    assert len(suggestions) == 1
    assert "memory" in suggestions[0].title.lower() or "oom" in suggestions[0].title.lower()

def test_generate_repair_body_contains_hint():
    failures = [FailureMatch("out_of_memory", "CUDA OOM", 1, "critical", "reduce batch size")]
    suggestions = generate_repair_suggestions(failures)
    assert "batch" in suggestions[0].body.lower()

def test_format_pr_body():
    failures = [FailureMatch("out_of_memory", "CUDA OOM", 1, "critical", "reduce batch")]
    suggestions = generate_repair_suggestions(failures)
    body = format_pr_body(suggestions, "train-model", 12345)
    assert "train-model" in body
    assert "12345" in body
    assert "Self-Healing" in body


# --- API tests ---

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_webhook_workflow_run_completed_failure():
    payload = {
        "action": "completed",
        "workflow_run": {
            "id": 99,
            "name": "Train Model",
            "conclusion": "failure",
            "logs_url": "https://api.github.com/repos/test/repo/actions/runs/99/logs",
            "head_commit": {"message": "CUDA out of memory during training"},
        },
        "repository": {"full_name": "test/repo"},
    }
    resp = client.post(
        "/webhook",
        json=payload,
        headers={"x-github-event": "workflow_run"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processed"

def test_webhook_ignored_non_failure():
    payload = {
        "action": "completed",
        "workflow_run": {
            "id": 100,
            "name": "Train Model",
            "conclusion": "success",
            "head_commit": {"message": "Training done"},
        },
        "repository": {"full_name": "test/repo"},
    }
    resp = client.post(
        "/webhook",
        json=payload,
        headers={"x-github-event": "workflow_run"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ignored"

def test_analyze_logs_endpoint():
    resp = client.post(
        "/analyze-logs",
        json={"log_text": "RuntimeError: CUDA out of memory. Try reducing batch size."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_critical"] is True
    assert len(data["failures"]) > 0

def test_webhook_signature_verify():
    import hashlib, hmac as _hmac
    secret = "testsecret"
    payload = b'{"test": true}'
    sig = "sha256=" + _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(payload, sig, secret)

def test_webhook_signature_invalid():
    assert not verify_webhook_signature(b"payload", "sha256=invalidsig", "secret")
