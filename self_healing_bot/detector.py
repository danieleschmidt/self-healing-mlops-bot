"""ML pipeline failure detector.

Parses workflow logs and check run outputs for common ML failure patterns.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FailureMatch:
    pattern_name: str
    matched_text: str
    line_number: int
    severity: str  # "critical", "warning", "info"
    repair_hint: str


# Common ML failure patterns
FAILURE_PATTERNS = [
    {
        "name": "out_of_memory",
        "patterns": [
            r"CUDA out of memory",
            r"RuntimeError: CUDA error: out of memory",
            r"torch\.cuda\.OutOfMemoryError",
            r"OOM when allocating tensor",
            r"Killed\s*$",  # OOM kill signal
        ],
        "severity": "critical",
        "hint": "Reduce batch_size or use gradient checkpointing. Add 'torch.cuda.empty_cache()' before training loop.",
    },
    {
        "name": "cuda_error",
        "patterns": [
            r"CUDA error: device-side assert triggered",
            r"RuntimeError: CUDA error:",
            r"cudaErrorIllegalAddress",
            r"CUDA kernel launch failed",
        ],
        "severity": "critical",
        "hint": "Set CUDA_LAUNCH_BLOCKING=1 for better error messages. Check tensor device consistency.",
    },
    {
        "name": "data_not_found",
        "patterns": [
            r"FileNotFoundError:.*\.(csv|json|parquet|pkl|pt|pth|h5|hdf5)",
            r"No such file or directory",
            r"Dataset not found",
            r"OSError: \[Errno 2\]",
            r"HuggingFace dataset.*not found",
        ],
        "severity": "critical",
        "hint": "Verify data paths in config. Add data existence check at pipeline start. Consider using relative paths.",
    },
    {
        "name": "dependency_error",
        "patterns": [
            r"ModuleNotFoundError: No module named",
            r"ImportError:",
            r"cannot import name",
        ],
        "severity": "critical",
        "hint": "Pin dependency versions in requirements.txt. Run 'pip install -r requirements.txt' in CI.",
    },
    {
        "name": "nan_loss",
        "patterns": [
            r"loss is nan",
            r"Loss became NaN",
            r"nan loss detected",
            r"gradient is nan",
        ],
        "severity": "warning",
        "hint": "Add gradient clipping (clip_grad_norm_). Check learning rate (try 10x smaller). Add loss.isnan() guard.",
    },
    {
        "name": "timeout",
        "patterns": [
            r"TimeoutError",
            r"Job exceeded maximum time",
            r"Error: The operation was canceled",
        ],
        "severity": "warning",
        "hint": "Increase timeout in workflow YAML or reduce dataset size. Add checkpoint/resume logic.",
    },
    {
        "name": "permission_denied",
        "patterns": [
            r"PermissionError: \[Errno 13\]",
            r"Permission denied",
            r"Access denied",
        ],
        "severity": "warning",
        "hint": "Check file permissions and secrets configuration. Ensure GITHUB_TOKEN has required scopes.",
    },
]


def detect_failures(log_text: str) -> List[FailureMatch]:
    """Scan log text for known ML failure patterns."""
    matches = []
    lines = log_text.split("\n")

    for pattern_def in FAILURE_PATTERNS:
        for line_num, line in enumerate(lines, start=1):
            for pattern in pattern_def["patterns"]:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    matches.append(FailureMatch(
                        pattern_name=pattern_def["name"],
                        matched_text=line.strip(),
                        line_number=line_num,
                        severity=pattern_def["severity"],
                        repair_hint=pattern_def["hint"],
                    ))
                    break  # one match per pattern per line is enough

    # Deduplicate by pattern_name (keep first occurrence)
    seen = set()
    deduped = []
    for m in matches:
        if m.pattern_name not in seen:
            seen.add(m.pattern_name)
            deduped.append(m)

    return deduped


def has_critical_failures(matches: List[FailureMatch]) -> bool:
    return any(m.severity == "critical" for m in matches)
