"""Repair suggestion generator for ML pipeline failures."""
from typing import List, Dict, Any
from dataclasses import dataclass
from .detector import FailureMatch


@dataclass
class RepairSuggestion:
    failure_name: str
    title: str
    body: str
    files_to_modify: List[str]
    priority: int  # 1=highest


def generate_repair_suggestions(failures: List[FailureMatch]) -> List[RepairSuggestion]:
    """Generate repair suggestions for detected failures."""
    suggestions = []

    for failure in failures:
        if failure.pattern_name == "out_of_memory":
            suggestions.append(RepairSuggestion(
                failure_name="out_of_memory",
                title="fix: reduce memory usage to address OOM error",
                body=f"""## Auto-detected: Out of Memory Error

**Detected at line {failure.line_number}:** `{failure.matched_text}`

### Suggested fixes:

```python
# Option 1: Reduce batch size
batch_size = 16  # was 32 or higher — halve it

# Option 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Option 3: Clear cache before training
import torch
torch.cuda.empty_cache()

# Option 4: Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)
```

**Hint:** {failure.repair_hint}
""",
                files_to_modify=["train.py", "config.yaml"],
                priority=1,
            ))

        elif failure.pattern_name == "cuda_error":
            suggestions.append(RepairSuggestion(
                failure_name="cuda_error",
                title="fix: address CUDA runtime error",
                body=f"""## Auto-detected: CUDA Error

**Detected at line {failure.line_number}:** `{failure.matched_text}`

### Suggested fixes:

```bash
# Add to .env or workflow env:
CUDA_LAUNCH_BLOCKING=1

# In Python code, add device consistency check:
assert tensor.device == model.device, f"Device mismatch: {{tensor.device}} vs {{model.device}}"
```

**Hint:** {failure.repair_hint}
""",
                files_to_modify=[".env", "train.py"],
                priority=1,
            ))

        elif failure.pattern_name == "data_not_found":
            suggestions.append(RepairSuggestion(
                failure_name="data_not_found",
                title="fix: add data validation and path guards",
                body=f"""## Auto-detected: Data Not Found

**Detected at line {failure.line_number}:** `{failure.matched_text}`

### Suggested fix:

```python
import os

def validate_data_paths(config):
    \"\"\"Validate all data paths exist before training starts.\"\"\"
    required_paths = [
        config.get("train_data"),
        config.get("val_data"),
        config.get("test_data"),
    ]
    for path in required_paths:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {{path}}")
    print("All data paths validated OK")

# Call at pipeline start:
validate_data_paths(config)
```

**Hint:** {failure.repair_hint}
""",
                files_to_modify=["pipeline.py", "train.py"],
                priority=1,
            ))

        elif failure.pattern_name == "dependency_error":
            suggestions.append(RepairSuggestion(
                failure_name="dependency_error",
                title="fix: pin missing dependencies in requirements.txt",
                body=f"""## Auto-detected: Missing Dependency

**Detected at line {failure.line_number}:** `{failure.matched_text}`

### Suggested fix:

Add to `requirements.txt` and pin versions:
```
# Pin all deps to avoid resolution issues
torch==2.1.0
transformers==4.35.0
datasets==2.14.0
```

Also add to CI workflow:
```yaml
- name: Install dependencies
  run: pip install -r requirements.txt --no-cache-dir
```

**Hint:** {failure.repair_hint}
""",
                files_to_modify=["requirements.txt", ".github/workflows/train.yml"],
                priority=1,
            ))

        else:
            suggestions.append(RepairSuggestion(
                failure_name=failure.pattern_name,
                title=f"fix: address {failure.pattern_name.replace('_', ' ')}",
                body=f"""## Auto-detected: {failure.pattern_name.replace('_', ' ').title()}

**Detected at line {failure.line_number}:** `{failure.matched_text}`

**Hint:** {failure.repair_hint}
""",
                files_to_modify=[],
                priority=2,
            ))

    return sorted(suggestions, key=lambda s: s.priority)


def format_pr_body(suggestions: List[RepairSuggestion], workflow_name: str, run_id: int) -> str:
    """Format a GitHub PR body from repair suggestions."""
    body = f"""# 🤖 Self-Healing MLOps Bot — Auto-Repair PR

Automatically generated repair for failed workflow: **{workflow_name}** (run #{run_id})

---

"""
    for i, suggestion in enumerate(suggestions, 1):
        body += f"## Fix {i}: {suggestion.title}\n\n"
        body += suggestion.body
        body += "\n---\n\n"

    body += """
> **Note:** Review these suggestions carefully before merging.
> This PR was auto-generated by the self-healing MLOps bot.
"""
    return body
