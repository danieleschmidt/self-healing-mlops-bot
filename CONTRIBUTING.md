# Contributing to Self-Healing MLOps Bot

Thank you for your interest in contributing to the Self-Healing MLOps Bot! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/danieleschmidt/self-healing-mlops-bot.git
   cd self-healing-mlops-bot
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Follow the existing code style
   - Add tests for new functionality

3. **Run quality checks**
   ```bash
   # Format code
   black self_healing_bot/ tests/
   isort self_healing_bot/ tests/
   
   # Run linting
   flake8 self_healing_bot/ tests/
   
   # Type checking
   mypy self_healing_bot/
   
   # Security scanning
   bandit -r self_healing_bot/
   
   # Run tests
   pytest --cov=self_healing_bot
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new detector for deployment failures"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with the following modifications:
- Line length: 88 characters (Black default)
- Use double quotes for strings
- Use type hints for all function signatures

### Code Structure

```python
"""Module docstring explaining the purpose."""

import standard_library
import third_party_packages

from . import local_imports

# Constants
CONSTANT_VALUE = "value"

# Classes and functions with proper docstrings
class ExampleClass:
    """Class docstring following Google style."""
    
    def __init__(self, param: str) -> None:
        """Initialize the class.
        
        Args:
            param: Description of parameter.
        """
        self.param = param
    
    def example_method(self, value: int) -> str:
        """Example method with proper documentation.
        
        Args:
            value: Input value to process.
            
        Returns:
            Processed string value.
            
        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Value must be non-negative")
        return str(value)
```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Example:
```
feat(detectors): add support for custom metric thresholds

- Allow users to configure custom thresholds for detectors
- Add validation for threshold ranges
- Update documentation with examples

Closes #123
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

from self_healing_bot.core.bot import SelfHealingBot


class TestSelfHealingBot:
    """Test suite for SelfHealingBot."""
    
    def test_process_event_success(self, mock_context):
        """Test successful event processing."""
        # Arrange
        bot = SelfHealingBot()
        event_data = {"type": "workflow_run", "conclusion": "failure"}
        
        # Act
        result = bot.process_event("workflow_run", event_data)
        
        # Assert
        assert result is not None
        assert result.has_error() is False
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test asynchronous operations."""
        # Test async code here
        pass
```

### Test Categories

- **Unit Tests**: Test individual functions/methods in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test scalability and performance

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bot.py

# Run with coverage
pytest --cov=self_healing_bot --cov-report=html

# Run only unit tests
pytest -m unit

# Run with verbose output
pytest -v
```

## Contributing Types

### 1. Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant logs or error messages

### 2. Feature Requests

For new features:
- Describe the use case and problem it solves
- Provide examples of how it would be used
- Consider backward compatibility
- Discuss implementation approach if applicable

### 3. Documentation

We welcome improvements to:
- Code documentation and docstrings
- README and setup instructions
- Architecture documentation
- Examples and tutorials
- API documentation

### 4. Code Contributions

#### Adding New Detectors

1. Create a new file in `self_healing_bot/detectors/`
2. Inherit from `BaseDetector`
3. Implement required methods
4. Add comprehensive tests
5. Update documentation

Example:
```python
from .base import BaseDetector

class CustomDetector(BaseDetector):
    """Detect custom issues in ML pipelines."""
    
    def get_supported_events(self) -> List[str]:
        return ["workflow_run", "push"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        # Implementation here
        pass
```

#### Adding New Playbooks

1. Create a new playbook class
2. Register with the playbook registry
3. Implement trigger conditions and actions
4. Add tests and documentation

Example:
```python
from self_healing_bot.core.playbook import Playbook, Action

@Playbook.register("custom_handler")
class CustomPlaybook(Playbook):
    """Handle custom ML pipeline issues."""
    
    def should_trigger(self, context: Context) -> bool:
        return context.event_type == "custom_event"
    
    @Action(order=1)
    def fix_issue(self, context: Context) -> str:
        # Implementation here
        return "Issue fixed"
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release PR
- [ ] Tag release after merge
- [ ] Publish to PyPI
- [ ] Update Docker images

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord**: For real-time chat (link in README)

### Communication Guidelines

- Be respectful and inclusive
- Provide context and details
- Use clear, descriptive titles
- Search existing issues before creating new ones
- Follow up on your contributions

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Annual contributor awards

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Questions?

If you have questions about contributing, please:
1. Check this document and existing issues
2. Ask in GitHub Discussions
3. Reach out to maintainers directly

Thank you for contributing to making ML pipelines more reliable and autonomous!