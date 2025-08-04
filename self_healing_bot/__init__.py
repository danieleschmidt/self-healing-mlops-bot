"""
Self-Healing MLOps Bot - Autonomous ML Pipeline Repair and Drift Detection

This package provides a comprehensive framework for monitoring, detecting, and
automatically repairing issues in ML pipelines and deployments.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core.bot import SelfHealingBot
from .core.playbook import Playbook, Action
from .core.context import Context
from .detectors.base import BaseDetector
from .actions.base import BaseAction
from .integrations.github import GitHubIntegration

__all__ = [
    "SelfHealingBot",
    "Playbook",
    "Action",
    "Context",
    "BaseDetector",
    "BaseAction",
    "GitHubIntegration",
]