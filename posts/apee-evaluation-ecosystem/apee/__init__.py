"""
APEE - Adaptive Poly-Agentic Evaluation Ecosystem

A comprehensive framework for evaluating multi-agent AI systems.
"""

__version__ = "0.1.0"
__author__ = "ahjavid"

from apee.agents.base import Agent, AgentRole
from apee.agents.ollama import OllamaAgent
from apee.coordination.coordinator import Coordinator
from apee.evaluation.evaluator import Evaluator
from apee.evaluation.report import EvaluationReport

__all__ = [
    "Agent",
    "AgentRole", 
    "OllamaAgent",
    "Coordinator",
    "Evaluator",
    "EvaluationReport",
]
