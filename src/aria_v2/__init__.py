"""
ARIA v2: Language-Guided Meta-Learning Architecture

Core insight: Understand games through language reasoning, not end-to-end neural prediction.

Architecture:
    Observation → Visual Grounding → Language Description
                        ↓
               Event Detection → "Player touched diamond, score +1"
                        ↓
               LLM Reasoning → "Diamonds are collectibles"
                        ↓
               Subgoal Executor → Navigate to next diamond

See docs/current/ARCHITECTURE.md for full specification.
"""

__version__ = "0.1.0"

# Components will be added as implemented:
# from .visual_grounding import VisualGroundingModule
# from .event_detector import EventDetector
# from .game_memory import GameStateMemory
# from .llm_reasoning import LLMReasoningEngine
# from .subgoal_executor import SubgoalExecutor
# from .agent import ARIAv2Agent
