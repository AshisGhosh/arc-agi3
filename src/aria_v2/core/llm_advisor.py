"""
LLM Advisor for ARIA v2.

Uses an LLM to provide strategic advice based on belief state.
The LLM is advisory only - it doesn't control actions directly.

Supports:
- Anthropic Claude API
- OpenAI API
- Ollama (local)
- Fallback heuristics (no API needed)
"""

import os
from dataclasses import dataclass
from typing import Optional
import hashlib
from collections import OrderedDict

from .belief_state import BeliefState


@dataclass
class LLMAdvice:
    """Advice from the LLM."""
    strategy: str  # e.g., "explore", "collect", "find_goal"
    reasoning: str  # Why this strategy
    target_color: Optional[int]  # Color to focus on
    confidence: float
    source: str  # "llm", "heuristic", "cached"


class LRUCache:
    """Simple LRU cache for responses."""

    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, LLMAdvice] = OrderedDict()

    def get(self, key: str) -> Optional[LLMAdvice]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: LLMAdvice):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value


class LLMAdvisor:
    """
    LLM-based advisor for game strategy.

    Analyzes belief state and provides strategic advice.
    """

    def __init__(
        self,
        provider: str = "auto",  # "anthropic", "openai", "ollama", "heuristic", "auto"
        model: str = None,
        cache_size: int = 100,
    ):
        self.provider = provider
        self.model = model
        self.cache = LRUCache(cache_size)
        self.client = None

        # Auto-detect provider
        if provider == "auto":
            self._auto_detect_provider()
        else:
            self._init_provider(provider)

    def _auto_detect_provider(self):
        """Auto-detect available LLM provider."""
        # Try Anthropic first
        if os.getenv("ANTHROPIC_API_KEY"):
            self._init_provider("anthropic")
            return

        # Try OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self._init_provider("openai")
            return

        # Try Ollama (local)
        try:
            import ollama
            self._init_provider("ollama")
            return
        except ImportError:
            pass

        # Fallback to heuristics
        self.provider = "heuristic"
        print("LLMAdvisor: No LLM provider available, using heuristics")

    def _init_provider(self, provider: str):
        """Initialize the specified provider."""
        self.provider = provider

        if provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.model = self.model or "claude-3-haiku-20240307"
                print(f"LLMAdvisor: Using Anthropic ({self.model})")
            except ImportError:
                print("LLMAdvisor: anthropic not installed, falling back to heuristics")
                self.provider = "heuristic"

        elif provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
                self.model = self.model or "gpt-3.5-turbo"
                print(f"LLMAdvisor: Using OpenAI ({self.model})")
            except ImportError:
                print("LLMAdvisor: openai not installed, falling back to heuristics")
                self.provider = "heuristic"

        elif provider == "ollama":
            try:
                import ollama
                self.client = ollama
                # Auto-detect available model if not specified
                if self.model is None:
                    try:
                        models = ollama.list()
                        if models.get("models"):
                            self.model = models["models"][0]["name"]
                        else:
                            self.model = "qwen2.5:1.5b"
                    except:
                        self.model = "qwen2.5:1.5b"
                print(f"LLMAdvisor: Using Ollama ({self.model})")
            except ImportError:
                print("LLMAdvisor: ollama not installed, falling back to heuristics")
                self.provider = "heuristic"

    def _make_cache_key(self, belief_state: BeliefState) -> str:
        """Create cache key from belief state."""
        # Hash key components
        components = [
            str(belief_state.player_identified),
            str(len(belief_state.colors_tested)),
            str(belief_state.levels_completed),
        ]
        for color, belief in sorted(belief_state.color_beliefs.items()):
            components.append(f"{color}:{belief.is_blocker:.1f}:{belief.is_collectible:.1f}")
        return hashlib.md5("|".join(components).encode()).hexdigest()

    def _belief_to_prompt(self, belief_state: BeliefState) -> str:
        """Convert belief state to LLM prompt."""
        prompt = """You are advising an AI agent playing a puzzle game. Analyze the current knowledge and suggest a strategy.

Current Knowledge:
"""
        prompt += f"- Player identified: {belief_state.player_identified}\n"
        prompt += f"- Player position: {belief_state.player_position}\n"
        prompt += f"- Actions taken: {belief_state.total_actions}\n"
        prompt += f"- Levels completed: {belief_state.levels_completed}\n"
        prompt += f"- Colors tested: {len(belief_state.colors_tested)}\n"
        prompt += f"- Positions visited: {len(belief_state.positions_visited)}\n"

        if belief_state.color_beliefs:
            prompt += "\nColor Analysis:\n"
            for color, belief in sorted(belief_state.color_beliefs.items()):
                if belief.total_observations > 0:
                    prompt += f"  Color {color}: "
                    if belief.is_blocker >= 0.7:
                        prompt += "BLOCKER "
                    if belief.is_collectible >= 0.7:
                        prompt += "COLLECTIBLE "
                    if belief.is_goal >= 0.5:
                        prompt += "POSSIBLE_GOAL "
                    prompt += f"(observations: {belief.total_observations})\n"

        prompt += """
Based on this knowledge, what should the agent do next?
Reply in this exact format:
STRATEGY: [explore|collect|find_goal|test_color]
TARGET_COLOR: [number or none]
REASONING: [one sentence explanation]
"""
        return prompt

    def get_advice(self, belief_state: BeliefState) -> LLMAdvice:
        """Get strategic advice based on belief state."""
        # Check cache
        cache_key = self._make_cache_key(belief_state)
        cached = self.cache.get(cache_key)
        if cached:
            cached.source = "cached"
            return cached

        # Generate advice
        if self.provider == "heuristic":
            advice = self._heuristic_advice(belief_state)
        else:
            try:
                advice = self._llm_advice(belief_state)
            except Exception as e:
                print(f"LLM error: {e}, falling back to heuristics")
                advice = self._heuristic_advice(belief_state)

        # Cache and return
        self.cache.put(cache_key, advice)
        return advice

    def _llm_advice(self, belief_state: BeliefState) -> LLMAdvice:
        """Get advice from LLM."""
        prompt = self._belief_to_prompt(belief_state)

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content

        elif self.provider == "ollama":
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response["message"]["content"]

        else:
            return self._heuristic_advice(belief_state)

        # Parse response
        return self._parse_response(text)

    def _parse_response(self, text: str) -> LLMAdvice:
        """Parse LLM response into advice."""
        lines = text.strip().split("\n")
        strategy = "explore"
        target_color = None
        reasoning = "LLM advice"

        for line in lines:
            line = line.strip()
            if line.startswith("STRATEGY:"):
                strategy = line.split(":", 1)[1].strip().lower()
            elif line.startswith("TARGET_COLOR:"):
                val = line.split(":", 1)[1].strip().lower()
                if val != "none" and val.isdigit():
                    target_color = int(val)
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return LLMAdvice(
            strategy=strategy,
            reasoning=reasoning,
            target_color=target_color,
            confidence=0.7,
            source="llm",
        )

    def _heuristic_advice(self, belief_state: BeliefState) -> LLMAdvice:
        """Generate advice using simple heuristics."""
        # If player not found, explore
        if not belief_state.player_identified:
            return LLMAdvice(
                strategy="explore",
                reasoning="Need to identify player first",
                target_color=None,
                confidence=0.9,
                source="heuristic",
            )

        # If we have untested colors, test them
        all_tested = len(belief_state.colors_tested)
        if all_tested < 3:
            return LLMAdvice(
                strategy="test_color",
                reasoning=f"Only tested {all_tested} colors, need more information",
                target_color=None,
                confidence=0.8,
                source="heuristic",
            )

        # If we have known collectibles, collect them
        collectibles = belief_state.get_confident_collectibles()
        if collectibles:
            return LLMAdvice(
                strategy="collect",
                reasoning=f"Found {len(collectibles)} collectible colors",
                target_color=collectibles[0],
                confidence=0.7,
                source="heuristic",
            )

        # If we've been stuck (many actions, no progress), try random exploration
        if belief_state.total_actions > 50 and belief_state.levels_completed == 0:
            return LLMAdvice(
                strategy="explore",
                reasoning="Stuck - try exploring new areas",
                target_color=None,
                confidence=0.5,
                source="heuristic",
            )

        # Default: keep exploring
        return LLMAdvice(
            strategy="explore",
            reasoning="Continue systematic exploration",
            target_color=None,
            confidence=0.6,
            source="heuristic",
        )


def test_advisor():
    """Test the LLM advisor."""
    from .belief_state import BeliefState

    # Create test belief state
    belief = BeliefState()
    belief.player_identified = True
    belief.player_position = (32, 32)
    belief.colors_tested = {1, 2, 3}
    belief.get_color_belief(4).times_blocked_movement = 5
    belief.get_color_belief(5).times_disappeared_on_touch = 2

    # Get advice
    advisor = LLMAdvisor(provider="heuristic")  # Use heuristics for testing
    advice = advisor.get_advice(belief)

    print(f"Strategy: {advice.strategy}")
    print(f"Reasoning: {advice.reasoning}")
    print(f"Target color: {advice.target_color}")
    print(f"Confidence: {advice.confidence}")
    print(f"Source: {advice.source}")


if __name__ == "__main__":
    test_advisor()
