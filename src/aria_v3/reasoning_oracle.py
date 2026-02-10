"""
Layer 3: Reasoning Oracle.

Takes structured game reports from Layer 2, feeds them to an LLM,
and produces actionable strategy for Layer 1.

The LLM interprets evidence — it does NOT observe the game directly.
It reads text descriptions of what Layer 2 found and reasons about
what kind of game this is and what the agent should do.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = """\
You are a game analysis AI. You receive structured observations from an agent \
playing an unknown game on a 64x64 pixel grid with 16 colors. Actions are \
numbered 1-7 (1-5 are simple/directional, 6 is click at x,y coordinates, \
7 is undo which is disabled).

Your job: analyze the observations and output a JSON strategy.

Rules:
- Be concise. Output ONLY valid JSON, no explanation.
- "movement_map" maps action IDs to [dx, dy] pixel displacements (if detected).
- "player" identifies the controllable entity by its color.
- "targets" lists objects the agent should interact with.
- "plan" is one of: "pathfind_to_targets", "click_all_targets", "explore_frontier", "systematic_click".
- "click_targets" lists specific (x, y) positions to click (for click games).
- If you can't determine something, omit that field.

Example output:
{"game_type": "navigation", "player": {"color": 12}, "movement_map": {"1": [0, -8], "2": [0, 8], "3": [-8, 0], "4": [8, 0]}, "targets": [{"color": 9, "type": "collect"}], "plan": "pathfind_to_targets"}
"""

USER_PROMPT_TEMPLATE = """\
{report_text}

Analyze this game. What type of game is it? What should the agent do? \
Output ONLY a JSON strategy object."""


@dataclass
class Strategy:
    """Parsed strategy from the LLM."""
    game_type: str = "unknown"
    interpretation: str = ""
    player_color: int | None = None
    movement_map: dict[int, tuple[int, int]] | None = None  # action_id → (dx, dy)
    targets: list[dict] = field(default_factory=list)  # [{"color": int, "type": str}]
    plan: str = "explore_frontier"  # pathfind_to_targets, click_all_targets, explore_frontier, systematic_click
    click_targets: list[tuple[int, int]] | None = None  # [(x, y), ...]
    raw_json: dict = field(default_factory=dict)
    confidence: float = 0.0
    generation_time_ms: float = 0.0


class ReasoningOracle:
    """Layer 3: LLM-based game reasoning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 300,
        load_in_4bit: bool = True,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._load_in_4bit = load_in_4bit

        # Track conversation for context
        self._previous_strategies: list[Strategy] = []

    def load(self) -> None:
        """Load the LLM. Called lazily on first use."""
        if self._loaded:
            return

        print(f"Loading {self.model_name}...")
        start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self._load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        self._loaded = True
        elapsed = time.time() - start
        print(f"Loaded {self.model_name} in {elapsed:.1f}s")

        # Print VRAM usage
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM used: {mem:.1f}GB")

    def analyze(self, report_text: str) -> Strategy:
        """Analyze a game report and produce a strategy.

        Args:
            report_text: Natural language game report from Layer 2

        Returns:
            Parsed Strategy object
        """
        self.load()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(report_text=report_text)},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)

        start = time.time()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy for consistency
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed_ms = (time.time() - start) * 1000

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse JSON from response
        strategy = self._parse_strategy(response)
        strategy.generation_time_ms = elapsed_ms

        self._previous_strategies.append(strategy)
        return strategy

    def _parse_strategy(self, response: str) -> Strategy:
        """Parse LLM response into a Strategy object."""
        strategy = Strategy()

        # Try to extract JSON from the response
        json_str = self._extract_json(response)
        if json_str is None:
            strategy.interpretation = f"Failed to parse: {response[:200]}"
            return strategy

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            strategy.interpretation = f"Invalid JSON: {json_str[:200]}"
            return strategy

        strategy.raw_json = data
        strategy.game_type = data.get("game_type", "unknown")
        strategy.interpretation = data.get("interpretation", "")
        strategy.plan = data.get("plan", "explore_frontier")

        # Player
        player = data.get("player")
        if isinstance(player, dict):
            strategy.player_color = player.get("color")

        # Movement map
        mm = data.get("movement_map")
        if isinstance(mm, dict):
            strategy.movement_map = {}
            for k, v in mm.items():
                try:
                    action_id = int(k)
                    dx, dy = int(v[0]), int(v[1])
                    strategy.movement_map[action_id] = (dx, dy)
                except (ValueError, IndexError, TypeError):
                    continue

        # Targets
        targets = data.get("targets")
        if isinstance(targets, list):
            strategy.targets = targets

        # Click targets
        ct = data.get("click_targets")
        if isinstance(ct, list):
            strategy.click_targets = []
            for item in ct:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    strategy.click_targets.append((int(item[0]), int(item[1])))

        # Confidence heuristic: more fields filled = more confident
        filled = sum([
            strategy.game_type != "unknown",
            strategy.player_color is not None,
            strategy.movement_map is not None and len(strategy.movement_map) > 0,
            len(strategy.targets) > 0,
            strategy.plan != "explore_frontier",
        ])
        strategy.confidence = filled / 5.0

        return strategy

    def _extract_json(self, text: str) -> str | None:
        """Extract a JSON object from potentially messy LLM output."""
        # Try the whole text first
        text = text.strip()
        if text.startswith("{"):
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(text):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[:i+1]

        # Try to find JSON in markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find any JSON object
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return None

    def get_stats(self) -> dict:
        """Get oracle performance statistics."""
        if not self._previous_strategies:
            return {"calls": 0}

        times = [s.generation_time_ms for s in self._previous_strategies]
        return {
            "calls": len(self._previous_strategies),
            "mean_time_ms": sum(times) / len(times),
            "last_confidence": self._previous_strategies[-1].confidence,
            "last_plan": self._previous_strategies[-1].plan,
            "last_game_type": self._previous_strategies[-1].game_type,
        }
