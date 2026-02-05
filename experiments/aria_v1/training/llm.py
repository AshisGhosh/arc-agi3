"""
ARIA-Lite LLM Integration

Integrates a local LLM (Llama 3.2 1B) for goal hypothesis generation.
Uses llama-cpp-python for efficient inference with GGUF models.

Features:
- Goal hypothesis generation from grid observations
- Response caching for efficiency
- Configurable temperature and context length
- Graceful fallback when model unavailable

Target: ~1GB VRAM for int4 quantized model
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .config import ARIALiteConfig, LLMConfig

# Lazy import to avoid dependency issues
_llama_cpp = None


def _get_llama_cpp():
    """Lazy import llama-cpp-python."""
    global _llama_cpp
    if _llama_cpp is None:
        try:
            import llama_cpp
            _llama_cpp = llama_cpp
        except ImportError:
            _llama_cpp = False
    return _llama_cpp if _llama_cpp else None


@dataclass
class LLMResponse:
    """Response from LLM."""

    text: str
    goal_embedding: Optional[torch.Tensor]  # [goal_dim]
    tokens_used: int
    cached: bool


class LRUCache:
    """Simple LRU cache for LLM responses."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)


class GoalEncoder(torch.nn.Module):
    """Simple encoder to convert text goals to embeddings."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Simple bag-of-words style encoding
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.projection = torch.nn.Linear(embed_dim, embed_dim)

        # Simple tokenizer: hash words to vocab indices
        self.vocab_size = vocab_size

    def tokenize(self, text: str) -> torch.Tensor:
        """Simple hash-based tokenization."""
        words = text.lower().split()
        indices = [hash(w) % self.vocab_size for w in words]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        tokens = self.tokenize(text).unsqueeze(0)
        embedded = self.embedding(tokens)
        return self.projection(embedded).squeeze(0)


class LLMInterface:
    """
    Interface to local LLM for goal hypothesis generation.

    Uses llama-cpp-python for efficient local inference.
    Falls back to simple heuristics if model unavailable.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.model = None
        self.cache = LRUCache(config.cache_size)
        self.goal_encoder = GoalEncoder(embed_dim=64)

        # Try to load model
        self._load_model()

    def _load_model(self):
        """Load the LLM model."""
        llama_cpp = _get_llama_cpp()

        if llama_cpp is None:
            print("Warning: llama-cpp-python not installed. Using fallback mode.")
            return

        model_path = self.config.model_path
        if not model_path or not Path(model_path).exists():
            print(f"Warning: Model not found at {model_path}. Using fallback mode.")
            return

        try:
            self.model = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False,
            )
            print(f"Loaded LLM: {self.config.model_name}")
        except Exception as e:
            print(f"Warning: Failed to load LLM: {e}. Using fallback mode.")
            self.model = None

    def _make_cache_key(self, prompt: str) -> str:
        """Create cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _grid_to_text(self, grid: torch.Tensor) -> str:
        """Convert grid tensor to text description."""
        if grid.dim() == 2:
            H, W = grid.shape
        else:
            _, H, W = grid.shape
            grid = grid[0]

        # Color names
        colors = [
            "black", "blue", "red", "green", "yellow",
            "gray", "pink", "orange", "cyan", "brown",
            "purple", "lime", "olive", "navy", "teal", "white"
        ]

        # Count colors
        color_counts = {}
        for c in range(16):
            count = (grid == c).sum().item()
            if count > 0:
                color_counts[colors[c]] = count

        # Build description
        desc = f"Grid size: {H}x{W}. "
        desc += "Colors present: " + ", ".join(
            f"{c}({n})" for c, n in sorted(color_counts.items(), key=lambda x: -x[1])
        )

        return desc

    def generate_goal_hypothesis(
        self,
        input_grid: torch.Tensor,
        output_grid: Optional[torch.Tensor] = None,
        context: str = "",
    ) -> LLMResponse:
        """
        Generate goal hypothesis from grid observation.

        Args:
            input_grid: [H, W] or [B, H, W] input grid
            output_grid: Optional [H, W] expected output (for training)
            context: Additional context string

        Returns:
            LLMResponse with goal text and embedding
        """
        # Build prompt
        input_desc = self._grid_to_text(input_grid)

        prompt = f"""Analyze this ARC puzzle grid and hypothesize the transformation goal.

Input: {input_desc}
{f"Context: {context}" if context else ""}

What transformation rule might convert this input to the output? Be concise (1-2 sentences)."""

        if output_grid is not None:
            output_desc = self._grid_to_text(output_grid)
            prompt += f"\nOutput: {output_desc}"

        # Check cache
        cache_key = self._make_cache_key(prompt)
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            goal_embedding = self.goal_encoder(cached_response)
            return LLMResponse(
                text=cached_response,
                goal_embedding=goal_embedding,
                tokens_used=0,
                cached=True,
            )

        # Generate response
        if self.model is not None:
            response = self._generate_with_model(prompt)
        else:
            response = self._generate_fallback(input_grid, output_grid)

        # Cache response
        self.cache.put(cache_key, response["text"])

        # Encode goal
        goal_embedding = self.goal_encoder(response["text"])

        return LLMResponse(
            text=response["text"],
            goal_embedding=goal_embedding,
            tokens_used=response["tokens"],
            cached=False,
        )

    def _generate_with_model(self, prompt: str) -> dict:
        """Generate response using loaded model."""
        output = self.model(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=["\n\n", "Input:", "Output:"],
        )

        text = output["choices"][0]["text"].strip()
        tokens = output["usage"]["total_tokens"]

        return {"text": text, "tokens": tokens}

    def _generate_fallback(
        self,
        input_grid: torch.Tensor,
        output_grid: Optional[torch.Tensor] = None,
    ) -> dict:
        """Generate simple heuristic response when model unavailable."""
        if input_grid.dim() == 3:
            input_grid = input_grid[0]

        H, W = input_grid.shape

        # Simple pattern detection
        unique_colors = input_grid.unique().numel()
        is_symmetric_h = torch.allclose(input_grid, input_grid.flip(0))
        is_symmetric_v = torch.allclose(input_grid, input_grid.flip(1))

        hypotheses = []

        if unique_colors <= 3:
            hypotheses.append("simple color transformation")
        if is_symmetric_h or is_symmetric_v:
            hypotheses.append("exploit symmetry")
        if H == W:
            hypotheses.append("possible rotation or reflection")
        if unique_colors > 5:
            hypotheses.append("pattern recognition or filtering")

        if output_grid is not None:
            if output_grid.dim() == 3:
                output_grid = output_grid[0]
            oH, oW = output_grid.shape
            if oH < H or oW < W:
                hypotheses.append("extract or crop region")
            elif oH > H or oW > W:
                hypotheses.append("expand or tile pattern")

        text = "Goal: " + (hypotheses[0] if hypotheses else "transform grid pattern")

        return {"text": text, "tokens": 0}

    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
        }

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.model is not None


def create_llm_interface(config: Optional[ARIALiteConfig] = None) -> LLMInterface:
    """Factory function to create LLM interface from full config."""
    if config is None:
        config = ARIALiteConfig()
    return LLMInterface(config.llm)
