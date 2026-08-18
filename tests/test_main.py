"""Tests for the activation-viz Tk application.

TokenManager's Tk-widget wiring (create_sidebar/create_display, popups,
background-thread playback) and ProfiledSmolLM inference require a live
display and a downloaded model, respectively — neither is available in CI, so
only the pure computation on tensors/graph text is covered here. Manual
verification uses `USE_MOCK_LLM=1 uv run activation-viz`.
"""

import tkinter as tk
from typing import cast

from activation_viz.llm import ProfiledToken
from activation_viz.main import GraphConfig, TokenManager


def test_graph_config_fields():
    """Test that GraphConfig stores its fields as given."""
    config = GraphConfig(x_size=30, y_size=32, threshold=0.33)
    assert config.x_size == 30
    assert config.y_size == 32
    assert config.threshold == 0.33


def _manager(threshold: float) -> TokenManager:
    """Build a TokenManager without a real Tk root, for testing pure computation only."""
    mgr = TokenManager(root=cast(tk.Tk, None), graph_config=GraphConfig(x_size=2, y_size=2, threshold=threshold))
    mgr.tokens = [ProfiledToken(text="x", tensors=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])]
    mgr.curr_token = 0
    return mgr


def test_compress_tensors_batches_by_mean():
    """Test that tensors are downsampled to y_size via batch means."""
    mgr = _manager(threshold=3.0)
    assert mgr._compress_tensors() == [[0.5, 2.5], [4.5, 6.5]]


def test_compress_tensors_raises_on_uneven_division():
    """Test that a tensor length not evenly divisible by y_size raises ValueError."""
    mgr = _manager(threshold=3.0)
    mgr.tokens = [ProfiledToken(text="x", tensors=[[0.0, 1.0, 2.0]])]
    try:
        mgr._compress_tensors()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_graph_weights_marks_activations_above_threshold():
    """Test that graph text marks activations above the threshold as X and others as a dot."""
    mgr = _manager(threshold=3.0)
    compressed = mgr._compress_tensors()
    text = mgr._graph_weights(compressed)
    assert text == " 00 01\n  .  X\n  .  X\n"
