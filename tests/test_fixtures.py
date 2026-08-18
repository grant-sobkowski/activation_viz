"""Tests for mock activation fixture data."""

from activation_viz.fixtures import TOKENS, _make_tensors, get_default_graph_text

EXPECTED_TOKEN_TEXT = ["The", " capital", " of", " France", " is", " Paris", "."]


def test_make_tensors_is_deterministic():
    """Test that the same seed always produces the same tensors."""
    assert _make_tensors(0) == _make_tensors(0)


def test_make_tensors_different_seeds_differ():
    """Test that different seeds produce different tensors."""
    assert _make_tensors(0) != _make_tensors(1)


def test_make_tensors_shape():
    """Test that tensors have 30 layers of 768 values each, matching GraphConfig.x_size."""
    tensors = _make_tensors(0)
    assert len(tensors) == 30
    for tensor in tensors:
        assert len(tensor) == 768


def test_make_tensors_values_bounded():
    """Test that all tensor values fall within [0.0, 1.0]."""
    tensors = _make_tensors(0)
    for tensor in tensors:
        for value in tensor:
            assert 0.0 <= value <= 1.0


def test_tokens_has_expected_text():
    """Test that TOKENS contains the expected canned response text, in order."""
    assert [text for text, _tensors in TOKENS] == EXPECTED_TOKEN_TEXT


def test_tokens_each_have_full_tensor_shape():
    """Test that every token's tensors match the 30-layer x 768-value shape."""
    for _text, tensors in TOKENS:
        assert len(tensors) == 30
        assert all(len(tensor) == 768 for tensor in tensors)


def test_get_default_graph_text_header():
    """Test that the default graph text's header lists columns 00 through 29."""
    lines = get_default_graph_text().splitlines()
    assert lines[0] == " ".join(f"{i:02d}" for i in range(30))


def test_get_default_graph_text_row_count():
    """Test that the default graph text has one header row plus 31 data rows."""
    lines = get_default_graph_text().splitlines()
    assert len(lines) == 32
