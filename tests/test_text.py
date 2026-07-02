"""Tests for the streaming sentence chunker that drives TTS latency."""

from chuchote.text import drain_sentences


def test_splits_completed_sentences_and_keeps_remainder():
    sentences, rest = drain_sentences("Hello there. How are you? I am")
    assert sentences == ["Hello there.", "How are you?"]
    assert rest == " I am"


def test_no_boundary_yet_buffers_everything():
    sentences, rest = drain_sentences("no boundary yet")
    assert sentences == []
    assert rest == "no boundary yet"


def test_newline_is_a_boundary():
    sentences, rest = drain_sentences("Line one\nLine two")
    assert sentences == ["Line one"]
    assert rest == "Line two"


def test_streamed_pieces_accumulate_across_calls():
    buffer = ""
    out = []
    for piece in ["Hel", "lo.", " Bye", "!"]:
        buffer += piece
        sentences, buffer = drain_sentences(buffer)
        out += sentences
    assert out == ["Hello.", "Bye!"]
    assert buffer == ""


def test_ellipsis_and_multiple_terminators():
    sentences, rest = drain_sentences("Wait... really?! Yes")
    assert sentences == ["Wait...", "really?!"]
    assert rest == " Yes"
