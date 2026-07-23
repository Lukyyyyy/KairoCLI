from kairocli.thought_display import MAX_THOUGHT_DETAIL_CHARS, ThoughtDisplay


def test_thought_display_collapses_and_expands_the_latest_turn() -> None:
    ticks = iter((0.0, 10.0, 11.01))
    display = ThoughtDisplay(lambda: next(ticks))

    display.start()
    display.add("model reasoning")
    display.add("✓ 1 tool(s) · 2 ms")
    display.finish()

    assert display.summary() == "Thought for 1s (ctrl+o to expand)"
    assert display.render() == display.summary()
    assert display.toggle() is True
    assert display.render() == (
        "Thought for 1s (ctrl+o to collapse)\n"
        "model reasoning\n✓ 1 tool(s) · 2 ms"
    )
    assert display.toggle() is True
    assert display.render() == "Thought for 1s (ctrl+o to expand)"


def test_thought_display_is_bounded_redacted_and_terminal_safe() -> None:
    display = ThoughtDisplay(lambda: 1.0)
    display.start()
    display.add("Authorization: Bearer secret\x1b[31m")
    display.add("x" * (MAX_THOUGHT_DETAIL_CHARS + 10))
    display.finish()
    display.toggle()

    rendered = display.render()
    assert "secret" not in rendered
    assert "\x1b" not in rendered
    assert len(display.details) <= MAX_THOUGHT_DETAIL_CHARS + 1


def test_thought_display_cannot_expand_before_a_turn_finishes() -> None:
    display = ThoughtDisplay(lambda: 1.0)

    assert display.toggle() is False
    assert display.render() == ""


def test_dismiss_hides_the_committed_turn_without_losing_history() -> None:
    display = ThoughtDisplay(lambda: 1.0)
    display.start("prompt")
    display.add("reasoning")
    display.finish("answer")
    display.toggle()

    display.dismiss()

    assert display.finished is False
    assert display.expanded is False
    assert display.turns[-1].answer == "answer"


def test_finish_marks_an_answer_as_already_streamed() -> None:
    display = ThoughtDisplay(lambda: 1.0)
    display.start("question")
    display.finish("answer", answer_streamed=True)

    assert display.turns[-1].answer == "answer"
    assert display.turns[-1].answer_streamed is True
    assert display.toggle() is True


def test_live_summary_freezes_when_answer_streaming_starts() -> None:
    now = [0.0]
    display = ThoughtDisplay(lambda: now[0])
    display.start()

    now[0] = 2.9
    assert display.live_summary() == "Thought for 2s (ctrl+o to expand)"

    display.finish_thinking()
    now[0] = 9.9

    assert display.live_summary() == "Thought for 2s (ctrl+o to expand)"
    display.finish("answer")
    assert display.elapsed_seconds == 2


def test_reasoning_deltas_are_concatenated_without_forced_line_breaks() -> None:
    display = ThoughtDisplay(lambda: 1.0)
    display.start()

    display.add_delta("The ")
    display.add_delta("user ")
    display.add_delta("is asking.")

    assert display.details == "The user is asking."


def test_detailed_transcript_contains_only_thought_details() -> None:
    display = ThoughtDisplay(lambda: 1.0)
    display.start("first prompt")
    display.add("first reasoning")
    display.finish("first answer")
    display.start("second prompt")
    display.add("second tool")
    display.finish("second answer")

    transcript = display.render_transcript()

    assert transcript == "  first reasoning\n\n  second tool"
    assert "first prompt" not in transcript
    assert "first answer" not in transcript
    assert "Thought for" not in transcript
