from kairocli.text_safety import bound_utf8, safe_text


class HostileText:
    def __str__(self) -> str:
        raise KeyboardInterrupt


def test_safe_text_repairs_surrogates_and_hostile_string_conversion() -> None:
    assert safe_text("before\ud800after") == "before?after"
    assert safe_text(HostileText(), fallback="stable fallback") == "stable fallback"


def test_bound_utf8_never_splits_characters_or_exceeds_budget() -> None:
    bounded = bound_utf8("界" * 10, 10, "...")
    assert bounded == "界界..."
    assert len(bounded.encode("utf-8")) <= 10
    assert bound_utf8("payload", 2, "marker") == "ma"
    assert bound_utf8("payload", 0, "marker") == ""
