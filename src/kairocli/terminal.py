_BIDI_CONTROLS = {
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}


def sanitize_terminal_text(value: str) -> str:
    """Remove terminal control protocols from untrusted display text.

    This intentionally leaves ordinary formatting characters (including Markdown
    and brackets) untouched. Renderers that support markup must still disable
    markup interpretation for the returned value.
    """

    sanitizer = TerminalStreamSanitizer()
    return sanitizer.feed(value) + sanitizer.finish()


class TerminalStreamSanitizer:
    """Stateful sanitizer for terminal protocols split across arbitrary chunks."""

    def __init__(self) -> None:
        self._state = "normal"
        self._after_cr = False

    def feed(self, value: str) -> str:
        output: list[str] = []
        for character in value:
            if self._state == "escape":
                if character == "[":
                    self._state = "csi"
                elif character in "]PX^_":
                    self._state = "string"
                else:
                    self._state = "normal"
                continue
            if self._state == "csi":
                if 0x40 <= ord(character) <= 0x7E:
                    self._state = "normal"
                continue
            if self._state == "string":
                if character in {"\x07", "\x9c"}:
                    self._state = "normal"
                elif character == "\x1b":
                    self._state = "string_escape"
                continue
            if self._state == "string_escape":
                if character == "\\":
                    self._state = "normal"
                elif character != "\x1b":
                    self._state = "string"
                continue
            if self._after_cr:
                self._after_cr = False
                if character == "\n":
                    continue
            if character == "\r":
                output.append("\n")
                self._after_cr = True
                continue
            if character == "\x1b":
                self._state = "escape"
                continue
            codepoint = ord(character)
            if character == "\x9b":
                self._state = "csi"
                continue
            if character in {"\x90", "\x98", "\x9d", "\x9e", "\x9f"}:
                self._state = "string"
                continue
            if character in _BIDI_CONTROLS:
                continue
            if (codepoint < 0x20 and character not in {"\n", "\t"}) or (0x7F <= codepoint <= 0x9F):
                continue
            output.append(character)
        return "".join(output)

    def finish(self) -> str:
        self._state = "normal"
        self._after_cr = False
        return ""
