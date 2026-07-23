MAX_USER_INPUT_BYTES = 1024 * 1024


class UserInputError(ValueError):
    pass


def normalize_user_input(value: str, max_bytes: int | None = MAX_USER_INPUT_BYTES) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    try:
        size = len(normalized.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise UserInputError("Input contains invalid Unicode surrogate characters") from exc
    if max_bytes is not None and size > max_bytes:
        raise UserInputError(f"Input exceeds the {max_bytes}-byte limit")
    return normalized


def normalize_interactive_submission(value: str) -> str:
    normalized = normalize_user_input(value, max_bytes=None)
    try:
        return normalize_user_input(normalized)
    except UserInputError:
        if "@image:data:" in normalized:
            return normalized
        raise
