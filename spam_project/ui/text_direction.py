from __future__ import annotations

import re


ARABIC_CHARACTER_PATTERN = re.compile(
    r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]"
)


def contains_arabic(text: str) -> bool:
    """Return whether *text* contains an Arabic-script character."""
    return bool(ARABIC_CHARACTER_PATTERN.search(text))


def format_arabic_for_display(text: str) -> str:
    """Convert logical Arabic text to a Tk-compatible visual representation.

    This function is only for display. The returned value must not be sent to a
    classifier because its character order and glyph forms are presentation
    oriented rather than the original Unicode input.
    """
    if not contains_arabic(text):
        return text

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as error:
        raise RuntimeError(
            "Arabic display support requires arabic-reshaper and python-bidi. "
            "Install them with: pip install arabic-reshaper python-bidi"
        ) from error

    # Process each line independently so paragraph direction does not leak
    # across line breaks in a multi-line SMS message.
    return "\n".join(
        get_display(arabic_reshaper.reshape(line))
        for line in text.split("\n")
    )
