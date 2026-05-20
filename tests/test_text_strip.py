"""Unit tests for utils.text_strip — char + substring modes (#484)."""

from camelot.utils import text_strip


def test_text_strip_empty_returns_text_unchanged():
    assert text_strip("hello", "") == "hello"


def test_text_strip_str_is_per_character():
    # "abc" means strip ANY of a, b, c — long-standing behaviour kept.
    assert text_strip("cabbage and cake", "abc") == "ge nd ke"


def test_text_strip_list_strips_whole_substrings():
    # Substring mode: "[1]" goes, but bare "[" and "]" stay.
    assert (
        text_strip("Footnote [1] here and [2] there", ["[1]", "[2]"])
        == "Footnote  here and  there"
    )


def test_text_strip_tuple_also_accepted():
    assert text_strip("abXYcdXYef", ("XY",)) == "abcdef"


def test_text_strip_list_with_overlapping_substrings():
    # Alternation is left-greedy; "abc" matches before "ab" at the same
    # position, but for non-overlapping inputs both strip cleanly.
    assert text_strip("abc-ab-c", ["abc", "ab"]) == "--c"


def test_text_strip_list_unicode():
    # Input has spaces around "café" and "crème"; stripping just the two
    # words leaves those spaces in place: "[ ]café[ ]crème[ ]—[ ]espresso"
    # becomes "[ ][ ]—[ ]espresso".
    assert text_strip("café crème — espresso", ["café", "crème"]) == "  — espresso"


def test_text_strip_list_with_empty_entry_skipped():
    # Empty strings would otherwise match everywhere; we drop them silently.
    assert text_strip("hello world", ["", "world"]) == "hello "


def test_text_strip_list_with_only_empties_is_noop():
    assert text_strip("hello world", ["", ""]) == "hello world"


def test_text_strip_list_regex_metacharacters_escaped():
    # "." in the substring should match literal ".", not "any char".
    assert text_strip("a.b.c xyz", [".b"]) == "a.c xyz"
