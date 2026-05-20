"""text_replace + read_pdf replace_text= integration tests (#482)."""

from camelot.utils import text_replace


def test_text_replace_empty_is_noop():
    assert text_replace("hello", None) == "hello"
    assert text_replace("hello", {}) == "hello"
    assert text_replace("hello", []) == "hello"


def test_text_replace_basic_dict():
    assert text_replace("foo bar baz", {"bar": "BAR"}) == "foo BAR baz"


def test_text_replace_multiple_keys():
    assert text_replace("foo bar baz", {"foo": "FOO", "baz": "BAZ"}) == "FOO bar BAZ"


def test_text_replace_soft_newline_use_case():
    """Issue #481 motivating example: join soft-broken words."""
    assert text_replace("hyphen-\nated", {"-\n": ""}) == "hyphenated"
    assert text_replace("line one \nline two", {" \n": " "}) == "line one line two"


def test_text_replace_list_of_pairs_also_accepted():
    assert text_replace("ab cd", [("ab", "AB"), ("cd", "CD")]) == "AB CD"


def test_text_replace_longest_match_wins():
    # If both 'abc' and 'ab' could match at the same position, the longer
    # one should be picked (the obvious user expectation).
    assert text_replace("abcd", {"abc": "X", "ab": "Y"}) == "Xd"


def test_text_replace_empty_key_skipped():
    assert text_replace("hello", {"": "X", "ll": "LL"}) == "heLLo"


def test_text_replace_regex_metachars_literal():
    # User-supplied keys are escaped, so '.' matches literal dot only.
    assert text_replace("a.b ab", {".b": "Z"}) == "aZ ab"
    assert text_replace("foo(x)", {"(x)": "<X>"}) == "foo<X>"


def test_text_replace_unicode():
    assert text_replace("café crème", {"café": "coffee"}) == "coffee crème"


def test_text_replace_preserves_unmatched_text():
    assert text_replace("nothing to replace here", {"missing": "X"}) == (
        "nothing to replace here"
    )
