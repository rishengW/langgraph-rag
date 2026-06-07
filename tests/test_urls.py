from __future__ import annotations

from src.utils.urls import parse_url_input


def test_parse_url_input_empty_values():
    assert parse_url_input(None) is None
    assert parse_url_input("") is None
    assert parse_url_input(" , ") is None
    assert parse_url_input([]) is None


def test_parse_url_input_accepts_csv_and_lists():
    assert parse_url_input("https://a.test, https://b.test") == [
        "https://a.test",
        "https://b.test",
    ]
    assert parse_url_input(["https://a.test, https://b.test", " https://c.test "]) == [
        "https://a.test",
        "https://b.test",
        "https://c.test",
    ]

