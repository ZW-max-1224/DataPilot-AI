"""Translation-catalog integrity.

These checks are cheap and they prevent a whole class of defects that unit tests
normally miss: a UI key that exists in one language only, a `t["key"]` lookup that
would raise KeyError on a page the test suite does not render, or UI text written
directly in Python so it can never be translated at all.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = PROJECT_ROOT / "i18n"
UI_SOURCE = PROJECT_ROOT / "app" / "ui" / "streamlit_app.py"
LOCALES = ("zh-CN", "en-US")
CJK = re.compile(r"[\u4e00-\u9fff]")

# The language picker shows every language in its own language on purpose, so this
# literal is bilingual by design rather than an untranslated string.
ALLOWED_CJK_LITERALS = {"中文"}


def load_catalog(locale: str) -> dict[str, str]:
    return json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))


def test_both_locales_define_exactly_the_same_keys_in_the_same_order():
    zh, en = load_catalog("zh-CN"), load_catalog("en-US")
    assert set(zh) == set(en), f"key drift: {sorted(set(zh) ^ set(en))}"
    assert list(zh) == list(en), "key order drift makes translation diffs noisy"


def test_every_key_used_by_the_ui_resolves_in_every_locale():
    source = UI_SOURCE.read_text(encoding="utf-8")
    used = set(re.findall(r't\["([A-Za-z0-9_]+)"\]', source))
    used |= set(re.findall(r'translations\["([A-Za-z0-9_]+)"\]', source))
    used |= set(re.findall(r'translations\.get\("([A-Za-z0-9_]+)"', source))
    assert used, "no i18n lookups found; the extraction pattern must have gone stale"
    for locale in LOCALES:
        catalog = load_catalog(locale)
        missing = sorted(key for key in used if key not in catalog)
        assert not missing, f"{locale}.json is missing UI keys: {missing}"


def test_ui_code_has_no_hardcoded_chinese_strings():
    """A hardcoded string is invisible to the translation catalog.

    Roughly 120 lines of UI text were once written directly in Python, and several
    literals contained both languages at once so they could never render in a single
    language. This check keeps that from coming back: any Chinese character inside a
    string literal in the UI module must be explicitly allowlisted.
    """
    tree = ast.parse(UI_SOURCE.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if CJK.search(node.value) and node.value not in ALLOWED_CJK_LITERALS:
                offenders.append(f"line {node.lineno}: {node.value[:70]!r}")
    assert not offenders, (
        "Hardcoded Chinese found in app/ui/streamlit_app.py; move it into "
        "i18n/zh-CN.json and i18n/en-US.json and read it via t[...]:\n" + "\n".join(offenders)
    )


def test_catalogs_are_non_empty_strings():
    for locale in LOCALES:
        for key, value in load_catalog(locale).items():
            assert isinstance(value, str) and value.strip(), f"{locale}.{key} is empty"
