"""Translation-catalog integrity.

These two checks are cheap and they prevent a whole class of defects that unit tests
normally miss: a UI key that exists in one language only, or a `t["key"]` lookup that
would raise KeyError on a page the test suite does not render.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = PROJECT_ROOT / "i18n"
LOCALES = ("zh-CN", "en-US")


def load_catalog(locale: str) -> dict[str, str]:
    return json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))


def test_both_locales_define_exactly_the_same_keys_in_the_same_order():
    zh, en = load_catalog("zh-CN"), load_catalog("en-US")
    assert set(zh) == set(en), f"key drift: {sorted(set(zh) ^ set(en))}"
    assert list(zh) == list(en), "key order drift makes translation diffs noisy"


def test_every_key_used_by_the_ui_resolves_in_every_locale():
    source = (PROJECT_ROOT / "app" / "ui" / "streamlit_app.py").read_text(encoding="utf-8")
    used = set(re.findall(r't\["([A-Za-z0-9_]+)"\]', source))
    used |= set(re.findall(r'translations\["([A-Za-z0-9_]+)"\]', source))
    used |= set(re.findall(r'translations\.get\("([A-Za-z0-9_]+)"', source))
    assert used, "no i18n lookups found; the extraction pattern must have gone stale"
    for locale in LOCALES:
        catalog = load_catalog(locale)
        missing = sorted(key for key in used if key not in catalog)
        assert not missing, f"{locale}.json is missing UI keys: {missing}"


def test_catalogs_are_non_empty_strings():
    for locale in LOCALES:
        for key, value in load_catalog(locale).items():
            assert isinstance(value, str) and value.strip(), f"{locale}.{key} is empty"
