"""A pronunciation dictionary for every script: alias and phoneme rules, as Fish Audio's.

Fish Audio's TTS API takes pronunciation dictionaries (fishaudio/docs #292, FISH-1580);
the Studio keeps one, applied to every script before the engine sees it, the Studio's and
the API's:

- a phoneme rule replaces its key with <|phoneme_start|>value<|phoneme_end|>: the word
  read as those phonemes (CMU Arpabet, pinyin, romaji, or IPA, which Drama 3 accepts in
  any language);
- an alias rule replaces its key with its value, read like the rest of the script
  ("km/h" -> "kilometers per hour", "k8s" -> "Kubernetes").

Matching follows the docs: the first of duplicate keys wins (keys differing only in case
are duplicates unless the rule is case sensitive), the earliest match in the text wins and
at the same start the longest key, phoneme and alias rules together; replacements are not
matched again. Matching is case-insensitive unless a rule is case sensitive. Here a key
that starts or ends with a letter or digit also matches only whole words ("Dr." is not
replaced in "HDr."), and [directions], <tags>, words already given as phonemes and a
Direction: line are left alone.

The rules are kept in DRAMA3_PRONUNCIATION (a JSON list of {key, value, type,
case_sensitive}), read at startup and written on every change. PLS lexicons import
(<phoneme> and <alias> entries; one pair of surrounding / dropped from a phoneme) and
export.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

PATH = os.environ.get("DRAMA3_PRONUNCIATION", os.path.join("..", "cache", "pronunciation.json"))
TYPES = ("phoneme", "alias")
PLS_NS = "http://www.w3.org/2005/01/pronunciation-lexicon"
# Left as written: [directions], <tags> and <|tokens|>, words given as phonemes
KEEP = re.compile(r"<\|phoneme_start\|>.*?<\|phoneme_end\|>|<\|[^|>]*\|>|\[[^\[\]]*\]|<[^<>]*>", re.S)
ALIAS_KEY_FORBIDDEN = ("<|", "|>", "(", ")", "[", "]")


@dataclass(frozen=True)
class Rule:
    key: str
    value: str
    type: str = "phoneme"
    case_sensitive: bool = False


def problem(rule: Rule) -> str | None:
    """Why a rule cannot be used (the docs' limits), or None."""
    if not rule.key or not rule.value:
        return "needs both an input and an output"
    if rule.type not in TYPES:
        return f"type must be {' or '.join(TYPES)}"
    if any(marker in text for text in (rule.key, rule.value) for marker in ("<|", "|>")):
        return "cannot contain <| or |> (phoneme rules are wrapped for you)"
    if rule.type == "alias":
        if rule.key != rule.key.strip():
            return "an alias input cannot start or end with a space"
        if any(unicodedata.category(char) == "Cc" for char in rule.key):
            return "an alias input cannot contain control characters"
        if any(char in rule.key for char in ALIAS_KEY_FORBIDDEN):
            return "an alias input cannot contain ( ) [ ]"
    elif any(char in rule.value for char in "<>\n"):
        return "phonemes cannot contain < > or a line break"
    return None


def checked(rules: list[Rule]) -> tuple[list[Rule], list[str]]:
    """The usable rules, in order, and a message per rule left out."""
    kept, seen, errors = [], set(), []
    for number, rule in enumerate(rules, 1):
        if (why := problem(rule)) is not None:
            errors.append(f"rule {number} ({rule.key or 'empty'}): {why}")
            continue
        identity = rule.key if rule.case_sensitive else rule.key.lower()
        if identity in seen:
            errors.append(f"rule {number} ({rule.key}): repeats an earlier input, which wins")
            continue
        seen.add(identity)
        kept.append(rule)
    return kept, errors


class Dictionary:
    def __init__(self, path: str = PATH) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.rules: list[Rule] = []
        self.pattern: re.Pattern | None = None
        try:
            with open(path, encoding="utf-8") as f:
                self.set(rules_from_json(json.load(f)), save=False)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as error:
            print(f"[pronunciation] {path} not read ({type(error).__name__}: {error}); starting empty", flush=True)

    def set(self, rules: list[Rule], *, save: bool = True) -> list[str]:
        """Use these rules from the next take; the unusable ones are left out (returned)."""
        kept, errors = checked(rules)
        pattern = None
        if kept:
            # Longest keys first, so at one start the longest wins; finditer takes the earliest
            order = sorted(range(len(kept)), key=lambda i: -len(kept[i].key))
            pattern = re.compile("|".join(f"(?P<r{i}>{boundary(kept[i])})" for i in order))
        with self.lock:
            self.rules, self.pattern = kept, pattern
        if save:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([asdict(rule) for rule in kept], f, ensure_ascii=False, indent=1)
        return errors

    def apply(self, text: str) -> str:
        with self.lock:
            rules, pattern = self.rules, self.pattern
        if pattern is None:
            return text
        head = ""
        if text.startswith("Direction:") and "\nLine:" in text:  # the direction is not read
            head, text = text.split("\nLine:", 1)
            head += "\nLine:"

        def replaced(match: re.Match) -> str:
            rule = rules[int(match.lastgroup[1:])]
            return f"<|phoneme_start|>{rule.value}<|phoneme_end|>" if rule.type == "phoneme" else rule.value

        out, last = [], 0
        for kept in KEEP.finditer(text):
            out += [pattern.sub(replaced, text[last:kept.start()]), kept.group()]
            last = kept.end()
        out.append(pattern.sub(replaced, text[last:]))
        return head + "".join(out)


def boundary(rule: Rule) -> str:
    """The key as a pattern: whole words where it starts or ends with a letter or digit
    (not a CJK one: CJK text has no spaces between words)."""
    body = re.escape(rule.key)
    if not rule.case_sensitive:
        body = f"(?i:{body})"

    def wordy(char: str) -> bool:
        return char.isalnum() and ord(char) < 0x2E80

    before = r"(?<!\w)" if wordy(rule.key[0]) else ""
    after = r"(?!\w)" if wordy(rule.key[-1]) else ""
    return before + body + after


def rules_from_json(items) -> list[Rule]:
    return [Rule(key=str(item.get("key", "")), value=str(item.get("value", "")),
                 type=str(item.get("type") or "phoneme"), case_sensitive=bool(item.get("case_sensitive", False)))
            for item in items]


def rules_from_pls(data: bytes | str) -> list[Rule]:
    """<lexeme> entries of a PLS lexicon: each <grapheme> a rule, from its <phoneme> (one
    pair of surrounding / dropped) or <alias>. PLS has no case sensitivity: off."""
    root = ET.fromstring(data)
    rules = []
    for lexeme in root.iter():
        if lexeme.tag.rsplit("}", 1)[-1] != "lexeme":
            continue
        parts = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in lexeme}
        graphemes = [(child.text or "").strip() for child in lexeme if child.tag.rsplit("}", 1)[-1] == "grapheme"]
        if parts.get("phoneme"):
            value, kind = parts["phoneme"], "phoneme"
            if len(value) > 1 and value.startswith("/") and value.endswith("/"):
                value = value[1:-1]
        elif parts.get("alias"):
            value, kind = parts["alias"], "alias"
        else:
            continue
        rules += [Rule(key=grapheme, value=value, type=kind) for grapheme in graphemes if grapheme]
    return rules


def to_pls(rules: list[Rule]) -> str:
    ET.register_namespace("", PLS_NS)
    root = ET.Element(f"{{{PLS_NS}}}lexicon", {"version": "1.0", "alphabet": "ipa", "xml:lang": "en-US"})
    for rule in rules:
        lexeme = ET.SubElement(root, f"{{{PLS_NS}}}lexeme")
        ET.SubElement(lexeme, f"{{{PLS_NS}}}grapheme").text = rule.key
        ET.SubElement(lexeme, f"{{{PLS_NS}}}{rule.type}").text = rule.value
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


DICTIONARY: Dictionary | None = None


def dictionary() -> Dictionary:
    global DICTIONARY
    if DICTIONARY is None:
        DICTIONARY = Dictionary()
    return DICTIONARY


def install(server) -> None:
    """GET / PUT /api/pronunciation: the rules as JSON, {"rules": [{key, value, type,
    case_sensitive}, ...]}; a PUT replaces them all and answers with the rules in use and
    a message per rule left out."""
    from fastapi import Body, Depends

    app = server.app

    def listed() -> list[dict]:
        return [asdict(rule) for rule in dictionary().rules]

    @app.get("/api/pronunciation", dependencies=[Depends(server.identify)])
    def get_rules() -> dict:
        return {"rules": listed()}

    @app.put("/api/pronunciation", dependencies=[Depends(server.identify)])
    def put_rules(rules: list[dict] = Body(..., embed=True)) -> dict:
        errors = dictionary().set(rules_from_json(rules))
        return {"rules": listed(), "errors": errors}

    for _ in range(2):  # ahead of the Studio's mount at /
        app.router.routes.insert(0, app.router.routes.pop())
