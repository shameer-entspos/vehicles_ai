"""Pure-Python plate text helpers (no ML / OpenCV dependencies)."""
import re

TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}


def normalize(text):
    """Uppercase letters+digits only (drops spaces, hyphens, junk)."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _matches(s, patterns):
    return any(re.fullmatch(p, s) for p in patterns)


def fix_by_position(s, patterns):
    """Fix OCR mix-ups by position (letters first, digits after).
    Returns (text, valid) where valid = matches one of the patterns."""
    best = None
    for k in range(2, 5):
        if len(s) <= k:
            continue
        pre = "".join(TO_LETTER.get(c, c) if c.isdigit() else c for c in s[:k])
        suf = "".join(TO_DIGIT.get(c, c) if c.isalpha() else c for c in s[k:])
        cand = pre + suf
        if pre.isalpha() and suf.isdigit() and _matches(cand, patterns):
            subs = sum(a != b for a, b in zip(s, cand))
            if best is None or subs < best[1]:
                best = (cand, subs)
    if best:
        return best[0], True
    return s, _matches(s, patterns)


def _edit_leq1(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def similar(a, b):
    """True if two plate strings are equal or differ by one character (OCR noise)."""
    return bool(a) and bool(b) and _edit_leq1(a, b)
