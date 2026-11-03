"""벤치마크 지표 (음성 인식·안내 해석 공통).

- 글자 오류율(CER): 공백·문장부호를 뺀 글자 단위 편집 거리 ÷ 정답 글자 수
- 메뉴 키 정확도: 정답 선택지 (키, 이름) 마다 인식 문장에서 이름을 찾고, 그 뒤에 붙은 키가 맞는지
- 숫자열 정확도: 인식 문장에서 가장 긴 숫자열(아라비아 숫자·한국어 숫자)을 뽑아 정답 숫자열과 비교
- 판단 일치: 기대 판단 표기(press:3, ask, ask:ambiguous, nopress, secure, backtrack, end, any, A|B)와 실제 판단 비교
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

_DROP = re.compile(r"[\s.,!?·~\"'()\[\]{}\-:;…]")
_KO_DIGIT = {"공": "0", "영": "0", "빵": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
             "육": "6", "륙": "6", "칠": "7", "팔": "8", "구": "9"}


def norm(s: str | None) -> str:
    return _DROP.sub("", s or "")


def edit_distance(a, b) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)]


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(편집 거리, 정답 글자 수). 여러 단락을 합산해 전체 CER 을 낸다."""
    r, h = norm(ref), norm(hyp)
    return edit_distance(r, h), len(r)


# ---------------------------------------------------------------------------
# 메뉴 키
# ---------------------------------------------------------------------------
_KEY_TOKEN = re.compile(r"([0-9])|(\*|별표|별)|(#|샵|샾|우물정자|우물정|우물|정자)|([공영일이삼사오육칠팔구])(?=번)")


def key_after(text: str, pos: int, window: int = 10) -> str | None:
    """text[pos:] 앞쪽 window 글자 안에서 처음 나오는 키 (0-9, *, #)."""
    m = _KEY_TOKEN.search(text[pos:pos + window])
    if not m:
        return None
    if m.group(1):
        return m.group(1)
    if m.group(2):
        return "*"
    if m.group(3):
        return "#"
    return _KO_DIGIT[m.group(4)]


def find_label(text: str, label: str, threshold: float = 0.7) -> int | None:
    """정규화한 text 에서 label 과 가장 비슷한 구간을 찾아 그 끝 위치를 돌려준다 (비슷한 정도가 threshold 미만이면 None)."""
    lab = norm(label)
    if not lab or not text:
        return None
    if lab in text:
        return text.index(lab) + len(lab)
    best, best_end = 0.0, None
    for w in {max(1, len(lab) - 2), len(lab) - 1, len(lab), len(lab) + 1, len(lab) + 2}:
        for i in range(0, max(1, len(text) - w + 1)):
            r = SequenceMatcher(None, lab, text[i:i + w]).ratio()
            if r > best:
                best, best_end = r, i + w
    return best_end if best >= threshold else None


def menu_pairs(hyp: str, options: list) -> tuple[int, int, list[dict]]:
    """(맞은 쌍, 전체 쌍, 쌍별 결과). options 는 [[키, 이름], ...]."""
    t = norm(hyp)
    detail = []
    ok = 0
    for key, label in options:
        end = find_label(t, label)
        got = key_after(t, end) if end is not None else None
        hit = got == str(key)
        ok += hit
        detail.append({"key": str(key), "label": label, "found": end is not None, "got": got, "ok": hit})
    return ok, len(options), detail


# ---------------------------------------------------------------------------
# 숫자열
# ---------------------------------------------------------------------------
_DIGIT_RUN = re.compile(r"[0-9공영빵일이삼사오육륙칠팔구]{3,}")


def digits_of(text: str) -> str:
    """인식 문장에서 가장 긴 숫자열을 숫자로 바꿔 돌려준다 ('구 오 공 일' → '9501'). 없으면 빈 문자열."""
    runs = _DIGIT_RUN.findall(norm(text))
    if not runs:
        return ""
    best = max(runs, key=len)
    return "".join(_KO_DIGIT.get(c, c) for c in best)


def digit_result(expected: str, hyp: str) -> dict:
    got = digits_of(hyp)
    exp = re.sub(r"[^0-9]", "", expected or "")
    ed = edit_distance(exp, got)
    return {"expected_len": len(exp), "got": got, "exact": got == exp, "edits": ed}


# ---------------------------------------------------------------------------
# 판단
# ---------------------------------------------------------------------------
def scored(expect: str | None) -> bool:
    return bool(expect) and expect != "any"


def decision_match(expect: str, pred: str) -> bool:
    for e in expect.split("|"):
        e = e.strip()
        if e == "any":
            return True
        if e == "nopress" and not pred.startswith("press"):
            return True
        if e in ("ask", "end", "backtrack", "secure", "none") and pred.split(":")[0] == e:
            return True
        if pred == e:
            return True
    return False


def expects_ask(expect: str) -> bool:
    """기대가 '사용자에게 묻기'만 허용하는지 (되묻기 판정 평가용)."""
    alts = [e.strip() for e in expect.split("|")]
    return all(a.startswith("ask") for a in alts)


def dangerous(expect: str, pred: str) -> bool:
    """누르면 안 되는 곳에서 눌렀거나 다른 키를 누른 경우."""
    if not pred.startswith("press") or not scored(expect):
        return False
    return not decision_match(expect, pred)


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------
def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def fmt(v, kind: str = "pct") -> str:
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "s":
        return f"{v:.2f}초"
    return str(v)
