"""벤치마크 도구 시험 (실제 STT·Gemini 호출 없음)."""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

from clawops.agent._audio import ulaw_to_pcm16

from audio import SILENCE_FRAME, hum_ulaw
from bench import metrics as M


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def test_cer_ignores_spaces_and_punctuation():
    assert M.cer_counts("잔액조회는 3번, 거래내역은 4번.", "잔액 조회는 3번 거래내역은 4번") == (0, 14)
    assert M.cer_counts("카드 업무", "하드 업무") == (1, 4)


def test_menu_pairs_find_label_then_key():
    opts = [["3", "잔액조회"], ["4", "거래내역조회"], ["0", "상담직원 연결"], ["#", "빠른 말 서비스"], ["*", "분실신고"]]
    hyp = "분실신고는 별표, 잔액 조회는 3번, 거래 내역 조회는 3번, 상담 직원 연결은 0번, 빠른 말 서비스는 샵 버튼"
    ok, n, detail = M.menu_pairs(hyp, opts)
    assert n == 5 and ok == 4
    assert [d["ok"] for d in detail] == [True, False, True, True, True]   # 거래내역조회 4번을 3번으로 인식


def test_digits_from_spoken_and_arabic():
    assert M.digits_of("입력하신 번호는 구 오 공 일 공 일 일 이 삼 사 오 육 칠 입니다") == "9501011234567"
    assert M.digits_of("입력하신 번호는 950101-1234567 입니다") == "9501011234567"
    assert M.digits_of("이번 달 결제 금액") == ""                     # '이번' 은 숫자로 보지 않음 (3자 미만)
    r = M.digit_result("0200000000", "팩스번호는 공이 공공공공 공공공공 입니다")
    assert r["exact"] and r["edits"] == 0
    r = M.digit_result("0012", "공일이")                               # '공 공' 을 하나로 줄여 받아 적은 경우
    assert not r["exact"] and r["edits"] == 1


def test_decision_tokens():
    assert M.decision_match("press:3", "press:3") and not M.decision_match("press:3", "press:4")
    assert M.decision_match("ask", "ask:ambiguous") and M.decision_match("ask:ambiguous", "ask:ambiguous")
    assert not M.decision_match("ask:ambiguous", "ask:low_confidence")
    assert M.decision_match("nopress", "end:not_found") and M.decision_match("nopress", "ask:low_confidence")
    assert not M.decision_match("nopress", "press:0")
    assert M.decision_match("press:5|ask", "ask:low_confidence")
    assert M.dangerous("nopress", "press:0") and M.dangerous("press:3", "press:4") and not M.dangerous("press:3", "ask:ambiguous")
    assert M.expects_ask("ask") and not M.expects_ask("press:1|ask")


# ---------------------------------------------------------------------------
# 단락 나누기 (make_clips)
# ---------------------------------------------------------------------------
def _audio(*parts) -> bytes:
    """('hum', 초) / ('sil', 초) 를 이어 붙인 μ-law."""
    out = b""
    for kind, sec in parts:
        n = int(sec * 50)
        out += (hum_ulaw() if kind == "hum" else SILENCE_FRAME) * n
    return out


def test_make_clips_marks_zone_decision_and_digits(tmp_path, monkeypatch):
    from bench import make_clips

    # 메뉴(0.8초) → 키 → 입력 안내(0.8초) → 번호 입력 → 되읽기(0.8초)
    ulaw = _audio(("sil", 0.5), ("hum", 0.8), ("sil", 2.0), ("hum", 0.8), ("sil", 2.0), ("hum", 0.8), ("sil", 2.0))
    raw = tmp_path / "raw"
    raw.mkdir()
    with wave.open(str(raw / "N2-1.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(ulaw_to_pcm16(ulaw))
    meta = {"scenario": "N2", "take": 1, "wav": "N2-1.wav", "actions": [
        {"step": 0, "type": "key", "keys": "2", "digits": None, "at_sample": int(3.0 * 8000)},
        {"step": 1, "type": "input", "keys": "9913321234567", "digits": "9913321234567", "at_sample": int(6.0 * 8000)},
    ], "captions": []}
    (raw / "N2-1.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "data"
    path = run(make_clips.make(raw / "N2-1.json", out, None, False, "cpu"))
    clips = json.loads(path.read_text(encoding="utf-8"))["clips"]
    assert len(clips) == 3
    assert [c["zone"] for c in clips] == ["A", "A", "B"]
    assert [c["decision"] for c in clips] == [0, 1, None]
    assert clips[2]["digits"] == "9913321234567"
    assert all((out / "N2-1" / c["file"]).exists() for c in clips)


# ---------------------------------------------------------------------------
# 음성 인식 평가 (stt_eval)
# ---------------------------------------------------------------------------
def _dataset(tmp_path: Path) -> Path:
    """단락 3개짜리 녹음 폴더: 메뉴, 숫자 되읽기, 무음."""
    folder = tmp_path / "data" / "N2-1"
    folder.mkdir(parents=True)
    clips = [
        {"file": "a.wav", "text": "건강보험증 발급은 2번을 눌러 주세요.", "zone": "A", "kind": "menu",
         "options": [["2", "건강보험증 발급"]], "decision": 0},
        {"file": "b.wav", "text": "입력하신 번호는 구 구 일 삼 입니다. 맞으면 1번, 틀리면 2번", "zone": "B", "kind": "confirm",
         "options": [["1", "맞으면"], ["2", "틀리면"]], "digits": "9913", "decision": 1},
        {"file": "c.wav", "text": "", "zone": "B"},
    ]
    for c in clips:
        with wave.open(str(folder / c["file"]), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(ulaw_to_pcm16(_audio(("hum", 0.5))))
    (folder / "clips.json").write_text(json.dumps({"scenario": "N2", "clips": clips}, ensure_ascii=False), encoding="utf-8")
    return tmp_path / "data"


class ScriptRecognizer:
    is_local = True

    def __init__(self, texts):
        self.texts = list(texts)

    async def recognize(self, ulaw):
        return self.texts.pop(0) if self.texts else ""


def test_stt_eval_reports_cer_menu_digits_and_hallucination(tmp_path):
    from bench import stt_eval
    from bench.data import load_recordings

    recs = load_recordings(_dataset(tmp_path))
    texts = ["워밍업", "건강보험증 발급은 2번을 눌러 주세요", "입력하신 번호는 구 일 삼 입니다 맞으면 1번 틀리면 2번", "감사합니다"]
    r = run(stt_eval.eval_model("fake", recs, "cpu", make=lambda n: ScriptRecognizer(texts)))
    assert r["clips"] == 3
    assert r["menu_pairs_n"] == "3/3"
    assert r["digits_n"] == "0/1" and r["digit_error"] == 0.25        # '구 구' 를 하나로 줄임
    assert r["halluc"] == "1/1"                                        # 무음 단락에서 '감사합니다'
    assert r["cer_A"] == 0.0 and r["cer_B"] > 0
    assert "| fake |" in stt_eval.table([r])


# ---------------------------------------------------------------------------
# 해석·판단 평가 (interp_eval)
# ---------------------------------------------------------------------------
class TableInterpreter:
    """정해 둔 표대로 답하는 해석기 (Interpreter 프로토콜)."""

    def __init__(self, table, scores):
        self.table, self.scores, self.goal = table, scores, None

    async def summarize_goal(self, goal):
        self.goal = goal
        return goal

    async def classify(self, text):
        from interpreter import Prompt

        for k, d in self.table.items():
            if k in text:
                return Prompt.from_dict(text, d)
        return Prompt(kind="info", text=text)

    async def score(self, goal, options, path=None):
        return {o.key: self.scores.get((goal, o.label), 0.0) for o in options}

    async def answer_confirm(self, goal, prompt):
        return None


def test_interp_eval_scores_decisions_ask_and_danger(tmp_path):
    import config
    from bench import interp_eval
    from bench.data import load_recordings

    recs = load_recordings(_dataset(tmp_path))
    scenario = {"N2": {"id": "N2", "route": [{"key": "2"}, {"key": "2", "para": 1}],
                       "goals": [{"goal": "건강보험증 발급", "expect": ["press:2", "ask"]},
                                 {"goal": "자동차 보험", "expect": ["nopress", "ask"]}]}}
    table = {"건강보험증 발급은": {"kind": "menu", "options": [{"key": "2", "label": "건강보험증 발급"}]},
             "입력하신 번호는": {"kind": "confirm", "options": [{"key": "1", "label": "맞음"}, {"key": "2", "label": "틀림"}]}}
    scores = {("건강보험증 발급", "건강보험증 발급"): 0.9, ("자동차 보험", "건강보험증 발급"): 0.6}
    st = config.Settings()
    r = run(interp_eval.evaluate("table", lambda: TableInterpreter(table, scores), recs, scenario, st))
    # 목적 1: 2번 누름(정답), 되읽기 확인은 항상 질문(정답). 목적 2: 0.6 이라 2번을 눌러 버림(위험), 되읽기 질문(정답)
    assert r["decision_n"] == "3/4" and r["dangerous"] == 1
    assert r["ask_recall"] == 1.0 and r["ask_precision"] == 1.0
    assert r["kind_n"] == "4/4"
    assert "| table |" in interp_eval.table([r])
