"""
오디오·통화 계층 시험 (실통화 없음)

  python -m pytest -q test_audio.py
"""

from __future__ import annotations

import array
import random

from clawops.agent._audio import pcm16_to_ulaw, ulaw_to_pcm16

from audio import FRAME, Segmenter, detect_dtmf, dtmf_ulaw, hum_ulaw
from stt import _clean
from telephony import ClawOpsBackend


def _noisy(ulaw: bytes, sd: float) -> bytes:
    a = array.array("h", ulaw_to_pcm16(ulaw))
    for i in range(len(a)):
        a[i] = max(-32768, min(32767, a[i] + int(random.gauss(0, sd))))
    return pcm16_to_ulaw(a.tobytes())


def test_dtmf_tones_decode_all_keys():
    keys = "0123456789*#"
    assert detect_dtmf(dtmf_ulaw(keys)) == keys
    assert detect_dtmf(dtmf_ulaw("1122")) == "1122"  # 같은 키 연속도 각각 인식 (톤 사이 무음)


def test_dtmf_survives_line_noise():
    random.seed(1)
    for _ in range(5):
        assert detect_dtmf(_noisy(dtmf_ulaw("5#9*0"), 2000)) == "5#9*0"


def test_dtmf_not_detected_in_tone_or_silence():
    assert detect_dtmf(hum_ulaw(2000)) == ""
    assert detect_dtmf(b"\xff" * 8000) == ""


def test_dtmf_frames_are_20ms():
    tone = dtmf_ulaw("123")
    assert len(tone) % FRAME == 0  # ClawOps 로 160바이트 단위로 나눠 보낸다


def test_segmenter_cuts_paragraph_after_silence():
    seg = Segmenter(threshold=500, silence_ms=1200)
    loud = dtmf_ulaw("5")[:FRAME]  # 큰 소리 프레임
    quiet = b"\xff" * FRAME
    out = [seg.feed(loud) for _ in range(50)] + [seg.feed(quiet) for _ in range(59)]
    assert not any(out)  # 1.2초가 지나기 전에는 단락을 끝내지 않는다
    assert seg.feed(quiet) is not None


def test_stt_drops_hallucinations():
    assert _clean("시청해 주셔서 감사합니다.") == ""
    assert _clean("감사합니다.") == ""
    assert _clean("잔액조회는  3번") == "잔액조회는 3번"


def test_clawops_fail_reason_mapping():
    m = ClawOpsBackend.FAIL_REASON
    assert m["busy"] == "busy" and m["no-answer"] == "no_answer" and m["rejected"] == "no_answer"
    assert m["canceled"] == "local_hangup" and m["failed"] == "failed"
