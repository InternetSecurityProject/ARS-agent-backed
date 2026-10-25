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


def test_segmenter_hands_chunks_at_pauses():
    calls = []
    seg = Segmenter(threshold=500, silence_ms=1200)
    seg.on_chunk = lambda prefix, start: calls.append((len(prefix), start))
    loud, quiet = dtmf_ulaw("5")[:FRAME], b"\xff" * FRAME
    for _ in range(2):  # 2초 말하고 0.4초 쉼 (단락 안의 쉼)
        for _ in range(100):
            seg.feed(loud)
        for _ in range(20):
            seg.feed(quiet)
    assert len(calls) == 2 and calls[0][1] == 0 and calls[1][1] == calls[0][0]


def test_whisper_prefetch_reuses_recognized_prefix():
    import asyncio

    from stt import WhisperRecognizer

    rec = WhisperRecognizer()
    runs = []

    def fake_run(ulaw, context=""):
        runs.append(len(ulaw))
        return f"조각{len(runs)}"

    rec._run = fake_run
    loud, quiet = dtmf_ulaw("5")[:FRAME], b"\xff" * FRAME
    a = loud * 100
    b = a + loud * 100
    whole = b + quiet * 45  # 마지막 조각 뒤 단락 끝 무음

    async def go():
        await rec.prefetch(a, 0)
        await rec.prefetch(b, len(a))
        return await rec.recognize(whole)

    assert asyncio.run(go()) == "조각1 조각2"
    assert runs == [len(a), len(b) - len(a)]  # 앞부분은 다시 인식하지 않고, 끝 무음은 인식하지 않는다


def test_stt_fixes_ars_terms():
    # 실제 KB 녹음에서 나온 오인식
    assert _clean("빠른 말 서비스는 샷 버튼 잔액조회는 3번") == "빠른 말 서비스는 샵 버튼 잔액조회는 3번"
    assert _clean("국민의 평생 금융 파트너 킬로바이트 국민은행입니다") == "국민의 평생 금융 파트너 KB 국민은행입니다"
    assert _clean("케이 비 청년 미래 적금 안내는 8번") == "KB 청년 미래 적금 안내는 8번"
    assert _clean("본 뱅킹 전용 메뉴는 7번") == "폰뱅킹 전용 메뉴는 7번"
    assert _clean("케이크는 3번") == "케이크는 3번"  # 일반 단어는 바꾸지 않는다
