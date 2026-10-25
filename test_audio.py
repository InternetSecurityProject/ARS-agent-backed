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


def test_mic_audio_keeps_remainder_across_messages():
    from audio import FrameChunker

    ch = FrameChunker()
    data = bytes(range(256)) * 16  # 4096바이트를 앱처럼 1024바이트씩 보낸다
    frames = []
    for i in range(0, len(data), 1024):
        frames += ch.feed(data[i:i + 1024])
    assert all(len(f) == FRAME for f in frames)
    assert b"".join(frames) + ch.rest == data  # 버리는 바이트 없이 순서대로
    assert len(frames) == 4096 // FRAME and len(ch.rest) == 4096 % FRAME


def _speech_like(seconds: float, seed: int = 0) -> bytes:
    """말소리처럼 크기가 출렁이는 소리 (음절 120~200ms + 짧은 쉼)."""
    import math
    import struct

    rnd = random.Random(seed)
    pcm = []
    while len(pcm) < seconds * 8000:
        n, amp, f = int(rnd.uniform(0.12, 0.2) * 8000), rnd.uniform(3000, 9000), rnd.uniform(150, 300)
        pcm += [int(amp * math.sin(2 * math.pi * f * i / 8000) * math.sin(math.pi * i / n)) for i in range(n)]
        pcm += [0] * int(rnd.uniform(0.04, 0.12) * 8000)
    return pcm16_to_ulaw(struct.pack(f"<{len(pcm)}h", *pcm))


def _feed_all(seg: Segmenter, ulaw: bytes) -> list[tuple[float, int]]:
    out = []
    for i in range(0, len(ulaw) - FRAME + 1, FRAME):
        r = seg.feed(ulaw[i:i + FRAME])
        if r is not None:
            out.append((i / 8000, len(r)))
    return out


def test_segmenter_ends_announcement_when_hold_music_starts():
    from sim_voice import hold_music

    seg = Segmenter(threshold=500, silence_ms=1200)
    speech = _speech_like(6)
    out = _feed_all(seg, speech + hold_music(20) + _speech_like(3, seed=1) + b"\xff" * 16000)
    # 안내 단락은 음악이 시작되고 몇 초 안에 끝나고, 음악만 있는 단락은 STT 로 보내지 않으며, 상담원 목소리는 따로 나온다
    assert len(out) == 2
    ann_end, ann_len = out[0]
    assert len(speech) / 8000 < ann_end < len(speech) / 8000 + 5
    assert out[1][0] > len(speech) / 8000 + 20


def test_segmenter_does_not_mistake_speech_for_music():
    seg = Segmenter(threshold=500, silence_ms=1200)
    out = _feed_all(seg, _speech_like(25) + b"\xff" * 16000)
    assert len(out) == 1 and not seg.music  # 25초 말소리는 한 단락 그대로


def test_no_prefetch_for_music_only_chunks():
    from sim_voice import hold_music

    calls = []
    seg = Segmenter(threshold=500, silence_ms=1200)
    seg.on_chunk = lambda prefix, start: calls.append(len(prefix))
    _feed_all(seg, b"\xff" * 8000 + hold_music(20) + b"\xff" * 16000)
    assert calls == []  # 대기 음악만 나올 때는 미리 인식을 하지 않는다
