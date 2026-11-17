"""
오디오·통화 계층 시험 (실제 발신 없음)

  python -m pytest -q test_audio.py
"""

from __future__ import annotations

import array
import asyncio
import random

from clawops.agent._audio import pcm16_to_ulaw, ulaw_to_pcm16

from audio import FRAME, Segmenter, detect_dtmf, dtmf_ulaw, hum_ulaw
from stt import _clean, _plausible
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
    assert len(tone) % FRAME == 0  # ClawOps 로 160바이트 단위 분할 전송


def test_segmenter_cuts_paragraph_after_silence():
    seg = Segmenter(threshold=500, silence_ms=1200)
    loud = dtmf_ulaw("5")[:FRAME]  # 큰 소리 프레임
    quiet = b"\xff" * FRAME
    out = [seg.feed(loud) for _ in range(50)] + [seg.feed(quiet) for _ in range(59)]
    assert not any(out)  # 무음 1.2초 전에는 단락 종료 없음
    assert seg.feed(quiet) is not None
    assert seg.last_forced is False


def test_segmenter_marks_paragraph_cut_at_max_length():
    """무음 없이 최대 길이(max_s)를 넘긴 단락의 절단 반환과 last_forced 표시(말하는 도중 절단) 확인."""
    seg = Segmenter(threshold=500, silence_ms=1200, max_s=2.0)
    loud = dtmf_ulaw("5")[:FRAME]
    quiet = b"\xff" * FRAME
    out = [seg.feed(loud) for _ in range(120)]
    cut = [i for i, x in enumerate(out) if x is not None]
    assert cut and seg.last_forced is True


def test_segmenter_cuts_long_paragraph_at_its_last_pause():
    """최대 길이에서 자를 때 마지막 쉼(0.3초)에서 절단, 쉼 뒤 소리는 버리지 않고 다음 단락의 앞부분으로 이월되는지 확인."""
    seg = Segmenter(threshold=500, silence_ms=1200, max_s=3.0)
    loud = dtmf_ulaw("5")[:FRAME]
    quiet = b"\xff" * FRAME
    frames = [loud] * 60 + [quiet] * 20 + [loud] * 100            # 1.2초 말, 0.4초 쉼, 계속 말함 (3초에서 잘림)
    out = [x for x in (seg.feed(f) for f in frames) if x is not None]
    assert len(out) == 1 and seg.last_forced is True
    assert len(out[0]) // FRAME < 90                               # 쉼 자리(약 1.5초)에서 절단 (최대 3초 지점이 아님)
    assert seg.speaking and len(seg.buf) // FRAME > 60             # 쉼 뒤 소리는 다음 단락으로 이월
    total = len(out[0]) + len(seg.buf)
    assert total // FRAME >= 175                                   # 버린 소리 없음 (앞 200ms 버퍼를 빼고 거의 전부)


def test_stt_drops_hallucinations():
    assert _clean("시청해 주셔서 감사합니다.") == ""
    assert _clean("감사합니다.") == ""
    assert _clean("잔액조회는  3번") == "잔액조회는 3번"
    assert _clean("MBC 뉴스 김민수입니다.") == ""


def test_repetition_tail_does_not_drop_real_menu():
    """긴 메뉴 안내 끝에 반복 환각(같은 말 × 22)이 붙어 압축률 기준에 걸려도 구간을 통째로 버리지 않고 반복만 줄여 메뉴를 살리는지 확인.
    숫자 되읽기 반복은 축약하지 않음."""
    from types import SimpleNamespace
    from stt import _compression_ratio, _cut_loops, _segment_text

    menu = "반품 소포 접수는 1번, 국내 등기 우편 조회는 2번, 업무 시간 및 각종 안내는 5번을 눌러주십시오. "
    looped = menu + "우체국 소포, " * 22
    seg = SimpleNamespace(text=looped, no_speech_prob=0.0, avg_logprob=-0.05, compression_ratio=_compression_ratio(looped))
    assert seg.compression_ratio > 2.4 and not _plausible(seg)
    assert _segment_text(seg).startswith(menu.strip()) and _segment_text(seg).count("우체국 소포") == 1
    # 말이 아닌 소리·확신 없는 문장은 그대로 폐기
    assert _segment_text(SimpleNamespace(text=looped, no_speech_prob=0.9, avg_logprob=-0.05, compression_ratio=2.9)) == ""
    assert _segment_text(SimpleNamespace(text=looped, no_speech_prob=0.0, avg_logprob=-1.5, compression_ratio=2.9)) == ""
    # 같은 숫자가 이어지는 되읽기·숫자열은 축약하지 않음
    assert _cut_loops("입력하신 번호는 일 일 일 일 이 이 삼 사 번입니다") == "입력하신 번호는 일 일 일 일 이 이 삼 사 번입니다"
    assert _cut_loops("1111 2222 번입니다") == "1111 2222 번입니다"


def test_repetition_split_into_many_segments_is_judged_per_window():
    """faster-whisper 는 압축률 등을 디코딩 창 전체로 계산해 그 창의 모든 구간에 부여. 반복이 구간 22개로 나뉘어 나와도
    창 단위로 합쳐 반복을 줄이고, 줄인 것이 없는 반복(숫자)은 폐기하는지 확인."""
    from types import SimpleNamespace
    from stt import _window_texts

    menu = "반품 소포 접수는 1번, 업무 시간 및 각종 안내는 5번을 눌러주십시오."
    win = dict(seek=0, no_speech_prob=0.05, avg_logprob=-0.2, compression_ratio=3.7)
    segs = [SimpleNamespace(text=menu, **win)] + [SimpleNamespace(text=" 우체국 소포,", **win) for _ in range(22)]
    out = " ".join(_window_texts(segs))
    assert out.startswith(menu) and out.count("우체국 소포") == 1
    digits = [SimpleNamespace(text="1 1 1 1 1 1 1 1 1 1 1 1", seek=0, no_speech_prob=0.59, avg_logprob=-0.99,
                              compression_ratio=5.5) for _ in range(3)]
    assert _window_texts(digits) == []
    # 창이 정상이면 구간 문장을 그대로 연결
    ok = [SimpleNamespace(text="잔액 조회는 3번,", seek=0, no_speech_prob=0.0, avg_logprob=-0.1, compression_ratio=1.3),
          SimpleNamespace(text="상담원 연결은 0번", seek=0, no_speech_prob=0.0, avg_logprob=-0.1, compression_ratio=1.3)]
    assert _window_texts(ok) == ["잔액 조회는 3번, 상담원 연결은 0번"]


def test_stt_hallucination_drops_only_that_sentence():
    # 실제 안내 뒤에 지어낸 문장이 붙어도 안내는 유지
    assert _clean("잔액조회는 1번, 분실신고는 2번을 눌러 주세요. 시청해 주셔서 감사합니다.") == \
        "잔액조회는 1번, 분실신고는 2번을 눌러 주세요."
    assert _clean("시청해 주셔서 감사합니다. 감사합니다.") == ""
    assert _clean("감사합니다. 감사합니다.") == "" and _clean("시청해 주셔서 감사합니다. 감사합니다. 감사합니다.") == ""
    assert _clean("지금까지 MBC 뉴스였습니다.") == ""
    # 방송사 이름이 든 방송사 ARS 인사는 유지
    assert _clean("KBS 뉴스 제보센터 입니다. 제보는 1번") == "KBS 뉴스 제보센터 입니다. 제보는 1번"
    # 미리 인식 조각에는 단락 단위 판정 미적용 (쉼 뒤 따로 인식된 끝맺음 '감사합니다' 보존)
    assert _clean("감사합니다.", piece=True) == "감사합니다."
    # 안내 끝의 '감사합니다' 는 유지 (결과·업무시간 외 판정 규칙이 읽음)
    assert _clean("이용해 주셔서 감사합니다. 감사합니다.") == "이용해 주셔서 감사합니다. 감사합니다."
    from gemini_interpreter import rule_classify
    assert rule_classify(_clean("고객님의 현재 등급은 골드 등급입니다. 감사합니다.")).kind == "result"
    assert rule_classify(_clean("지금은 상담 시간이 아닙니다. 감사합니다.")).kind == "out_of_hours"
    # 메뉴 끝의 '…입니다. 감사합니다.' 는 결과로 보지 않음
    assert rule_classify(_clean("잔액 조회는 1번, 카드 분실 신고는 2번, 상담원 연결은 0번입니다. 감사합니다.")).kind == "menu"
    # 소수점은 문장 경계 아님
    assert _clean("금리는 연 3.5%입니다. 자막 제공 및 자막 제작") == "금리는 연 3.5%입니다."
    # 방송사 ARS 의 실제 안내 보존, 인사말도 문장 일부면 유지
    assert _clean("KBS 뉴스 제보는 1번, 시청자 상담은 2번입니다.") == "KBS 뉴스 제보는 1번, 시청자 상담은 2번입니다."
    assert _clean("감사합니다 고객님, 1번을 눌러 주세요.") == "감사합니다 고객님, 1번을 눌러 주세요."


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
    assert runs == [len(a), len(b) - len(a)]  # 앞부분 재인식 없음, 끝 무음 인식 없음


def test_stt_does_not_rewrite_specific_words():
    """특정 기관·업무 낱말에 맞춘 교정 규칙 미사용 확인 (범용성 유지, 인식 문장 그대로)."""
    for text in ("국민의 평생 금융 파트너 킬로바이트 국민은행입니다", "본 뱅킹 전용 메뉴는 7번", "분실 보이스티싱은 8번",
                 "케이크는 3번", "샷을 찍어 주세요"):
        assert _clean(text) == text


def test_stt_fixes_common_key_names_only():
    """모든 ARS 공통 키 이름의 교정 확인 ('샷 버튼' → '샵 버튼'). 해석기의 '#' 선택지 인식용."""
    assert _clean("빠른 말 서비스는 샷 버튼 잔액조회는 3번") == "빠른 말 서비스는 샵 버튼 잔액조회는 3번"
    assert _clean("다른 서비스는 샾버튼을 눌러 주십시오") == "다른 서비스는 샵 버튼을 눌러 주십시오"


def test_mic_audio_keeps_remainder_across_messages():
    from audio import FrameChunker

    ch = FrameChunker()
    data = bytes(range(256)) * 16  # 4096바이트를 앱처럼 1024바이트씩 전송
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
    # 안내 단락은 음악 시작 뒤 몇 초 안에 종료, 음악만 있는 단락은 STT 로 보내지 않음, 상담원 목소리는 별도 단락
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
    assert calls == []  # 대기 음악만 나올 때는 미리 인식 없음


def test_stt_native_number_keys_in_menu():
    # "1번" 을 "한 번" 으로 적은 인식 결과에서 AI 가 선택지를 놓치는 경우 대비
    assert _clean("보험료 관련 문의는 한 번 자격 및 증명서 발급은 2번") == "보험료 관련 문의는 1번 자격 및 증명서 발급은 2번"
    assert _clean("최근거래내역은 한 번 기간별 거래내역은 두 번을 눌러 주십시오") == "최근거래내역은 1번 기간별 거래내역은 2번을 눌러 주십시오"
    assert _clean("잔액조회는 세 번, 거래내역조회는 네 번") == "잔액조회는 3번, 거래내역조회는 4번"
    assert _clean("한 번 더 눌러 주세요") == "한 번 더 눌러 주세요"  # 메뉴 안내 형태가 아니면 그대로
    assert _clean("이번 달 결제 예정 금액은 두 번 나누어") == "이번 달 결제 예정 금액은 두 번 나누어"


def test_stt_drops_made_up_segments():
    from types import SimpleNamespace as Seg

    real = Seg(no_speech_prob=0.05, compression_ratio=1.4, avg_logprob=-0.3)  # ARS 안내를 받아 적은 조각
    assert _plausible(real)
    # 잡음에서 지어낸 조각: 말이 아니라고 판단 / "일어나서 일어나서 …" 반복 / 확신 없음
    assert not _plausible(Seg(no_speech_prob=0.8, compression_ratio=1.5, avg_logprob=-0.5))
    assert not _plausible(Seg(no_speech_prob=0.1, compression_ratio=5.2, avg_logprob=-0.2))
    assert not _plausible(Seg(no_speech_prob=0.3, compression_ratio=1.6, avg_logprob=-1.4))


def test_whisper_uses_voice_activity_filter_by_default(monkeypatch):
    from stt import WhisperRecognizer

    monkeypatch.delenv("STT_VAD", raising=False)
    assert WhisperRecognizer().vad is True  # 말소리 없는 구간 인식 제외 (잡음에서 문장 생성 방지)
    monkeypatch.setenv("STT_VAD", "0")
    assert WhisperRecognizer().vad is False


def test_long_audio_is_split_at_longest_pause():
    """25초 넘는 음성을 그 안의 가장 긴 쉼에서 나누는지 확인 (Whisper 에 30초 넘는 음성을 넣으면 앞부분이 버려짐)."""
    from audio import SILENCE_FRAME
    from stt import split_at_pauses

    speech = hum_ulaw(1000)
    short_pause, long_pause = SILENCE_FRAME * 15, SILENCE_FRAME * 40      # 0.3초, 0.8초 쉼
    audio = speech * 12 + short_pause + speech * 3 + long_pause + speech * 10 + short_pause + speech * 8
    pieces = split_at_pauses(audio, 25.0, quiet_rms=50)
    assert b"".join(pieces) == audio and all(len(p) <= 25 * 8000 for p in pieces)
    long_start = len(speech * 12 + short_pause + speech * 3)
    assert long_start < len(pieces[0]) < long_start + len(long_pause)        # 가장 긴 쉼(0.8초) 안에서 절단
    assert split_at_pauses(speech * 5, 25.0) == [speech * 5]               # 짧으면 그대로


def test_cut_short_transcript_is_warned():
    """받아쓴 마지막 시각이 조각의 말소리 끝보다 CUT_SHORT_WARN_S(6초) 넘게 앞설 때의 누락 의심 경고 확인 (동작 변경 없음)."""
    from types import SimpleNamespace

    from audio import SILENCE_FRAME
    from stt import _warn_if_cut_short, voiced_end_s

    speech = hum_ulaw(1000)                                                    # 1초 소리
    audio = speech * 10 + SILENCE_FRAME * 50                                   # 10초 말, 1초 무음
    assert abs(voiced_end_s(audio, 50) - 10.0) < 0.05
    assert _warn_if_cut_short([SimpleNamespace(end=2.0)], audio, 50) > 7       # 2초까지만 받아 적음 → 경고
    assert _warn_if_cut_short([SimpleNamespace(end=9.5)], audio, 50) is None   # 끝까지 받아 적음
    assert _warn_if_cut_short([SimpleNamespace(end=7.0)], audio, 50) is None   # 끝 시각을 3초 이르게 적는 것은 정상
    assert _warn_if_cut_short([], audio, 50) is None


def test_cut_short_piece_is_recognized_once_more_and_replaced_only_if_longer(monkeypatch):
    """누락 의심 조각의 재인식과, 받아 적은 범위가 3초 넘게 늘었을 때만 그 결과로 교체(덧붙이지 않음)하는지 확인.
    이어 인식(STT_FILL_UNCOVERED)을 끈 경우의 동작. 켠 경우는 test_short_piece_rest_* 시험."""
    import types
    from audio import SILENCE_FRAME
    from stt import WhisperRecognizer
    monkeypatch.setenv("STT_FILL_UNCOVERED", "0")

    def seg(text, end):
        return types.SimpleNamespace(text=text, end=end, no_speech_prob=0.0, avg_logprob=-0.1, compression_ratio=1.0, seek=0)

    class FakeModel:
        def __init__(self, outs):
            self.outs, self.calls = list(outs), 0

        def transcribe(self, pcm, **kw):
            self.calls += 1
            return self.outs.pop(0), None

    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec._beam, rec.vad, rec.quiet_rms = 1, True, 50
    rec._model = FakeModel([[seg("잔액조회는 3번입니다.", 2.0)], [seg("잔액조회는 3번, 거래내역은 4번입니다.", 9.8)]])
    assert "4번" in rec._run(audio) and rec._model.calls == 2                 # 재인식해 더 받아 적은 쪽 사용
    rec._model = FakeModel([[seg("잔액조회는 3번입니다.", 2.0)], [seg("잔액조회는 3번.", 2.5)]])
    assert rec._run(audio) == "잔액조회는 3번입니다." and rec._model.calls == 2  # 늘지 않으면 처음 결과
    rec._model = FakeModel([[seg("잔액조회는 3번, 거래내역은 4번입니다.", 9.8)]])
    rec._run(audio)
    assert rec._model.calls == 1                                               # 누락 의심이 아니면 한 번만


def _fill_seg(text, start, end, seek=0, words=None, ratio=1.0):
    import types
    return types.SimpleNamespace(text=text, start=start, end=end, no_speech_prob=0.0, avg_logprob=-0.1, compression_ratio=ratio,
                                 seek=seek, words=words)


class _FillModel:
    """조각 길이·단어 시각 여부·힌트를 기록하고 정해 둔 구간 목록을 차례로 돌려주는 가짜 Whisper."""

    def __init__(self, outs):
        self.outs, self.calls = list(outs), []

    def transcribe(self, pcm, **kw):
        self.calls.append({"sec": round(len(pcm) / 16000, 2), "words": bool(kw.get("word_timestamps")),
                           "prompt": kw.get("initial_prompt"), "beam": kw.get("beam_size"),
                           "no_ts": kw.get("without_timestamps"), "lang": kw.get("language"), "vad": kw.get("vad_filter")})
        return self.outs.pop(0), None


def _fill_rec(model):
    from stt import WhisperRecognizer
    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec._beam, rec.vad, rec.quiet_rms, rec._model = 3, True, 50, model
    return rec


def test_jump_piece_rest_is_recognized_from_last_word_end(monkeypatch):
    """끝 시각 점프(마지막 구간 끝 > 조각 길이 + 1초): 단어 시각의 마지막 단어 끝부터 남은 소리만 앞 문장을 문맥으로 다시 인식해 덧붙이는지 확인."""
    import types
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 20                                                # 20초 말소리
    word = types.SimpleNamespace(start=0.4, end=2.6, word="eight.")
    model = _FillModel([[_fill_seg("For foreign language services, press 8.", 0.0, 29.98)],
                        [_fill_seg("For foreign language services, press 8.", 0.0, 2.6, words=[word])],
                        [_fill_seg("소포 접수 1번, 배달 조회 2번, 우체국 정보 4번을 눌러주십시오.", 0.0, 17.2)]])
    out = _fill_rec(model)._run(audio)
    assert out == "For foreign language services, press 8. 소포 접수 1번, 배달 조회 2번, 우체국 정보 4번을 눌러주십시오."
    assert [c["words"] for c in model.calls] == [False, True, False]
    assert model.calls[2]["sec"] == 17.4                                       # 2.6초 뒤 소리만
    assert model.calls[2]["prompt"].startswith("별표, 샵 버튼, 우물정자.") and "press 8." in model.calls[2]["prompt"]


def test_short_piece_rest_is_recognized_from_segment_end_without_word_pass(monkeypatch):
    """일찍 끝냄(같은 설정 다시 인식 뒤에도 말소리가 6초 넘게 남음): 구간 끝부터 남은 소리만 다시 인식하고 단어 시각 디코딩은 하지 않는지
    확인 (단어 시각 디코딩은 같은 글을 조각 끝 쪽 시각으로 옮겨 찍어 받아 적은 끝을 잘못 잡을 수 있음)."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50                           # 10초 말, 1초 무음
    model = _FillModel([[_fill_seg("잔액조회는 3번.", 0.0, 2.0)], [_fill_seg("잔액조회는 3번.", 0.0, 2.0)],
                        [_fill_seg("거래내역은 4번입니다.", 0.0, 7.5)]])
    out = _fill_rec(model)._run(audio)
    assert out == "잔액조회는 3번. 거래내역은 4번입니다."
    assert [c["words"] for c in model.calls] == [False, False, False] and model.calls[2]["sec"] == 9.0


def test_segment_beyond_audio_is_dropped_only_when_its_end_is_impossible(monkeypatch):
    """끝이 조각 길이 + 1초를 넘고 시작이 말소리 끝 뒤인 구간(들은 소리가 없는 곳에 적은 글)의 제거 확인. 마침표 없이 이어진 앞 문장이
    지어낸 문장 제거에 함께 걸려 사라지지 않음. 끝 시각이 있을 수 있는 구간·말소리 안에서 시작한 구간은 제거하지 않음."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 20                           # 10초 말, 0.4초 무음
    model = _FillModel([[_fill_seg("카드 발급은 1번 분실 신고는 2번", 0.0, 10.2),
                         _fill_seg("시청해 주셔서 감사합니다", 10.2, 40.2, seek=1020)]])
    assert _fill_rec(model)._run(audio) == "카드 발급은 1번 분실 신고는 2번" and len(model.calls) == 1
    monkeypatch.setenv("STT_FILL_UNCOVERED", "0")                            # 이어 인식을 끄면 창이 달라도 마침표 없이 이어져 함께 빠짐
    model = _FillModel([[_fill_seg("카드 발급은 1번 분실 신고는 2번", 0.0, 10.2),
                         _fill_seg("시청해 주셔서 감사합니다", 10.2, 40.2, seek=1020)]])
    assert _fill_rec(model)._run(audio) == ""
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    model = _FillModel([[_fill_seg("카드 발급은 1번", 0.0, 9.8), _fill_seg("분실 신고는 2번", 9.8, 11.0)]])
    assert _fill_rec(model)._run(audio) == "카드 발급은 1번 분실 신고는 2번" and len(model.calls) == 1  # 끝 11.0 ≤ 10.4 + 1


def test_beyond_audio_tail_no_longer_takes_unpunctuated_sentence_with_it(monkeypatch):
    """같은 창에 붙은 소리 밖 구간을 제거해 앞 문장이 남는지 확인 (제거하지 않으면 앞 문장까지 한 문장으로 지어낸 문장 제거에 걸려 통째로 사라짐)."""
    from audio import SILENCE_FRAME
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 20
    outs = lambda: [[_fill_seg("카드 발급은 1번 분실 신고는 2번", 0.0, 10.2), _fill_seg("시청해 주셔서 감사합니다", 10.2, 40.2)]]  # noqa: E731
    monkeypatch.setenv("STT_FILL_UNCOVERED", "0")
    assert _fill_rec(_FillModel(outs()))._run(audio) == ""                     # 이어 인식을 끈 경우: 실제 안내까지 빠짐
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    assert _fill_rec(_FillModel(outs()))._run(audio) == "카드 발급은 1번 분실 신고는 2번"


def test_fill_repeats_at_most_three_times_and_stops_without_progress_or_rest(monkeypatch):
    """남은 소리에서 다시 멈추면 최대 3번까지 되풀이하는지 확인. 받아 적은 끝 뒤 말소리가 1초 이하이거나 단어가 없으면 다시 인식하지 않음."""
    import types
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 30
    model = _FillModel([[_fill_seg("가", 0.0, 1.0)], [_fill_seg("가", 0.0, 1.0)]]
                       + [[_fill_seg(t, 0.0, 1.0)] for t in ("나", "다", "라")])  # 다시 인식마다 1초만 받아 적음
    assert _fill_rec(model)._run(audio) == "가. 나. 다. 라" and len(model.calls) == 5     # 첫 인식·같은 설정 다시 인식·이어 인식 3번
    assert [c["sec"] for c in model.calls] == [30.0, 30.0, 29.0, 28.0, 27.0]         # 회차마다 받아 적은 끝 뒤 소리만
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("끝까지", 0.0, 9.2)]])                      # 남은 말소리 0.8초: 신호 없음
    assert _fill_rec(model)._run(audio) == "끝까지" and len(model.calls) == 1
    model = _FillModel([[_fill_seg("점프", 0.0, 29.9)], [_fill_seg("점프", 0.0, 2.0, words=[])]])
    assert _fill_rec(model)._run(audio) == "점프" and len(model.calls) == 2    # 단어 시각이 없으면 멈춤
    word = types.SimpleNamespace(start=0.0, end=9.6, word="점프")
    model = _FillModel([[_fill_seg("점프", 0.0, 29.9)], [_fill_seg("점프", 0.0, 9.6, words=[word])]])
    assert _fill_rec(model)._run(audio) == "점프" and len(model.calls) == 2    # 단어 끝 뒤 말소리 0.4초


def test_hallucinated_fill_pass_does_not_delete_earlier_unpunctuated_text(monkeypatch):
    """이어 인식이 지어낸 문장('시청해 주셔서 감사합니다')을 내도 마침표 없이 끝난 앞 회차 문장이 남는지 확인(회차마다 따로 뺀 뒤 이음).
    점프 조각의 첫 회차가 지어낸 문장이어도 이어 인식으로 되찾은 메뉴 문장은 남음."""
    import types
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("잔액조회는 3번", 0.0, 2.0)], [_fill_seg("잔액조회는 3번", 0.0, 2.0)],
                        [_fill_seg("시청해 주셔서 감사합니다.", 0.0, 3.0)]])
    assert _fill_rec(model)._run(audio) == "잔액조회는 3번"
    assert "시청해" not in (model.calls[2]["prompt"] or "")
    audio = hum_ulaw(1000) * 20
    word = types.SimpleNamespace(start=0.0, end=2.6, word="감사합니다")
    model = _FillModel([[_fill_seg("시청해 주셔서 감사합니다", 0.0, 29.98)],
                        [_fill_seg("시청해 주셔서 감사합니다", 0.0, 2.6, words=[word])],
                        [_fill_seg("소포 접수 1번, 배달 조회 2번입니다.", 0.0, 17.0)]])
    assert _fill_rec(model)._run(audio) == "소포 접수 1번, 배달 조회 2번입니다."
    assert "시청해" not in model.calls[2]["prompt"]                            # 다음 회차 문맥에도 넣지 않음


def test_jump_uses_only_first_window_words_when_word_pass_decodes_more_windows(monkeypatch):
    """단어 시각 디코딩이 뒤 창까지 디코딩해도 받아 적은 끝을 첫 창(첫 구간의 seek) 단어로만 잡는지 확인."""
    import types
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 20
    w1, w2 = types.SimpleNamespace(start=0.4, end=2.6, word="8."), types.SimpleNamespace(start=3.0, end=19.6, word="번")
    model = _FillModel([[_fill_seg("press 8.", 0.0, 29.98)],
                        [_fill_seg("press 8.", 0.0, 2.6, seek=0, words=[w1]), _fill_seg("1번 2번", 2.6, 19.6, seek=260, words=[w2])],
                        [_fill_seg("1번 2번 3번을 눌러주십시오.", 0.0, 17.0)]])
    assert _fill_rec(model)._run(audio) == "press 8. 1번 2번 3번을 눌러주십시오." and model.calls[2]["sec"] == 17.4


def test_fill_stops_when_covered_end_is_before_minimum_progress(monkeypatch):
    """받아 적은 끝이 0.3초보다 앞이면(같은 소리를 거의 그대로 다시 넣는 것) 이어 인식하지 않는지 확인."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("네", 0.0, 0.2)], [_fill_seg("네", 0.0, 0.2)]])
    assert _fill_rec(model)._run(audio) == "네" and len(model.calls) == 2


def test_drop_beyond_audio_needs_both_impossible_end_and_start_after_speech(monkeypatch):
    """소리 밖 구간 제거는 두 조건이 모두 맞을 때만 적용되는지 확인: 시작이 말소리 끝 뒤라도 끝 시각이 있을 수 있으면 남기고, 끝 시각이
    있을 수 없어도 말소리 안에서 시작했으면 남김(그 경우는 끝 시각 점프로 이어 인식 여부 판단)."""
    import types
    from audio import SILENCE_FRAME
    from stt import _drop_beyond_audio
    a, b = _fill_seg("앞", 0.0, 9.0), _fill_seg("뒤", 10.2, 11.0)
    assert _drop_beyond_audio([a, b], 10.4, 10.0) == [a, b]                   # 시작 10.2 ≥ 말소리 끝 10.0, 끝 11.0 ≤ 11.4
    c = _fill_seg("말소리 안에서 시작", 9.5, 40.0)
    assert _drop_beyond_audio([a, c], 10.4, 10.0) == [a, c]                   # 끝 40 은 있을 수 없으나 시작 9.5 < 10.0
    d = _fill_seg("지어냄", 10.2, 40.0)
    assert [s.text for s in _drop_beyond_audio([a, d], 10.4, 10.0)] == ["앞"]   # 같은 창의 남은 구간은 압축률을 다시 잰 사본
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 20
    model = _FillModel([[a, c]])
    assert _fill_rec(model)._run(audio) == "앞 말소리 안에서 시작" and len(model.calls) == 1  # 점프 구간 시작 9.5 뒤 말소리 0.5초: 멈춤
    e = _fill_seg("작은 소리의 실제 말", 10.1, 40.0)                          # 말소리 끝(RMS) 뒤지만 조각 끝 1초 안에서 시작하지 않음
    assert _drop_beyond_audio([a, e], 12.0, 10.0) == [a, e]
    assert [s.text for s in _drop_beyond_audio([a, e], 10.4, 10.0)] == ["앞"]


def test_default_fill_still_uses_longer_same_settings_retry_first(monkeypatch):
    """이어 인식을 켠 기본 설정에서도 누락 의심 조각은 먼저 같은 설정으로 다시 인식하고, 끝이 3초 넘게 늘었으면 그 결과를 쓰는지 확인.
    그 결과가 말소리 끝까지 덮으면 이어 인식하지 않음."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("잔액조회는 3번입니다.", 0.0, 2.0)], [_fill_seg("잔액조회는 3번, 거래내역은 4번입니다.", 0.0, 9.8)]])
    assert _fill_rec(model)._run(audio) == "잔액조회는 3번, 거래내역은 4번입니다." and len(model.calls) == 2


def test_fill_trims_text_repeated_at_the_seam(monkeypatch):
    """일찍 끝냄의 구간 끝이 실제보다 이르게 찍혀 이어 인식이 앞 문장 끝을 다시 받아 적으면(글자·숫자 8자 이상 같음) 그 부분을 제거하는지 확인.
    짧게 같은 것('번을')은 제거하지 않음. 이어 인식은 첫 인식과 같은 언어·빔·시각 토큰·VAD 설정 사용."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("카드 발급은 1번, 분실과 도난 신고는 2번", 0.0, 2.5)],
                        [_fill_seg("카드 발급은 1번, 분실과 도난 신고는 2번", 0.0, 2.5)],
                        [_fill_seg("분실과 도난 신고는 2번, 해외 결제는 3번입니다.", 0.0, 7.0)]])
    assert _fill_rec(model)._run(audio) == "카드 발급은 1번, 분실과 도난 신고는 2번. 해외 결제는 3번입니다."
    rest = model.calls[2]
    assert (rest["beam"], rest["no_ts"], rest["lang"], rest["vad"], rest["words"]) == (3, False, "ko", True, False)
    from stt import _trim_overlap
    assert _trim_overlap("상담원 연결은 0번을", "번을 눌러주세요") == "번을 눌러주세요"


def test_short_identical_digit_overlap_after_early_end_is_trimmed_by_words(monkeypatch):
    """일찍 끝냄 뒤 이음매에서 낱말 단위로 그대로 겹친 짧은 끝('5678', '7번')은 8자 미만이어도 제거하는지 확인(되읽기 자릿수가 늘지 않게).
    점프(단어 시각) 뒤에는 낱말 단위로 제거하지 않음."""
    from audio import SILENCE_FRAME
    from stt import _trim_overlap
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 12 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("고객님 휴대폰 번호 010 1234 5678", 0.0, 4.0)], [_fill_seg("고객님 휴대폰 번호 010 1234 5678", 0.0, 4.0)],
                        [_fill_seg("5678 로 인증번호를 보내드렸습니다.", 0.0, 7.5)]])
    out = _fill_rec(model)._run(audio)
    assert sum(ch.isdigit() for ch in out) == 11 and out.endswith("로 인증번호를 보내드렸습니다.")
    assert _trim_overlap("잔액 조회는 3번, 분실 신고는 7번,", "신고는 7번, 거래 내역은 4번", tokens=True) == "거래 내역은 4번"
    assert _trim_overlap("잔액 조회는 3번", "3번 거래", tokens=True) == "3번 거래"                     # 2자: 기준 3자 미만
    assert _trim_overlap("번호는 5678", "5678 입니다") == "5678 입니다"                                  # 낱말 단위는 일찍 끝냄 뒤만


def test_revived_window_hallucination_does_not_take_previous_window_text(monkeypatch):
    """소리 밖 꼬리를 빼 압축률이 낮아진 창에 지어낸 문장이 남아도 앞 창의 마침표 없는 실제 문장을 함께 지우지 않는지 확인(창마다 따로 뺌).
    남은 반복은 이어 인식을 끈 경우처럼 한 번으로 축약."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 100
    model = _FillModel([[_fill_seg("카드 발급은 1번 분실 신고는 2번", 0.0, 9.8, seek=0, ratio=0.8),
                         _fill_seg("시청해 주셔서 감사합니다.", 9.8, 11.0, seek=980, ratio=2.85),
                         _fill_seg("Thank you. " * 12, 11.0, 30.0, seek=980, ratio=2.85)]])
    assert _fill_rec(model)._run(audio) == "카드 발급은 1번 분실 신고는 2번"
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 20
    rep = "거래 내역은 4번입니다."
    model = _FillModel([[_fill_seg("잔액 조회는 3번", 0.0, 4.0, ratio=3.23)] + [_fill_seg(rep, t, t + 2.0, ratio=3.23) for t in (4.0, 6.0, 8.0)]
                        + [_fill_seg(rep, t, t + 4.0, ratio=3.23) for t in (10.0, 14.0, 18.0)]])
    out = _fill_rec(model)._run(audio)
    assert out.count("거래 내역은 4번입니다") == 1 and out.startswith("잔액 조회는 3번")


def test_greeting_only_first_pass_is_not_filled_and_errors_fall_back_to_first_result(monkeypatch):
    """첫 인식이 '감사합니다' 뿐이면 이어 인식하지 않는지 확인(대기 음악일 가능성, 이어 인식을 끈 경우에도 단락 전체가 빠짐). 이어 인식 중
    오류가 나면 이어 인식을 끈 경우와 같은 첫 인식 결과 사용. 겹침을 뺀 뒤 남은 글이 힌트 되풀이면 폐기."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 20
    model = _FillModel([[_fill_seg("감사합니다.", 0.0, 2.0)], [_fill_seg("감사합니다.", 0.0, 2.0)], [_fill_seg("네.", 0.0, 1.5)]])
    assert _fill_rec(model)._run(audio) == "감사합니다." and len(model.calls) == 2

    class Boom(_FillModel):
        def transcribe(self, pcm, **kw):
            if len(self.calls) >= 2:
                raise RuntimeError("CUDA out of memory")
            return super().transcribe(pcm, **kw)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    assert _fill_rec(Boom([[_fill_seg("잔액조회는 3번.", 0.0, 2.0)], [_fill_seg("잔액조회는 3번.", 0.0, 2.0)]]))._run(audio) == "잔액조회는 3번."
    model = _FillModel([[_fill_seg("카드 발급은 1번, 분실과 도난 신고는 2번", 0.0, 2.5)], [_fill_seg("카드 발급은 1번, 분실과 도난 신고는 2번", 0.0, 2.5)],
                        [_fill_seg("분실과 도난 신고는 2번. 별표, 샵 버튼, 우물정자.", 0.0, 7.0)]])
    assert _fill_rec(model)._run(audio) == "카드 발급은 1번, 분실과 도난 신고는 2번"


def test_jump_that_is_not_the_first_segment_is_left_as_when_switched_off(monkeypatch):
    """점프 구간 앞에 다른 구간이 있으면(첫 인식의 둘째 창 등) 이어 인식하지 않고 이어 인식을 끈 경우와 같게 두는지 확인: 첫 창 단어 끝부터
    다시 인식하면 이미 적은 글이 두 번 들어갈 수 있음. 첫 구간의 점프는 시작이 0.5초여도(앞 무음) 단어 시각으로 이어 인식."""
    import types
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 25
    model = _FillModel([[_fill_seg("안녕하십니까 우체국 고객센터입니다.", 0.0, 12.4, seek=0),
                         _fill_seg("외국어 안내는 8번, 소포 접수는 1번", 12.4, 42.38, seek=1240)]])
    assert _fill_rec(model)._run(audio) == "안녕하십니까 우체국 고객센터입니다. 외국어 안내는 8번, 소포 접수는 1번" and len(model.calls) == 1
    word = types.SimpleNamespace(start=0.5, end=3.1, word="8.")
    model = _FillModel([[_fill_seg("press 8.", 0.5, 29.98)], [_fill_seg("press 8.", 0.5, 3.1, words=[word])],
                        [_fill_seg("소포 접수 1번입니다.", 0.0, 20.0)]])
    assert _fill_rec(model)._run(audio) == "press 8. 소포 접수 1번입니다." and model.calls[1]["words"] and model.calls[2]["sec"] == 21.9


def test_jump_on_a_fill_pass_uses_word_pass_on_the_remaining_audio(monkeypatch):
    """이어 인식 회차에서 점프가 나도 단어 시각 디코딩은 남은 소리·그 회차 힌트로 하고, 그 단어 끝부터 다시 이어 인식하는지 확인."""
    import types
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 20 + SILENCE_FRAME * 10
    word = types.SimpleNamespace(start=0.0, end=2.0, word="8.")
    model = _FillModel([[_fill_seg("안녕하세요.", 0.0, 3.0)], [_fill_seg("안녕하세요.", 0.0, 3.0)],
                        [_fill_seg("press 8.", 0.0, 29.98)], [_fill_seg("press 8.", 0.0, 2.0, words=[word])],
                        [_fill_seg("소포 접수 1번입니다.", 0.0, 14.0)]])
    assert _fill_rec(model)._run(audio) == "안녕하세요. press 8. 소포 접수 1번입니다."
    assert [c["sec"] for c in model.calls] == [20.2, 20.2, 17.2, 17.2, 15.2]
    assert model.calls[3]["words"] and model.calls[3]["prompt"] == model.calls[2]["prompt"] == "별표, 샵 버튼, 우물정자. 안녕하세요."


def test_window_that_loses_a_beyond_audio_tail_is_kept_or_dropped_as_when_switched_off(monkeypatch):
    """소리 밖 구간이 빠진 창을 남길지 버릴지를 이어 인식을 끈 경우와 똑같이 창 전체로 정하는지 확인. 창 전체가 그럴듯하면 남은 구간만
    쓰고(빠진 구간의 글만 없어짐), 그럴듯하지 않으면 끈 경우의 결과(버리거나 반복을 줄여 살림)를 그대로 사용: 되살아나는 지어낸 조각·두 벌
    남는 반복 없음."""
    from audio import SILENCE_FRAME
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 20

    def both(segs_fn):
        out = {}
        for v in ("0", "1"):
            monkeypatch.setenv("STT_FILL_UNCOVERED", v)
            out[v] = _fill_rec(_FillModel([segs_fn()]))._run(audio)
        return out["0"], out["1"]
    off, on = both(lambda: [_fill_seg("카드 발급은 1번, 분실 신고는 2번을 눌러 주십시오.", 0.0, 10.0, ratio=2.6),
                            _fill_seg("우체국 소포, " * 12, 10.2, 40.0, ratio=2.6)])
    assert on == off and on.startswith("카드 발급은 1번")                    # 반복을 줄여 살린 끈 경우 결과 그대로
    s = "고객님 번호는 010 1234 5678 입니다."
    off, on = both(lambda: [_fill_seg(s, 0.0, 6.0, ratio=5.23), _fill_seg(s, 6.0, 9.6, ratio=5.23)]
                   + [_fill_seg(s, t, t + 4.0, ratio=5.23) for t in (10.0, 14.0, 18.0, 22.0, 26.0)])
    assert on == off and on.count("5678") == 1                                 # 소리 안 두 벌이 남지 않음
    off, on = both(lambda: [_fill_seg("앞 안내입니다.", 0.0, 9.0, ratio=1.0), _fill_seg("뒤 안내", 9.0, 10.0, ratio=1.0),
                            _fill_seg("지어낸 꼬리", 10.2, 40.0, ratio=1.0)])
    assert off.endswith("지어낸 꼬리") and on == "앞 안내입니다. 뒤 안내"         # 그럴듯한 창: 빠진 구간의 글만 없어짐


def test_fill_pass_with_only_greeting_hint_echo_or_punctuation_is_not_appended(monkeypatch):
    """이어 인식 글이 '감사합니다.' 뿐(대기 음악 꼬리)이거나, 힌트 문장을 되풀이했거나, 문장부호뿐이면 덧붙이지 않는지 확인."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    for tail in ("감사합니다.", "감사합니다. 감사합니다.", "감사합니다,", "별표, 샵 버튼, 우물정자.", "잔액조회는 3번입니다.", " . "):
        model = _FillModel([[_fill_seg("잔액조회는 3번입니다.", 0.0, 2.0)], [_fill_seg("잔액조회는 3번입니다.", 0.0, 2.0)],
                            [_fill_seg(tail, 0.0, 3.0)]])
        assert _fill_rec(model)._run(audio) == "잔액조회는 3번입니다.", tail
    for head, tail, want in (("이전 메뉴는", "별표.", "이전 메뉴는. 별표."), ("상담원 연결은", "0번.", "상담원 연결은. 0번."),
                             ("번호 끝자리는 1111", "2222", "번호 끝자리는 1111 2222")):  # 짧은 정상 글은 남김(숫자끼리는 끊지 않음)
        model = _FillModel([[_fill_seg(head, 0.0, 2.0)], [_fill_seg(head, 0.0, 2.0)], [_fill_seg(tail, 0.0, 3.0)]])
        assert _fill_rec(model)._run(audio) == want, tail


def test_beyond_audio_tail_does_not_revive_a_window_switched_off_drops(monkeypatch):
    """이어 인식을 끈 경우 창째 버려지는 지어낸 조각('감사합니다.', 숫자)이 소리 밖 꼬리를 빼도 되살아나지 않는지 확인."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 100
    for left in ("감사합니다.", "Thank you.", "1."):
        model = _FillModel([[_fill_seg("카드 발급은 1번 분실 신고는 2번", 0.0, 9.8, seek=0, ratio=0.8),
                             _fill_seg(left, 9.8, 11.0, seek=980, ratio=2.85), _fill_seg("Thank you. " * 12, 11.0, 30.0, seek=980, ratio=2.85)]])
        assert _fill_rec(model)._run(audio) == "카드 발급은 1번 분실 신고는 2번", left


def test_review6_cases_match_or_beat_switched_off(monkeypatch):
    """경계 입력 확인: (1) 압축률 기준에 걸린 창의 소리 밖 반복 꼬리 + 일찍 끝냄 → 창을 건드리지 않아 이어 인식을 끈 경우와 같음(번호 두 번 없음).
    (2) 이어 인식 두 창에 걸쳐 나뉜 지어낸 문장이 번호 가운데 이음매 앞 회차 글을 지우지 않음. (3) 창 끝의 'KBS 뉴스' 는 끈 경우처럼
    남음. (4) 띄어 받아 적은 되읽기 끝 두 자리('7 8')는 일찍 끝냄 이음매에서 제거."""
    from audio import SILENCE_FRAME
    from stt import _trim_overlap

    def run(v, outs, audio):
        monkeypatch.setenv("STT_FILL_UNCOVERED", v)
        return _fill_rec(_FillModel(outs()))._run(audio)
    audio = hum_ulaw(1000) * 12 + SILENCE_FRAME * 50
    outs = lambda: [[_fill_seg("입력하신 번호는 010 1234 5678", 0.0, 4.0, ratio=3.58)]  # noqa: E731
                    + [_fill_seg("감사합니다.", 13.0 + 1.4 * i, 14.4 + 1.4 * i, ratio=3.58) for i in range(12)],
                    [_fill_seg("5678 입니다. 맞으면 1번 틀리면 2번을 눌러 주십시오.", 0.0, 7.5)]]
    assert run("1", outs, audio) == run("0", outs, audio)
    outs = lambda: [[_fill_seg("고객님 번호 공일공 일이삼사", 0.0, 3.0)], [_fill_seg("고객님 번호 공일공 일이삼사", 0.0, 3.0)],  # noqa: E731
                    [_fill_seg("오육칠팔 시청해", 0.0, 4.0, seek=0), _fill_seg("주셔서 감사합니다", 4.0, 8.0, seek=400)]]
    on = run("1", outs, audio)
    assert on.startswith("고객님 번호 공일공 일이삼사") and "시청" not in on
    audio2 = hum_ulaw(1000) * 20
    two = lambda: [_fill_seg("안녕하십니까 시청자 상담실입니다 지금 들으시는 방송은", 0.0, 4.0, seek=0),  # noqa: E731
                   _fill_seg("KBS 뉴스", 4.2, 5.0, seek=420)]
    outs = lambda: [two(), two(), [_fill_seg("프로그램 문의는 1번입니다.", 0.0, 10.0)]]  # noqa: E731
    assert "KBS 뉴스" in run("1", outs, audio2)
    assert _trim_overlap("입력하신 번호는 0 1 0 1 2 3 4 5 6 7 8", "7 8 입니다", tokens=True) == "입니다"
    assert _trim_overlap("상담원 연결은 0번", "8 입니다", tokens=True) == "8 입니다"


def test_seam_inside_a_number_gets_no_sentence_break(monkeypatch):
    """번호 가운데서 이어 인식하면(앞 글 끝과 이어 인식 첫 글자가 모두 숫자) 회차 사이에 마침표를 넣지 않는지 확인(숫자열 가림이 끊기지 않게)."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 12 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("고객님 번호 공일공 일이삼사", 0.0, 3.0)], [_fill_seg("고객님 번호 공일공 일이삼사", 0.0, 3.0)],
                        [_fill_seg("일이삼사 오육칠팔 로 인증번호를 보내드렸습니다.", 0.0, 8.5)]])
    assert _fill_rec(model)._run(audio) == "고객님 번호 공일공 일이삼사 오육칠팔 로 인증번호를 보내드렸습니다."
    for head, tail in (("입력하신 번호는 0, 1, 0, 3, 0, 8,", "1, 9, 7, 7, 3 번입니다."), ("번호는 010-1234-", "5678 입니다.")):
        model = _FillModel([[_fill_seg(head, 0.0, 3.0)], [_fill_seg(head, 0.0, 3.0)], [_fill_seg(tail, 0.0, 8.5)]])
        assert _fill_rec(model)._run(audio) == f"{head} {tail}", head                # 쉼표·하이픈으로 끝난 번호 가운데도 끊지 않음


def test_seam_overlap_also_trims_repeated_numbers_and_strips_leading_punctuation(monkeypatch):
    """겹친 번호도 제거하는지 확인: 되읽기 번호가 두 번 들어가면 자릿수가 틀림(이음매에서 실제로 두 번 읽었어도 한 벌은 남음).
    뺀 뒤 앞에 남은 문장부호는 삭제."""
    from audio import SILENCE_FRAME
    from stt import _trim_overlap
    assert _trim_overlap("입력하신 번호는 1234 5678", "1234 5678 맞으면 1번") == "맞으면 1번"
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("입력하신 번호는 010 1234 5678", 0.0, 2.5)], [_fill_seg("입력하신 번호는 010 1234 5678", 0.0, 2.5)],
                        [_fill_seg("1234 5678 맞으면 1번, 틀리면 2번", 0.0, 7.0)]])
    out = _fill_rec(model)._run(audio)
    assert out == "입력하신 번호는 010 1234 5678. 맞으면 1번, 틀리면 2번" and out.count("1234") == 1
    assert _trim_overlap("분실과 도난 신고는", "분실과 도난 신고는? 2번입니다") == "2번입니다"
    assert _trim_overlap("분실과 도난 신고는", ") 분실과 도난 신고는) 2번") == "2번"


def test_word_without_end_time_does_not_break_the_jump_path(monkeypatch):
    import types
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 20
    model = _FillModel([[_fill_seg("press 8.", 0.0, 29.98)],
                        [_fill_seg("press 8.", 0.0, 2.6, words=[types.SimpleNamespace(start=0.0, end=None, word="8.")])]])
    assert _fill_rec(model)._run(audio) == "press 8." and len(model.calls) == 2


def test_fill_joins_cleaned_passes_without_rechecking_across_the_seam_and_hints_with_corrected_text(monkeypatch):
    """회차마다 지어낸 문장을 뺀 뒤에는 이어 붙인 글을 다시 검사하지 않는지 확인(회차 경계에 걸친 일치로 앞 회차 글이 지워지지 않게).
    이어 인식 힌트는 키 이름 교정을 마친 글로 생성."""
    from audio import SILENCE_FRAME
    monkeypatch.delenv("STT_FILL_UNCOVERED", raising=False)
    audio = hum_ulaw(1000) * 10 + SILENCE_FRAME * 50
    model = _FillModel([[_fill_seg("샷 버튼 안내와 KBS", 0.0, 2.0)], [_fill_seg("샷 버튼 안내와 KBS", 0.0, 2.0)],
                        [_fill_seg("뉴스 홍길동입니다", 0.0, 5.0)]])
    out = _fill_rec(model)._run(audio)
    assert out == "샵 버튼 안내와 KBS. 뉴스 홍길동입니다"
    from stt import _clean
    assert _clean(out) == out                                                # 단락 끝 정리(recognize)도 회차 경계를 넘어 지우지 않음
    assert "샵 버튼" in model.calls[2]["prompt"] and "샷" not in model.calls[2]["prompt"]


def test_fill_switch_off_keeps_previous_behavior(monkeypatch):
    """STT_FILL_UNCOVERED=0 이면 첫 인식(과 누락 의심 다시 인식)만 쓰는지 확인."""
    monkeypatch.setenv("STT_FILL_UNCOVERED", "0")
    audio = hum_ulaw(1000) * 20
    model = _FillModel([[_fill_seg("For foreign language services, press 8.", 0.0, 29.98)]])
    assert _fill_rec(model)._run(audio) == "For foreign language services, press 8." and len(model.calls) == 1


def test_whisper_long_paragraph_is_recognized_in_pieces():
    from stt import WhisperRecognizer

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.quiet_rms, rec._prefix, rec._lock = 50, [], asyncio.Lock()
    seen = []
    rec._run = lambda audio, ctx="": (seen.append((len(audio), ctx)), f"조각{len(seen)}")[1]
    from audio import SILENCE_FRAME
    audio = hum_ulaw(1000) * 20 + SILENCE_FRAME * 40 + hum_ulaw(1000) * 15
    assert asyncio.run(rec.recognize(audio)) == "조각1 조각2"
    assert len(seen) == 2 and seen[1][1] == "조각1"                         # 앞 조각 문장을 다음 조각의 문맥으로


def test_whisper_keeps_timestamp_tokens():
    """Whisper 의 시각 정보 토큰 사용 확인 (끄면 한 자씩 끊어 읽는 번호 되읽기를 통째로 건너뜀). 힌트는 공통 키 이름만."""
    import types
    from stt import WhisperRecognizer

    seen = {}

    class FakeModel:
        def transcribe(self, pcm, **kw):
            seen.update(kw)
            return [types.SimpleNamespace(text="번입니다", no_speech_prob=0.0, avg_logprob=-0.1, compression_ratio=1.0)], None

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec._model, rec._beam, rec.vad = FakeModel(), 1, True
    rec._run(hum_ulaw(1000))
    assert seen["without_timestamps"] is False and seen["initial_prompt"] == "별표, 샵 버튼, 우물정자."  # 공통 키 이름 힌트만
