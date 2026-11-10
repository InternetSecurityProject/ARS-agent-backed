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


def test_cut_short_piece_is_recognized_once_more_and_replaced_only_if_longer():
    """누락 의심 조각의 재인식과, 받아 적은 범위가 3초 넘게 늘었을 때만 그 결과로 교체(덧붙이지 않음)하는지 확인."""
    import types
    from audio import SILENCE_FRAME
    from stt import WhisperRecognizer

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
