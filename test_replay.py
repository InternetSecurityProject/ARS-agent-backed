"""녹음 기반 가상 ARS(replay_ars.ReplayEngine) 시험. 작은 묶음을 직접 만들어 프레임 단위로 실행 (실통화·녹음 없음)."""

import json
import pathlib
from datetime import datetime

import pytest

from replay_ars import FRAME_S, KST, ReplayBundle, ReplayEngine, pick_start

FRAME = 160


def clip(seconds: float, loud: bool = True) -> bytes:
    """μ-law 조각: 소리(0x00 근처 큰 값) 또는 무음(0xff)."""
    n = int(seconds / FRAME_S) * FRAME
    return (b"\x10" if loud else b"\xff") * n


@pytest.fixture
def bundle(tmp_path):
    org = {
        "org": "시험", "start": "N0", "answer_delay_s": 0.0, "first_audio_s": {"median": 1.0}, "noise": None,
        "barge_stop_s": 0.2, "barge_extra_s": 1.0,
        "start_rules": [{"days": [0, 1, 2, 3, 4], "from": "09:00", "to": "18:00", "state": "D0"}],
        "states": {
            "D0": {"clip": "o/D0.ulaw", "barge_in": "yes", "keys": {"1": {"next": "M1", "delay_s": {"median": 0.5}}},
                   "after_audio": {"next": "D0_TO", "timeout_s": {"median": 5.0}}},
            "N0": {"clip": "o/N0.ulaw", "barge_in": "yes", "keys": {"1": {"next": "M1", "delay_s": {"median": 0.5}},
                                                                     "7": {"next": "M7", "delay_s": {"median": 0.6}},
                                                                     "77": {"next": "M77", "delay_s": {"median": 0.3}},
                                                                     "2": {"next": "IN", "delay_s": {"median": 0.5}},
                                                                     "9": {"next": "P", "delay_s": {"median": 0.5}}},
                   "after_audio": {"next": "N0_TO", "timeout_s": {"median": 5.0}, "third_fail_next": "END",
                                   "third_fail_timeout_s": {"median": 4.0}}},
            "N0_TO": {"clip": "o/TO.ulaw", "barge_in": "yes", "keys": {"1": {"next": "M1", "delay_s": {"median": 0.5}}},
                      "after_audio": {"next": "N0_TO", "timeout_s": {"median": 5.0}, "third_fail_next": "END",
                                      "third_fail_timeout_s": {"median": 4.0}}},
            "D0_TO": {"clip": "o/TO.ulaw"},
            "END": {"clip": "o/END.ulaw", "after_audio": {"next": "hangup", "timeout_s": {"median": 0.5}}},
            "M1": {"clip": "o/M1.ulaw", "barge_in": "no", "keys": {"3": {"next": "M7", "delay_s": {"median": 0.5}}}},
            "M7": {"clip": "o/M1.ulaw"}, "M77": {"clip": "o/M1.ulaw"},
            "IN": {"clip": "o/IN.ulaw", "input": {"len": 4, "terminator": None, "digit_timeout_s": 6.5}, "input_next": "P",
                   "input_delay_s": {"median": 1.0}, "after_audio": {"next": "IN_TO", "timeout_s": {"median": 5.0}}},
            "IN_TO": {"clip": "o/TO.ulaw"},
            "P": {"clip": None, "personal": True},
        },
    }
    (tmp_path / "o").mkdir()
    for name, sec in (("D0", 2.0), ("N0", 3.0), ("TO", 1.0), ("END", 1.0), ("M1", 2.0), ("IN", 1.0)):
        (tmp_path / "o" / f"{name}.ulaw").write_bytes(clip(sec))
    (tmp_path / "bundle.json").write_text(json.dumps({"version": 1, "phones": {"15880000": "o"}, "orgs": {"o": org}}),
                                          encoding="utf-8")
    return ReplayBundle(tmp_path)


NIGHT = datetime(2026, 11, 9, 21, 0, tzinfo=KST)
DAY = datetime(2026, 11, 9, 14, 0, tzinfo=KST)


def run_until(eng: ReplayEngine, t: float, keys: dict[float, str] | None = None) -> list[bytes]:
    keys = dict(keys or {})
    out = []
    while eng.frame * FRAME_S < t and not eng.ended:
        now = eng.frame * FRAME_S
        for kt in sorted(k for k in keys if k <= now):
            eng.key(keys.pop(kt))
        out.append(eng.tick())
    return out


def states(eng):
    return [p["state"] for p in eng.path if "state" in p]


def test_start_state_follows_time_of_day(bundle):
    """시간대에 따라 다른 첫 상태(평일 낮 메뉴·그 밖). 규칙에 맞지 않으면 기본 시작 상태."""
    org = bundle.orgs["o"]
    assert pick_start(org, DAY) == "D0" and pick_start(org, NIGHT) == "N0"
    assert pick_start(org, datetime(2026, 11, 8, 14, 0, tzinfo=KST)) == "N0"  # 일요일 낮
    eng = ReplayEngine(bundle, "o", clock=DAY)
    run_until(eng, 1.5)
    assert states(eng) == ["D0"] and eng.speaking


def test_audio_timing_and_key_after_audio(bundle):
    """첫 소리는 first_audio_s 뒤. 안내가 끝난 뒤 누른 키는 그 연결의 지연 뒤 다음 상태."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    out = run_until(eng, 0.99)
    assert all(f == b"\xff" * FRAME for f in out)  # 무음 (디지털 무음 회선)
    run_until(eng, 4.5, {4.2: "1"})  # 안내 1~4초, 키 4.2초
    run_until(eng, 4.6)
    assert states(eng) == ["N0"]
    run_until(eng, 4.75)
    assert states(eng) == ["N0", "M1"] and abs(eng.path[-1]["t"] - 4.7) < 0.05


def test_barge_in_stops_after_barge_stop_and_adds_extra_delay(bundle):
    """안내 도중 키: barge_stop_s(0.2초) 더 말하다 멈추고, 도중 지연을 잰 값이 없으면 기관 단위 차이(1.0초)를 더함."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 2.0)
    eng.key("1")
    run_until(eng, 2.15)
    assert eng.speaking
    run_until(eng, 2.3)
    assert not eng.speaking
    run_until(eng, 3.45)
    assert states(eng) == ["N0"]
    run_until(eng, 3.6)
    assert states(eng)[-1] == "M1"


def test_no_barge_in_waits_until_audio_end(bundle):
    """끼어들기가 안 되는 상태: 안내가 끝난 뒤 키를 처리."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 4.5, {4.2: "1"})
    run_until(eng, 5.0)  # M1 4.7초부터 2초
    eng.key("3")
    run_until(eng, 6.6)
    assert states(eng)[-1] == "M1" and eng.speaking
    run_until(eng, 7.3)
    assert states(eng)[-1] == "M7"


def test_two_digit_key(bundle):
    """'7' 과 '77' 이 함께 있는 메뉴: 한 자리 연결 전에 둘째 자리가 오면 '77'."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 6.0, {4.1: "7", 4.35: "7"})
    assert states(eng) == ["N0", "M77"]
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 6.0, {4.1: "7"})
    assert states(eng) == ["N0", "M7"]


def test_timeouts_repeat_then_third_fail_and_hangup(bundle):
    """입력이 없으면 시간 초과 안내를 되풀이하고, 연속 세 번째는 종료 안내 뒤 ARS 가 끊음."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 40.0)
    assert states(eng) == ["N0", "N0_TO", "N0_TO", "END"] and eng.ended and eng.note is None
    assert eng.end_reason == "remote_hangup"


def test_input_digits_reset_timeout_and_personal_state_ends(bundle):
    """입력 상태: 자리마다 안내를 멈추고 시간 초과를 마지막 자리부터 다시 잼. 다 입력하면 다음 상태로, 개인정보 상태는 재생하지
    않고 녹음 밖 종료."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 4.5, {4.1: "2"})            # IN 4.6초부터 1초
    run_until(eng, 4.7)
    eng.key("0")
    run_until(eng, 5.0)
    assert not eng.speaking                   # 입력 중 안내 멈춤
    run_until(eng, 10.0, {7.0: "0"})          # 안내 끝 + 5초가 지나도 시간 초과 없음 (마지막 자리 7.0초부터 6.5초)
    assert states(eng)[-1] == "IN"
    run_until(eng, 13.6)
    assert states(eng)[-1] == "IN_TO"
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 4.5, {4.1: "2"})
    run_until(eng, 8.0, {6.0: "1", 6.3: "2", 6.6: "3", 6.9: "4"})
    assert eng.ended and "개인정보" in eng.note


def test_unseen_key_ends_off_map(bundle):
    """녹음에 없는 키는 지어내지 않고 녹음 밖 종료."""
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 5.0, {4.2: "5"})
    assert eng.ended and "녹음에 없는 키" in eng.note


def test_variants_follow_choice(bundle):
    """실제 ARS 가 스스로 고르는 변형: choose_variant 로 지정하거나, 없으면 관측 비율대로 선택(씨앗으로 고정)."""
    bundle.orgs["o"]["states"]["N0"]["variants"] = [{"clip": "o/N0.ulaw", "name": "a", "weight": 3},
                                                    {"clip": "o/M1.ulaw", "name": "b", "weight": 1}]
    got = []
    for seed in range(200):
        eng = ReplayEngine(bundle, "o", clock=NIGHT, seed=seed)
        run_until(eng, 1.1)
        got.append(eng.path[0]["variant"])
    assert 0.6 < got.count("a") / len(got) < 0.9
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    eng.choose_variant = lambda sid, vs: 1
    run_until(eng, 1.5)
    assert eng.path[0]["variant"] == "b"


def test_self_repeat_beyond_recorded_count_ends_off_map(bundle):
    """같은 안내가 시간 초과로 녹음에서 본 횟수(max_plays, 없으면 2번)보다 많이 반복되면 지어내지 않고 녹음 밖 종료."""
    st = bundle.orgs["o"]["states"]
    st["N0_TO"]["after_audio"] = {"next": "N0_TO", "timeout_s": {"median": 5.0}}   # 세 번째 실패 규칙 없음
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 40.0)
    assert states(eng) == ["N0", "N0_TO", "N0_TO"] and eng.ended and "반복" in eng.note
    st["N0_TO"]["after_audio"]["max_plays"] = 3
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 40.0)
    assert states(eng) == ["N0", "N0_TO", "N0_TO", "N0_TO"] and "3번" in eng.note


def test_partial_input_without_timeout_record_ends_off_map(bundle):
    """입력 도중 멈췄을 때의 동작이 녹음에 없으면 무음으로 계속 기다리지 않고 기관의 시간 초과 상한 뒤 녹음 밖 종료."""
    st = bundle.orgs["o"]["states"]
    st["IN"].pop("after_audio")
    bundle.orgs["o"]["unseen_timeout_s"] = 6.0
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 30.0, {4.1: "2", 6.0: "1"})
    assert eng.ended and "입력이 없을 때" in eng.note and abs(eng.path[-1]["t"] - 12.0) < 0.1


def test_unverified_behaviors_are_flagged(bundle):
    """녹음에 근거가 부족한 동작은 path 에 표시: 끼어들기 여부를 모르는 상태의 안내 도중 키, 안내 끝 4.5초 뒤 키, 다음 안내
    전에 미리 누른 키, 지연을 재지 못한 연결."""
    st = bundle.orgs["o"]["states"]
    st["N0"]["barge_in"] = "unknown"
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 3.0, {2.0: "1"})
    assert "barge_unverified" in eng.flags
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 9.0, {8.7: "1"})              # 안내 끝(4초) 뒤 4.7초
    assert eng.flags == ["late_key"]
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 6.0, {4.2: "1", 4.4: "3"})    # 1 → M1 진입(4.7초) 전에 3
    assert "type_ahead" in eng.flags
    st["N0"]["keys"]["1"].pop("delay_s")
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 6.0, {4.2: "1"})
    assert eng.flags == ["unmeasured_delay"]
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 6.0, {4.1: "7", 4.35: "7"})   # 두 자리 키의 둘째 자리는 미리 누른 키가 아님
    assert "type_ahead" not in eng.flags


def test_delay_record_without_median_is_flagged(bundle):
    """지연 기록은 있으나 가운뎃값이 비면(관측 없음) 기본값을 쓰고 unmeasured_delay 로 표시."""
    bundle.orgs["o"]["states"]["N0"]["keys"]["1"]["delay_s"] = {"median": None, "n": 0}
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 6.0, {4.2: "1"})
    assert eng.flags == ["unmeasured_delay"] and states(eng) == ["N0", "M1"]


def test_repeated_partial_input_reaches_third_fail(bundle):
    """입력 도중 멈춤도 실패로 계산: 부분 입력을 되풀이해 같은 안내가 다시 나와도 반복 상한·세 번째 무입력 규칙 적용."""
    st = bundle.orgs["o"]["states"]
    st["IN"]["after_audio"] = {"next": "IN", "timeout_s": {"median": 5.0}, "third_fail_next": "END"}
    st["IN"]["input"]["digit_timeout_s"] = 2.0
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    keys = {4.1: "2"}
    t = 4.8
    for _ in range(6):                           # IN 이 나올 때마다 한 자리만 누르고 멈춤
        keys[t] = "1"
        t += 3.2
    run_until(eng, 40.0, keys)
    seq = states(eng)
    assert seq[:3] == ["N0", "IN", "IN"] and "END" in seq and seq.count("IN") <= 3


def test_personal_state_plays_only_from_local_personal_bundle(bundle, tmp_path):
    """개인정보 상태는 기본 묶음에서는 재생하지 않고 종료. 개인정보 포함 묶음(includes_personal)이고 조각이 있으면 실제 녹음을
    그대로 재생하고 다음 연결로 진행. 개인정보 묶음은 git 저장소 안에 만들지 않음."""
    import pytest

    from replay_build import build, inside_git
    st = bundle.orgs["o"]["states"]
    st["P"].update({"clip": "o/M1.ulaw", "keys": {"1": {"next": "M1", "delay_s": {"median": 0.5}}}})
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 8.0, {4.2: "9"})
    assert eng.ended and "개인정보" in eng.note
    bundle.meta["includes_personal"] = True
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 9.0, {4.2: "9", 7.5: "1"})
    assert not eng.ended or eng.note is None
    assert [p["state"] for p in eng.path if "state" in p][:3] == ["N0", "P", "M1"]
    assert any(p.get("personal") for p in eng.path if "state" in p)
    assert inside_git(pathlib.Path(__file__).parent)
    with pytest.raises(SystemExit):
        build(str(tmp_path), str(pathlib.Path(__file__).parent / "_personal_bundle_test"), [], set(), include_personal=True)
    assert not (pathlib.Path(__file__).parent / "_personal_bundle_test").exists()


def test_builder_drops_inferred_edges_and_reads_count_rule():
    """묶음 생성: 관측 없이 옮겨 적은 추정 연결(n=0, inferred_from)은 제외하고, count_rule 은 세 번째 무입력 규칙으로 옮김."""
    from replay_build import input_next, norm_after, norm_keys

    keys = norm_keys({"1": {"next": "A", "n": 3, "delay_s": {"median": 0.5}},
                      "3": {"next": "B", "n": 0, "inferred_from": "S0"},
                      "4": {"next": None, "n": 0, "inferred_next": "C"}})
    assert list(keys) == ["1"]
    a = norm_after({"next": "S_t", "timeout_s": {"median": 5.33}, "next_counts": {"S_t": 1, "S_final": 1},
                    "count_rule": "두 번째 반복 뒤 시간 초과면 S_final"})
    assert a["third_fail_next"] == "S_final"
    assert input_next({"input": {"outcomes": [{"next": "RB", "n": 0, "inferred_from": "X"}]}}) == (None, None)


def test_transcript_mode_emits_recorded_paragraphs_and_drops_unclosed_on_key(bundle):
    """받아쓰기 모드: 조각에 담긴 녹음 당시 받아쓰기를 같은 시각(단락 닫힘·인식 지연)에 내보냄. 단락이 닫히기 전에 키를 받으면
    그 단락은 내보내지 않음(실제 서버는 키를 보낸 뒤 말하던 단락을 버림)."""
    bundle.orgs["o"]["states"]["N0"]["paras"] = [{"text": "메뉴 안내", "start": 0.0, "end": 4.2, "at": 4.6}]
    eng = ReplayEngine(bundle, "o", clock=NIGHT)  # N0 은 1.0초에 시작: 닫힘 5.2초, 받아쓰기 5.6초
    run_until(eng, 2.0)
    assert eng.text_pending and eng.transcripts == []
    run_until(eng, 5.5)
    assert eng.transcripts == []
    run_until(eng, 5.7)
    # 시작 시각은 Segmenter 앞 여유(0.2초)를 뺀 단락 감지 시각
    assert eng.transcripts == [("메뉴 안내", 1.2, False)] and not eng.text_pending
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 5.0, {4.9: "1"})               # 닫히기(5.2초) 전 키, 키 전송이 끝난 때 폐기
    eng.keys_done()
    run_until(eng, 6.0)
    assert eng.transcripts == [] and states(eng) == ["N0", "M1"]


def test_transcript_paragraph_survives_state_change_and_flushes_at_hangup(bundle):
    """상태가 바뀌어도(시간 초과 자동 전환) 이미 시작된 단락은 유지. ARS 가 끊으면 듣던 단락을 flush 로 전달."""
    st = bundle.orgs["o"]["states"]
    st["END"]["paras"] = [{"text": "이용해 주셔서 감사합니다", "start": 0.0, "end": 2.2, "at": 2.6}]
    st["N0"]["paras"] = [{"text": "메뉴 안내", "start": 0.0, "end": 4.2, "at": 9.0}]   # 시간 초과 전환(9초) 뒤에 나옴
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 10.5)
    assert states(eng)[:2] == ["N0", "N0_TO"] and eng.transcripts and eng.transcripts[0][0] == "메뉴 안내"
    eng = ReplayEngine(bundle, "o", clock=NIGHT)
    run_until(eng, 40.0)                          # 세 번째 무입력 → END(1초) → 0.5초 뒤 끊음: END 단락은 닫히기 전
    assert eng.ended and eng.note is None
    assert [t for t, _, _ in eng.flush()] == ["이용해 주셔서 감사합니다"]


def test_replay_call_streams_frames_and_takes_keys(bundle):
    """서버용 ReplayCall: 연결 뒤 20ms 프레임을 실시간으로 내고, 키는 톤 길이만큼 보낸 뒤 0 을 반환(실통화 톤 방식과 같음).
    키를 받아 다음 상태로 가고, 녹음 밖이면 상대가 끊은 것으로 종료."""
    import asyncio

    from replay_ars import ReplayBackend
    from telephony import CallHandlers

    async def go():
        got = {"answer": 0, "frames": 0, "end": None}

        async def on_answer():
            got["answer"] += 1

        async def on_audio(f):
            got["frames"] += 1

        async def on_transcript(*a, **k):
            pass

        async def on_end(reason):
            got["end"] = reason

        be = ReplayBackend(str(bundle.dir), log_path=str(bundle.dir / "replay_log.jsonl"))
        be.bundle = bundle
        import os
        os.environ["REPLAY_CLOCK"] = "2026-11-09T21:00"
        try:
            call = await be.dial("1588-0000", CallHandlers(on_answer, on_audio, on_transcript, on_end))
        finally:
            os.environ.pop("REPLAY_CLOCK", None)
        await asyncio.sleep(1.5)
        # 소리 모드에서는 실통화처럼 '말하는 중' 을 알리지 않음(서버가 소리로 판단)
        assert got["answer"] == 1 and got["frames"] >= 50 and call.eng.speaking and not call.speaking
        dur = await call.send_keys("5")   # 녹음에 없는 키
        assert dur == 0.0
        await asyncio.sleep(0.3)
        assert got["end"] == "remote_hangup" and call.eng.note
        with pytest.raises(RuntimeError):
            await be.dial("1588-9999", CallHandlers(on_answer, on_audio, on_transcript, on_end))

    asyncio.run(go())


def test_replay_call_counts_transcript_until_receiver_takes_it(bundle):
    """받아쓰기 모드: 단락을 넘기는 순간(동기) transcripts_inflight 를 올리고, 받는 쪽이 가져갔다고 알리거나(taken) 처리를 마치면
    해제. 단락 대기 표시(speaking)가 꺼진 뒤 세션이 그 단락을 가져가기 전 사이도 세션이 ARS 활동으로 취급하도록 함."""
    import asyncio

    from replay_ars import ReplayCall
    from telephony import CallHandlers

    async def go():
        seen = {}
        gate = asyncio.Event()

        async def nop(*a, **k):
            pass

        async def on_transcript(text, started_at=None, forced=False, taken=None):
            seen["before"] = call.transcripts_inflight
            await gate.wait()
            taken()
            taken()  # 두 번 호출해도 한 번만 해제
            seen["after"] = call.transcripts_inflight

        call = ReplayCall(bundle, "o", CallHandlers(nop, nop, on_transcript, nop), clock=NIGHT, transcripts=True)
        call._hand_over("안내 문장", 0.0, False)
        assert call.transcripts_inflight == 1
        await asyncio.sleep(0)
        assert seen["before"] == 1
        gate.set()
        await asyncio.sleep(0.01)
        assert seen["after"] == 0 and call.transcripts_inflight == 0

        async def ignore(text, started_at=None, forced=False, taken=None):
            pass
        call.h = CallHandlers(nop, nop, ignore, nop)
        call._hand_over("안내 문장", 0.0, False)
        assert call.transcripts_inflight == 1
        await asyncio.sleep(0.01)
        assert call.transcripts_inflight == 0

    asyncio.run(go())


def test_replay_call_key_reaches_ars_at_tone_end(bundle):
    """서버 경로(ReplayCall.send_keys): 자리마다 톤(0.15초)이 끝난 때 ARS 가 받음. 검증(replay_check)·spec 지연의 기준점과 같음.
    두 번째 자리는 톤+쉼(0.25초) 뒤. 반환 시각은 마지막 쉼까지 마친 뒤."""
    import asyncio
    import time

    from replay_ars import KEY_TONE_ON_S, KEY_TONE_S, ReplayCall
    from telephony import CallHandlers

    async def go():
        async def nop(*a, **k):
            pass

        call = ReplayCall(bundle, "o", CallHandlers(nop, nop, nop, nop), clock=NIGHT)
        call._spawn(call.answer())
        await asyncio.sleep(1.2)
        got = []
        orig = call.eng.key
        call.eng.key = lambda ch: (got.append(time.monotonic()), orig(ch))
        t0 = time.monotonic()
        await call.send_keys("77")
        back = time.monotonic() - t0
        await call.hangup()
        assert len(got) == 2
        assert abs((got[0] - t0) - KEY_TONE_ON_S) < 0.05
        assert abs((got[1] - got[0]) - KEY_TONE_S) < 0.05
        assert abs(back - 2 * KEY_TONE_S) < 0.06

    asyncio.run(go())
