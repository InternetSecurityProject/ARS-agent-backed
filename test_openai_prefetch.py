"""Service-level prefetch invariants; synthetic audio, no external calls."""
import asyncio
import time
from types import SimpleNamespace
import pytest

from audio import FRAME, SILENCE_FRAME, hum_ulaw


def run(c): return asyncio.run(c)


async def settle(predicate):
    for _ in range(100):
        if predicate(): return
        await asyncio.sleep(.001)
    assert predicate(), 'background task did not settle'


def test_work_happens_before_final_and_ordered_overlap_is_not_duplicated():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[];texts=['잔액 조회는 3번, 카드 업무는', '카드 업무는 5번입니다.']
        async def transcribe(data): calls.append(data);return texts.pop(0)
        c=PrefetchedSegment(transcribe,min_chunk_s=1,overlap_s=.8)
        first=hum_ulaw(2000);full=first+hum_ulaw(1000)
        c.offer(first)
        await settle(lambda:len(calls)==1 and c.prefetched_bytes==len(first))
        assert c.prefetched_bytes==len(first)  # recognition completed while menu is still arriving
        result=await c.finish(full)
        assert result=='잔액 조회는 3번, 카드 업무는 5번입니다.'
        assert len(calls)==2 and len(calls[1])==14400
    run(go())


def test_pending_offers_are_coalesced_not_queued_unboundedly():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[];gate=asyncio.Event()
        async def transcribe(data):
            calls.append(len(data))
            if len(calls)==1: await gate.wait()
            return '메뉴 안내'
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        c.offer(hum_ulaw(1500));await settle(lambda:len(calls)==1)
        for n in range(2,11): c.offer(hum_ulaw(n*1000))
        gate.set();await settle(lambda:c.prefetched_bytes==80000)
        assert len(calls)==2
        c.discard()
    run(go())


def test_silent_tail_needs_no_extra_request_and_short_segment_falls_back():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[]
        async def transcribe(data): calls.append(data);return '안내는 7번.'
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        audio=hum_ulaw(2000);c.offer(audio)
        await settle(lambda:c.prefetched_bytes==len(audio))
        assert await c.finish(audio+SILENCE_FRAME*60)=='안내는 7번.'
        assert len(calls)==1
        short=PrefetchedSegment(transcribe)
        assert await short.finish(hum_ulaw(500))=='안내는 7번.'
        assert len(calls)==2
    run(go())


def test_failed_chunk_uses_full_segment_without_losing_words():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[]
        async def transcribe(data):
            calls.append(data)
            return '' if len(calls)==1 else '다시 듣기는 9번.'
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        c.offer(hum_ulaw(1500));await settle(lambda:len(calls)==1)
        full=hum_ulaw(2500)
        assert await c.finish(full)=='다시 듣기는 9번.'
        assert calls[-1]==full
    run(go())


def test_unalignable_overlap_repeated_key_retries_whole_paragraph():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[];texts=['예약은 2번을 눌러주시고.', '그러면 2번을 눌러주시고 조회는 8번.', '예약은 2번, 조회는 8번.']
        async def transcribe(data):
            calls.append(data);return texts.pop(0)
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        c.offer(hum_ulaw(2000));await settle(lambda:c.prefetched_bytes>0)
        whole=hum_ulaw(4000)
        assert await c.finish(whole)=='예약은 2번, 조회는 8번.'
        assert c.metrics['fallback'] and calls[-1]==whole
    run(go())


def test_tail_receives_only_its_own_paragraph_context():
    from stt_prefetch import PrefetchedSegment
    async def go():
        contexts=[]
        async def ordinary(data): return '잔액 조회는 3번.'
        async def contextual(data,previous):
            contexts.append(previous);return '잔액 조회는 3번. 카드 업무는 5번.'
        a=PrefetchedSegment(ordinary,transcribe_with_context=contextual,min_chunk_s=1)
        a.offer(hum_ulaw(1500));await settle(lambda:a.prefetched_bytes>0)
        assert await a.finish(hum_ulaw(2500))=='잔액 조회는 3번. 카드 업무는 5번.'
        assert contexts==['잔액 조회는 3번.']
        b=PrefetchedSegment(ordinary,transcribe_with_context=contextual)
        assert await b.finish(hum_ulaw(500))=='잔액 조회는 3번.'
        assert len(contexts)==1
    run(go())


def test_discard_cancels_work_and_never_uploads_a_fallback():
    from stt_prefetch import PrefetchedSegment
    async def go():
        started=asyncio.Event();cancelled=asyncio.Event();calls=[]
        async def transcribe(data):
            calls.append(data);started.set()
            try: await asyncio.Event().wait()
            finally: cancelled.set()
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        c.offer(hum_ulaw(1500));await started.wait();c.discard()
        await settle(cancelled.is_set)
        assert await c.finish(hum_ulaw(2500))==''
        assert len(calls)==1
    run(go())


def test_different_audio_cannot_reuse_a_previous_prefix():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[]
        async def transcribe(data): calls.append(data);return '새로운 안내'
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        c.offer(hum_ulaw(1500));await settle(lambda:c.prefetched_bytes>0)
        different=SILENCE_FRAME*10+hum_ulaw(2000)
        assert await c.finish(different)=='새로운 안내'
        assert calls[-1]==different
    run(go())


def test_final_wait_is_bounded_even_when_background_request_hangs():
    from stt_prefetch import PrefetchedSegment
    async def go():
        async def transcribe(data): await asyncio.Event().wait()
        c=PrefetchedSegment(transcribe,min_chunk_s=1,timeout_s=.02)
        audio=hum_ulaw(1500);c.offer(audio);await asyncio.sleep(0)
        before=time.monotonic();assert await c.finish(audio)==''
        assert time.monotonic()-before<.2
    run(go())


def test_stitch_handles_spacing_punctuation_and_keeps_distinct_keys():
    from stt_prefetch import stitch,OverlapConflict
    assert stitch('거래 내역 조회는 4번.', '거래내역조회는 4번, 카드 발급은 5번.')=='거래 내역 조회는 4번. 카드 발급은 5번.'
    assert stitch('신청은 1번', '조회는 7번')=='신청은 1번 조회는 7번'
    with pytest.raises(OverlapConflict): stitch('신청은 1번', '조회는 1번')


@pytest.mark.parametrize('before,after', [
    ('이전 메뉴는 *번', '이전 메뉴는 #번, 상담원은 0번.'),
    ('카드 발급은 3번', '카드 발급은 5번, 다른 서비스는 #번.'),
])
def test_conflicting_overlap_keys_require_rerecognition(before,after):
    from stt_prefetch import stitch,OverlapConflict
    with pytest.raises(OverlapConflict): stitch(before,after)


def test_conflicting_prefetch_overlap_uses_full_audio_result():
    from stt_prefetch import PrefetchedSegment
    async def go():
        calls=[];texts=['이전 메뉴는 *번','이전 메뉴는 #번, 상담원은 0번.','이전 메뉴는 #번, 상담원은 0번.']
        async def transcribe(data): calls.append(data);return texts.pop(0)
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        c.offer(hum_ulaw(1500));await settle(lambda:c.prefetched_bytes>0)
        full=hum_ulaw(2500)
        assert await c.finish(full)=='이전 메뉴는 #번, 상담원은 0번.'
        assert calls[-1]==full and c.metrics['fallback']
    run(go())


@pytest.mark.parametrize('mode',['short','tail','failed_prefix'])
def test_discard_during_final_request_cancels_it_and_returns_no_text(mode):
    from stt_prefetch import PrefetchedSegment
    async def go():
        started=asyncio.Event();cancelled=asyncio.Event();calls=[]
        async def transcribe(data):
            calls.append(data)
            if mode!='short' and len(calls)==1:
                return '' if mode=='failed_prefix' else '카드 업무는 5번.'
            started.set()
            try: await asyncio.Event().wait()
            finally: cancelled.set()
        c=PrefetchedSegment(transcribe,min_chunk_s=1)
        if mode!='short':
            c.offer(hum_ulaw(1500));await settle(lambda:len(calls)==1)
        future=asyncio.create_task(c.finish(hum_ulaw(2500)))
        await started.wait();c.discard()
        await settle(cancelled.is_set)
        assert await future==''
    run(go())


class ProbeRecognizer:
    is_local=False
    def __init__(self): self.calls=[];self.contexts=[]
    async def recognize(self,data): self.calls.append(data);return '잔액 조회는 3번, 카드 업무는 5번.'
    def new_segment(self):
        from stt_prefetch import PrefetchedSegment
        c=PrefetchedSegment(self.recognize,min_chunk_s=1.5);self.contexts.append(c);return c


def make_session():
    from test_flows import make_settings
    from app import build_deps
    from session import Session
    deps=build_deps(make_settings(stt_prefetch=True));rec=ProbeRecognizer();deps.recognizer=rec
    s=Session('5550000',deps)
    s.call=SimpleNamespace(provides_transcripts=False)
    s.status='listening';s.agent_active=True
    return s,rec


async def feed(s, frames):
    for f in frames:
        await s._on_audio(f);await asyncio.sleep(0)


def speech_frames():
    # Varying syllables avoid the music detector; 300ms pause after 2.4s of speech.
    voiced=[hum_ulaw(20)]*10+[SILENCE_FRAME]*5
    return voiced*8+[SILENCE_FRAME]*15


def test_session_starts_stt_early_but_never_decides_before_endpoint():
    async def go():
        s,rec=make_session();paragraphs=[]
        async def handle(text,started): paragraphs.append(text)
        s.on_paragraph=handle
        worker=s.spawn(s._stt_worker())
        await feed(s,speech_frames())
        await settle(lambda:len(rec.calls)>0)
        assert not paragraphs
        await feed(s,[SILENCE_FRAME]*60)
        await settle(lambda:len(paragraphs)==1)
        assert not s._prefetch_segments
        await s.finish('user_canceled',hangup=False)
        await worker
    run(go())


def test_privacy_blocks_early_and_final_upload_even_if_window_expires_mid_segment():
    async def go():
        s,rec=make_session();s.secure_window=True;s.secure_sent_at=time.monotonic()
        worker=s.spawn(s._stt_worker())
        await feed(s,speech_frames())
        assert not rec.calls
        s.secure_sent_at=time.monotonic()-100
        await feed(s,[SILENCE_FRAME]*60)
        await settle(lambda:s.stt_inflight==0)
        assert not rec.calls
        await s.finish('user_canceled',hangup=False);await worker
    run(go())


def test_session_finish_and_key_press_discard_only_owned_contexts():
    async def go():
        s,rec=make_session();other,other_rec=make_session()
        await feed(s,speech_frames());await feed(other,speech_frames())
        assert s._prefetch_segments and other._prefetch_segments
        await s.finish('user_canceled',hangup=False)
        assert not s._prefetch_segments and other._prefetch_segments
        async def keys(_): return .01
        other.call.send_keys=keys
        await other._send_keys('1')
        assert not other._prefetch_segments
        await other.finish('user_canceled',hangup=False)
    run(go())


def test_detector_dropped_paragraph_discards_its_prefetch_context():
    async def go():
        s,rec=make_session();await feed(s,speech_frames())
        assert s._prefetch_segments
        # Detector's supported discard result: speaking -> reset, no segment.
        def drop(frame): s.segmenter.reset();return None
        s.segmenter.feed=drop
        await s._on_audio(SILENCE_FRAME)
        assert not s._prefetch_segments and s._prefetch is None
        await s.finish('user_canceled',hangup=False)
    run(go())
