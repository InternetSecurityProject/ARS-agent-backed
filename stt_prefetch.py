"""Bounded, paragraph-owned STT prefetch. No decisions or telephony operations.

One job per paragraph; pending offers replace each other. Overlapping audio and
text stitching protect word boundaries. The owning Session gates private audio
BEFORE offer(), and discards this object on keys, private input and call end.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from audio import FRAME, rms

log = logging.getLogger('ars.stt')


class OverlapConflict(ValueError):
    pass


def stitch(previous: str, current: str) -> str:
    if not previous: return current.strip()
    if not current: return previous.strip()
    # A disagreement over the same overlapped phrase's DTMF key is not a safe
    # text deduplication opportunity. Re-recognize the whole paragraph instead.
    lpos=[i for i,c in enumerate(previous) if c.isalpha()]
    rpos=[i for i,c in enumerate(current) if c.isalpha()]
    letters_l=''.join(previous[i] for i in lpos)
    letters_r=''.join(current[i] for i in rpos)
    for size in range(min(100,len(lpos),len(rpos)),3,-1):
        if letters_l[-size:]==letters_r[:size]:
            a=previous[lpos[-size]:]
            b=current[:rpos[size] if size<len(rpos) else len(current)]
            keys_a=''.join(c for c in a if c in '0123456789*#')
            keys_b=''.join(c for c in b if c in '0123456789*#')
            if keys_a and keys_b and keys_a!=keys_b: raise OverlapConflict('overlap_key_conflict')
            break
    left=''.join(c for c in previous if c.isalnum() or c in '*#')
    positions=[i for i,c in enumerate(current) if c.isalnum() or c in '*#']
    right=''.join(current[i] for i in positions)
    for size in range(min(100,len(left),len(right)),3,-1):
        if left[-size:]==right[:size]:
            rest=current[positions[size-1]+1:].lstrip(' \t,，.。')
            return (previous.rstrip()+' '+rest).strip()
    # If a shared numbered choice cannot be aligned reliably, concatenation
    # could create two different labels for one key (later removed as unsafe).
    # Do not guess which rendering of the overlap is correct.
    keys = lambda s: {re.sub(r'\s+', '', k).replace('번','').replace('별표','*').replace('샵','#')
                      for k in re.findall(r'(?<!\d)\d{1,4}\s*번|[*#]|별표|샵', s)}
    if keys(previous) & keys(current):
        raise OverlapConflict('overlap_unaligned_key')
    return (previous.rstrip()+' '+current.strip()).strip()


class PrefetchedSegment:
    def __init__(self, transcribe, *, transcribe_with_context=None, min_chunk_s=6.0, overlap_s=2.0, timeout_s=10.0, quiet_rms=500):
        self.transcribe=transcribe
        self.transcribe_with_context=transcribe_with_context
        self.minimum=int(min_chunk_s*8000)
        self.overlap=int(overlap_s*8000)
        self.timeout_s=timeout_s
        self.quiet_rms=quiet_rms
        self._source=b''
        self._pending=None
        self._done=0
        self._text=''
        self._task=None
        self._final_task=None
        self._failed=False
        self._discarded=False
        self._finishing=False
        self.metrics={'requests':0,'uploaded_audio_s':0.0,'prefetched_audio_s':0.0,'fallback':False}

    @property
    def prefetched_bytes(self): return self._done

    def offer(self, prefix: bytes):
        if self._discarded or self._finishing or self._failed: return
        if not prefix.startswith(self._source):
            self._failed=True
            return
        self._source=prefix
        if len(prefix)-self._done<self.minimum: return
        self._pending=prefix  # coalesce instead of queueing every pause
        if self._task is None or self._task.done():
            self._task=asyncio.create_task(self._drain())

    async def _recognize(self, data, *, use_context=False):
        self.metrics['requests']+=1
        self.metrics['uploaded_audio_s']=round(self.metrics['uploaded_audio_s']+len(data)/8000,3)
        if use_context and self._text and self.transcribe_with_context is not None:
            return await self.transcribe_with_context(data,self._text[-240:])
        return await self.transcribe(data)

    async def _drain(self):
        try:
            while self._pending is not None and not self._discarded:
                prefix,self._pending=self._pending,None
                if len(prefix)-self._done<self.minimum: continue
                begin=max(0,self._done-self.overlap)
                text=await self._recognize(prefix[begin:],use_context=True)
                if self._discarded: return
                if not text:
                    self._failed=True
                    return
                self._text=stitch(self._text,text)
                self._done=len(prefix)
                self.metrics['prefetched_audio_s']=round(self._done/8000,2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed=True
            log.warning('STT 선처리 실패: %s',type(exc).__name__)

    def discard(self):
        self._discarded=True
        self._pending=None
        self._source=b''
        self._text=''
        if self._task is not None and not self._task.done(): self._task.cancel()
        if self._final_task is not None and not self._final_task.done(): self._final_task.cancel()

    async def finish(self, audio: bytes) -> str:
        if self._discarded: return ''
        self._finishing=True
        self._pending=None
        started=time.monotonic()
        try:
            async with asyncio.timeout(self.timeout_s):
                self._final_task=asyncio.create_task(self._finish(audio))
                return await self._final_task
        except asyncio.CancelledError:
            if self._discarded: return ''
            raise
        except TimeoutError:
            log.warning('STT 최종 대기 시간 초과')
            self.discard()
            return ''
        finally:
            self.metrics['finish_wait_s']=round(time.monotonic()-started,3)
            self._source=b''

    async def _finish(self, audio):
        if self._task is not None:
            try: await self._task
            except asyncio.CancelledError:
                if self._discarded: return ''
                raise
        if self._discarded: return ''
        if self._failed or not audio.startswith(self._source) or not self._done:
            self.metrics['fallback']=True
            text=await self._recognize(audio)
            return '' if self._discarded else text
        remaining=audio[self._done:]
        voiced=any(rms(remaining[i:i+FRAME])>self.quiet_rms for i in range(0,len(remaining)-FRAME+1,FRAME))
        if not voiced: return self._text
        text=await self._recognize(audio[max(0,self._done-self.overlap):],use_context=True)
        if self._discarded: return ''
        if not text:
            self.metrics['fallback']=True
            text=await self._recognize(audio)
            return '' if self._discarded else text
        try:
            return stitch(self._text,text)
        except OverlapConflict:
            self.metrics['fallback']=True
            text=await self._recognize(audio)
            return '' if self._discarded else text
