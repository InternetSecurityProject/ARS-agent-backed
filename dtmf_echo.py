"""Suppress only pure outgoing-tone echo; never blindly mute new ARS speech."""
import time
import numpy as np

from audio import DTMF_HZ, RATE, SILENCE_FRAME
from clawops.agent._audio import ulaw_to_pcm16


class DtmfEchoGuard:
    def __init__(self):
        self._keys=set()
        self._until=0.
        self._bases={}

    def clear(self):
        self._keys.clear()
        self._until=0.

    def start(self, keys):
        self._keys=set(keys) & DTMF_HZ.keys()
        self._until=time.monotonic()+len(keys)*.26+1.

    def finish(self):
        if self._keys:
            self._until=time.monotonic()+.12  # short echo tail, not a speech mute

    def filter(self, frame):
        if not self._keys:return frame
        if time.monotonic()>self._until:
            self.clear();return frame
        x=np.frombuffer(ulaw_to_pcm16(frame),dtype=np.int16).astype(np.float64)
        if not len(x):return frame
        x-=x.mean();energy=float(x@x)
        if energy/len(x)<300**2:return frame
        for key in self._keys:
            cache=(key,len(x))
            if cache not in self._bases:
                t=np.arange(len(x))/RATE
                basis=np.column_stack([f(2*np.pi*hz*t) for hz in DTMF_HZ[key]
                                       for f in (np.sin,np.cos)])
                self._bases[cache]=(basis,np.linalg.pinv(basis))
            basis,inverse=self._bases[cache];coeff=inverse@x
            residual=x-basis@coeff
            low=basis[:,:2]@coeff[:2];high=basis[:,2:]@coeff[2:]
            # Both expected frequencies must explain virtually all energy.
            # Mixed voice remains intact even if the tone is also audible.
            if (float(residual@residual)/energy<.06
                    and float(low@low)/energy>.12 and float(high@high)/energy>.12):
                return SILENCE_FRAME if len(frame)==len(SILENCE_FRAME) else b'\xff'*len(frame)
        return frame
