"""Local-only recognition for protected audio; no raw text reaches external AI.

The parser emits allowlisted field metadata or source-proven confirmation keys.
It does not infer business menus, numbers, or whether a private value is correct.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
import re
from interpreter import Prompt, Option
from readback_display import readback_options

_FIELDS = re.compile(r'계좌\s*비밀\s*번호|주민\s*등록\s*번호|주민\s*번호|사업자\s*등록\s*번호|'
                     r'계좌\s*번호|생년\s*월일|카드\s*번호|비밀\s*번호|인증\s*번호|'
                     r'휴대\s*(?:폰|전화)\s*번호|전화\s*번호|연락처|OTP|CVC', re.I)
_COMMAND = re.compile(r'입력(?:해\s*주|하(?:여|고|신\s*후|신\s*뒤|십시오|세요))|'
                      r'눌러\s*주|누르(?:신\s*후|신\s*뒤|세요|십시오|시고|고)')
_COUNT = re.compile(r'(앞|뒤)?\s*(\d{1,2}|열네|열세|열두|열한|열|아홉|여덟|일곱|여섯|다섯|네|세|두|한)\s*자리')
_NUMBERS={'한':1,'두':2,'세':3,'네':4,'다섯':5,'여섯':6,'일곱':7,'여덟':8,'아홉':9,'열':10,'열한':11,'열두':12,'열세':13,'열네':14}


def protected_prompt(text: str) -> Prompt | None:
    """Never return raw text in a parsed prompt, including mixed readbacks."""
    choices = readback_options(text, Prompt('confirm',''))
    if choices and re.search(r'입력하신|입력한|번호는|번호가|맞으(?:면|시면)|정확하(?:면|시면)',text):
        return Prompt('confirm','입력 정보 확인이 필요합니다.',choices,
                      critical=bool(re.search(r'결제|송금|이체|해지|해약|계약|가입',text)))
    for sentence in re.split(r'[.!?。;\n]',text):
        # A confirmation/re-entry option is not a fresh field-entry instruction.
        if re.search(r'맞으(?:면|시면)|틀리(?:면|시면)|하시려면|하려면|입력하지',sentence):
            continue
        fields=list(_FIELDS.finditer(sentence))
        if len(fields)!=1:
            continue
        field=fields[0]
        tail=sentence[field.end():]
        command=_COMMAND.search(tail)
        if command is None or re.search(r'\d\s*번',tail[:command.end()]):
            continue  # Business-menu keys do not request entry of the field.
        label=re.sub(r'\s+','',field.group())
        count=_COUNT.search(tail[:command.start()])
        if count:
            n=int(count[2]) if count[2].isdigit() else _NUMBERS[count[2]]
            label+=f' {count[1]+" " if count[1] else ""}{n}자리'
        # Keep only an explicitly heard, unambiguous symbol. Never assume #.
        from gemini_interpreter import normalize_keys
        normalized=normalize_keys(tail)
        symbols=set(re.findall(r'(?:후|뒤|다음|입력하고|누르고|마지막으로|와|과)\s*([#*])\s*(?:버튼|키)?(?:을|를)?\s*(?:눌러|누르)',normalized))
        term=next(iter(symbols)) if len(symbols)==1 else None
        text=f'{label}를 입력해 주세요.'
        if term:text+=f' 입력 후 {term} 버튼을 눌러 주세요.'
        return Prompt('sensitive',text,label=label,terminator=term)
    # Later results/menus also stay local; never re-upload merely because
    # the privacy timer expired. Display raw text only to the session owner.
    from gemini_interpreter import rule_classify
    rule=rule_classify(text)
    if rule.kind=='result':
        return Prompt('result','조회 결과를 화면에서 확인해 주세요.')
    if rule.kind=='input_error':
        return Prompt('input_error','ARS 입력 시간이 지났거나 입력이 거절되었어요.')
    if rule.kind=='menu':
        keys=list(dict.fromkeys(o.key for o in rule.all_options() if re.fullmatch(r'[0-9*#]',o.key)))
        if keys:
            return Prompt('menu','개인정보 보호 중인 ARS 메뉴입니다. 화면에서 번호를 선택해 주세요.',
                          [Option(key,f'{key}번') for key in keys],critical=True)
    return None


class ProtectedWhisperRecognizer:
    """Disk-only model, one worker, no transcript logs or speculative requests."""
    is_local=True
    def __init__(self,model_path: str,timeout_s: float=6):
        self.path=Path(model_path).resolve();self.timeout_s=timeout_s
        self._model=None;self._work=None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            if not (self.path/'model.bin').is_file():raise FileNotFoundError('Local STT model missing')
            self._model=WhisperModel(str(self.path),device='cpu',compute_type='int8',
                                     cpu_threads=4,local_files_only=True)
        return self._model

    async def warmup(self):
        await asyncio.to_thread(self._load)

    def _run(self,audio):
        import numpy as np
        from audio import ulaw8k_to_pcm16k
        pcm=np.frombuffer(ulaw8k_to_pcm16k(audio),dtype=np.int16).astype(np.float32)/32768
        segments,_=self._load().transcribe(pcm,language='ko',beam_size=3,
            initial_prompt='ARS 입력 안내. 계좌번호, 계좌 비밀번호, 주민등록번호, 사업자등록번호, 생년월일, 카드번호, 인증번호, 휴대폰번호, 별표, 샵 버튼. 실제 들리는 내용만 전사합니다.',
            condition_on_previous_text=False,temperature=0,without_timestamps=True,
            vad_filter=False)
        return ' '.join(s.text.strip() for s in segments if s.no_speech_prob<.6 and s.avg_logprob>-1 and s.compression_ratio<2.4)

    async def recognize(self,audio):
        # A timeout cannot stop the CPU thread. Do not queue overlapping work.
        if self._work is not None and not self._work.done():return ''
        self._work=asyncio.create_task(asyncio.to_thread(self._run,audio))
        self._work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        try:return await asyncio.wait_for(asyncio.shield(self._work),self.timeout_s)
        except Exception:return ''  # no external retry, no raw exception text
