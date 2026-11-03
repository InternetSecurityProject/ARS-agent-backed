"""
설정값 로드

.env 파일 또는 환경변수에서 읽는다. 시간 관련 값은 API 명세서 v0.3 의 기준값을 기본값으로 둔다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env(HERE / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # 인증
    app_key: str = field(default_factory=lambda: _env("APP_KEY", "dev-app-key"))
    token_ttl_s: int = 24 * 3600

    # 통화 백엔드: clawops(실제 발신) / sim(가상 ARS)
    backend: str = field(default_factory=lambda: _env("CALL_BACKEND", "clawops"))
    clawops_from: str = field(default_factory=lambda: _env("CLAWOPS_FROM", ""))
    # DTMF 전송: inband(톤을 통화 음성에 섞음, 실통화 확인됨) / sdk(ClawOps 미디어 채널 DTMF 이벤트)
    dtmf_mode: str = field(default_factory=lambda: _env("DTMF_MODE", "inband"))
    sim_tree: str = field(default_factory=lambda: _env("SIM_TREE", str(HERE / "sim_kb.json")))
    sim_speed: float = field(default_factory=lambda: float(_env("SIM_SPEED", "1.0")))
    sim_audio: bool = field(default_factory=lambda: _env("SIM_AUDIO", "1") == "1")
    # 음성 가상 ARS: 안내를 TTS 소리로 보내 STT·DTMF 톤까지 실통화와 같은 경로로 시험 (Windows)
    sim_voice: bool = field(default_factory=lambda: _env("SIM_VOICE", "0") == "1")

    # 음성 인식: whisper(로컬) / openai / rtzr / clova / gemini / qwen / none (후보 엔진 설정은 stt_engines.py, .env.example)
    stt: str = field(default_factory=lambda: _env("STT", "whisper"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "kimthegarden/whisper-small-ko-low-qual-voice-ct2"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "cpu"))
    # ARS 가 말하는 동안 단락 앞부분을 미리 인식 (단락이 끝난 뒤 인식 시간 단축)
    stt_prefetch: bool = field(default_factory=lambda: _env("STT_PREFETCH", "1") == "1")

    # AI 슬롯필링(팀원 담당). 기본은 목 데이터(MockInterpreter). "모듈:클래스" 형식으로 교체한다.
    interpreter: str = field(default_factory=lambda: _env("INTERPRETER", "interpreter:MockInterpreter"))

    # 발화 구간 검출
    vad_rms: int = field(default_factory=lambda: int(_env("VAD_RMS", "500")))
    paragraph_silence_ms: int = 1200     # 안내가 이 시간 멈추면 단락 종료 (6.1)

    # 탐색 기준 (6.3)
    match_threshold: float = 0.50        # 이 값 이상이면 일치
    low_threshold: float = 0.30          # 이 값 이상 일치 기준 미만이면 불확실(low_confidence 재질문), 미만이면 불일치
    margin: float = 0.10
    # 반복 판정 (명세 6.3). 지나온 메뉴에 다시 도착한 횟수 / 키 전송 횟수가 이 값에 이르면 사용자에게 묻는다. 0 이면 끔
    max_revisit: int = field(default_factory=lambda: int(_env("MAX_REVISIT", "3")))
    max_presses: int = field(default_factory=lambda: int(_env("MAX_PRESSES", "10")))

    # 시간 기준 (초)
    no_response_s: float = 8.0           # 키 전송 후 ARS 가 소리를 내지 않을 때 (7장 3항)
    clarify_expire_s: float = 30.0       # 재질문 expires_at (6.3 무응답 처리)
    clarify_drop_s: float = 60.0         # 재질문 무효 처리
    # 재질문에 60초 동안 답이 없을 때 상담원 연결 키를 누를지 (명세 6.3). 0 이면 요청을 유지한 채 계속 듣는다 (기본)
    drop_to_agent: bool = field(default_factory=lambda: _env("CLARIFY_DROP_TO_AGENT", "0") == "1")
    secure_expire_s: float = 60.0        # 민감정보 입력 expires_at
    user_key_hold_s: float = 10.0        # 사용자 직접 선택 후 AI 보류 (4.3 key)
    # 상대가 끊은 뒤 마지막 안내 인식·결과 요약을 기다리는 최대 시간 (7장 7항). AI 요약 최악 응답(제한 6초 × 재시도 2회 = 12초)보다 길게.
    # 넘기면 결과 안내 원문으로 요약을 대신하여 summary → ended 를 보낸다
    hangup_flush_s: float = 15.0
    # 단락의 유음(소리) 길이가 이보다 짧으면(초) 인식하지 않는다. 키 전송에 끊긴 조각·딸깍음이 글자로 남는 것을 막는다 (설계안 2.4)
    min_voiced_s: float = field(default_factory=lambda: float(_env("MIN_VOICED_S", "0.4")))
    # 키 전송 뒤 이 시간(초) 안에 끝난 일반 안내는 키에 잘린 조각으로 보고 자막으로 보내지 않는다 (판단은 한다)
    post_key_caption_s: float = field(default_factory=lambda: float(_env("POST_KEY_CAPTION_S", "3")))
    # STT=gemini 일 때 받아쓰기와 분류·점수를 Gemini 한 번으로 처리 (1=켬). 안내 종료 → 판단 지연이 호출 한 번만큼 준다.
    # 0 이면 원래 구조(받아쓰기 호출 + 해석기 호출)
    gemini_stt_combined: bool = field(default_factory=lambda: _env("GEMINI_STT_COMBINED", "0") == "1")
    # 외부 STT(openai·rtzr·clova·gemini) 사용 시 민감정보 입력 뒤 음성을 보내지 않는 시간(초). 되읽기가 외부로 나가지 않게 한다
    secure_skip_s: float = field(default_factory=lambda: float(_env("SECURE_SKIP_S", "20")))
    max_call_s: int = 600


settings = Settings()
