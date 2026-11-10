"""
설정값 로드

.env 파일 또는 환경변수에서 읽음. 시간 관련 기본값은 API 명세서 v0.3 의 기준값 사용.
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
# ARS_ENV_FILE=경로 지정 시 그 .env 도 추가로 읽음 (코드 폴더의 .env 에 없는 값 보충). 먼저 읽은 값 우선:
# 실행 환경변수 > 코드 폴더의 .env > ARS_ENV_FILE
if os.environ.get("ARS_ENV_FILE", "").strip():
    load_env(Path(os.environ["ARS_ENV_FILE"].strip()))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # 인증
    app_key: str = field(default_factory=lambda: _env("APP_KEY", "dev-app-key"))
    token_ttl_s: int = 24 * 3600

    # 통화 백엔드: clawops(실제 발신) / sim(가상 ARS) / replay(녹음 기반 가상 ARS)
    backend: str = field(default_factory=lambda: _env("CALL_BACKEND", "clawops"))
    clawops_from: str = field(default_factory=lambda: _env("CLAWOPS_FROM", ""))
    # 동시 통화용 발신 번호 목록 (쉼표 구분). 비어 있으면 CLAWOPS_FROM 번호 하나만 사용 (단일 번호 동작).
    # CLAWOPS_FROM 에 쉼표 목록 금지: 번호 목록을 지원하지 않는 서버가 같은 .env 를 읽으면 목록 전체를 번호 하나로 이어 붙임
    clawops_lines: str = field(default_factory=lambda: _env("CLAWOPS_LINES", ""))
    # DTMF 전송: inband(톤을 통화 음성에 섞음, 기본) / sdk(ClawOps 미디어 채널 DTMF 이벤트)
    dtmf_mode: str = field(default_factory=lambda: _env("DTMF_MODE", "inband"))
    sim_tree: str = field(default_factory=lambda: _env("SIM_TREE", str(HERE / "sim_kb.json")))
    sim_speed: float = field(default_factory=lambda: float(_env("SIM_SPEED", "1.0")))
    sim_audio: bool = field(default_factory=lambda: _env("SIM_AUDIO", "1") == "1")
    # 음성 가상 ARS: 안내를 TTS 소리로 보내 STT·DTMF 톤까지 실제 발신과 같은 경로로 시험 (Windows)
    sim_voice: bool = field(default_factory=lambda: _env("SIM_VOICE", "0") == "1")
    # 가상 ARS 의 가상 번호 수. 2 이상: 번호 여러 개의 배정(동시 통화)을 실제 발신 없이 시험
    sim_lines: int = field(default_factory=lambda: int(_env("SIM_LINES", "1") or "1"))
    # 녹음 기반 가상 ARS (CALL_BACKEND=replay): 묶음 폴더(replay_build.py 로 생성), 키 뒤 지연을 녹음 범위 안에서 흔들지 여부,
    # 통화마다 지나온 상태·키·종료 이유를 남길 파일(jsonl), 받아쓰기 모드(소리를 인식하지 않고 녹음 당시 받아쓰기를 전달). 번호 수는 SIM_LINES
    replay_bundle: str = field(default_factory=lambda: _env("REPLAY_BUNDLE", ""))
    replay_jitter: bool = field(default_factory=lambda: _env("REPLAY_JITTER", "0") == "1")
    replay_log: str = field(default_factory=lambda: _env("REPLAY_LOG", ""))
    replay_transcripts: bool = field(default_factory=lambda: _env("REPLAY_TRANSCRIPTS", "0") == "1")

    # 음성 인식: whisper(로컬) / openai / rtzr / clova / gemini / qwen / omni / none (후보 엔진 설정은 stt_engines.py, .env.example)
    stt: str = field(default_factory=lambda: _env("STT", "whisper"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "kimthegarden/whisper-small-ko-low-qual-voice-ct2"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "cpu"))
    # ARS 가 말하는 동안 단락 앞부분을 미리 인식 (단락이 끝난 뒤 인식 시간 단축)
    stt_prefetch: bool = field(default_factory=lambda: _env("STT_PREFETCH", "1") == "1")
    # 미리 인식으로 넘길 최소 조각 길이(초). 너무 짧으면 '분실'·'문의' 같은 낱말이 조각 경계에 걸려 앞뒤 소리 없이 인식되어
    # 오인식 증가. 너무 길면 긴 조각에서 Whisper 의 되풀이·누락 증가
    stt_prefetch_min_s: float = field(default_factory=lambda: float(_env("STT_PREFETCH_MIN_S", "5") or "5"))

    # AI 슬롯필링 해석기. 기본값은 목 데이터(MockInterpreter). "모듈:클래스" 형식으로 교체 가능
    interpreter: str = field(default_factory=lambda: _env("INTERPRETER", "interpreter:MockInterpreter"))

    # 발화 구간 검출
    vad_rms: int = field(default_factory=lambda: int(_env("VAD_RMS", "500")))
    paragraph_silence_ms: int = 1200     # 안내가 이 시간 멈추면 단락 종료 (6.1)

    # 탐색 기준 (6.3)
    match_threshold: float = 0.50        # 이 값 이상이면 일치
    low_threshold: float = 0.30          # 이 값 이상 일치 기준 미만이면 불확실(low_confidence 재질문), 미만이면 불일치
    margin: float = 0.10
    # 반복 판정 (명세 6.3). 지나온 메뉴 재도착 횟수 / 키 전송 횟수가 이 값에 이르면 사용자에게 질문. 0 이면 끔
    max_revisit: int = field(default_factory=lambda: int(_env("MAX_REVISIT", "3")))
    max_presses: int = field(default_factory=lambda: int(_env("MAX_PRESSES", "10")))

    # 시간 기준 (초)
    no_response_s: float = 8.0           # 키 전송 후 ARS 가 소리를 내지 않을 때 (7장 3항)
    clarify_expire_s: float = 30.0       # 재질문 expires_at (6.3 무응답 처리)
    clarify_drop_s: float = 60.0         # 재질문 무효 처리
    # 재질문에 60초 동안 답이 없을 때 상담원 연결 키 전송 여부 (명세 6.3). 0 이면 요청을 유지한 채 계속 청취 (기본)
    drop_to_agent: bool = field(default_factory=lambda: _env("CLARIFY_DROP_TO_AGENT", "0") == "1")
    secure_expire_s: float = 60.0        # 민감정보 입력 expires_at
    # 민감정보 입력값을 받았을 때 ARS 가 안내 중이면 안내가 끝날 때까지 기다렸다 보내는 최대 시간(초).
    # '입력 시간이 초과되었습니다' 처럼 키를 받지 않는 안내 중에 보내면 입력 무시됨. 0 이면 즉시 전송
    secure_wait_max_s: float = field(default_factory=lambda: float(_env("SECURE_WAIT_MAX_S", "15")))
    # 시험용 통화 기록(record=true)의 통화 종료 뒤 서버 메모리 보관 시간(초). 소유자가 받거나 지우면 즉시 삭제
    recording_keep_s: float = field(default_factory=lambda: float(_env("RECORDING_KEEP_S", "1800")))
    # 통화 종료 시 통화 기록의 서버 디스크 저장 여부 (1=저장, 기본). 폴더: RECORDING_DIR/<목적>_<전화번호>_<기관>_<날짜-시각>/
    # (call.wav, log.json, log.txt). git 제외(.gitignore). ARS 되읽기 소리와 그 인식 문장이 들어 있으므로 시험 뒤 삭제 필요
    recording_save: bool = field(default_factory=lambda: _env("RECORDING_SAVE", "1") != "0")
    recording_dir: str = field(default_factory=lambda: _env("RECORDING_DIR", "") or str(HERE / "recordings"))
    # 앱의 기록 요청이 없어도(record 없음, Android 앱 등) 모든 통화 기록 (1=켬, 기본 끔). 시험 기간 전용
    record_always: bool = field(default_factory=lambda: _env("RECORD_ALWAYS", "0") == "1")
    user_key_hold_s: float = 10.0        # 사용자 직접 선택 후 AI 보류 (4.3 key)
    # 상대가 끊은 뒤 마지막 안내 인식·결과 요약을 기다리는 최대 시간 (7장 7항). AI 요약 최악 응답(제한 6초 × 시도 2회 = 12초)보다 길게 설정.
    # 초과 시 결과 안내 원문으로 요약을 대신하여 summary → ended 전송
    hangup_flush_s: float = 15.0
    # 단락의 유음(소리) 길이가 이보다 짧으면(초) 인식 생략. 키 전송에 끊긴 조각·딸깍음이 글자로 남는 것 방지
    min_voiced_s: float = field(default_factory=lambda: float(_env("MIN_VOICED_S", "0.4")))
    # 키 전송 뒤 이 시간(초) 안에 끝난 일반 안내: 키에 잘린 조각으로 보고 자막 전송 생략 (판단은 수행)
    post_key_caption_s: float = field(default_factory=lambda: float(_env("POST_KEY_CAPTION_S", "3")))
    # STT=gemini 일 때 받아쓰기와 분류·점수를 Gemini 한 번으로 처리 (1=켬). 안내 종료 → 판단 지연을 호출 한 번만큼 단축.
    # 0 이면 분리 구조(받아쓰기 호출 + 해석기 호출)
    gemini_stt_combined: bool = field(default_factory=lambda: _env("GEMINI_STT_COMBINED", "0") == "1")
    # 외부 STT(openai·rtzr·clova·gemini) 사용 시 민감정보 입력 뒤 음성을 보내지 않는 시간(초). 되읽기 음성의 외부 전송 방지
    secure_skip_s: float = field(default_factory=lambda: float(_env("SECURE_SKIP_S", "20")))
    max_call_s: int = 600
    # 발신 뒤 이 시간(초) 안에 상대가 받지 않으면 끊고 no_answer 로 종료. 통신망이 울림(ringing) 뒤 아무 신호를 주지 않을 때
    # 통화가 '전화를 걸고 있습니다' 상태로 계속 남는 것 방지
    dial_timeout_s: float = field(default_factory=lambda: float(_env("DIAL_TIMEOUT_S", "60")))


settings = Settings()
