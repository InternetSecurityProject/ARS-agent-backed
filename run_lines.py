"""발신 번호 여러 개로 동시 통화 서버 실행.

  python run_lines.py --env path/to/.env --port 8001 --no-reload

- --env       : 키가 든 .env 경로. 이 폴더에 .env 가 없을 때 사용 (ARS_ENV_FILE 로 전달. 값 출력 없음).
- 계정의 발신 번호 목록을 ClawOps 에서 조회해 CLAWOPS_LINES 로 전달. 화면에는 번호 수와 끝 2자리만 표시.
- 기본으로 CLAWOPS_FROM 번호 제외: 다른 서버(운영 서버 등)가 사용 중일 가능성. ClawOps 는 번호마다 연결 자리가
  하나뿐: 같은 번호를 두 프로세스가 쓰면 나중에 붙은 쪽이 자리를 가져가 먼저 쓰던 서버의 통화 끊김.
  --keep-from 지정 시 그 번호도 사용 (다른 서버를 내린 뒤에만).
- CLAWOPS_LINES 가 실행 환경이나 .env 에 이미 있으면 계정 목록 조회 없이 그 목록 사용. 이때도 CLAWOPS_FROM 번호는 제외.
- 나머지 인자(--port, --no-reload, --debug 등)는 app.py 에 그대로 전달. 동시 시험 중에는 --no-reload 사용.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


def account_numbers() -> list[str]:
    """계정의 번호 목록 (숫자만). ClawOps 키는 SDK 가 환경변수(CLAWOPS_API_KEY 등)에서 읽음."""
    from clawops import AsyncClawOps

    async def go():
        async with AsyncClawOps() as cl:
            return await cl.numbers.list()

    out = []
    for it in asyncio.run(go()):
        num = it.get("number") if isinstance(it, dict) else getattr(it, "number", None)
        d = "".join(ch for ch in str(num or "") if ch.isdigit())
        if d:
            out.append(d)
    return out


def select_lines(nums: list[str], own: str, keep_from: bool) -> tuple[list[str], bool]:
    """사용할 번호 (중복 제거, 국제 표기 82… 와 국내 표기 0… 는 같은 번호로 취급) 와 CLAWOPS_FROM 번호 제외 여부."""
    from telephony import parse_numbers

    key = lambda d: "0" + d[2:] if d.startswith("82") else d
    mine = key("".join(ch for ch in own if ch.isdigit()))
    use, dropped = [], False
    for n in parse_numbers(",".join(nums)):
        if not keep_from and mine and key(n) == mine:
            dropped = True
            continue
        use.append(n)
    return use, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", help="키가 든 .env 경로 (ARS_ENV_FILE)")
    ap.add_argument("--keep-from", action="store_true", help="CLAWOPS_FROM 번호도 쓴다 (다른 서버가 그 번호를 쓰지 않을 때만)")
    a, rest = ap.parse_known_args()
    if a.env:
        if not os.path.exists(a.env):
            raise SystemExit(f".env 파일이 없습니다: {a.env}")
        os.environ["ARS_ENV_FILE"] = os.path.abspath(a.env)
    import config  # noqa: F401  .env 를 환경변수로 로드

    if os.environ.get("CALL_BACKEND", "clawops") != "sim":
        preset = os.environ.get("CLAWOPS_LINES", "").strip()
        nums = preset.split(",") if preset else account_numbers()
        own = os.environ.get("CLAWOPS_FROM", "")
        use, dropped = select_lines(nums, own, a.keep_from)
        if not use:
            raise SystemExit("쓸 수 있는 번호가 없습니다 (계정 번호가 없거나 CLAWOPS_FROM 하나뿐).")
        os.environ["CLAWOPS_LINES"] = ",".join(use)
        tail = "".join(ch for ch in own if ch.isdigit())[-2:]
        skipped = f", CLAWOPS_FROM 번호(끝 {tail}) 제외" if dropped else ""
        src = "CLAWOPS_LINES 설정" if preset else "계정 번호 목록"
        print(f"발신 번호 {len(use)}개 사용 ({src}, 끝자리 {', '.join(n[-2:] for n in use)}{skipped})", flush=True)
    sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), *rest]
    import app

    app.main()


if __name__ == "__main__":
    main()
