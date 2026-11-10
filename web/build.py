"""app.src.html + Preact 단독 모듈 + 앱 아이콘 → static/app.html (외부 스크립트 없이 한 파일).

준비: htm 3.1.1 패키지를 web/vendor 에 푼다 (npm pack htm@3.1.1 → tar xzf htm-3.1.1.tgz -C web/vendor)
실행: python web/build.py <ars-agent-frontend>/app/src/main/assets/figma static/app.html
"""
import json
import re
import sys
from pathlib import Path

here = Path(__file__).parent
src = (here / "app.src.html").read_text(encoding="utf-8")
# npm ci is the reproducible frontend setup; keep the older npm-pack layout usable.
vendor_path = here / "node_modules/htm/preact/standalone.module.js"
if not vendor_path.exists():
    vendor_path = here / "vendor/package/preact/standalone.module.js"
vendor = vendor_path.read_text(encoding="utf-8")
icons_dir = Path(sys.argv[1])
out = Path(sys.argv[2])

m = re.search(r"export\{([^}]*)\};?\s*$", vendor)
assert m, "export 구문을 찾지 못함"
pairs = []
for part in m.group(1).split(","):
    local, _, name = part.strip().partition(" as ")
    pairs.append(f"{name}:{local}")
preact = vendor[: m.start()] + "return{" + ",".join(pairs) + "};"

icons = {}
for f in sorted(icons_dir.glob("*.svg")):
    svg = f.read_text(encoding="utf-8")
    svg = re.sub(r'\s(id|style|preserveAspectRatio|overflow)="[^"]*"', "", svg)
    svg = re.sub(r">\s+<", "><", svg).strip()
    icons[f.stem] = svg

html = src.replace("/*__PREACT__*/", preact).replace("/*__ICONS__*/", json.dumps(icons, ensure_ascii=False))
assert "/*__" not in html
out.write_text(html, encoding="utf-8")
print(out, len(html), "bytes,", len(icons), "icons")
