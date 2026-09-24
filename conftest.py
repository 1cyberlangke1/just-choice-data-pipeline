"""pytest 公共配置：把 src/ 与仓库根放进 sys.path。

本仓库用 src 布局（`src/jc` 是通用层，`workflows/` 是各 workflow），
所以测试要能同时 import 到两者。装了 `pip install -e .` 之后这一步是冗余的，
但保留它可以让「没装包」的环境也能直接跑 pytest。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
