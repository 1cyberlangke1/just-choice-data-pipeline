"""支持 `python -m jc`：命令行入口。

输入：sys.argv。输出：进程退出码。
预期行为：**用户输入不对**（workflow 名写错、切分名写错、文件不在）这一类只打印一行人话就收场，
退出码 2；其余异常照旧抛出去——那是 bug，打印一行「出错了」把它藏起来比崩掉更难查
（范式 §9.94 那条「别把 bug 藏起来」的同一条原则）。
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, FileNotFoundError) as exc:
        # KeyError 的 str() 自带一层引号，取 args[0] 才是干净的那句人话。
        detail = exc.args[0] if exc.args else exc
        print(f"错误：{detail}", file=sys.stderr)
        sys.exit(2)
