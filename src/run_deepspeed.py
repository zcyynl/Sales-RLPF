#!/usr/bin/env python
"""
run_deepspeed.py
deepspeed 0.11.1 + pydantic v2 兼容启动器。
在 deepspeed 任何模块导入之前先打 patch，然后调用 deepspeed 原生 launcher。

用法（替代 deepspeed 命令）：
  python run_deepspeed.py --include=localhost:0,1 train_grpo_rlpf.py [args...]
"""

# ── pydantic v2 compat patch，必须在所有 deepspeed import 之前 ──
import pydantic as _pydantic
if int(_pydantic.VERSION.split('.')[0]) >= 2:
    from pydantic.fields import FieldInfo as _FieldInfo
    if not hasattr(_FieldInfo, 'required'):
        _FieldInfo.required = property(lambda self: self.is_required())
# ──────────────────────────────────────────────────────────────

from deepspeed.launcher.runner import main

if __name__ == '__main__':
    main()
