"""支持从源码检出直接运行测试；安装后的使用不依赖此文件。"""
from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
