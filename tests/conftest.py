"""
pytest配置文件
用于设置测试环境和通用测试夹具
"""
import sys
import os
from pathlib import Path

# 将项目根目录添加到Python路径中，以便能够导入项目模块
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root) 