#!/usr/bin/env python
"""
测试运行脚本
用于方便地运行项目中的所有单元测试
"""
import sys
import os
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


def run_all_tests():
    """运行项目中的所有单元测试"""
    # 找到tests目录
    test_dir = Path(__file__).parent.absolute()
    
    # 使用unittest发现并运行所有测试
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 根据测试结果返回合适的退出码
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code) 