# 单元测试文档

## 测试结构

项目的测试结构采用模块化设计，与主项目结构保持一致：

```
tests/
  ├── core/                 # 核心模块测试
  │   ├── vector/           # 向量存储相关测试
  │   │   ├── __init__.py
  │   │   └── test_es_vector_store.py  # ES向量存储测试
  │   └── __init__.py
  ├── __init__.py           # 测试包初始化文件
  ├── conftest.py           # pytest配置文件
  └── run_tests.py          # 测试运行脚本
```

## 运行测试

### 方法一：使用测试运行脚本

在项目根目录下运行：

```bash
python tests/run_tests.py
```

### 方法二：使用unittest

在项目根目录下运行：

```bash
python -m unittest discover -s tests
```

### 方法三：运行单个测试文件

运行特定的测试文件：

```bash
python -m unittest tests/core/vector/test_es_vector_store.py
```

### 方法四：使用pytest（如果已安装）

在项目根目录下运行：

```bash
pytest tests/
```

或运行特定测试：

```bash
pytest tests/core/vector/test_es_vector_store.py
```

## 编写新测试

1. 在适当的目录下创建测试文件，文件名应以`test_`开头
2. 测试类应继承`unittest.TestCase`或`unittest.IsolatedAsyncioTestCase`（用于异步测试）
3. 测试方法应以`test_`开头
4. 使用`unittest.mock`来模拟外部服务和依赖

示例：

```python
import unittest
from unittest.mock import patch, MagicMock

class TestSomeClass(unittest.TestCase):
    def setUp(self):
        # 测试前的准备工作
        pass
        
    def test_some_method(self):
        # 测试代码
        pass
``` 