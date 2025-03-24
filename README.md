# Wolf 智能代理系统

Wolf是一个基于大型语言模型（LLM）的智能代理系统，集成了向量检索、嵌入计算、重排序等功能，用于构建高效的人工智能应用。

## 项目结构

```
/
├── client.py          # 客户端配置和初始化
├── config.py          # 全局配置文件
├── core/              # 核心功能模块
│   ├── Agent/         # 智能代理实现
│   ├── embeddings/    # 嵌入向量计算
│   ├── llms/          # 大语言模型接口
│   ├── mem/           # 记忆/状态管理
│   ├── openai_types.py # OpenAI接口类型定义
│   ├── rags/          # 检索增强生成系统
│   ├── ranks/         # 结果重排序功能
│   ├── runner/        # 运行器实现
│   ├── util.py        # 核心工具函数
│   └── vector/        # 向量数据库接口
├── logs/              # 日志目录
├── tests/             # 测试目录
└── utils/             # 通用工具
    ├── log.py         # 日志工具
    └── retry.py       # 重试机制
```

## 核心功能

### 大语言模型 (LLMs)

系统支持连接到多种大语言模型服务，包括：
- DeepSeek V3
- DeepSeek R1
- 其他兼容OpenAI接口的LLM

### 嵌入计算

提供多种嵌入计算方式：
- Silicon嵌入代理（基于BGE-M3模型）
- HTTP嵌入代理（通用HTTP接口）

### 向量存储

实现了高效的向量数据管理和检索功能：
- Milvus向量数据库接口
- 支持稠密向量和稀疏向量

### 检索增强生成 (RAG)

提供检索增强生成系统，结合向量检索和大型语言模型：
- 支持上下文检索
- 结果重排序优化

## 快速开始

### 环境要求

- Python 3.10+
- 相关依赖库（建议使用虚拟环境）

### 安装依赖

```bash
pip install -r requirements.txt  # 需要创建此文件
```

### 基本使用

1. 配置客户端

```python
from core.Agent.assient import AsyncAssistant
import asyncio
from config import *
from core.llms.openai_llm import OpenAICoT

async def main():
    llm = OpenAICoT(
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
    assistant = AsyncAssistant(
        llm=llm,
        memory=None,
        function_list=[],
        instruction="以懂王特朗普的口吻回答问题",
        name="Assistant",
        stream=True
    )
    response = await assistant.run("如何看待LGBTQ")
    async for thinking, content in response:
        if thinking:
            print(thinking)
        if content:
            print(content)
    
if __name__ == "__main__":
    asyncio.run(main())
```

## 测试

项目包含多种测试用例，用于验证各个组件的功能：
- `agent_test.py`: 代理功能测试
- `function_test.py`: 函数调用测试
- `mcp_test.py`: 主控制协议测试
- `open_test.py`: 开放功能测试

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

[需要添加许可证信息] 