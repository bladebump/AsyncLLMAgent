# 异步智能代理系统

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

系统提供了灵活的大语言模型接口，支持多种模型实现：

#### 基础模型接口
- 支持基本的对话功能
- 支持函数调用
- 支持流式输出
- 支持最大长度限制

#### 链式思考模型
- 支持思考过程输出
- 支持流式和非流式响应
- 支持提示词和消息格式

#### 具体实现
- OpenAI 实现
  - 支持 OpenAI API
  - 支持流式输出
  - 支持函数调用
  - 支持思考过程输出

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

- Python 3.11+
- 相关依赖库（建议使用虚拟环境）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

1. 配置客户端
2. 初始化 LLM 和助手
3. 运行对话

## 测试

项目包含完整的测试用例，用于验证各个组件的功能。

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

[需要添加许可证信息] 