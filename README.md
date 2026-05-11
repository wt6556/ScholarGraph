# ScholarGraph

本地优先的 AI 论文研究与分析方法系统，基于 **Agent + RAG (Retrieval-Augmented Generation) 架构**，支持多论文理解、关系挖掘、知识拓扑构建。

![方法拓扑图示例](figs/topology.png)

> **图 1**: LoRA 相关方法的演进关系拓扑图。节点表示方法，边表示方法间的改进关系及改进方向。

## 核心优势

| 优势 | 说明 |
|------|------|
| **本地优先** | 所有数据存储在本地，无云依赖，保障隐私安全 |
| **方法演进追踪** | 独特的拓扑图功能，自动从论文中提取方法改进关系，构建知识演进图谱 |
| **多论文联合理解** | 支持批量摄入论文，联合理解方法之间的关联与演化 |
| **混合 RAG 检索** | 向量 + 关键词 + 重排三重检索，保证答案准确性 |
| **开箱即用** | 一条命令完成论文解析、索引、查询全流程 |

## 特性

- **多 Agent 协作**: Orchestrator、QueryAgent、CriticAgent 分工明确
- **混合 RAG**: FAISS 向量检索 + BM25 关键词检索 + Reranker 重排
- **方法拓扑图**: 追踪方法演进关系，构建知识图谱
- **CCF 评级映射**: 自动识别论文学术级别
- **多 LLM 支持**: Anthropic Claude、MiniMax、OpenAI、Ollama 本地模型

## 架构

![架构图](figs/Architecture.png = 300x)

**核心组件**:
- **Orchestrator**: Controller（工作流编排，调度 Functions 执行论文摄取/查询流程）
- **Agent 层**: QueryAgent（查询分析）、CriticAgent（答案评估）
- **Function 层**: 9 个无状态 Function（PDFParser、Understanding、Enrichment、Classification、Relation、Retrieval、Synthesis、Embedding、CCFParser）
- **RAG 层**: VectorSearch、BM25Search、Reranker、Fusion
- **Storage 层**: PaperStore（SQLite）、ChunkStore（FAISS）、TopologyStore

## 安装

### 1. 进入项目目录

```bash
cd ScholarGraph
```

### 2. 安装依赖

```bash
# 使用 conda 环境
conda activate agent

# 安装项目
pip install -e .
```

### 3. 下载模型

系统需要两个模型，均从 `config/config.yaml` 读取：

```bash
# 下载 Embedding 模型 + Reranker 模型到 ./models 目录
python -m ScholarGraph.cli download-model

```

### 4. 安装 Graphviz（用于生成拓扑图）

Graphviz 用于生成方法拓扑图。

1. **安装 Python 包**：

```bash
pip install graphviz
```

2. **安装 Graphviz 软件**：
   - macOS: `brew install graphviz`
   - Linux: `sudo apt install graphviz` 或 `sudo yum install graphviz`
   - Windows: 从 https://graphviz.org/download/ 下载安装

安装后将 Graphviz 的 bin 目录添加到系统 PATH 环境变量。

### 5. 配置

编辑 `config/config.yaml`：

```yaml
# LLM 配置（必需）
llm:
  provider: "anthropic"  # 或 "openai" / "ollama"
  model: "MiniMax-M2.7-highspeed"
  api_key: "your_api_key_here"
  base_url: "https://api.minimaxi.com/anthropic"
  timeout: 60

# Embedding 配置
embedding:
  provider: "transformers"
  model: "./models/sentence-transformers_all-MiniLM-L6-v2"  # 本地路径，或 HuggingFace 模型名
  dimension: 384
  device: "cpu"

# PDF 解析配置
pdf_parser:
  primary: "pdfplumber"  # 或 "grobid"
```

## 使用

### 分类体系

系统使用 field → subfield 层级分类体系，支持以下来源：

| 来源    | 说明                                 | 来源链接                                                         |
| ------- | ------------------------------------ | ---------------------------------------------------------------- |
| ccs     | ACM Computing Classification System  | https://www.acm.org/publications/computing-classification-system |
| ccf     | 中国计算机学会推荐学术会议与期刊列表 | https://ccf-cccr.ccf.org.cn/                                     |
| journal | SCI 期刊分类                         | https://journals.clarivate.com/                                  |
| custom  | 自定义 YAML 文件                     | 用户自行编写                                                     |

当前分类体系包含 5 个领域、40+ 个子领域：

```yaml
fields:
  - name: Computer Science
    subfields:
      - Artificial Intelligence
      - Computer Vision
      - Natural Language Processing
      - Machine Learning
      - Large Language Model Fine-tuning
      - Parameter-Efficient Fine-Tuning
      # ... 更多子领域
  - name: Systems & Security
  - name: Media & Interaction
  - name: Theory & Algorithms
  - name: Interdisciplinary
```

### CLI 命令

```bash
# 初始化项目（创建数据目录和默认分类体系）
python -m ScholarGraph.cli init

# 下载 embedding 模型到本地（推荐，避免每次运行时下载）
python -m ScholarGraph.cli download-model

# 解析单篇论文
python -m ScholarGraph.cli parse paper.pdf

# 批量解析论文目录
python -m ScholarGraph.cli ingest ./papers/

# 查询论文
python -m ScholarGraph.cli query "LoRA 方法的改进有哪些？"

# 更新分类体系（从 ACM CCS 来源获取，会自动重新分类已有论文）
python -m ScholarGraph.cli update-taxonomy ccs

# 更新分类体系（从 CCF 来源获取）
python -m ScholarGraph.cli update-taxonomy ccf

# 更新分类体系（使用自定义 YAML 文件）
python -m ScholarGraph.cli update-taxonomy custom --file ./my_taxonomy.yaml

# 更新分类体系（不重新分类已有论文）
python -m ScholarGraph.cli update-taxonomy ccs --no-reclassify

# 更新 CCF 评级映射
python -m ScholarGraph.cli update-ccf ./CCF目录.pdf

# 生成方法拓扑图
python -m ScholarGraph.cli topology

# 启动交互模式
python -m ScholarGraph.cli repl
```

### 拓扑图生成

方法拓扑图用于可视化论文方法之间的改进关系。

**命令行生成：**

```bash
# 生成拓扑图（PNG 格式）
python -m ScholarGraph.cli topology

# 指定输出路径
python -m ScholarGraph.cli topology --output-path ./output/my_topology.png

# 导出 JSON 格式（包含节点和边的详细数据）
python -m ScholarGraph.cli topology --format json --output-path ./output/topology.json
```

**REPL 中生成：**

在交互模式下，输入包含以下关键词时会自动生成拓扑图：

- 拓扑图、关系图、方法图
- topology、graph
- 关系网络、方法关系

```
ScholarGraph> 生成 LoRA 相关方法的拓扑图
[INFO] 正在生成拓扑图...
[INFO] 拓扑图已保存到: ./output/topology.png
[INFO] 拓扑数据已保存到: ./output/topology.json
```

## 项目结构

```
ScholarGraph/
├── config/          # 配置文件（LLM、分类体系、CCF评级）
├── data/           # SQLite 数据库、FAISS 索引
├── papers/         # PDF 论文
├── output/         # 生成的拓扑图
├── models/         # 下载的 embedding 和 reranker 模型
├── tests/          # pytest 测试
└── README.md, pyproject.toml, requirements.txt
```

核心代码位于 `src/ScholarGraph/`，按功能分为 agents、functions、storage、rag、llm、config 等模块。
