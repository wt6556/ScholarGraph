"""CLI 入口 - ScholarGraph 命令行工具"""

import os
import sys
import glob
import logging
import typer
import yaml
from typing import Optional, List
from pathlib import Path

# 配置 logging，确保 INFO 级别日志输出到 stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# 添加 src 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ScholarGraph.shared_env import SharedEnvironment
from ScholarGraph.models.paper import Paper
from ScholarGraph.config import load_taxonomy, TaxonomyNotFoundError, get_config
from ScholarGraph.llm import get_llm_client
from ScholarGraph.functions.classification import Classification
from ScholarGraph.agents.controller import Controller

app = typer.Typer(help="ScholarGraph - 本地优先的 AI 论文研究与分析系统")

DEBUG_MODE = False


def get_env() -> SharedEnvironment:
    """获取 SharedEnvironment 实例"""
    config = get_config()
    env = SharedEnvironment(
        db_path=config.storage.sqlite_path,
        vector_index_path=config.storage.vector_index_path,
        embedding_dimension=config.embedding.dimension
    )
    return env


@app.command()
def parse(
    pdf_path: str = typer.Argument(..., help="PDF 文件路径"),
    title: Optional[str] = typer.Option(None, help="论文标题（可选，自动从 PDF 提取）"),
    source: str = typer.Option("manual", help="论文来源")
):
    """解析单篇论文"""
    typer.echo(f"解析论文: {pdf_path}")

    env = get_env()

    # 调用 Controller 执行摄取流程
    controller = Controller(env)
    result = controller.run_ingestion(pdf_path)

    if result.success:
        typer.echo(f"[INFO] 论文解析成功")
        if result.data and result.data.get("paper_id"):
            typer.echo(f"[INFO] 论文 ID: {result.data.get('paper_id')}")

        # 打印解析出的所有字段
        paper = env.get_current_paper()
        if paper:
            typer.echo("\n=== 解析出的论文字段 ===")
            fields = [
                ("id", paper.id),
                ("title", paper.title),
                ("authors", paper.authors),
                ("year", paper.year),
                ("venue", paper.venue),
                ("ccf_rating", paper.ccf_rating),
                ("doi", paper.doi),
                ("abstract", paper.abstract[:200] + "..." if paper.abstract and len(paper.abstract) > 200 else paper.abstract),
                ("task", paper.task),
                ("method", paper.method),
                ("paper_field", paper.paper_field),
                ("subfield", paper.subfield),
            ]
            for field_name, field_value in fields:
                typer.echo(f"  {field_name}: {field_value}")
    else:
        typer.echo(f"[ERROR] 论文解析失败: {result.error}", err=True)
        raise typer.Exit(1)


@app.command()
def ingest(
    directory: str = typer.Argument(..., help="论文目录路径"),
    recursive: bool = typer.Option(True, help="递归扫描子目录"),
    source: str = typer.Option("arxiv", help="论文来源")
):
    """批量解析论文目录"""
    typer.echo(f"批量解析论文目录: {directory}")

    env = get_env()

    # 查找 PDF 文件
    if recursive:
        pattern = os.path.join(directory, "**/*.pdf")
        pdf_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, "*.pdf")
        pdf_files = glob.glob(pattern)

    if not pdf_files:
        typer.echo(f"[WARNING] 未找到 PDF 文件: {directory}")
        return

    typer.echo(f"[INFO] 找到 {len(pdf_files)} 个 PDF 文件")

    # 批量解析
    controller = Controller(env)
    success_count = 0
    failed_count = 0

    with typer.progressbar(pdf_files, label="解析进度") as bar:
        for pdf_path in bar:
            try:
                result = controller.run_ingestion(pdf_path)
                if result.success:
                    success_count += 1
                else:
                    failed_count += 1
                    typer.echo(f"\n[WARNING] 解析失败: {pdf_path}: {result.error}")
            except Exception as e:
                failed_count += 1
                typer.echo(f"\n[ERROR] 解析异常: {pdf_path}: {e}")

    typer.echo(f"\n[INFO] 批量解析完成: 成功 {success_count}, 失败 {failed_count}")


@app.command()
def query(
    question: str = typer.Argument(..., help="查询问题")
):
    """查询论文"""
    typer.echo(f"查询: {question}")

    env = get_env()

    from ScholarGraph.agents.query_agent import QueryAgent
    from ScholarGraph.agents.critic_agent import CriticAgent
    from ScholarGraph.memory.conversation_history import ConversationHistory
    from ScholarGraph.memory.pipeline_state import PipelineState
    from ScholarGraph.functions.retrieval import Retrieval
    from ScholarGraph.functions.synthesis import Synthesis
    from ScholarGraph.models.query import QueryAnalysisResult

    # 1. 查询分析
    query_agent = QueryAgent(ConversationHistory())
    analysis_result = query_agent.execute(env, question)

    if not analysis_result.success:
        typer.echo(f"[ERROR] 查询分析失败: {analysis_result.error}", err=True)
        raise typer.Exit(1)

    typer.echo(f"[INFO] 查询类型: {analysis_result.data.query_type}")
    typer.echo(f"[INFO] 扩展查询: {analysis_result.data.original_query}")

    # 2. 检索相关 Chunk
    retrieval_func = Retrieval()
    query_analysis = analysis_result.data
    retrieval_result = retrieval_func.execute(env, query_analysis)

    if not retrieval_result.success:
        typer.echo(f"[ERROR] 检索失败: {retrieval_result.error}", err=True)
        raise typer.Exit(1)

    # 解包 RetrievalResult
    if hasattr(retrieval_result.data, 'chunks') and hasattr(retrieval_result.data, 'scores'):
        chunks = retrieval_result.data.chunks
        scores = retrieval_result.data.scores
        original_query = getattr(retrieval_result.data, 'original_query', query_analysis.original_query)
        translated_query = getattr(retrieval_result.data, 'translated_query', query_analysis.original_query)
    else:
        chunks = []
        scores = []
        original_query = query_analysis.original_query
        translated_query = query_analysis.original_query

    typer.echo(f"[INFO] 检索到 {len(chunks)} 个相关 Chunk（来自 {len(set(c.paper_id for c in chunks))} 篇论文）")

    if not chunks:
        typer.echo(f"[WARNING] 没有找到相关 Chunk")
        return

    # 3. 合成答案（传递原始查询用于语言控制，传递扩展后查询用于检索）
    synthesis_func = Synthesis()
    answer_result = synthesis_func.execute(env, chunks, scores, query_analysis,
                                           original_query=question,  # 用户原始输入，用于语言控制
                                           translated_query=query_analysis.original_query)  # 扩展后查询用于检索

    if not answer_result.success:
        typer.echo(f"[ERROR] 答案合成失败: {answer_result.error}", err=True)
        raise typer.Exit(1)

    answer = answer_result.data
    typer.echo(f"\n[答案] {answer.conclusion}")

    # 显示证据/引用
    if answer.evidence:
        typer.echo("\n[引用]")
        for i, ev in enumerate(answer.evidence, 1):
            paper_id = ev.get("source", "unknown")
            paper = env.get_paper(paper_id) if paper_id != "unknown" else None
            title = paper.title if paper else paper_id
            typer.echo(f"  {i}. [{paper_id[:8]}] {title}")
            if ev.get("statement"):
                typer.echo(f"     {ev['statement']}")

    if answer.summary:
        typer.echo(f"\n[摘要] {answer.summary}")

    # 4. 评估答案
    critic_agent = CriticAgent(PipelineState())
    critique_result = critic_agent.execute(env, answer, question)

    if critique_result.success:
        if critique_result.data.get("passed"):
            typer.echo(f"[评估] 答案通过检查")
        else:
            typer.echo(f"[评估] 答案未通过检查: {critique_result.data.get('failed_checks')}")
    else:
        typer.echo(f"[WARNING] 评估过程出错: {critique_result.error}")


@app.command()
def update_ccf(
    mappings_file: str = typer.Argument(..., help="CCF PDF 文件路径")
):
    """
    更新 CCF 评级映射

    从 CCF PDF 文件中解析会议/期刊评级信息，
    生成 venue → CCF rating 映射表并保存到 config/ccf_mappings.yaml

    示例：
        python -m ScholarGraph.cli update-ccf ./CCF目录.pdf
    """
    typer.echo(f"正在解析 CCF PDF: {mappings_file}")

    from ScholarGraph.functions.ccf_parser import CCFParser

    parser = CCFParser()
    result = parser.execute(
        env=get_env(),
        ccf_pdf_path=mappings_file,
        output_path="./config/ccf_mappings.yaml"
    )

    if not result.success:
        typer.echo(f"[ERROR] 解析 CCF PDF 失败: {result.error}", err=True)
        raise typer.Exit(1)

    mappings = result.data
    total_venues = len(mappings.get("venue_ratings", {}))

    typer.echo(f"[INFO] CCF 评级映射已更新")
    typer.echo(f"[INFO] 版本: {mappings.get('version', 'unknown')}")
    typer.echo(f"[INFO] 共提取 {total_venues} 个会议/期刊")

    # 按评级统计
    ratings = {"CCF-A": 0, "CCF-B": 0, "CCF-C": 0}
    for rating in mappings.get("venue_ratings", {}).values():
        if rating in ratings:
            ratings[rating] += 1

    typer.echo(f"[INFO] CCF-A: {ratings['CCF-A']}, CCF-B: {ratings['CCF-B']}, CCF-C: {ratings['CCF-C']}")
    typer.echo(f"[INFO] 已保存到: ./config/ccf_mappings.yaml")


@app.command()
def update_taxonomy(
    source: str = typer.Argument(
        ...,
        help="分类来源: ccs (ACM Computing Classification System) / ccf (中国计算机学会) / journal (SCI期刊) / custom (自定义YAML)"
    ),
    file_path: Optional[str] = typer.Option(
        None,
        "--file", "-f",
        help="自定义分类体系 YAML 文件路径（仅 source=custom 时需要）"
    ),
    reclassify: bool = typer.Option(True, help="是否重新分类已有论文")
):
    """
    更新分类体系

    支持从以下来源自动获取分类：
    - ccs:  ACM Computing Classification System (ACM 计算机分类体系)
    - ccf: 中国计算机学会（CCF）推荐国际学术会议与期刊列表
    - journal: SCI 期刊分类
    - custom: 自定义分类（需要 --file 指定 YAML 文件）

    示例：
        python -m ScholarGraph.cli update-taxonomy ccs
        python -m ScholarGraph.cli update-taxonomy ccf
        python -m ScholarGraph.cli update-taxonomy custom --file ./my_taxonomy.yaml
    """
    valid_sources = ["ccs", "ccf", "journal", "custom"]

    if source not in valid_sources:
        typer.echo(
            f"[ERROR] 无效的来源: {source}\n"
            f"支持的来源: {', '.join(valid_sources)}",
            err=True
        )
        raise typer.Exit(1)

    if source == "custom" and not file_path:
        typer.echo(
            "[ERROR] source=custom 时需要提供 --file 参数指定 YAML 文件",
            err=True
        )
        raise typer.Exit(1)

    if source != "custom" and file_path:
        typer.echo(
            f"[WARNING] --file 参数仅在 source=custom 时有效，将忽略",
            err=False
        )

    typer.echo(f"正在从 {source} 来源获取分类体系...")

    from ScholarGraph.functions.taxonomy_updater import TaxonomyUpdater

    updater = TaxonomyUpdater()
    result = updater.execute(
        env=get_env(),
        source_name=source,
        file_path=file_path,
        output_path="./config/taxonomy.yaml"
    )

    if not result.success:
        typer.echo(f"[ERROR] 更新分类体系失败: {result.error}", err=True)
        raise typer.Exit(1)

    # 显示新分类体系
    typer.echo(f"\n[INFO] 分类体系已更新 (来源: {source})")
    typer.echo(f"[INFO] 共 {result.data['fields_count']} 个领域")

    # 备份旧文件
    taxonomy_path = "./config/taxonomy.yaml"
    if os.path.exists(taxonomy_path + ".backup"):
        backup_files = glob.glob(taxonomy_path + ".backup.*")
        if backup_files:
            typer.echo(f"[INFO] 旧分类体系已备份")

    # 重新分类已有论文
    if reclassify:
        typer.echo(f"\n[INFO] 重新分类已有论文...")
        env = get_env()

        # 重置 taxonomy 缓存以加载新的
        from ScholarGraph.config import reset_taxonomy
        reset_taxonomy()

        # 获取所有论文
        papers = env.get_all_papers(limit=10000)
        total = len(papers)
        typer.echo(f"[INFO] 共 {total} 篇论文需要重新分类")

        if total == 0:
            typer.echo("[INFO] 无论文需要分类")
            return

        # 重新分类
        classification_func = Classification()
        success_count = 0
        failed_count = 0

        with typer.progressbar(papers, label="分类进度") as bar:
            for paper in bar:
                try:
                    result = classification_func.execute(env, paper)
                    if result.success:
                        success_count += 1
                    else:
                        failed_count += 1
                        typer.echo(f"\n[WARNING] 分类失败 {paper.id}: {result.error}")
                except TaxonomyNotFoundError:
                    typer.echo(f"\n[ERROR] 分类体系文件丢失: {taxonomy_path}", err=True)
                    raise typer.Exit(1)
                except Exception as e:
                    failed_count += 1
                    typer.echo(f"\n[WARNING] 分类异常 {paper.id}: {e}")

        typer.echo(f"\n[INFO] 分类完成: 成功 {success_count}, 失败 {failed_count}")
    else:
        typer.echo(f"\n[INFO] 已跳过重新分类（使用 --no-reclassify）")


@app.command()
def repl(debug: bool = typer.Option(False, "--debug", help="打印调试信息")):
    """启动交互模式"""
    import os
    os.environ["SCHOLARGRAPH_DEBUG"] = "1" if debug else "0"
    if debug:
        typer.echo("[DEBUG] 调试模式已开启")
    typer.echo("ScholarGraph 交互模式 (输入 'exit' 或 'quit' 退出)")

    env = get_env()
    typer.echo(f"[INFO] SharedEnvironment: {env}")

    # 导入查询相关模块
    from ScholarGraph.agents.query_agent import QueryAgent
    from ScholarGraph.memory.conversation_history import ConversationHistory
    from ScholarGraph.functions.retrieval import Retrieval
    from ScholarGraph.functions.synthesis import Synthesis
    from ScholarGraph.models.query import QueryAnalysisResult
    from ScholarGraph.utils.topology_visualizer import generate_topology_graph, export_topology_json

    query_agent = QueryAgent(ConversationHistory())
    retrieval_func = Retrieval()
    synthesis_func = Synthesis()

    # 检测是否需要生成拓扑图的关键词
    topology_keywords = ["拓扑图", "关系图", "方法图", "topology", "graph", "关系网络", "方法关系"]

    while True:
        try:
            user_input = typer.prompt("\nScholarGraph> ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ('exit', 'quit', 'q'):
            break

        if not user_input.strip():
            continue

        # 检测是否需要生成拓扑图
        needs_topology = any(kw in user_input.lower() for kw in [k.lower() for k in topology_keywords])

        if needs_topology:
            # 生成拓扑图
            typer.echo("[INFO] 正在生成拓扑图...")

            output_path = "./output/topology.png"
            os.makedirs("./output", exist_ok=True)

            # 生成图片
            success = generate_topology_graph(env, output_path)
            if success:
                typer.echo(f"[INFO] 拓扑图已保存到: {output_path}")

                # 同时导出 JSON
                json_path = "./output/topology.json"
                export_topology_json(env, json_path)
                typer.echo(f"[INFO] 拓扑数据已保存到: {json_path}")
            else:
                typer.echo("[ERROR] 生成拓扑图失败", err=True)
            continue

        # 执行普通查询流程
        typer.echo("[INFO] 正在分析查询...")

        # 1. 查询分析
        analysis_result = query_agent.execute(env, user_input)
        if not analysis_result.success:
            typer.echo(f"[ERROR] 查询分析失败: {analysis_result.error}")
            continue

        typer.echo(f"[INFO] 扩展查询: {analysis_result.data.expanded_query}")

        # 2. 检索相关 Chunk
        query_analysis = analysis_result.data
        retrieval_result = retrieval_func.execute(env, query_analysis)

        if not retrieval_result.success:
            typer.echo(f"[ERROR] 检索失败: {retrieval_result.error}")
            continue

        # 解包 RetrievalResult
        if hasattr(retrieval_result.data, 'chunks') and hasattr(retrieval_result.data, 'scores'):
            chunks = retrieval_result.data.chunks
            scores = retrieval_result.data.scores
            # original_query 用于检索语言控制，translated_query 是扩展后的查询
            original_query = getattr(retrieval_result.data, 'original_query', user_input)
            translated_query = getattr(retrieval_result.data, 'translated_query', user_input)
        else:
            chunks = []
            scores = []
            original_query = user_input
            translated_query = user_input

        typer.echo(f"[INFO] 检索到 {len(chunks)} 个相关 Chunk")

        if not chunks:
            typer.echo("[WARNING] 没有找到相关 Chunk")
            continue

        # 3. 合成答案（传递原始查询用于语言控制，传递扩展后查询用于检索）
        typer.echo("[INFO] 正在合成答案...")
        answer_result = synthesis_func.execute(env, chunks, scores, query_analysis,
                                               original_query=user_input,  # 用户原始输入，用于语言控制
                                               translated_query=query_analysis.original_query)  # 扩展后查询用于检索

        if not answer_result.success:
            typer.echo(f"[ERROR] 答案合成失败: {answer_result.error}")
            continue

        answer = answer_result.data
        typer.echo(f"\n[答案] {answer.conclusion}")

        # 显示证据/引用
        if answer.evidence:
            typer.echo("\n[引用]")
            for i, ev in enumerate(answer.evidence, 1):
                paper_id = ev.get("source", "unknown")
                paper = env.get_paper(paper_id) if paper_id != "unknown" else None
                title = paper.title if paper else paper_id
                typer.echo(f"  {i}. [{paper_id[:8]}] {title}")
                if ev.get("statement"):
                    typer.echo(f"     {ev['statement']}")

        if answer.summary:
            typer.echo(f"\n[摘要] {answer.summary}")

    typer.echo("\n再见!")


@app.command()
def init():
    """初始化数据目录和配置"""
    typer.echo("初始化 ScholarGraph...")

    # 创建数据目录
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./config", exist_ok=True)
    typer.echo("[INFO] 数据目录已创建: ./data, ./config")

    # 检查分类体系
    taxonomy_path = "./config/taxonomy.yaml"
    if not os.path.exists(taxonomy_path):
        # 创建默认分类体系
        default_taxonomy = {
            'fields': [
                {
                    'name': 'Computer Science',
                    'subfields': [
                        'Artificial Intelligence',
                        'Machine Learning',
                        'Computer Vision',
                        'Natural Language Processing',
                        'Data Mining',
                        'Computer Graphics',
                        'Computer Networks',
                        'Databases',
                        'Software Engineering'
                    ]
                },
                {
                    'name': 'Mathematics',
                    'subfields': [
                        'Statistics',
                        'Optimization',
                        'Probability Theory',
                        'Numerical Analysis'
                    ]
                },
                {
                    'name': 'Physics',
                    'subfields': [
                        'Quantum Computing',
                        'Condensed Matter Physics'
                    ]
                }
            ],
            'default_field': 'Computer Science',
            'default_subfield': 'Artificial Intelligence'
        }
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_taxonomy, f, allow_unicode=True, sort_keys=False)
        typer.echo(f"[INFO] 默认分类体系已创建: {taxonomy_path}")
    else:
        typer.echo(f"[INFO] 分类体系已存在: {taxonomy_path}")

    # 检查配置文件
    config_path = "./config/config.yaml"
    if not os.path.exists(config_path):
        typer.echo(f"[INFO] 请创建配置文件: {config_path}")
    else:
        typer.echo(f"[INFO] 配置文件已存在: {config_path}")

    typer.echo("\n初始化完成!")


@app.command()
def download_model(
    output_dir: str = typer.Option("./models", help="本地模型保存路径")
):
    """
    下载 Embedding 模型和 Reranker 模型到本地

    示例：
        python -m ScholarGraph.cli download-model
    """
    try:
        config = get_config()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # ============================================================
        # 1. 下载 Embedding 模型
        # ============================================================
        embedding_model_name = config.embedding.model
        # 如果是本地路径，提取出模型名
        # 本地路径格式: ./models/xxx, models/xxx, C:\path\xxx 等
        # HuggingFace 格式: org/model-name 或 model-name
        # 本地路径特征：包含路径分隔符或下划线（模型名通常不带路径分隔符）
        is_local_path = (
            "./" in embedding_model_name or
            "../" in embedding_model_name or
            "C:\\" in embedding_model_name or
            embedding_model_name.startswith("/") or
            (embedding_model_name.startswith("models") and ("_" in embedding_model_name or "/" in embedding_model_name)) or
            (len(embedding_model_name.split("/")) == 2 and ("_" in embedding_model_name.split("/")[-1] or "-" in embedding_model_name.split("/")[-1]))
        )

        if is_local_path:
            # 本地路径格式: ./models/microsoft/harrier-oss-v1-0.6b 或 models/microsoft/harrier-oss-v1-0.6b
            # 下载时需要转换为 HuggingFace 格式: microsoft/harrier-oss-v1-0.6b

            # 处理 models/microsoft/harrier-oss-v1-0.6b 格式（os.path.basename 会丢失 microsoft/）
            if embedding_model_name.startswith("models/"):
                # 提取 models/ 后面的部分: microsoft/harrier-oss-v1-0.6b
                path_after_models = embedding_model_name[len("models/"):]
                hf_model_name = path_after_models  # 直接使用，因为已经是 org/model 格式
            else:
                basename = os.path.basename(embedding_model_name.rstrip("/"))
                hf_model_name = basename

            # 如果 hf_model_name 包含下划线，需要转换为 org/model 格式
            # 例如 microsoft_harrier-oss-v1-0.6b -> microsoft/harrier-oss-v1-0.6b
            if "_" in hf_model_name and "/" not in hf_model_name:
                known_orgs = ["microsoft", "bert", "roberta", "t5", "gpt", "llama", "bloom", "clip", "sbert"]
                for org in known_orgs:
                    if hf_model_name.startswith(f"{org}_"):
                        hf_model_name = f"{org}/{hf_model_name[len(org)+1:]}"
                        break
        else:
            hf_model_name = embedding_model_name
        typer.echo(f"[1/2] 正在下载 Embedding 模型: {hf_model_name}")
        typer.echo(f"[DEBUG] config.embedding.model = {config.embedding.model}")
        typer.echo(f"[DEBUG] is_local_path = {is_local_path}")
        from transformers import AutoTokenizer, AutoModel

        embedding_local_path = os.path.join(output_dir, hf_model_name.replace("/", "_"))
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        embedding_model = AutoModel.from_pretrained(hf_model_name)
        embedding_model.save_pretrained(embedding_local_path)
        tokenizer.save_pretrained(embedding_local_path)
        typer.echo(f"[INFO] Embedding 模型已保存到: {embedding_local_path}")

        # ============================================================
        # 2. 下载 Reranker 模型
        # ============================================================
        reranker_model = config.rag.reranker.model
        # 处理本地路径格式
        if reranker_model.startswith("models/"):
            hf_reranker_model = reranker_model[len("models/"):]
        else:
            hf_reranker_model = reranker_model
        reranker_name = hf_reranker_model.replace("/", "_")
        typer.echo(f"[2/2] 正在下载 Reranker 模型: {hf_reranker_model}")
        from sentence_transformers import CrossEncoder

        reranker_local_path = os.path.join(output_dir, reranker_name)
        reranker = CrossEncoder(hf_reranker_model)
        reranker.save(reranker_local_path)
        typer.echo(f"[INFO] Reranker 模型已保存到: {reranker_local_path}")

        typer.echo(f"\n[SUCCESS] 两个模型均已下载完成")
        typer.echo(f"请在 config.yaml 中设置 embedding.model 为: {embedding_local_path}")

    except ImportError as e:
        typer.echo(f"[ERROR] 缺少依赖: {e}", err=True)
        typer.echo("安装命令: pip install transformers torch sentence-transformers", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[ERROR] 下载失败: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def topology(
    output_path: str = typer.Option("./output/topology.png", help="拓扑图输出路径"),
    format: str = typer.Option("png", help="输出格式: png, json")
):
    """
    生成方法拓扑图

    示例：
        python -m ScholarGraph.cli topology
        python -m ScholarGraph.cli topology --output-path ./output/my_topology.png
        python -m ScholarGraph.cli topology --format json
    """
    typer.echo("正在生成拓扑图...")

    env = get_env()

    if format == "json":
        from ScholarGraph.utils.topology_visualizer import export_topology_json
        result = export_topology_json(env, output_path.replace(".png", ".json"))
    else:
        from ScholarGraph.utils.topology_visualizer import generate_topology_graph
        result = generate_topology_graph(env, output_path)

    if result:
        typer.echo(f"[INFO] 拓扑图已保存到: {output_path}")
    else:
        typer.echo("[ERROR] 生成拓扑图失败", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
