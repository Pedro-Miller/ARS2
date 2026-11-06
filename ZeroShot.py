import sys
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
import networkx as nx
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Configuração de Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ZeroShotPipeline")

# ---------------------------------------------------------------------------
# Estruturas de Auditoria e Dados
# ---------------------------------------------------------------------------
@dataclass
class AuditMetrics:
    total_raw_records: int = 0
    communities_count: int = 0
    posts_count: int = 0
    comments_count: int = 0
    ignored_records: int = 0
    orphan_comments: int = 0
    missing_parents: List[str] = field(default_factory=list)
    initial_nodes_count: int = 0
    pruned_nodes_count: int = 0
    retained_nodes_count: int = 0
    structural_context_nodes: int = 0
    retention_rate: float = 0.0
    # Métrica estendida: Distribuição sociológica de Stance no Grafo Retido
    stance_distribution: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeData:
    id: str
    parsed_id: str
    data_type: str  # 'community', 'post', 'comment'
    text_content: str
    author: Optional[str] = None
    parent_id: Optional[str] = None
    post_id: Optional[str] = None
    community_name: Optional[str] = None
    stance_absoluto: Optional[str] = None  # PRO, ANTI, NEUTRAL
    stance_score: Optional[float] = None
    is_context_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Filtro de Slang (Gírias) - Task 1.3
# ---------------------------------------------------------------------------
SLANG_DISAGREE_KEYWORDS = {
    "cap", "crap", "bs", "bullshit", "lmao no", "cope", "ratio",
    "brainrot", "delusional", "fake news", "nah", "wth", "tf"
}

def check_slang_override(text: str) -> bool:
    """Retorna True se o texto contiver gírias explícitas de discordância."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SLANG_DISAGREE_KEYWORDS)

# ---------------------------------------------------------------------------
# Pipeline Principal (ZeroShot + NLI + Bottom-Up + Gephi)
# ---------------------------------------------------------------------------
class ZeroShotDebatePipeline:
    def __init__(
        self,
        deberta_model_name: str = "cross-encoder/nli-deberta-v3-base",
        zero_shot_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        candidate_labels: List[str] = None
    ):
        self.audit = AuditMetrics()
        self.nodes: Dict[str, NodeData] = {}
        self.graph = nx.DiGraph()
        self.device = 0 if torch.cuda.is_available() else -1

        # Candidate labels bidirecionais (Suporte + Ataque)
        self.candidate_labels = candidate_labels or [
            "supporting generative AI, praising AI art, or criticizing anti-AI artists",
            "opposing generative AI, criticizing AI art, or attacking pro-AI users",
            "neutral, off-topic, spam, or purely informational comment"
        ]
        self.label_mapping = {
            self.candidate_labels[0]: "PRO",
            self.candidate_labels[1]: "ANTI",
            self.candidate_labels[2]: "NEUTRAL"
        }

        # Classificador Zero-Shot para Nó
        logger.info(f"Carregando Zero-Shot Classifier ({zero_shot_model_name})...")
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=zero_shot_model_name,
            device=self.device
        )

        # Modelo NLI para Arestas
        logger.info(f"Carregando NLI Cross-Encoder ({deberta_model_name})...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name)
        if torch.cuda.is_available():
            self.nli_model = self.nli_model.to("cuda")
        self.nli_model.eval()

    def load_data(self, file_path: str):
        logger.info(f"Carregando dados de: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.audit.total_raw_records = len(data)

        for entry in data:
            data_type = entry.get("dataType")
            node_id = entry.get("id")

            if not node_id or not data_type:
                self.audit.ignored_records += 1
                continue

            if data_type == "community":
                self.audit.communities_count += 1
                text = f"{entry.get('title', '')} {entry.get('description', '')}".strip()
                self.nodes[node_id] = NodeData(
                    id=node_id,
                    parsed_id=entry.get("parsedId", node_id),
                    data_type=data_type,
                    text_content=text,
                    metadata={"url": entry.get("url")}
                )

            elif data_type == "post":
                self.audit.posts_count += 1
                title = entry.get("title", "")
                body = entry.get("body", "")
                text = f"{title}\n{body}".strip()
                self.nodes[node_id] = NodeData(
                    id=node_id,
                    parsed_id=entry.get("parsedId", node_id),
                    data_type=data_type,
                    text_content=text,
                    author=entry.get("username"),
                    community_name=entry.get("parsedCommunityName") or entry.get("communityName"),
                    metadata={"url": entry.get("url"), "upVotes": entry.get("upVotes")}
                )

            elif data_type == "comment":
                self.audit.comments_count += 1
                body = entry.get("body", "").strip()
                parent_id = entry.get("parentId")
                post_id = entry.get("postId")
                effective_parent = parent_id if parent_id else post_id

                self.nodes[node_id] = NodeData(
                    id=node_id,
                    parsed_id=entry.get("parsedId", node_id),
                    data_type=data_type,
                    text_content=body,
                    author=entry.get("username"),
                    parent_id=effective_parent,
                    post_id=post_id,
                    community_name=entry.get("communityName"),
                    metadata={"depth": entry.get("depth", 0), "upVotes": entry.get("upVotes")}
                )

        logger.info(f"Dados carregados. Total de nós válidos: {len(self.nodes)}")

    def classify_nodes_zeroshot(self, batch_size: int = 32):
        logger.info("Iniciando classificação Zero-Shot dos nós...")
        node_list = list(self.nodes.values())
        prompts = []

        for node in node_list:
            if node.parent_id and node.parent_id in self.nodes:
                parent_text = self.nodes[node.parent_id].text_content
                formatted_input = (
                    f"Target Topic: Generative AI and AI Art Debate\n"
                    f"[CONTEXTO PAI]: {parent_text}\n"
                    f"[RESPOSTA FILHO]: {node.text_content}"
                )
            else:
                formatted_input = (
                    f"Target Topic: Generative AI and AI Art Debate\n"
                    f"[POST/CONTEÚDO]: {node.text_content}"
                )
            prompts.append(formatted_input)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            results = self.zero_shot_classifier(
                batch_prompts,
                candidate_labels=self.candidate_labels,
                multi_label=False
            )

            if isinstance(results, dict):
                results = [results]

            for j, res in enumerate(results):
                node_obj = node_list[i + j]
                top_label = res["labels"][0]
                top_score = res["scores"][0]

                neutral_label = self.candidate_labels[2]
                if top_label == neutral_label and top_score < 0.50:
                    for label, score in zip(res["labels"], res["scores"]):
                        if label != neutral_label:
                            top_label = label
                            top_score = score
                            break

                node_obj.stance_absoluto = self.label_mapping.get(top_label, "NEUTRAL")
                node_obj.stance_score = float(top_score)

        logger.info("Classificação Zero-Shot de nós concluída.")

    def _predict_nli_relation(self, parent_text: str, child_text: str) -> int:
        inputs = self.nli_tokenizer(
            parent_text,
            child_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            label_id = torch.argmax(probs).item()

        if label_id == 0:
            return 1
        elif label_id == 2:
            return -1
        return 0

    def build_graph_edges(self):
        logger.info("Construindo arestas NLI e aplicando pré-filtro de slang...")

        for node_id, node in self.nodes.items():
            self.graph.add_node(
                node_id,
                parsed_id=node.parsed_id,
                data_type=node.data_type,
                author=node.author or "unknown",
                stance_absoluto=node.stance_absoluto,
                stance_score=node.stance_score,
                is_context_only=False,
                text_content=node.text_content
            )

        for node_id, node in self.nodes.items():
            if node.parent_id and node.parent_id in self.nodes:
                parent_node = self.nodes[node.parent_id]

                if check_slang_override(node.text_content):
                    edge_sign = -1
                else:
                    edge_sign = self._predict_nli_relation(
                        parent_text=parent_node.text_content,
                        child_text=node.text_content
                    )

                self.graph.add_edge(
                    node.parent_id,
                    node_id,
                    sign=edge_sign,
                    weight=abs(edge_sign) if edge_sign != 0 else 0.1
                )
            elif node.data_type == "comment":
                self.audit.orphan_comments += 1

    def prune_bottom_up(self):
        logger.info("Executando Poda Bottom-Up e resgate de nós estruturais...")
        self.audit.initial_nodes_count = self.graph.number_of_nodes()

        topological_depth = {}
        for node_id in self.graph.nodes():
            ancestors = nx.ancestors(self.graph, node_id)
            topological_depth[node_id] = len(ancestors)

        nodes_sorted = sorted(
            self.graph.nodes(data=True),
            key=lambda x: topological_depth[x[0]],
            reverse=True
        )

        has_polarized_descendant: Dict[str, bool] = {}
        nodes_to_remove: Set[str] = set()

        for node_id, data in nodes_sorted:
            stance = data.get("stance_absoluto")
            successors = list(self.graph.successors(node_id))

            is_polarized = stance in ["PRO", "ANTI"]
            descendants_polarized = any(
                has_polarized_descendant.get(succ, False) for succ in successors
            )

            is_useful = is_polarized or descendants_polarized
            has_polarized_descendant[node_id] = is_useful

            if stance == "NEUTRAL":
                if descendants_polarized:
                    self.graph.nodes[node_id]["is_context_only"] = True
                    self.nodes[node_id].is_context_only = True
                    self.audit.structural_context_nodes += 1
                else:
                    nodes_to_remove.add(node_id)

        for n_id in nodes_to_remove:
            self.graph.remove_node(n_id)

        self.audit.pruned_nodes_count = len(nodes_to_remove)
        self.audit.retained_nodes_count = self.graph.number_of_nodes()
        self.audit.retention_rate = (
            self.audit.retained_nodes_count / max(1, self.audit.initial_nodes_count)
        ) * 100

        logger.info(
            f"Poda concluída: {self.audit.pruned_nodes_count} nós removidos. "
            f"Retenção final: {self.audit.retention_rate:.2f}% "
            f"({self.audit.structural_context_nodes} nós neutros estruturais salvos)."
        )

    def aggregate_user_stances(self):
        user_stances = {}
        for _, data in self.graph.nodes(data=True):
            author = data.get("author")
            stance = data.get("stance_absoluto")
            if author and author != "unknown" and stance:
                user_stances.setdefault(author, []).append(stance)

        user_mode = {}
        for author, stances in user_stances.items():
            mode_stance = Counter(stances).most_common(1)[0][0]
            user_mode[author] = mode_stance

        for node_id, data in self.graph.nodes(data=True):
            author = data.get("author")
            self.graph.nodes[node_id]["user_stance_mode"] = user_mode.get(author, "NEUTRAL")

    def _compute_stance_metrics(self):
        """Calcula as contagens e porcentagens de Stance no Grafo Retido."""
        counts = {"PRO": 0, "ANTI": 0, "NEUTRAL": 0}
        total = self.graph.number_of_nodes()

        for _, data in self.graph.nodes(data=True):
            stance = data.get("stance_absoluto", "NEUTRAL")
            if stance in counts:
                counts[stance] += 1
            else:
                counts["NEUTRAL"] += 1

        pcts = {
            k: round((v / max(1, total)) * 100, 2)
            for k, v in counts.items()
        }

        self.audit.stance_distribution = {
            "counts": counts,
            "percentages": pcts
        }

    def export_gephi(self, output_gexf_path: str = "debate_graph.gexf", audit_path: str = "audit.json"):
        self.aggregate_user_stances()
        self._compute_stance_metrics()

        for node_id, data in self.graph.nodes(data=True):
            self.graph.nodes[node_id]["include_in_analysis"] = str(not data.get("is_context_only", False))
            self.graph.nodes[node_id]["is_context_only"] = str(data.get("is_context_only", False))

        logger.info(f"Exportando grafo para Gephi em {output_gexf_path}...")
        nx.write_gexf(self.graph, output_gexf_path)

        logger.info(f"Exportando relatório de auditoria em {audit_path}...")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.audit), f, indent=2, ensure_ascii=False)

        logger.info("Pipeline concluído com sucesso!")

# ---------------------------------------------------------------------------
# Execução Principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "BigScrape.json"
    output_gexf = sys.argv[2] if len(sys.argv) > 2 else "debate_graph.gexf"

    pipeline_runner = ZeroShotDebatePipeline()

    pipeline_runner.load_data(input_file)
    pipeline_runner.classify_nodes_zeroshot(batch_size=32)
    pipeline_runner.build_graph_edges()
    pipeline_runner.prune_bottom_up()
    pipeline_runner.export_gephi(
        output_gexf_path=output_gexf,
        audit_path="audit.json"
    )