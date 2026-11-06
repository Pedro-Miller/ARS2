import os
import sys
import json
import re
import torch
import networkx as nx
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------------------------------------------------------------------
# 1. Configuração do Ambiente e LLM (OpenRouter via BYOK)
# ---------------------------------------------------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

if not OPENROUTER_API_KEY:
    raise ValueError("A variável OPENROUTER_API_KEY não foi encontrada no arquivo .env!")

llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ---------------------------------------------------------------------------
# 2. Estruturas de Auditoria e Dados
# ---------------------------------------------------------------------------
@dataclass
class AuditMetrics:
    total_raw_records: int = 0
    posts_count: int = 0
    comments_count: int = 0
    ignored_records: int = 0
    orphan_comments: int = 0
    initial_nodes_count: int = 0
    pruned_nodes_count: int = 0
    retained_nodes_count: int = 0
    structural_context_nodes: int = 0
    retention_rate: float = 0.0
    stance_distribution: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeData:
    id: str
    author: str
    text_content: str
    data_type: str
    parent_id: Optional[str] = None
    stance_absoluto: str = "NEUTRAL"
    stance_score: float = 0.0
    is_context_only: bool = False

# ---------------------------------------------------------------------------
# 3. Filtro Determinístico de Slangs (Regex + Janela de Negação)
# ---------------------------------------------------------------------------
OPPOSITION_SLANG = [
    r"bait", r"baits", r"baiting", r"baited", r"rage\s*bait", 
    r"low\s*iq", r"copium", r"clowns?", r"trolls?", r"trolling", 
    r"trolled", r"delusional", r"schizo", r"1st\s*quartile",
    r"cap", r"crap", r"bs", r"bullshit", r"lmao no", r"cope", 
    r"ratio", r"brainrot", r"fake news", r"nah", r"wth", r"tf"
]
NEGATIONS = {"not", "no", "isn't", "isnt", "neither", "never", "não", "nao"}

def check_slang_contradiction(text: str) -> bool:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    for slang in OPPOSITION_SLANG:
        match = re.search(rf"\b{slang}\b", text_lower)
        if match:
            matched_word = match.group(0).split()[0]
            if matched_word in words:
                idx = words.index(matched_word)
                prev_words = words[max(0, idx - 3):idx]
                if any(neg in prev_words for neg in NEGATIONS):
                    return False
            return True
    return False

# ---------------------------------------------------------------------------
# 4. Pipeline Híbrido
# ---------------------------------------------------------------------------
class HybridPipeline:
    def __init__(self):
        self.audit = AuditMetrics()
        self.nodes: Dict[str, NodeData] = {}
        self.comment_tree = nx.DiGraph()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Modelo NLI Large
        nli_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        print(f"Carregando NLI Large ({nli_name})...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name)
        self.nli_model.eval().to(self.device)

        # 2. Zero-Shot NLI Base
        zs_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        print(f"Carregando Zero-Shot NLI ({zs_name})...")
        self.zs_classifier = pipeline(
            "zero-shot-classification", model=zs_name, device=0 if torch.cuda.is_available() else -1
        )
        self.zs_labels = [
            "supporting generative AI, praising AI art, or criticizing anti-AI artists",
            "opposing generative AI, criticizing AI art, or attacking pro-AI users",
            "neutral, off-topic, or purely informational"
        ]
        self.zs_mapping = {self.zs_labels[0]: "PRO", self.zs_labels[1]: "ANTI", self.zs_labels[2]: "NEUTRAL"}

    def classify_root_llm(self, text: str) -> str:
        prompt = (
            "Você é um classificador especialista em análise de postura (stance classification) no Reddit (r/aiwars).\n"
            "Analise a atitude do autor em relação ao tema: 'Uso de Inteligência Artificial na Arte'.\n\n"
            "Categorias permitidas:\n"
            "- PRO: Apoia, defende ou usa sarcasmo/memes para zombar dos críticos de IA ('antis').\n"
            "- ANTI: Rejeita, critica ou usa sarcasmo/memes para zombar de entusiastas de IA ('pro-AI' / 'AI bros').\n"
            "- NEUTRAL: Informativo, dúvidas, perguntas abertas ou sem opinião clara.\n\n"
            "Instrução: Responda estritamente com UMA única palavra (PRO, ANTI ou NEUTRAL).\n"
            f"Texto: \"{text}\""
        )
        try:
            response = llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=OPENROUTER_MODEL, temperature=0.0, max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            if result.startswith("PRO"): return "PRO"
            if result.startswith("ANTI"): return "ANTI"
            return "NEUTRAL"
        except Exception as e:
            print(f"[ERRO LLM] {e}")
            return "NEUTRAL"

    def zero_shot_stance(self, text: str) -> tuple[str, float]:
        res = self.zs_classifier(text, candidate_labels=self.zs_labels, multi_label=False)
        top_label = res["labels"][0]
        top_score = res["scores"][0]
        
        # Se a classificação neutra estiver com score baixo, pega o polo mais forte
        if top_label == self.zs_labels[2] and top_score < 0.55:
            for label, score in zip(res["labels"], res["scores"]):
                if label != self.zs_labels[2]:
                    top_label, top_score = label, score
                    break
        return self.zs_mapping[top_label], top_score

    def evaluate_nli_batch(self, pairs: list, batch_size: int = 32) -> list:
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = self.nli_tokenizer([p[0] for p in batch], [p[1] for p in batch], 
                                        truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                preds = torch.argmax(self.nli_model(**inputs).logits, dim=1).cpu().tolist()
            for pred in preds:
                label = self.nli_model.config.id2label[pred].lower()
                if "entailment" in label: results.append(1)
                elif "contradiction" in label: results.append(-1)
                else: results.append(0)
        return results

    def load_data(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.audit.total_raw_records = len(data)
        for entry in data:
            if not isinstance(entry, dict) or entry.get("dataType") == "community":
                continue
            
            item_id = entry.get("id")
            author = entry.get("username") or "[deleted]"
            if not item_id or author in ["[deleted]", "AutoModerator"]:
                self.audit.ignored_records += 1
                continue
                
            title, body = entry.get("title", ""), entry.get("body", "").strip()
            text = f"{title}\n{body}".strip() if title else body
            if not text:
                self.audit.ignored_records += 1
                continue

            parent_id = entry.get("parentId") or entry.get("postId")
            
            node = NodeData(
                id=item_id, author=author, text_content=text, 
                data_type=entry.get("dataType"), parent_id=parent_id
            )
            self.nodes[item_id] = node
            
            if node.data_type == "post": self.audit.posts_count += 1
            else: self.audit.comments_count += 1

    def build_tree_and_classify(self):
        print("Classificando árvore e construindo nós...")
        
        # 1. Construir Grafo de Comentários Temporário
        for nid, node in self.nodes.items():
            self.comment_tree.add_node(nid, stance="NEUTRAL")
            if node.parent_id and node.parent_id in self.nodes:
                self.comment_tree.add_edge(node.parent_id, nid)
            elif node.data_type == "comment":
                self.audit.orphan_comments += 1

        # 2. Classificação Nível por Nível (Raízes primeiro)
        depth_cache = {}
        def get_level(nid, seen=None):
            if nid in depth_cache: return depth_cache[nid]
            seen = seen or set()
            if nid in seen: return 0
            seen.add(nid)
            node = self.nodes[nid]
            if node.data_type == "post" or not node.parent_id or node.parent_id not in self.nodes:
                depth_cache[nid] = 0
                return 0
            lvl = get_level(node.parent_id, seen) + 1
            depth_cache[nid] = lvl
            return lvl

        levels = defaultdict(list)
        for nid in self.nodes:
            levels[get_level(nid)].append(nid)

        nli_queue = []
        nli_ids = []
        zs_queue = []
        zs_ids = []

        for lvl in sorted(levels.keys()):
            for nid in levels[lvl]:
                node = self.nodes[nid]
                
                # A) LLM EXCLUSIVO NA RAIZ (Posts)
                if node.data_type == "post":
                    stance = self.classify_root_llm(node.text_content)
                    self.comment_tree.nodes[nid]["stance"] = stance
                    self.nodes[nid].stance_absoluto = stance
                    continue

                # B) COMENTÁRIOS
                parent_exists = node.parent_id and node.parent_id in self.nodes
                parent_stance = self.nodes[node.parent_id].stance_absoluto if parent_exists else "NEUTRAL"
                parent_text = self.nodes[node.parent_id].text_content if parent_exists else ""

                if not parent_exists or parent_stance == "NEUTRAL":
                    # Se é órfão ou pai é neutro, usa Zero-Shot NLI no próprio texto
                    zs_queue.append(node.text_content)
                    zs_ids.append(nid)
                else:
                    # Pai tem posição. Enfileira para o NLI Batch (Relação).
                    nli_queue.append((parent_text[:1000], node.text_content))
                    nli_ids.append(nid)

        # 3. Processar Zero-Shot (Neutros e Órfãos)
        if zs_queue:
            print(f"Processando {len(zs_queue)} classificações Zero-Shot...")
            zs_results = self.zs_classifier(zs_queue, candidate_labels=self.zs_labels, multi_label=False, batch_size=32)
            for nid, res in zip(zs_ids, zs_results):
                top_label = res["labels"][0]
                top_score = res["scores"][0]
                if top_label == self.zs_labels[2] and top_score < 0.55:
                    for label, score in zip(res["labels"], res["scores"]):
                        if label != self.zs_labels[2]:
                            top_label, top_score = label, score
                            break
                stance = self.zs_mapping[top_label]
                self.nodes[nid].stance_absoluto = stance
                self.nodes[nid].stance_score = top_score
                self.comment_tree.nodes[nid]["stance"] = stance

        # 4. Processar NLI (Relações com Pais Polarizados)
        if nli_queue:
            print(f"Processando {len(nli_queue)} relações NLI...")
            batch_rels = self.evaluate_nli_batch(nli_queue)
            for nid, rel in zip(nli_ids, batch_rels):
                node = self.nodes[nid]
                parent_stance = self.nodes[node.parent_id].stance_absoluto
                
                # Filtro de Slang força contradição
                if check_slang_contradiction(node.text_content):
                    rel = -1
                
                if rel == 1:
                    node.stance_absoluto = parent_stance
                elif rel == -1:
                    node.stance_absoluto = "ANTI" if parent_stance == "PRO" else "PRO"
                else:
                    node.stance_absoluto = "NEUTRAL"
                
                self.comment_tree.nodes[nid]["stance"] = node.stance_absoluto

    def prune_bottom_up(self):
        print("Executando Poda Bottom-Up (Estrutural)...")
        self.audit.initial_nodes_count = self.comment_tree.number_of_nodes()
        
        # Ordenação Topológica Reversa (Folhas para Raízes)
        topo_order = list(nx.topological_sort(self.comment_tree))
        nodes_sorted = reversed(topo_order)

        has_polarized_descendant = {}
        nodes_to_remove = set()

        for nid in nodes_sorted:
            node = self.nodes[nid]
            successors = list(self.comment_tree.successors(nid))
            
            is_polarized = node.stance_absoluto in ["PRO", "ANTI"]
            desc_polarized = any(has_polarized_descendant.get(s, False) for s in successors)
            
            is_useful = is_polarized or desc_polarized
            has_polarized_descendant[nid] = is_useful

            if node.stance_absoluto == "NEUTRAL":
                if desc_polarized:
                    node.is_context_only = True
                    self.audit.structural_context_nodes += 1
                else:
                    nodes_to_remove.add(nid)

        for nid in nodes_to_remove:
            self.comment_tree.remove_node(nid)

        self.audit.pruned_nodes_count = len(nodes_to_remove)
        self.audit.retained_nodes_count = self.comment_tree.number_of_nodes()
        self.audit.retention_rate = (self.audit.retained_nodes_count / max(1, self.audit.initial_nodes_count)) * 100

    def aggregate_users_to_gephi(self, output_gephi: str, output_audit: str):
        print("Agregando usuários e gerando GEXF final...")
        
        G = nx.DiGraph()
        user_stances = defaultdict(list)
        interacoes = defaultdict(list)

        for nid in self.comment_tree.nodes():
            node = self.nodes[nid]
            
            # Agrega apenas se não for contexto puro
            if not node.is_context_only and node.stance_absoluto != "NEUTRAL":
                user_stances[node.author].append(node.stance_absoluto)

            # Adiciona aresta de interação
            if node.parent_id and node.parent_id in self.comment_tree.nodes():
                parent_author = self.nodes[node.parent_id].author
                if parent_author != node.author and parent_author not in ["[deleted]", "AutoModerator"]:
                    # Lógica de Sinal de Interação
                    if node.stance_absoluto == "NEUTRAL" or self.nodes[node.parent_id].stance_absoluto == "NEUTRAL":
                        rel = 0
                    elif node.stance_absoluto == self.nodes[node.parent_id].stance_absoluto:
                        rel = 1
                    else:
                        rel = -1
                    
                    interacoes[(node.author, parent_author)].append(rel)

        # Nós (Aggregation por Moda)
        for author, stances in user_stances.items():
            mode_stance = Counter(stances).most_common(1)[0][0]
            pro = stances.count("PRO") / len(stances)
            anti = stances.count("ANTI") / len(stances)
            G.add_node(author, stance=mode_stance, total=len(stances), pro_ratio=round(pro, 2), anti_ratio=round(anti, 2))

        # Arestas (Correção de conflito e sinal neutro)
        for (u1, u2), rels in interacoes.items():
            if u1 in G.nodes and u2 in G.nodes:
                peso = len(rels)
                soma = sum(rels)
                
                if soma > 0:
                    sinal = 1
                elif soma < 0:
                    sinal = -1
                else:
                    sinal = 0 
                
                conflict = round(abs(soma) / peso, 2) if soma != 0 else 0.0
                
                G.add_edge(u1, u2, weight=peso, sign=sinal, conflict_intensity=conflict)

        # Estatísticas Finais
        counts = {"PRO": 0, "ANTI": 0, "NEUTRAL": 0}
        for _, data in G.nodes(data=True):
            counts[data["stance"]] += 1
        self.audit.stance_distribution = {"counts": counts, "percentages": {k: round((v/max(1, G.number_of_nodes()))*100, 2) for k, v in counts.items()}}

        nx.write_gexf(G, output_gephi)
        with open(output_audit, "w", encoding="utf-8") as f:
            json.dump(asdict(self.audit), f, indent=2, ensure_ascii=False)
        
        print(f"Sucesso! Grafo: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")

# ---------------------------------------------------------------------------
# Execução Principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "dados.json"
    output_gexf = sys.argv[2] if len(sys.argv) > 2 else "grafo_final.gexf"

    pipeline_runner = HybridPipeline()
    pipeline_runner.load_data(input_file)
    pipeline_runner.build_tree_and_classify()
    pipeline_runner.prune_bottom_up()
    pipeline_runner.aggregate_users_to_gephi(output_gexf, "audit.json")