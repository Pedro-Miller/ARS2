import os
import sys
import json
import re
import time
import logging
import random
import torch
import numpy as np
import networkx as nx
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------------------------------------------------------------------
# 0. Configuração de Ambiente e Reprodutibilidade
# ---------------------------------------------------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

if not OPENROUTER_API_KEY:
    raise ValueError("A variável OPENROUTER_API_KEY não foi encontrada no arquivo .env!")

def set_seed(seed=42):
    """
    Configura sementes globais para garantir que o experimento seja reprodutível.
    Essencial para a validade científica do artigo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------------------------------------------------------
# 1. Estruturas de Dados para Auditoria e Nós
# ---------------------------------------------------------------------------
@dataclass
class AuditMetrics:
    """Armazena as métricas macro da rede para exportação no audit.json."""
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
    """Armazena as informações individuais de cada nó para auditoria e exportação bruta."""
    id: str
    author: str
    text_content: str
    data_type: str
    parent_id: Optional[str] = None
    stance_absoluto: str = "NEUTRAL"
    stance_score: float = 0.0
    method: str = "UNKNOWN" # LLM, NLI, ZERO_SHOT, FILTER
    is_context_only: bool = False

# ---------------------------------------------------------------------------
# 2. Funções Utilitárias e Filtro Determinístico
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove ruídos comuns do Reddit (URLs, menções, markdown) antes do NLP."""
    if not text: return ""
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'[\*\_\>]', '', text)
    return text.strip()

# Lista de gírias que indicam ataque/oposição direta no Reddit
OPPOSITION_SLANG = [
    r"bait", r"baits", r"baiting", r"baited", r"rage\s*bait", 
    r"low\s*iq", r"copium", r"clowns?", r"trolls?", r"trolling", 
    r"trolled", r"delusional", r"schizo", r"1st\s*quartile",
    r"cap", r"crap", r"bs", r"bullshit", r"lmao no", r"cope", 
    r"ratio", r"brainrot", r"fake news", r"nah", r"wth", r"tf"
]
# Termos de negação para evitar falsos positivos (ex: "This is not bait")
NEGATIONS = {"not", "no", "isn't", "isnt", "neither", "never", "não", "nao"}

def check_slang_contradiction(text: str) -> bool:
    """
    Verifica se há gíria de oposição com limite de palavra (\\b) 
    e valida se não há termos de negação nas 3 palavras anteriores.
    """
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
# 3. Pipeline Híbrido Principal
# ---------------------------------------------------------------------------
class HybridPipeline:
    def __init__(self, logger):
        self.logger = logger
        self.audit = AuditMetrics()
        self.nodes: Dict[str, NodeData] = {}
        self.comment_tree = nx.DiGraph() # Grafo temporário para cálculo de níveis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        
        # 1. Modelo NLI Large (Relações)
        nli_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.logger.info(f"Carregando NLI Large ({nli_name})...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name)
        self.nli_model.eval().to(self.device)

        # 2. Modelo Zero-Shot Base (Classificação Independente)
        zs_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.logger.info(f"Carregando Zero-Shot NLI ({zs_name})...")
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
        """
        LLM com Few-Shot e Retry. Usada EXCLUSIVAMENTE para os posts raiz.
        O retry com backoff exponencial protege contra falhas da API do OpenRouter.
        """
        prompt = (
            "Você é um classificador especialista em análise de postura (stance classification) no Reddit (r/aiwars).\n"
            "Analise a atitude do autor em relação ao tema: 'Uso de Inteligência Artificial na Arte'.\n\n"
            "Exemplos de classificação:\n"
            "Texto: 'Meu boneco de palito feito no paint é melhor que isso' -> PRO (Sarcasmo contra anti-IA)\n"
            "Texto: 'Artistas estão sendo roubados por máquinas' -> ANTI (Defesa de direitos autorais)\n"
            "Texto: 'Não sei se isso é arte, o que vocês acham?' -> NEUTRAL\n\n"
            "Categorias permitidas:\n"
            "- PRO: Apoia, defende ou usa sarcasmo/memes para zombar dos críticos de IA ('antis').\n"
            "- ANTI: Rejeita, critica ou usa sarcasmo/memes para zombar de entusiastas de IA ('pro-AI').\n"
            "- NEUTRAL: Informativo, dúvidas, perguntas abertas.\n\n"
            "Instrução: Responda estritamente com UMA única palavra (PRO, ANTI ou NEUTRAL).\n"
            f"Texto: \"{text}\""
        )
        for attempt in range(3):
            try:
                response = self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=OPENROUTER_MODEL, temperature=0.0, max_tokens=10
                )
                result = response.choices[0].message.content.strip().upper()
                if result.startswith("PRO"): return "PRO"
                if result.startswith("ANTI"): return "ANTI"
                if result.startswith("NEUTRAL"): return "NEUTRAL"
                return "NEUTRAL"
            except Exception as e:
                self.logger.warning(f"[ERRO LLM] Tentativa {attempt+1} falhou: {e}")
                time.sleep(2 ** attempt)
        self.logger.error("[ERRO LLM] Falha crítica após 3 tentativas. Retornando NEUTRAL.")
        return "NEUTRAL"

    def zero_shot_stance(self, text: str) -> tuple[str, float]:
        """
        Classifica via Zero-Shot respeitando o threshold científico.
        Se a classificação neutra estiver com score baixo (dúvida), 
        pega o polo mais forte para evitar falsos neutros.
        """
        res = self.zs_classifier(text, candidate_labels=self.zs_labels, multi_label=False)
        top_label = res["labels"][0]
        top_score = res["scores"][0]
        
        # Threshold de 0.65: Se a dúvida for grande, respeita o NEUTRAL
        if top_label == self.zs_labels[2] and top_score < 0.65:
            for label, score in zip(res["labels"], res["scores"]):
                if label != self.zs_labels[2]:
                    top_label, top_score = label, score
                    break
        return self.zs_mapping[top_label], top_score

    def evaluate_nli_batch(self, pairs: list, batch_size: int = 32) -> list:
        """Avalia relações em lote na GPU para máxima performance."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            # Truncamos o pai para proteger o texto do filho (que contém a opinião principal)
            inputs = self.nli_tokenizer(
                [p[0][:1000] for p in batch], [p[1][:1000] for p in batch], 
                truncation=True, padding=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                preds = torch.argmax(self.nli_model(**inputs).logits, dim=1).cpu().tolist()
            for pred in preds:
                label = self.nli_model.config.id2label[pred].lower()
                if "entailment" in label: results.append(1)
                elif "contradiction" in label: results.append(-1)
                else: results.append(0)
        return results

    def load_data(self, filepath: str):
        """Carrega o JSON, limpa o texto e filtra ruídos (deletados, bots, imagens puras)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.audit.total_raw_records = len(data)
        for entry in data:
            if not isinstance(entry, dict) or entry.get("dataType") == "community":
                continue
            
            item_id = entry.get("id")
            author = entry.get("username") or "[deleted]"
            # Filtro de bots e usuários deletados
            if not item_id or author in ["[deleted]", "AutoModerator"]:
                self.audit.ignored_records += 1
                continue
                
            title, body = entry.get("title", ""), entry.get("body", "").strip()
            raw_text = f"{title}\n{body}".strip() if title else body
            text = clean_text(raw_text)
            
            # Filtro de mídia pura (imagens sem texto)
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
        """
        Constrói a árvore BFS e classifica os nós de forma independente.
        Evita o "Efeito Dominó" de erros em cascata.
        """
        self.logger.info("Construindo árvore e calculando níveis (BFS Iterativo)...")
        
        # 1. Construir Grafo de Comentários
        for nid, node in self.nodes.items():
            self.comment_tree.add_node(nid, stance="NEUTRAL")
            if node.parent_id and node.parent_id in self.nodes:
                self.comment_tree.add_edge(node.parent_id, nid)
            elif node.data_type == "comment":
                self.audit.orphan_comments += 1

        # 2. BFS Iterativo (Evita RecursionError em threads profundas)
        levels = defaultdict(list)
        queue = deque()
        for nid, node in self.nodes.items():
            if node.data_type == "post" or not node.parent_id or node.parent_id not in self.nodes:
                levels[0].append(nid)
                queue.append((nid, 0))
                
        while queue:
            curr, lvl = queue.popleft()
            for child in self.comment_tree.successors(curr):
                levels[lvl + 1].append(child)
                queue.append((child, lvl + 1))

        nli_queue = []
        nli_ids = []
        zs_queue = []
        zs_ids = []

        # 3. Classificação Nível por Nível
        for lvl in sorted(levels.keys()):
            for nid in levels[lvl]:
                node = self.nodes[nid]
                
                # LLM EXCLUSIVO NA RAIZ (Posts)
                if node.data_type == "post":
                    stance = self.classify_root_llm(node.text_content)
                    self.comment_tree.nodes[nid]["stance"] = stance
                    self.nodes[nid].stance_absoluto = stance
                    self.nodes[nid].method = "LLM"
                    continue

                # COMENTÁRIOS: Verifica se o pai tem posição clara
                parent_exists = node.parent_id and node.parent_id in self.nodes
                parent_stance = self.nodes[node.parent_id].stance_absoluto if parent_exists else "NEUTRAL"
                parent_text = self.nodes[node.parent_id].text_content if parent_exists else ""

                if not parent_exists or parent_stance == "NEUTRAL":
                    # Se o pai é neutro/órfão, usa Zero-Shot no próprio texto
                    zs_queue.append(node.text_content)
                    zs_ids.append(nid)
                else:
                    # Pai tem posição. Enfileira para o NLI Batch (Relação).
                    nli_queue.append((parent_text, node.text_content))
                    nli_ids.append(nid)

        # 4. Executar Zero-Shot em Lote
        if zs_queue:
            self.logger.info(f"Processando {len(zs_queue)} classificações Zero-Shot...")
            zs_results = self.zs_classifier(zs_queue, candidate_labels=self.zs_labels, multi_label=False, batch_size=32)
            for nid, res in zip(zs_ids, zs_results):
                top_label = res["labels"][0]
                top_score = res["scores"][0]
                if top_label == self.zs_labels[2] and top_score < 0.65:
                    for label, score in zip(res["labels"], res["scores"]):
                        if label != self.zs_labels[2]:
                            top_label, top_score = label, score
                            break
                stance = self.zs_mapping[top_label]
                self.nodes[nid].stance_absoluto = stance
                self.nodes[nid].stance_score = top_score
                self.nodes[nid].method = "ZERO_SHOT"
                self.comment_tree.nodes[nid]["stance"] = stance

        # 5. Executar NLI Batch
        if nli_queue:
            self.logger.info(f"Processando {len(nli_queue)} relações NLI...")
            batch_rels = self.evaluate_nli_batch(nli_queue)
            for nid, rel in zip(nli_ids, batch_rels):
                node = self.nodes[nid]
                parent_stance = self.nodes[node.parent_id].stance_absoluto
                
                # Filtro de Slang força contradição (-1)
                if check_slang_contradiction(node.text_content):
                    rel = -1
                
                if rel == 1:
                    node.stance_absoluto = parent_stance
                elif rel == -1:
                    node.stance_absoluto = "ANTI" if parent_stance == "PRO" else "PRO"
                else:
                    node.stance_absoluto = "NEUTRAL"
                
                node.method = "NLI"
                self.comment_tree.nodes[nid]["stance"] = node.stance_absoluto

    def prune_bottom_up(self):
        """
        Poda Bottom-Up Estrutural.
        Remove nós NEUTRAL que não têm descendentes polarizados.
        Mantém os neutros que servem de "ponte" para discussões PRO/ANTI.
        """
        self.logger.info("Executando Poda Bottom-Up (Estrutural)...")
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

    def export_raw_predictions(self, filepath: str):
        """
        Salva as predições brutas para a seção de Error Analysis do artigo.
        Permite cruzar o método usado (LLM, NLI, ZERO_SHOT) com o texto original.
        """
        records = []
        for nid, node in self.nodes.items():
            records.append({
                "id": nid,
                "author": node.author,
                "data_type": node.data_type,
                "stance": node.stance_absoluto,
                "method": node.method,
                "text": node.text_content
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Predições brutas salvas em: {filepath}")

    def aggregate_users_to_gephi(self, output_gephi: str, output_audit: str):
        """
        Agrega os nós por usuário (Moda) e cria as arestas.
        Exporta o GEXF final com métricas corretas de conflito e sinal.
        """
        self.logger.info("Agregando usuários e gerando GEXF final...")
        
        G = nx.DiGraph()
        user_stances = defaultdict(list)
        interacoes = defaultdict(list)

        for nid in self.comment_tree.nodes():
            node = self.nodes[nid]
            
            # Agrega apenas se não for contexto puro (nós neutros que foram salvos)
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
                
                if soma > 0: sinal = 1
                elif soma < 0: sinal = -1
                else: sinal = 0 
                
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
        
        self.logger.info(f"Sucesso! Grafo: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")

# ---------------------------------------------------------------------------
# 4. Execução Principal e Gestão de Experimentos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Argumentos de linha de comando
    input_file = sys.argv[1] if len(sys.argv) > 1 else "dados.json"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "v3_default"

    # Cria a pasta do experimento (ex: resultados/v3_final_limiar65/)
    output_dir = f"resultados/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configura o Logger para salvar dentro da pasta do experimento
    logger = logging.getLogger("HybridStancePipeline")
    logger.setLevel(logging.INFO)
    
    # Limpa handlers antigos para evitar duplicação de logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(f"{output_dir}/pipeline.log", mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Iniciando experimento: {experiment_name}")
    logger.info(f"Arquivo de entrada: {input_file}")

    # Execução do Pipeline
    pipeline_runner = HybridPipeline(logger)
    pipeline_runner.load_data(input_file)
    pipeline_runner.build_tree_and_classify()
    pipeline_runner.prune_bottom_up()
    
    # Exportação dos Artefatos
    pipeline_runner.export_raw_predictions(f"{output_dir}/predictions_raw.json")
    pipeline_runner.aggregate_users_to_gephi(
        output_gephi=f"{output_dir}/grafo.gexf", 
        output_audit=f"{output_dir}/audit.json"
    )
    
    logger.info(f"Experimento '{experiment_name}' finalizado com sucesso.")