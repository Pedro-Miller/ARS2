import os
import sys
import json
import re
import torch
import networkx as nx
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. Configuração do Ambiente e LLM (OpenRouter via BYOK)
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

if not OPENROUTER_API_KEY:
    raise ValueError("A variável OPENROUTER_API_KEY não foi encontrada no arquivo .env!")

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

print(f"-> Provedor LLM selecionado: OpenRouter ({OPENROUTER_MODEL})")

# ---------------------------------------------------------------------------
# 2. Carregamento do Modelo NLI
# ---------------------------------------------------------------------------
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    print("\n[AVISO] Nenhuma GPU (CUDA) foi detectada pelo PyTorch. Executando na CPU...")

print(f"Carregando modelo NLI local ({NLI_MODEL_NAME}) no dispositivo: {device.type.upper()}...")
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
model.eval()
model.to(device)

# ---------------------------------------------------------------------------
# 3. Filtro Determinístico de Slangs (Regex)
# ---------------------------------------------------------------------------
OPPOSITION_SLANG = [
    r"bait", r"baits", r"baiting", r"baited",
    r"rage\s*bait", r"low\s*iq", r"copium",
    r"clowns?", r"trolls?", r"trolling", r"trolled",
    r"delusional", r"schizo", r"1st\s*quartile"
]

def check_slang_contradiction(text: str) -> bool:
    text_lower = text.lower()
    for slang in OPPOSITION_SLANG:
        if re.search(rf"\b{slang}\b", text_lower):
            return True
    return False

# ---------------------------------------------------------------------------
# 4. Classificação LLM e Inferência em Lote NLI
# ---------------------------------------------------------------------------
def classify_root_stance(text: str) -> str:
    """Classifica a postura do post raiz via LLM (Uso exclusivo para raízes)."""
    prompt = (
        "Você é um classificador especialista em análise de postura (stance classification) no Reddit (r/aiwars).\n"
        "Analise a atitude do autor em relação ao tema: 'Uso de Inteligência Artificial na Arte'.\n\n"
        "Categorias permitidas:\n"
        "- PRO: Apoia, defende ou usa sarcasmo/memes para zombar dos críticos de IA ('antis').\n"
        "- ANTI: Rejeita, critica ou usa sarcasmo/memes para zombar de entusiastas de IA ('pro-AI' / 'AI bros').\n"
        "- NEUTRAL: Informativo, dúvidas, perguntas abertas ou sem opinião clara.\n\n"
        "Atenção a Sarcasmo/Ironia: Frases debochadas como 'meu boneco de palito é melhor' ou 'antis estão obcecados' são PRO-IA.\n\n"
        "Instrução: Responda estritamente com UMA única palavra (PRO, ANTI ou NEUTRAL).\n\n"
        f"Texto: \"{text}\""
    )
    
    try:
        response = llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=OPENROUTER_MODEL,
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[ERRO LLM] Falha ao chamar OpenRouter: {e}")
        return "NEUTRAL"

    if result.startswith("PRO"):
        return "PRO"
    if result.startswith("ANTI"):
        return "ANTI"
    if result.startswith("NEUTRAL"):
        return "NEUTRAL"

    print(f"[AVISO LLM] Resposta não-conforme: {result!r} — classificando como NEUTRAL")
    return "NEUTRAL"

def evaluate_nli_batch(pairs: list, batch_size: int = 32) -> list:
    results = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        premises = [p[0] for p in batch_pairs]
        hypotheses = [p[1] for p in batch_pairs]
        
        inputs = tokenizer(
            premises, hypotheses,
            truncation=True, padding=True, max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            
        id2label = model.config.id2label
        for pred in predictions:
            label = id2label[pred].lower()
            if "entailment" in label:
                results.append(1)
            elif "contradiction" in label:
                results.append(-1)
            else:
                results.append(0)
    return results

# ---------------------------------------------------------------------------
# 5. Processamento BFS, Poda e Construção do Grafo
# ---------------------------------------------------------------------------
def build_stance_graph(json_file_path: str, output_gexf_path: str):
    with open(json_file_path, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    
    # Indexação dos itens
    items_by_id = {}
    for item in posts_data:
        if isinstance(item, dict) and item.get("dataType") != "community":
            iid = item.get("id")
            if iid:
                items_by_id[iid] = item
    
    # Cálculo de profundidade real
    depth_cache = {}
    def get_true_level(item_id, visiting=None):
        if item_id in depth_cache:
            return depth_cache[item_id]
        if visiting is None:
            visiting = set()
        if item_id in visiting:
            depth_cache[item_id] = 0
            return 0
        visiting.add(item_id)
        
        item = items_by_id.get(item_id)
        if not item:
            depth_cache[item_id] = 0
            return 0
        if item.get("dataType") == "post":
            depth_cache[item_id] = 0
            return 0
        
        parent_id = item.get("parentId")
        if not parent_id or parent_id not in items_by_id:
            depth_cache[item_id] = 1
            return 1
        
        level = get_true_level(parent_id, visiting) + 1
        depth_cache[item_id] = level
        return level
    
    # Agrupamento por nível
    items_by_level = defaultdict(list)
    for item_id in items_by_id:
        items_by_level[get_true_level(item_id)].append(items_by_id[item_id])
    
    print(f"Processando {len(items_by_level)} níveis de profundidade...")
    
    # Estruturas de dados
    nodes_stance = {}
    node_text_map = {}
    node_author_map = {}
    user_stance_counts = defaultdict(lambda: {"PRO": 0, "ANTI": 0, "NEUTRAL": 0})
    interacoes = defaultdict(list)
    pruned_items = set()
    
    # Processamento Nível por Nível (BFS)
    for level in sorted(items_by_level.keys()):
        level_items = items_by_level[level]
        nli_pairs = []
        nli_item_data = []
        n_pruned_this_level = 0
        
        for item in level_items:
            item_id = item.get("id")
            author = item.get("username") or "[deleted]"
            
            title = item.get("title", "")
            body = item.get("body", "").strip()
            text = f"{title}\n{body}".strip() if title else body
            
            # Filtro de mídia pura / moderadores
            if not item_id or author in ["[deleted]", "AutoModerator"] or not text:
                if item_id:
                    pruned_items.add(item_id)
                    n_pruned_this_level += 1
                continue
            
            parent_id = item.get("parentId") or item.get("postId")
            node_text_map[item_id] = text
            node_author_map[item_id] = author
            
            # Post Raiz (LLM - Única exceção permitida)
            if item.get("dataType") == "post":
                stance = classify_root_stance(text)
                nodes_stance[item_id] = stance
                
                # Poda de Raiz Neutra (Moved up to prevent counting)
                if stance == "NEUTRAL":
                    pruned_items.add(item_id)
                    n_pruned_this_level += 1
                    continue
                
                # Contabiliza o voto apenas se a raiz tiver posição
                user_stance_counts[author][stance] += 1
                continue
            
            # Poda de Órfãos (Pai não tem texto ou foi descartado)
            parent_text = node_text_map.get(parent_id, "")
            if not parent_text or parent_id in pruned_items:
                pruned_items.add(item_id)
                n_pruned_this_level += 1
                continue
                
            # Cinto de segurança: Pai não classificado
            if parent_id not in nodes_stance:
                print(f"[AVISO] Pai {parent_id} não classificado — pulando filho {item_id}")
                pruned_items.add(item_id)
                n_pruned_this_level += 1
                continue
            
            # Pronto para NLI
            nli_pairs.append((parent_text[:1000], text))
            nli_item_data.append({
                "item_id": item_id,
                "author": author,
                "parent_id": parent_id,
                "text": text
            })
        
        # Processa NLI do nível atual
        if nli_pairs:
            batch_relations = evaluate_nli_batch(nli_pairs, batch_size=32)
            
            for item_data, batch_rel in zip(nli_item_data, batch_relations):
                item_id = item_data["item_id"]
                parent_id = item_data["parent_id"]
                author = item_data["author"]
                
                parent_stance = nodes_stance[parent_id]
                relation = -1 if check_slang_contradiction(item_data["text"]) else batch_rel
                
                if relation == 1:
                    new_stance = parent_stance
                elif relation == -1:
                    new_stance = "ANTI" if parent_stance == "PRO" else "PRO"
                else:
                    new_stance = "NEUTRAL"
                
                nodes_stance[item_id] = new_stance
                
                # Poda de Neutros
                if new_stance == "NEUTRAL":
                    pruned_items.add(item_id)
                    n_pruned_this_level += 1
                    continue
                
                # Acumulando votos e interações
                user_stance_counts[author][new_stance] += 1
                
                parent_author = node_author_map.get(parent_id)
                if parent_author and parent_author not in ["[deleted]", "AutoModerator"] and parent_author != author:
                    interacoes[(author, parent_author)].append(relation)
        
        print(f"  Nível {level}: {len(level_items)} itens | {n_pruned_this_level} podados | {len(nli_pairs)} via NLI")
    
    # -----------------------------------------------------------------------
    # Construção final do grafo
    # -----------------------------------------------------------------------
    print("\nAgregando stances de usuários e construindo o grafo final...")
    
    G = nx.DiGraph()
    
    # Nós (Moda do stance)
    for user, counts in user_stance_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        final_stance = max(counts, key=counts.get)
        G.add_node(
            user,
            stance=final_stance,
            total_comments=total,
            pro_ratio=round(counts["PRO"] / total, 2),
            anti_ratio=round(counts["ANTI"] / total, 2),
            neutral_ratio=round(counts["NEUTRAL"] / total, 2)
        )
    
    # Arestas (Acúmulo + conflict_intensity)
    for (u1, u2), relacoes in interacoes.items():
        if u1 not in G.nodes or u2 not in G.nodes:
            continue
        peso = len(relacoes)
        soma = sum(relacoes)
        sinal = 1 if soma >= 0 else -1
        G.add_edge(
            u1, u2,
            weight=peso,
            sign=sinal,
            conflict_intensity=round(abs(soma) / peso, 2)
        )
    
    print(f"Exportando grafo com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas...")
    nx.write_gexf(G, output_gexf_path)
    print(f"Sucesso! Arquivo gerado em: {output_gexf_path}")

# ---------------------------------------------------------------------------
# 6. Ponto de Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_json = sys.argv[1] if len(sys.argv) > 1 else "dados.json"
    output_gexf = sys.argv[2] if len(sys.argv) > 2 else "grafo_teste.gexf"
    
    if os.path.exists(input_json):
        build_stance_graph(input_json, output_gexf)
    else:
        print(f"Erro: Arquivo de entrada '{input_json}' não foi encontrado.")