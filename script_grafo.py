import os
import sys
import json
import networkx as nx
import torch
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
# 2. Carregamento do Modelo NLI de Alta Precisão (Large)
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
# 3. Regra Determinística de Slangs & Provocações (Solução 2)
# ---------------------------------------------------------------------------
OPPOSITION_SLANG = [
    "bait", "rage bait", "low iq", "copium", "clown", 
    "troll", "delusional", "schizo", "1st quartile"
]

def check_slang_contradiction(text: str) -> bool:
    """Retorna True se o comentário for uma provocação/ataque direto em fórum."""
    text_lower = text.lower()
    return any(slang in text_lower for slang in OPPOSITION_SLANG)


# ---------------------------------------------------------------------------
# 4. Classificação LLM e Inferência em Lote NLI (GPU)
# ---------------------------------------------------------------------------
def classify_root_stance(text: str) -> str:
    """Classifica a postura do post/comentário raiz via LLM."""
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
            max_tokens=10,  # Previne erro 402 no OpenRouter
        )
        result = response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[ERRO LLM] Falha ao chamar OpenRouter: {e}")
        return "NEUTRAL"

    if "PRO" in result:
        return "PRO"
    elif "ANTI" in result:
        return "ANTI"
    return "NEUTRAL"


def evaluate_nli_batch(pairs: list, batch_size: int = 32) -> list:
    """
    Avalia uma lista de pares [(parent_text, reply_text), ...] em lotes na GPU.
    Retorna uma lista de inteiros (+1 para Entailment, -1 para Contradiction, 0 para Neutral).
    """
    results = []
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        premises = [p[0] for p in batch_pairs]
        hypotheses = [p[1] for p in batch_pairs]
        
        inputs = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=512,
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
# 5. Processamento da Árvore e Construção do Grafo (.gexf)
# ---------------------------------------------------------------------------
def build_stance_graph(json_file_path: str, output_gexf_path: str):
    with open(json_file_path, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
        
    G = nx.DiGraph()
    nodes_stance = {}     # item_id -> Stance
    node_text_map = {}    # item_id -> Texto
    node_author_map = {}  # item_id -> Autor (username)
    
    print("Processando árvore de comentários e separando nós...")
    
    def get_sort_key(x):
        if not isinstance(x, dict):
            return (99, 99)
        dt = x.get("dataType")
        depth = x.get("depth") if x.get("depth") is not None else 0
        if dt == "post":
            return (0, 0)
        return (1, depth)

    items = sorted(posts_data, key=get_sort_key)
    nli_queue = []

    for item in items:
        if not isinstance(item, dict) or item.get("dataType") == "community":
            continue

        item_id = item.get("id")
        author = item.get("username") or "[deleted]"
        
        title = item.get("title", "")
        body = item.get("body", "").strip()
        text = f"{title}\n{body}".strip() if title else body

        # Filtra moderadores, deletados e comentários sem texto (imagens puras)
        if not item_id or author in ["[deleted]", "AutoModerator"] or not text:
            continue

        depth = item.get("depth")
        parent_id = item.get("parentId")
        post_id = item.get("postId")

        if not parent_id and post_id:
            parent_id = post_id

        node_text_map[item_id] = text
        node_author_map[item_id] = author

        is_root = (item.get("dataType") == "post") or (depth == 0 and not parent_id)
        
        if is_root:
            stance = classify_root_stance(text)
            nodes_stance[item_id] = stance
            if not G.has_node(author):
                G.add_node(author, stance=stance)
        else:
            parent_text = node_text_map.get(parent_id, "")
            parent_author = node_author_map.get(parent_id)
            
            if parent_text:
                nli_queue.append({
                    "item_id": item_id,
                    "author": author,
                    "parent_author": parent_author,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "text": text
                })

    # Inferência em lote NLI + Aplicação da Solução 2 (Pre-filtro de Slangs)
    if nli_queue:
        pairs = [(item["parent_text"], item["text"]) for item in nli_queue]
        print(f"Processando {len(pairs)} relações NLI em lote na GPU (Batching)...")
        batch_relations = evaluate_nli_batch(pairs, batch_size=32)

        for item_data, batch_rel in zip(nli_queue, batch_relations):
            item_id = item_data["item_id"]
            author = item_data["author"]
            parent_author = item_data["parent_author"]
            parent_stance = nodes_stance.get(item_data["parent_id"], "NEUTRAL")

            # Aplicação da Solução 2: Se contiver gíria de ataque, força contradição (-1)
            if check_slang_contradiction(item_data["text"]):
                relation = -1
            else:
                relation = batch_rel

            # Propagação do Stance
            if relation == 1:
                nodes_stance[item_id] = parent_stance
            elif relation == -1:
                nodes_stance[item_id] = "ANTI" if parent_stance == "PRO" else ("PRO" if parent_stance == "ANTI" else "NEUTRAL")
            else:
                nodes_stance[item_id] = "NEUTRAL"

            if not G.has_node(author):
                G.add_node(author, stance=nodes_stance[item_id])

            if parent_author and parent_author not in ["[deleted]", "AutoModerator"] and parent_author != author:
                G.add_edge(author, parent_author, weight=relation, sign=relation)

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