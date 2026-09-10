import os
import sys
import json
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. Configuração do Provedor LLM (Groq vs Gemini)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_PROVIDER = None

if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    LLM_PROVIDER = "GROQ"
    print("-> Provedor LLM selecionado: Groq (llama-3.3-70b-versatile)")
elif GEMINI_API_KEY:
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    LLM_PROVIDER = "GEMINI"
    print("-> Provedor LLM selecionado: Google AI Studio (gemini-2.5-flash)")
else:
    raise ValueError("Nenhuma chave de API encontrada. Defina GROQ_API_KEY ou GEMINI_API_KEY no terminal.")


# ---------------------------------------------------------------------------
# 2. Carregamento do Modelo NLI (DeBERTa-v3)
# ---------------------------------------------------------------------------
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
print(f"Carregando modelo NLI local: {NLI_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    print("\n[AVISO] Nenhuma GPU (CUDA) foi detectada pelo PyTorch.")
    print("[AVISO] O modelo DeBERTa-v3 será executado na CPU, o que pode ser significativamente lento para bases grandes.")
    
    # Se estiver rodando interativamente no terminal, pede confirmação
    if sys.stdin.isatty():
        resposta = input("Deseja continuar executando na CPU mesmo assim? [s/N]: ").strip().lower()
        if resposta not in ["s", "sim", "y", "yes"]:
            print("Execução interrompida pelo usuário.")
            sys.exit(0)
    else:
        print("Continuando execução em modo não-interativo...")

print(f"\nCarregando modelo NLI local ({NLI_MODEL_NAME}) no dispositivo: {device.type.upper()}...")
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
model.eval()
model.to(device)




# ---------------------------------------------------------------------------
# 3. Funções de Incerteza e Classificação
# ---------------------------------------------------------------------------
def classify_root_stance(text: str) -> str:
    """Classifica a postura do post/comentário raiz via Groq ou Gemini."""
    prompt = (
        "Você é um classificador especialista em análise de postura (stance classification).\n"
        "Analise a atitude ou intenção do autor do texto abaixo em relação ao tema: 'Uso de Inteligência Artificial na Arte'.\n\n"
        "Categorias permitidas:\n"
        "- PRO: O autor apoia, incentiva ou defende o uso de IA na arte.\n"
        "- ANTI: O autor critica, rejeita ou se opõe ao uso de IA na arte.\n"
        "- NEUTRAL: O texto é meramente informativo, ambíguo, traz dúvidas ou não expressa opinião clara sobre o tema.\n\n"
        "Instrução: Responda estritamente com UMA única palavra (PRO, ANTI ou NEUTRAL).\n\n"
        f"Texto: \"{text}\""
    )
    
    result = ""
    
    if LLM_PROVIDER == "GROQ":
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().upper()
        
    elif LLM_PROVIDER == "GEMINI":
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        result = response.text.strip().upper()

    if "PRO" in result:
        return "PRO"
    elif "ANTI" in result:
        return "ANTI"
    return "NEUTRAL"


def evaluate_nli_relation(parent_text: str, reply_text: str) -> int:
    """
    Avalia a relação de NLI entre o texto pai (premise) e a resposta (hypothesis).
    Retorna:
     +1 -> Concordância / Suporte
     -1 -> Discordância / Conflito
      0 -> Neutro
    """
    inputs = tokenizer(
        parent_text,
        reply_text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        prediction = torch.argmax(probabilities).item()
        
    id2label = model.config.id2label
    predicted_label = id2label[prediction].lower()
    
    if "entailment" in predicted_label:
        return 1
    elif "contradiction" in predicted_label:
        return -1
    else:
        return 0


# ---------------------------------------------------------------------------
# 4. Construção do Grafo e Exportação (.gexf)
# ---------------------------------------------------------------------------
def build_stance_graph(json_file_path: str, output_gexf_path: str):
    with open(json_file_path, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
        
    G = nx.DiGraph()
    nodes_stance = {}   # Mapeia ID do item -> Stance
    node_text_map = {}  # Mapeia ID do item -> Texto
    
    print("Processando árvore de comentários e propagando stances...")
    
    # Ordenar por profundidade (depth) para garantir processamento da raiz até as folhas
    items = sorted(posts_data, key=lambda x: x.get("depth", 0))
    
    for item in items:
        item_id = item.get("id") or item.get("postId")
        author = item.get("author") or item.get("authorName", "[deleted]")
        text = item.get("body") or item.get("text") or item.get("title", "")
        parent_id = item.get("parentId")
        depth = item.get("depth", 0)
        
        if not item_id or author == "[deleted]":
            continue
            
        node_text_map[item_id] = text
        
        # Ancoragem Raiz (Depth 0) via LLM
        if depth == 0 or not parent_id:
            stance = classify_root_stance(text)
            nodes_stance[item_id] = stance
        else:
            # Avalia NLI em relação ao nó pai
            parent_text = node_text_map.get(parent_id, "")
            parent_stance = nodes_stance.get(parent_id, "NEUTRAL")
            
            if parent_text:
                relation = evaluate_nli_relation(parent_text, text)
                
                # Regra de Propagação do Stance
                if relation == 1:
                    nodes_stance[item_id] = parent_stance
                elif relation == -1:
                    nodes_stance[item_id] = "ANTI" if parent_stance == "PRO" else ("PRO" if parent_stance == "ANTI" else "NEUTRAL")
                else:
                    nodes_stance[item_id] = "NEUTRAL"
                
                # Gravação da Aresta no Grafo de Usuários
                parent_author = item.get("parentAuthor")
                
                if not G.has_node(author):
                    G.add_node(author, stance=nodes_stance[item_id])
                
                if parent_author and parent_author != "[deleted]" and parent_author != author:
                    G.add_edge(author, parent_author, weight=relation, sign=relation)

    print(f"Exportando grafo com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas...")
    nx.write_gexf(G, output_gexf_path)
    print(f"Sucesso! Arquivo gerado em: {output_gexf_path}")


# ---------------------------------------------------------------------------
# 5. Execução Principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    argc = len(sys.argv)
    
    # Valores padrão
    input_json = "dados.json"
    output_gexf = "rede_usuarios_assinada.gexf"
    
    # Lógica de argumentos por posição
    if argc > 1:
        input_json = sys.argv[1]
    if argc > 2:
        output_gexf = sys.argv[2]
        
    if os.path.exists(input_json):
        build_stance_graph(input_json, output_gexf)
    else:
        print(f"Erro: O arquivo de entrada '{input_json}' não foi localizado.")
        print(f"Uso: python3 {sys.argv[0]} [arquivo_entrada.json] [arquivo_saida.gexf]")