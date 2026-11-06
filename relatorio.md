# Relatório de Desenvolvimento: Pipeline Híbrido de Classificação de Stance (r/aiwars)

**Data:** 2026-11-06  
**Tags:** #nlp #graph-analysis #stance-classification #zero-shot #deberta #openrouter #reddit #gexf #gephi  
**Status:** 🟢 100% Funcional (Classificação Independente + Poda Bottom-Up + Correção de Arestas)  
**Acurácia Estimada:** ~80% (Amostragem Manual)

---

## 1. Visão Geral da Arquitetura Atualizada

O pipeline evoluiu de uma abordagem de **Propagação em Cascata** (onde um erro no topo invertia toda a árvore) para uma arquitetura **Híbrida e Independente**. O objetivo continua sendo mapear a polarização no `r/aiwars` e exportar um grafo assinado (`.gexf`) para o Gephi.

### Princípios de Design

* **LLM Restrita à Raiz:** A LLM (Gemma 4 via OpenRouter) atua **exclusivamente** na classificação de posts raiz (`dataType == "post"`). Isso elimina o custo de milhares de chamadas de API e garante que o "âncora" do debate seja preciso.
* **Classificação Independente (Zero-Shot):** Se um comentário responde a um pai `NEUTRAL` ou é um órfão, ele não herda o neutro. Ele é reclassificado do zero via Zero-Shot NLI, lendo apenas o próprio texto.
* **NLI Large para Relações:** Se o pai tem posição (`PRO`/`ANTI`), o comentário é avaliado pelo DeBERTa-large para descobrir se ele concorda (+1) ou contradiz (-1).
* **Poda Bottom-Up Estrutural:** O grafo é podado das folhas para as raízes. Nós `NEUTRAL` são removidos, **a menos que** tenham descendentes polarizados. Nesse caso, são salvos como `is_context_only=True` (pontes estruturais), preservando a topologia.
* **Matemática Correta de Arestas:** O cálculo de `conflict_intensity` e `sign` agora lida corretamente com interações neutras (sinal 0, conflito 0.0).

---

## 2. Histórico de Correções e Evolução

### 2.1. Fim do Efeito Dominó (Falsos Anti/Pro em Cascata)
* **O que deu errado:** No modelo anterior, se o NLI errasse a relação de um pai, o filho era invertido, e o neto herdava o erro. Isso gerou falsos positivos (ex: posts PRO classificados como ANTI).
* **Solução:** Classificação independente com Zero-Shot. Se o pai é NEUTRAL, o filho é classificado por conta própria. Se o pai é PRO/ANTI, o NLI decide a relação, mas sem inverter cegamente (o sinal é calculado pela lógica de concordância).

### 2.2. Poda Bottom-Up (Preservação Topológica)
* **O que deu errado:** A poda BFS tradicional cortava o neutro e todos os seus filhos, destruindo 60% do grafo e perdendo discussões valiosas abaixo de comentários informativos.
* **Solução:** Implementação da função `prune_bottom_up`. Ela verifica se o nó neutro tem filhos polarizados. Se tiver, ele é mantido como `context_only`. Isso aumentou a retenção para ~40% e preservou 17 nós estruturais.

### 2.3. Correção do Bug de Arestas no GEXF
* **O que deu errado:** Arestas com `rel = 0` (neutras) estavam gerando `conflict_intensity="1.0"` (conflito máximo) e `sign="1"` (concordância), o que distorcia as métricas no Gephi.
* **Solução:** A lógica foi ajustada para:
  * `soma > 0` → `sign=1` (Concordância)
  * `soma < 0` → `sign=-1` (Contradição)
  * `soma == 0` → `sign=0` (Neutro) e `conflict_intensity=0.0`
  * `conflict_intensity` agora é `abs(soma) / peso`.

### 2.4. Upgrade de Modelos e Filtro de Slang
* O NLI foi mantido no modelo **Large** para máxima precisão.
* O Zero-Shot passou a usar batching (32) para não estourar a GPU.
* O filtro de slang (Regex + Janela de Negação) foi mantido e ampliado para evitar falsos positivos como "inhabit" ativando "bait".

---

## 3. Resultados e Métricas (Audit)

A execução no dataset de 240 registros apresentou os seguintes resultados:

| Métrica | Valor |
| :--- | :--- |
| **Total de Registros Brutos** | 240 |
| **Posts (Raízes)** | 39 |
| **Comentários** | 183 |
| **Nós Iniciais (Grafo Bruto)** | 222 |
| **Nós Podados** | 133 |
| **Nós Retidos** | 89 |
| **Nós de Contexto (Pontes)** | 17 |
| **Taxa de Retenção** | ~40.1% |

### Distribuição de Stance (Grafo Final Agregado)
*   **PRO:** 24 nós (~38.1%)
*   **ANTI:** 39 nós (~61.9%)
*   **NEUTRAL:** 0 nós (todos foram podados ou viraram contexto)

### Análise de Acurácia
A acurácia subiu de ~50% (modelo anterior) para **~80%**. Os erros restantes (20%) são majoritariamente:
1. Piadas internas com contexto externo (ex: discussões sobre Star Trek).
2. Sarcasmo puro em respostas curtas.
3. Comentários neutros em threads off-topic que são forçados a PRO/ANTI pelo Zero-Shot.

---

## 4. Próximos Passos (Visualização no Gephi)

1. **Particionamento (Partition):** Colorir nós por `stance` (PRO / ANTI).
2. **Aparência (Ranking):** Dimensionar nós por `total` (número de comentários).
3. **Cor das Arestas:** Colorir por `sign` (1=Verde, -1=Vermelho, 0=Cinza).
4. **Espessura das Arestas:** Escalar por `conflict_intensity`.
5. **Layout:** Aplicar ForceAtlas 2 para visualizar as trincheiras ideológicas.

---

## 5. Configuração de Ambiente

**requirements.txt atualizado:**
    networkx>=3.0
    transformers>=4.37.0
    torch>=2.0.0
    tqdm>=4.66.1
    scipy>=1.11.0
    openai>=1.0.0
    python-dotenv>=1.0.0
    accelerate>=0.20.0
    sentencepiece>=0.1.99

**Execução:**
    source .ARS/bin/activate
    python ZeroShot.py BigScrape.json grafo_big1.gexf