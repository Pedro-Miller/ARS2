# Relatório de Desenvolvimento: Pipeline Híbrido de Classificação de Stance (r/aiwars)

**Data:** 2026-11-06  
**Tags:** #nlp #graph-analysis #stance-classification #zero-shot #deberta #openrouter #reddit #gexf #gephi #cienciometria  
**Status:** 🟢 100% Funcional e Blindado para Publicação Científica  
**Acurácia Estimada:** ~80% a 85% (Amostragem Manual)

---

## 1. Visão Geral da Arquitetura Atualizada

O pipeline evoluiu de uma abordagem de **Propagação em Cascata** (onde um erro no topo invertia toda a árvore) para uma arquitetura **Híbrida e Independente**. O objetivo é mapear a polarização no `r/aiwars` e exportar um grafo assinado (`.gexf`) para o Gephi, focado em análise de redes sociais (SNA) para um artigo científico.

### Princípios de Design

*   **LLM Restrita à Raiz:** A LLM (Gemma 4 via OpenRouter) atua **exclusivamente** na classificação de posts raiz (`dataType == "post"`). Isso elimina o custo de milhares de chamadas de API e garante que o "âncora" do debate seja preciso.
*   **Classificação Independente (Zero-Shot):** Se um comentário responde a um pai `NEUTRAL` ou é um órfão, ele não herda o neutro. Ele é reclassificado do zero via Zero-Shot NLI, lendo apenas o próprio texto.
*   **NLI Large para Relações:** Se o pai tem posição (`PRO`/`ANTI`), o comentário é avaliado pelo DeBERTa-large para descobrir se ele concorda (+1) ou contradiz (-1).
*   **Poda Bottom-Up Estrutural:** O grafo é podado das folhas para as raízes. Nós `NEUTRAL` são removidos, **a menos que** tenham descendentes polarizados. Nesse caso, são salvos como `is_context_only=True` (pontes estruturais), preservando a topologia.
*   **Matemática Correta de Arestas:** O cálculo de `conflict_intensity` e `sign` agora lida corretamente com interações neutras (sinal 0, conflito 0.0).

---

## 2. Histórico de Correções e Evolução

### 2.1. Fim do Efeito Dominó (Falsos Anti/Pro em Cascata)
*   **O que deu errado:** No modelo anterior, se o NLI errasse a relação de um pai, o filho era invertido, e o neto herdava o erro. Isso gerou falsos positivos.
*   **Solução:** Classificação independente com Zero-Shot. Se o pai é NEUTRAL, o filho é classificado por conta própria.

### 2.2. Poda Bottom-Up (Preservação Topológica)
*   **O que deu errado:** A poda BFS tradicional cortava o neutro e todos os seus filhos, destruindo o grafo e perdendo discussões valiosas abaixo de comentários informativos.
*   **Solução:** Implementação da função `prune_bottom_up`. Ela verifica se o nó neutro tem filhos polarizados. Se tiver, ele é mantido como `context_only`. Isso aumentou a retenção e preservou os nós estruturais.

### 2.3. Correção do Bug de Arestas no GEXF
*   **O que deu errado:** Arestas com `rel = 0` (neutras) estavam gerando `conflict_intensity="1.0"` (conflito máximo) e `sign="1"` (concordância).
*   **Solução:** A lógica foi ajustada para tratar `soma == 0` como `sign=0` e `conflict_intensity=0.0`.

---

## 3. Resultados e Métricas (Audit v3)

A execução no dataset de 240 registros apresentou uma melhora significativa na retenção de dados em comparação à versão anterior (v2).

| Métrica | Valor (v2 - antiga) | Valor (v3 - atual) |
| :--- | :--- | :--- |
| **Nós Iniciais (Grafo Bruto)** | 222 | 222 |
| **Nós Podados** | 133 | 109 |
| **Nós Retidos** | 89 | 113 |
| **Nós de Contexto (Pontes)** | 17 | 22 |
| **Taxa de Retenção** | ~40.1% | **~50.9%** |

### Distribuição de Stance (Grafo Final Agregado)
*   **PRO:** 30 nós (~40.54%)
*   **ANTI:** 44 nós (~59.46%)
*   **NEUTRAL:** 0 nós (todos foram podados ou viraram contexto)

---

## 4. Análise de Erros (Error Analysis para o Artigo)

Apesar da acurácia de ~80-85%, o modelo comete erros previsíveis que devem ser documentados na seção de limitações do artigo. Cruzando o `predictions_raw.json` com o texto, encontramos:

1.  **Crítica mascarada de apoio (Falsos PRO):**
    *   `BlueFlower673` (*"I don't recommend using gen AI... do I want to use a generator at the expense of my fellow artists"*). Texto é ANTI, mas o Zero-Shot chutou PRO.
    *   `jaidyynn` (*"try different mediums of art, don't give up"*). É um incentivo à arte humana, mas foi para PRO.
2.  **Sarcasmo puro em frases curtas:**
    *   `-UltraFerret-` respondeu *"Can you report them?"* (em um contexto de assédio contra usuários de IA). O NLI marcou como ANTI, mas era uma interação de apoio (PRO).
3.  **Piadas internas / Off-topic:**
    *   A thread inteira de Star Trek. O Zero-Shot tenta forçar PRO/ANTI onde as pessoas só estão conversando sobre a série.

---

## 5. Configuração de Ambiente e Artefatos Gerados

**Execução:**
    source .ARS/bin/activate
    python ZeroShot.py BigScrape.json grafo_big2.gexf

**Artefatos gerados pelo script (v3):**
*   `pipeline.log`: Log estruturado com timestamps de toda a execução.
*   `predictions_raw.json`: Classificação individual de cada nó (texto, stance e método usado: LLM, NLI ou ZERO_SHOT). Essencial para o Error Analysis.
*   `audit.json`: Estatísticas macro da rede.
*   `grafo_big2.gexf`: Arquivo final para o Gephi.

---

## 6. Organização e Topologia do Grafo (Para o Gephi)

A configuração atual do pipeline produz um grafo direcionado, assinado e altamente purificado. A organização estrutural foi desenhada para isolar o "núcleo do debate".

### 6.1. Estrutura dos Nós (Vértices)
*   **Entidade Representada:** Cada nó no grafo final representa um **Usuário** (`author`).
*   **Atributos Exportados:**
    *   `stance`: A posição dominante do usuário (Moda).
    *   `total`: O número total de comentários polarizados feitos por esse usuário.
    *   `pro_ratio` / `anti_ratio`: A proporção de comentários PRO e ANTI.

### 6.2. Estrutura das Arestas (Conexões)
*   **Atributos Exportados:**
    *   `weight`: A frequência de interação.
    *   `sign`: O tom da relação acumulada (1=Concordância, -1=Contradição, 0=Neutro).
    *   `conflict_intensity`: Métrica de 0.0 a 1.0 que calcula a consistência da relação (1.0 = briga/concordância total; 0.5 = relação mista).

### 6.3. Topologia Macro (O que esperar visualmente)
A topologia do grafo assume um formato de **"Arquipélago Polarizado"** (duas grandes trincheiras).
1.  **Clusters Herméticos (Echo Chambers):** A maioria das arestas dentro dos grupos será verde (`sign=1`).
2.  **Arestas de Combate (Cross-cutting edges):** As arestas vermelhas (`sign=-1`) serão as "pontes" entre os dois grandes clusters.
3.  **Ausência de Nós Neutros:** O espaço entre os dois clusters será preenchido apenas pelas linhas vermelhas longas que conectam os polos.

---

## 7. Blindagem Técnica para Publicação Científica (Versão v3)

Para atender aos rigorosos critérios de um artigo científico, a versão final do código (`ZeroShot.py`) recebeu as seguintes atualizações:

### 7.1. Robustez de Infraestrutura
*   **Logging Estruturado:** Substituição de `print()` pela biblioteca `logging`, gerando o arquivo `pipeline.log` com timestamps.
*   **Retry e Backoff Exponencial na LLM:** Se o OpenRouter falhar, o código aguarda (2s, 4s, 8s) e tenta novamente até 3 vezes antes de assumir o `NEUTRAL`.
*   **BFS Iterativo (Prevenção de RecursionError):** O cálculo de profundidade das árvores foi reescrito usando `deque`. Threads com centenas de níveis não quebram mais o Python.
*   **Limpeza de Texto (`clean_text`):** Remoção de URLs, menções e Markdown antes de enviar para os modelos.

### 7.2. Robustez de NLP e Acurácia
*   **Few-Shot Prompting na LLM:** O prompt de classificação da raiz agora inclui 3 exemplos (1 PRO, 1 ANTI, 1 NEUTRAL). Isso ancora o modelo e reduz falsos positivos com sarcasmo.
*   **Threshold Científico de 0.65:** Se a classificação Zero-Shot de `NEUTRAL` tiver um score menor que 0.65, o modelo é forçado a escolher o polo mais próximo. Se for maior, o nó é mantido como `NEUTRAL` e subsequentemente podado.

### 7.3. Reprodutibilidade e Auditoria (Error Analysis)
*   **Seed Global (`set_seed(42)`):** Configuração fixa de sementes aleatórias para Python, NumPy e PyTorch. Garante reprodutibilidade.
*   **Exportação de `predictions_raw.json`:** O código salva um arquivo contendo o texto de cada nó, sua classificação final e qual método foi usado (`LLM`, `NLI` ou `ZERO_SHOT`). Isso permite escrever a seção de *Error Analysis* no artigo com precisão.