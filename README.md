# Reddit Stance Detection Graph

Pipeline em Python para construção de Signed Reply Graphs (`.gexf`) a partir de exportações do Apify (Reddit), para visualização no Gephi. 

A solução utiliza LLMs (**Groq** ou **Google AI Studio**) para ancoragem de stance nos posts/comentários raiz e o modelo **DeBERTa-v3** (NLI) para inferir relações de concordância/discordância entre comentários dependentes.

---

## Requisitos de Sistema

- **Python 3.8+**
- Compatível com qualquer distribuição Linux (Ubuntu, Arch, Fedora, etc.)

---

## Estrutura do Repositório

```text
.
├── examples/
│   └── amostra_apify.json      # Dados de exemplo para testes rápidos
├── .gitignore                  # Arquivos ignorados pelo Git
├── install.sh                  # Script de instalação do ambiente (.venv)
├── README.md                   # Documentação do projeto
├── requirements.txt            # Dependências em Python
└── script_grafo.py             # Script principal da pipeline
```

---

## Instalação Rápida

Clone o repositório:

```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPO.git
cd NOME_DO_REPO
```

Dê permissão de execução e rode o script de instalação (cria o `.venv` e instala os pacotes):

```bash
chmod +x install.sh
./install.sh
```

> **Nota:** Na primeira execução do script Python, o modelo `cross-encoder/nli-deberta-v3-base` será baixado automaticamente pelo Hugging Face e mantido no cache local (`~/.cache/huggingface/`).

---

## Como Utilizar

### 1. Ativar o Ambiente Virtual

Sempre que abrir uma nova sessão no terminal, ative o ambiente virtual:

```bash
source .venv/bin/activate
```

### 2. Configurar a Chave de API

O script detecta automaticamente qual provedor utilizar. Configure a chave de pelo menos um dos serviços no seu terminal:

- Para usar o **Groq** (Llama 3.3 70B):

  ```bash
  export GROQ_API_KEY="sua_chave_groq_aqui"
  ```

- Para usar o **Google AI Studio** (Gemini 2.5 Flash):

  ```bash
  export GEMINI_API_KEY="sua_chave_gemini_aqui"
  ```

### 3. Executar o Script

O script aceita os arquivos de entrada e saída como argumentos de linha de comando por posição (`sys.argv`):

```bash
python3 script_grafo.py [arquivo_entrada.json] [arquivo_saida.gexf]
```

#### Exemplos de Uso:

- **Sem argumentos (Modo Padrão):**  
  Procura automaticamente por `dados_apify.json` (ou `examples/amostra_apify.json`) e gera o arquivo `rede_usuarios_assinada.gexf`.

  ```bash
  python3 script_grafo.py
  ```

- **Passando apenas o JSON de entrada:**

  ```bash
  python3 script_grafo.py meus_dados.json
  ```

- **Passando entrada e saída personalizadas:**

  ```bash
  python3 script_grafo.py exemplos/thread_reddit.json grafos/rede_final.gexf
  ```

### 4. Visualizar no Gephi

1. Abra o Gephi.
2. Vá em **Arquivo -> Abrir...** e selecione o arquivo `.gexf` gerado.
3. No painel de importação, confirme que o grafo é do tipo **Direcionado (Directed)**.
4. Use os atributos `stance` (nos nós) ou `sign` / `weight` (nas arestas) para aplicar cores e algoritmos de layout (como ForceAtlas 2).