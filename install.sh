#!/usr/bin/env bash

set -e

echo "==> Verificando dependências base..."

if ! command -v python3 &> /dev/null; then
    echo "Erro: python3 não está instalado no sistema. Instale o Python 3 antes de continuar."
    exit 1
fi

# Cria o ambiente virtual
echo "==> Criando ambiente virtual Python (.ARS)..."
if [ ! -d ".ARS" ]; then
    python3 -m venv .ARS
    echo "Ambiente virtual criado."
else
    echo "Ambiente virtual .ARS já existe."
fi

# Ativa e instala os pacotes isoladamente
echo "==> Instalação dos pacotes com Pip..."
source .ARS/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=========================================================="
echo " Instalação concluída com sucesso em $(uname -s)!"
echo " Para rodar o projeto:"
echo "   1. source .ARS/bin/activate"
echo "   2. export GROQ_API_KEY=\"sua_chave\" OU export GEMINI_API_KEY=\"sua_chave\""
echo "   3. python3 script_grafo.py"
echo "=========================================================="