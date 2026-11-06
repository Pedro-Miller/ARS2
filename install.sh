#!/usr/bin/env bash

set -e

echo "==> Verificando dependências base..."

if ! command -v python3 &> /dev/null; then
    echo "Erro: python3 não está instalado no sistema. Instale o Python 3 antes de continuar."
    exit 1
fi

# 1. Criar o ambiente virtual com garantia de pip
echo "==> Criando/verificando ambiente virtual Python (.ARS)..."
if [ ! -d ".ARS" ]; then
    python3 -m venv --with-pip .ARS
    echo "Ambiente virtual criado."
else
    echo "Ambiente virtual .ARS já existe."
fi

# 2. Ativar o venv
source .ARS/bin/activate

# 3. Reparar e atualizar o pip caso esteja corrompido
echo "==> Verificando e reparando a integridade do Pip..."
python3 -m ensurepip --upgrade 2>/dev/null || true
python3 -m pip install --upgrade pip

# 4. Instalar PyTorch compilado com suporte a CUDA 12.1 (NVIDIA)
echo "==> Instalação otimizada do PyTorch (CUDA 12.1)..."
python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu121

# 5. Instalar dependências adicionais do projeto
echo "==> Instalando dependências do projeto (requirements.txt)..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
else
    echo "Erro: requirements.txt não encontrado!"
    exit 1
fi

# 6. Checagem ativa de GPU/CUDA no PyTorch
echo "==> Testando suporte a GPU (CUDA) no PyTorch..."
CUDA_CHECK=$(python3 -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_CHECK" = "True" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "🟢 Sucesso: CUDA ativado com a GPU -> $GPU_NAME"
else
    echo "⚠️ AVISO: O PyTorch foi instalado, mas CUDA NÃO está disponível (rodará em CPU)."
    echo "Verifique os drivers da NVIDIA no Bazzite se desejar aceleração por hardware."
fi

echo "=========================================================="
echo " Instalação concluída com sucesso em $(uname -s)!"
echo " Para rodar a pipeline híbrida (Zero-Shot + NLI):"
echo "   1. source .ARS/bin/activate"
echo "   2. Configure as chaves no arquivo .env"
echo "   3. python zeroshot.py BigScrape.json grafo_big1.gexf"
echo "=========================================================="