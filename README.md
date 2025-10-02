# RAG Modular (versão final — pronto para testar)

## Requisitos
- Python 3.11+
- macOS/Linux/Windows
- Chave de API da OpenAI

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env && ${EDITOR:-nano} .env
```

## Execução
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload --env-file .env
```
Abra: http://127.0.0.1:8010/ui

## Teste rápido por cURL
```bash
# Saúde
curl -s http://127.0.0.1:8010/healthz | jq .

# Ingestão de amostra
curl -s -X POST http://127.0.0.1:8010/ingest   -F "files=@samples/cadline.txt" | jq .

# Pergunta (sem streaming)
curl -s -X POST http://127.0.0.1:8010/query   -F 'question=Em que ano a Cadline Produções foi fundada?'   -F 'top_k=4'   -F 'stream=false' | jq .

# Limpar
curl -s -X POST http://127.0.0.1:8010/clear | jq .
```

## Estrutura
```
rag_modular/
  app/                # módulos do pipeline
  ui/                 # UI mínima
  tools/              # scripts auxiliares
  samples/            # textos de teste
  requirements.txt
  .env.example
  README.md
```

## Notas
- Suporta `.txt, .md, .pdf, .docx`.
- Recuperação **híbrida** (BM25 + vetorial c/ RRF).
- Streaming via `/query` com `stream=true`.
- Persistência do Chroma em `CHROMA_DIR` (configurável no .env).
