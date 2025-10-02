#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://127.0.0.1:8010}
echo "Usando BASE=$BASE"

echo "==> /healthz"
curl -s "$BASE/healthz" | jq . || true

echo "==> /ingest (amostra)"
curl -s -X POST "$BASE/ingest" -F "files=@samples/cadline.txt" | jq . || true

echo "==> /query (sem streaming)"
curl -s -X POST "$BASE/query"   -F 'question=Em que ano a Cadline Produções foi fundada?'   -F 'top_k=4'   -F 'stream=false' | jq . || true

echo "==> /clear"
curl -s -X POST "$BASE/clear" | jq . || true

echo "Concluído ✅"
