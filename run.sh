#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f ".env" ]; then cp .env.example .env; fi
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload --env-file .env
