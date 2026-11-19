# MiniGPT Studio

A local FastAPI dashboard for uploading TXT/PDF/DOCX documents, searching their contents, and training a small GPT-style character-level Transformer.

## Requirements
Python 3.11 or 3.12 is recommended. Install dependencies in a virtual environment:

```bash
cd MiniGPT-Studio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## Use
1. Upload TXT, searchable PDF, or DOCX files.
2. Ask questions to retrieve matching passages immediately.
3. Click Start training to combine extracted text into `data/train.txt` and train the small model.
4. Model checkpoint is saved at `checkpoints/minigpt.pt`.
