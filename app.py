from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json, subprocess, sys, time, threading

BASE = Path(__file__).resolve().parent
UPLOADS, DATA, CHECKPOINTS = BASE/'uploads', BASE/'data', BASE/'checkpoints'
for folder in (UPLOADS, DATA, CHECKPOINTS): folder.mkdir(exist_ok=True)
DB = DATA/'documents.json'
TRAIN_TEXT = DATA/'train.txt'
app = FastAPI(title='MiniGPT Studio')
app.mount('/static', StaticFiles(directory=BASE/'static'), name='static')
training = {'running': False, 'status': 'Not started', 'log': ''}
lock = threading.Lock()

def load_db():
    if DB.exists():
        try: return json.loads(DB.read_text(encoding='utf-8'))
        except Exception: return []
    return []

def save_db(items): DB.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == '.txt': return path.read_text(encoding='utf-8', errors='ignore')
    if ext == '.pdf': return '\n'.join(page.extract_text() or '' for page in PdfReader(str(path)).pages)
    if ext == '.docx': return '\n'.join(p.text for p in DocxDocument(str(path)).paragraphs)
    raise ValueError('Supported formats are TXT, PDF and DOCX.')

def make_chunks(text, size=1000, overlap=150):
    text = ' '.join(text.split())
    return [text[i:i+size] for i in range(0, len(text), size-overlap)] if text else []

@app.get('/', response_class=HTMLResponse)
def home(): return (BASE/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/documents')
def documents():
    return [{'name': d['name'], 'chunks': len(d['chunks']), 'characters': d['characters']} for d in load_db()]

@app.post('/api/upload')
async def upload(files: list[UploadFile] = File(...)):
    db, added = load_db(), []
    for up in files:
        original = Path(up.filename or 'upload').name
        ext = Path(original).suffix.lower()
        if ext not in {'.txt', '.pdf', '.docx'}: raise HTTPException(400, f'{original}: supported formats are TXT, PDF and DOCX.')
        raw = await up.read()
        if len(raw) > 25 * 1024 * 1024: raise HTTPException(413, f'{original} exceeds 25 MB.')
        target = UPLOADS / original
        if target.exists(): target = UPLOADS / f'{target.stem}_{int(time.time())}{target.suffix}'
        target.write_bytes(raw)
        try: extracted = extract_text(target)
        except Exception as e:
            target.unlink(missing_ok=True)
            raise HTTPException(400, f'Could not read {original}: {e}')
        parts = make_chunks(extracted)
        if not parts:
            target.unlink(missing_ok=True)
            raise HTTPException(400, f'No selectable text found in {original}. Scanned PDFs need OCR first.')
        db.append({'name': original, 'stored_name': target.name, 'characters': len(extracted), 'chunks': parts})
        added.append({'name': original, 'chunks': len(parts), 'characters': len(extracted)})
    save_db(db)
    return {'uploaded': added}

class ChatRequest(BaseModel): question: str

@app.post('/api/chat')
def chat(req: ChatRequest):
    passages = [(d['name'], chunk) for d in load_db() for chunk in d['chunks']]
    if not passages: return {'answer': 'Upload documents first.', 'sources': []}
    try:
        vec = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform([p[1] for p in passages] + [req.question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError: return {'answer': 'No matching text found. Try another question.', 'sources': []}
    best = scores.argsort()[::-1][:3]
    matches = [(passages[i][0], passages[i][1]) for i in best if scores[i] > 0]
    if not matches: return {'answer': 'I could not find a relevant passage in your uploaded documents.', 'sources': []}
    answer = 'Relevant passages found (this starter uses retrieval, not generated LLM answers):\n\n' + '\n\n'.join('• ' + text for _, text in matches)
    return {'answer': answer, 'sources': list(dict.fromkeys(name for name, _ in matches))}

def run_training():
    with lock:
        training.update(running=True, status='Training', log='Preparing corpus...')
    try:
        docs = load_db()
        corpus = '\n\n'.join(' '.join(d['chunks']) for d in docs)
        if len(corpus.strip()) < 100: raise RuntimeError('Upload more text first (at least 100 characters).')
        TRAIN_TEXT.write_text(corpus, encoding='utf-8')
        proc = subprocess.run([sys.executable, str(BASE/'train.py')], cwd=BASE, capture_output=True, text=True, timeout=3600)
        training['log'] = (proc.stdout or '') + ('\n' + proc.stderr if proc.stderr else '')
        training['status'] = 'Training completed' if proc.returncode == 0 else 'Training failed'
    except Exception as e:
        training['status'] = f'Error: {e}'
        training['log'] += str(e)
    finally: training['running'] = False

@app.post('/api/train')
def start_training(background: BackgroundTasks):
    if training['running']: raise HTTPException(409, 'Training is already running.')
    background.add_task(run_training)
    return {'message': 'Training job started.'}

@app.get('/api/train-status')
def train_status(): return training

@app.get('/api/health')
def health(): return {'documents': len(load_db()), 'checkpoint_exists': (CHECKPOINTS/'minigpt.pt').exists()}
