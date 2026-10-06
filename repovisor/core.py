"""RepoVisor core: clone -> parse (AST) -> symbols/deps -> embeddings -> retrieval."""
import ast, base64, hashlib, json, os, pathlib, pickle, re, subprocess, time
from functools import lru_cache
import numpy as np

_env = pathlib.Path(__file__).parent.parent / ".env"   # simple .env loader (API key lives here)
if _env.exists():
    for _l in _env.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            if _v.strip():
                os.environ[_k.strip()] = _v.strip()

DATA = pathlib.Path(os.getenv("REPOVISOR_DATA", "data")); DATA.mkdir(exist_ok=True)
SKIP = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".next"}
EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".md", ".html", ".css", ".c", ".cpp", ".h", ".cs", ".php", ".rb", ".rs", ".kt", ".sql", ".yml", ".yaml"}
JS_IMPORT = re.compile(r"(?:import\s[^'\"]*?from\s+|import\s+|require\()\s*['\"]([^'\"]+)['\"]")
JS_SYM = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(function|class)\s+(\w+)", re.M)


def rid_for(src: str) -> str:
    return hashlib.md5(src.encode()).hexdigest()[:10]


def _git(root, *args, token=None):
    cfg = []
    if token:  # token goes in a header, never in the URL/.git/config or prompts
        b = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cfg = ["-c", f"http.extraHeader=Authorization: Basic {b}"]
    return subprocess.run(["git", *cfg, *args], capture_output=True, text=True, check=True).stdout


def clone(src: str, token: str | None = None):
    rid = rid_for(src)
    if os.getenv("PUBLIC_MODE") and not src.startswith("https://github.com/"):
        raise ValueError("This public demo only accepts https://github.com/... links")
    if not src.startswith("http"):
        return rid, pathlib.Path(src).resolve()
    dest = DATA / rid / "src"
    if dest.exists():
        _git(dest, "-C", str(dest), "pull", "-q", token=token)
    else:
        _git(None, "clone", "--depth", "300", "-q", src, str(dest), token=token)
    return rid, dest


def parse_py(rel, text):
    syms, imps, calls = [], [], []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return syms, imps, calls
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            syms.append(dict(name=n.name, kind=type(n).__name__, file=rel, start=n.lineno, end=n.end_lineno))
            for c in ast.walk(n):
                if isinstance(c, ast.Call):
                    f = c.func
                    nm = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                    if nm:
                        calls.append([n.name, nm, rel])
        elif isinstance(n, ast.Import):
            imps += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            imps.append("." * n.level + n.module)
    return syms, imps, calls


def parse_js(rel, text):
    syms = [dict(name=m.group(2), kind=m.group(1), file=rel,
                 start=text[:m.start()].count("\n") + 1, end=text[:m.start()].count("\n") + 1)
            for m in JS_SYM.finditer(text)]
    return syms, JS_IMPORT.findall(text), []


def resolve(imp, src, files):
    if src.endswith(".py"):
        cand = imp.lstrip(".").replace(".", "/")
    elif imp.startswith("."):
        cand = os.path.normpath(os.path.join(os.path.dirname(src), imp)).replace("\\", "/")
    else:
        return None
    for f in files:
        stem = f.rsplit(".", 1)[0]
        if stem == cand or stem.endswith("/" + cand) or stem in (cand + "/index", cand + "/__init__") \
                or stem.endswith("/" + cand + "/__init__") or stem.endswith("/" + cand + "/index"):
            return f
    return None


def chunk(rel, text, syms, size=50, overlap=10):
    lines, out = text.splitlines(), []
    for i in range(0, max(len(lines), 1), size - overlap):
        s, e = i + 1, min(i + size, len(lines))
        names = [x["name"] for x in syms if x["start"] <= e and x["end"] >= s][:8]
        body = "\n".join(lines[i:i + size])
        if body.strip():
            out.append(dict(file=rel, start=s, end=e, text=f"# {rel} | symbols: {', '.join(names)}\n{body}"))
        if i + size >= len(lines):
            break
    return out


BACKEND = os.getenv("EMBED_BACKEND", "tfidf")   # "tfidf" = light, no downloads; "neural" = better, needs sentence-transformers


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))


def _prep(t):  # split camelCase / snake_case so "getUserName" matches "user name"
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", t).replace("_", " ").lower()


def build_vectors(d, texts):
    if BACKEND == "neural":
        np.save(d / "emb.npy", _model().encode(texts, normalize_embeddings=True, batch_size=32))
        return
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(preprocessor=_prep, token_pattern=r"[A-Za-z0-9]{2,}", sublinear_tf=True,
                        max_features=60000, ngram_range=(1, 2))
    sp.save_npz(d / "emb.npz", v.fit_transform(texts))
    (d / "vec.pkl").write_bytes(pickle.dumps(v))


def index_repo(src: str, token: str | None = None) -> dict:
    rid, root = clone(src, token)
    files, syms, calls, chunks, raw_imps = [], [], [], [], {}
    for p in sorted(root.rglob("*")):
        if len(files) >= int(os.getenv("MAX_FILES", "3000")):
            break
        if p.suffix not in EXT or any(s in p.parts for s in SKIP) or not p.is_file() or p.stat().st_size > 400_000:
            continue
        rel = p.relative_to(root).as_posix()
        text = p.read_text(errors="ignore")
        s, i, c = parse_py(rel, text) if p.suffix == ".py" else parse_js(rel, text) if p.suffix != ".md" else ([], [], [])
        files.append(rel); syms += s; calls += c; raw_imps[rel] = i
        chunks += chunk(rel, text, s)
    edges = sorted({(f, t) for f, imps in raw_imps.items() for i in imps if (t := resolve(i, f, files)) and t != f})
    d = DATA / rid; d.mkdir(exist_ok=True)
    meta = dict(id=rid, source=src, root=str(root), files=files, symbols=syms, edges=edges,
                calls=calls, chunks=chunks, commit=_commit(root))
    (d / "index.json").write_text(json.dumps(meta))
    if chunks:
        for old in ("emb.npy", "emb.npz"):
            (d / old).unlink(missing_ok=True)
        build_vectors(d, [c["text"] for c in chunks])
    load.cache_clear()
    return {k: meta[k] for k in ("id", "commit")} | dict(files=len(files), symbols=len(syms), edges=len(edges), chunks=len(chunks))


def _commit(root):
    try:
        return _git(None, "-C", str(root), "rev-parse", "HEAD").strip()
    except Exception:
        return None


@lru_cache(maxsize=8)
def load(rid):
    d = DATA / rid
    meta = json.loads((d / "index.json").read_text())
    if (d / "emb.npy").exists():
        return meta, np.load(d / "emb.npy"), None
    if (d / "emb.npz").exists():
        import scipy.sparse as sp
        return meta, sp.load_npz(d / "emb.npz"), pickle.loads((d / "vec.pkl").read_bytes())
    return meta, None, None


def search(rid, query, k=6):
    meta, emb, vec = load(rid)
    if emb is None:
        return []
    if vec is None:
        sc = emb @ _model().encode([query], normalize_embeddings=True)[0]
    else:
        sc = (emb @ vec.transform([query]).T).toarray().ravel()
    toks = set(re.findall(r"\w{3,}", query.lower()))
    for i, c in enumerate(meta["chunks"]):  # light boost for exact identifier hits
        sc[i] += 0.03 * sum(t in c["text"].lower() for t in toks)
    return [dict(meta["chunks"][i], score=float(sc[i])) for i in np.argsort(-sc)[:k]]


def evaluate(rid, n=30, k=5):
    """Auto quality check: ask 'where is X implemented' for sampled symbols; is the right file in top k?"""
    import random
    syms = [s for s in load(rid)[0]["symbols"] if len(s["name"]) > 5 and not s["name"].startswith("__")]
    random.Random(0).shuffle(syms)
    syms = syms[:n]
    hit = rr = 0.0
    t0 = time.time()
    for s in syms:
        files = [h["file"] for h in search(rid, f"where is {s['name']} implemented", k)]
        if s["file"] in files:
            hit += 1
            rr += 1 / (files.index(s["file"]) + 1)
    q = max(len(syms), 1)
    return dict(questions=len(syms), recall_at_k=round(hit / q, 2), mrr=round(rr / q, 2),
                avg_ms=round((time.time() - t0) * 1000 / q))


def deps(rid, file):
    meta = load(rid)[0]
    return dict(imports=[t for f, t in meta["edges"] if f == file],
                imported_by=[f for f, t in meta["edges"] if t == file])


def find_symbol(rid, name):
    return [s for s in load(rid)[0]["symbols"] if name.lower() in s["name"].lower()][:20]


def callers(rid, name):
    return sorted({(c[0], c[2]) for c in load(rid)[0]["calls"] if c[1] == name})[:30]


def read_file(rid, rel, start=1, end=400):
    root = pathlib.Path(load(rid)[0]["root"]).resolve()
    p = (root / rel).resolve()
    if root not in p.parents:  # block path traversal
        raise ValueError("path outside repository")
    return "\n".join(p.read_text(errors="ignore").splitlines()[start - 1:end])


def history(rid, path=None, n=10):
    root = load(rid)[0]["root"]
    args = ["-C", root, "log", f"-{n}", "--pretty=%h|%an|%ad|%s", "--date=short"] + (["--", path] if path else [])
    try:
        return _git(None, *args).splitlines()
    except Exception:
        return []
