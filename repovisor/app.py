"""RepoVisor API + AI onboarding agent.  Run: uvicorn repovisor.app:app --reload"""
import json, os, pathlib, re, urllib.error, urllib.request
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import core

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
SYSTEM = ("You are RepoVisor, an onboarding assistant for one specific codebase. Answer ONLY from the "
          "evidence provided. Cite every claim as [path:start-end]. If the evidence is insufficient, say "
          "what is missing instead of guessing. Be concise and practical; give file paths and next steps.")

app = FastAPI(title="RepoVisor")


def _post(url, payload, headers):
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "repovisor/1.0", **headers})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


_hits = []


def _limited():  # protects your free Gemini key on a public site
    if not os.getenv("PUBLIC_MODE"):
        return False
    import time
    now = time.time()
    _hits[:] = [t for t in _hits if now - t < 3600]
    if len(_hits) >= int(os.getenv("ASK_LIMIT_PER_HOUR", "60")):
        return True
    _hits.append(now)
    return False


_gem_ok = None   # remembers the Gemini model that worked


def _gemini(user):
    """Tries several Gemini models (names change over time) and auto-discovers ones your key can use."""
    global _gem_ok
    key = os.environ["GEMINI_API_KEY"]
    cands = [_gem_ok] if _gem_ok else ([os.environ["GEMINI_MODEL"]] if os.getenv("GEMINI_MODEL") else [])
    if not cands:
        cands = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
        try:
            req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=100",
                                         headers={"x-goog-api-key": key, "User-Agent": "repovisor/1.0"})
            for m in json.loads(urllib.request.urlopen(req, timeout=20).read())["models"]:
                n = m["name"].split("/")[-1]
                if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n \
                        and not any(x in n for x in ("image", "tts", "live", "audio", "embed", "thinking")):
                    cands.append(n)
        except Exception:
            pass
    last = "no model available"
    for mod in dict.fromkeys(cands):
        try:
            r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent",
                      {"system_instruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"parts": [{"text": user}]}]},
                      {"x-goog-api-key": key})
            _gem_ok = mod
            return r["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            last = f"{mod}: HTTP {e.code} {e.read().decode(errors='ignore')[:250]}"
            if e.code != 404:   # 400/403 = bad key, 429 = quota: more models won't help
                raise RuntimeError(last)
    raise RuntimeError(last)


def llm(user):
    """Uses whichever key is in .env: Anthropic, Gemini (free tier) or Groq (free tier). None if no key."""
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        m = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=1200, system=SYSTEM,
                                                  messages=[dict(role="user", content=user)])
        return m.content[0].text
    if os.getenv("GEMINI_API_KEY"):
        return _gemini(user)
    if os.getenv("GROQ_API_KEY"):
        r = _post("https://api.groq.com/openai/v1/chat/completions",
                  {"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
                  {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]})
        return r["choices"][0]["message"]["content"]
    return None


class Connect(BaseModel):
    source: str          # GitHub https URL or local path
    token: str | None = None   # scoped, read-only GitHub token (private repos)


class Ask(BaseModel):
    question: str


@app.post("/api/repos")
def connect(body: Connect):
    try:
        return core.index_repo(body.source, body.token)
    except Exception as e:
        raise HTTPException(400, f"Indexing failed: {str(e)[:300]}")


def _repo(rid):
    try:
        return core.load(rid)[0]
    except FileNotFoundError:
        raise HTTPException(404, "Unknown repo id. Index it first.")


@app.get("/api/repos/{rid}/graph")
def graph(rid: str):
    m = _repo(rid)
    deg = Counter(x for e in m["edges"] for x in e)
    nodes = [dict(id=f, group=f.split("/")[0] if "/" in f else "(root)", deg=deg[f])
             for f in m["files"] if not f.endswith(".md")]
    return dict(nodes=nodes, links=[dict(source=a, target=b) for a, b in m["edges"]])


@app.get("/api/repos/{rid}/eval")
def quality(rid: str):
    _repo(rid)
    return core.evaluate(rid)


@app.get("/api/repos/{rid}/doc")
def doc(rid: str):
    m = _repo(rid)
    deg = Counter(t for _, t in m["edges"])
    lines = [f"# Architecture notes ({m['source']} @ {(m['commit'] or 'local')[:8]})", "",
             f"{len(m['files'])} files, {len(m['symbols'])} symbols, {len(m['edges'])} internal imports.", "",
             "## Most depended-on modules"]
    lines += [f"- `{f}` imported by {n} files" for f, n in deg.most_common(10)]
    mods = {}
    for s in m["symbols"]:
        mods.setdefault(s["file"], []).append(s["name"])
    lines += ["", "## Modules and symbols"]
    lines += [f"- `{f}`: {', '.join(n[:8])}" for f, n in sorted(mods.items())[:60]]
    return dict(markdown="\n".join(lines))


@app.post("/api/repos/{rid}/ask")
def ask(rid: str, body: Ask):
    m = _repo(rid)
    hits = core.search(rid, body.question)
    facts = []
    for f in dict.fromkeys(h["file"] for h in hits[:4]):
        d = core.deps(rid, f)
        facts.append(f"{f}: imports {d['imports'][:6]}; imported by {d['imported_by'][:6]}")
    ctx = "\n\n".join(f"[{h['file']}:{h['start']}-{h['end']}]\n{h['text']}" for h in hits)
    sources = [dict(file=h["file"], start=h["start"], end=h["end"], score=round(h["score"], 3)) for h in hits]
    prompt = ("Dependency facts:\n" + "\n".join(facts) + f"\n\nEvidence:\n{ctx}\n\nQuestion: {body.question}")
    if _limited():
        raise HTTPException(429, "Demo limit reached for this hour. Please try again later.")
    try:
        text = llm(prompt)
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"AI service error HTTP {e.code}: {e.read().decode(errors='ignore')[:250]}")
    except Exception as e:
        raise HTTPException(502, f"AI service error: {str(e)[:300]}")
    if text is None:
        return dict(answer="No API key found in .env - showing retrieved evidence only.", sources=sources,
                    unverified_citations=[], commit=m["commit"])
    cited = set(re.findall(r"\[([^\]:\s]+):\d+(?:-\d+)?\]", text))
    bad = sorted(cited - {h["file"] for h in hits})   # answer validation: flag citations not in evidence
    return dict(answer=text, sources=sources, unverified_citations=bad, commit=m["commit"])


app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parent / "static", html=True), name="ui")
