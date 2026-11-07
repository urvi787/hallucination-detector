import os, re, json, requests
import pandas as pd
import torch
import wikipediaapi
from dotenv import load_dotenv, find_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline

load_dotenv(find_dotenv())

MODEL = "openai/gpt-oss-120b"
CON_TOP = 1  # placeholder: top-1 vs top-3 gets decided on the WikiBio tuning split in Week 3

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
wiki = wikipediaapi.Wikipedia(user_agent="hallucination-detector/1.0 (student project)", language="en")

_embedder = None
_nli = None

def load_models():
    global _embedder, _nli
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if _nli is None:
        _nli = hf_pipeline("text-classification",
                           model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                           top_k=None,
                           device=0 if torch.cuda.is_available() else -1)

def _ask(prompt, max_tokens=1500, temperature=0):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens)
    return r.choices[0].message.content.strip()

def get_llm_answer(question):
    return _ask(f"Answer in 4-6 factual sentences.\n\nQuestion: {question}", 1000, 0.7)

def extract_claims(text):
    out = _ask(f"""Break the text below into independent, atomic factual claims.

Rules:
- One fact per claim.
- Replace pronouns (he, she, it, they) with the full name.
- Each claim must make sense on its own.
- Return ONLY a JSON list of strings, nothing else.

Text:
{text}""")
    m = re.search(r"\[.*\]", out, re.DOTALL)
    return json.loads(m.group(0)) if m else []

def find_entity(text):
    return _ask("Return ONLY the name of the main person, place or thing this text is about, nothing else.\n\n"
                f"Text: {text}", 500)

def find_wiki_page(name):
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": name, "format": "json", "srlimit": 1},
        headers={"User-Agent": "hallucination-detector/1.0 (student project)"})
    hits = resp.json()["query"]["search"]
    return hits[0]["title"] if hits else None

def resolve_subject(sent, title):
    last = title.split()[-1]
    s = re.sub(rf"^The {last}\b", f"The {title}", sent, flags=re.I)
    return s

def build_evidence(title):
    text = wiki.page(title).text
    sents = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip()
        if len(s.split()) >= 8 and s.endswith(".") and ":" not in s:
            sents.append(resolve_subject(s, title))
    return sents, _embedder.encode(sents, convert_to_tensor=True)

def get_evidence_from(claim, sents, emb, top_k=5):
    ce = _embedder.encode(claim, convert_to_tensor=True)
    scores = util.cos_sim(ce, emb)[0]
    top = scores.topk(min(top_k, len(sents)))
    return [(sents[i], float(s)) for s, i in zip(top.values, top.indices)]

def verify_claim(claim, evidence, thr=0.8, con_top=CON_TOP, min_sim=0.5):
    best_ent = {"score": 0.0, "evidence": ""}
    best_con = {"score": 0.0, "evidence": ""}
    for rank, (sent, sim) in enumerate(evidence):
        out = _nli({"text": sent, "text_pair": claim})
        out = out[0] if isinstance(out[0], list) else out
        sc = {o["label"].lower(): o["score"] for o in out}
        if sc.get("entailment", 0) > best_ent["score"]:
            best_ent = {"score": sc["entailment"], "evidence": sent}
        if rank < con_top and sim >= min_sim and sc.get("contradiction", 0) > best_con["score"]:
            best_con = {"score": sc["contradiction"], "evidence": sent}
    if best_ent["score"] >= thr:
        return {"label": "entailment", "score": best_ent["score"], "evidence": best_ent["evidence"]}
    if best_con["score"] >= thr:
        return {"label": "contradiction", "score": best_con["score"], "evidence": best_con["evidence"]}
    return {"label": "neutral", "score": max(best_ent["score"], best_con["score"]), "evidence": best_ent["evidence"]}

VERDICT = {"entailment": "Supported", "contradiction": "Contradicted", "neutral": "Not enough info"}

def run_pipeline(question=None, answer=None):
    """Give a question (Groq answers it) OR an answer you pasted. Returns (page title, answer, results table)."""
    load_models()
    if answer is None:
        answer = get_llm_answer(question)
    claims = extract_claims(answer)
    title = find_wiki_page(find_entity(question or answer))
    if not title or not wiki.page(title).exists():
        raise ValueError("Could not find a Wikipedia page for this topic.")
    sents, emb = build_evidence(title)
    rows = []
    for c in claims:
        r = verify_claim(c, get_evidence_from(c, sents, emb))
        rows.append({"claim": c, "verdict": VERDICT[r["label"]],
                     "confidence": round(r["score"], 2), "evidence": r["evidence"]})
    return title, answer, pd.DataFrame(rows)