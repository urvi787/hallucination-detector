import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flask import Flask, request, jsonify, render_template
from jinja2 import TemplateNotFound
import pipeline
from cached_pipeline import run_cached

app = Flask(__name__)
pipeline.load_models()   # load once at startup, not on every request

MAX_CHARS = 3000

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "Backend is running. The web page comes in Step 4."

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "question")
    text = (data.get("text") or "").strip()

    if mode not in ("question", "answer"):
        return jsonify({"error": "mode must be 'question' or 'answer'"}), 400
    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Text is too long (max {MAX_CHARS} characters)."}), 400

    try:
        if mode == "question":
            title, answer, df, from_cache = run_cached(question=text)
        else:
            title, answer, df, from_cache = run_cached(answer=text)
    except ValueError as e:          # e.g. no Wikipedia page found
        return jsonify({"error": str(e)}), 404
    except Exception as e:           # Groq down, bad key, etc.
        return jsonify({"error": f"Something went wrong: {e}"}), 500

    counts = df["verdict"].value_counts().to_dict()
    summary = {
        "total": int(len(df)),
        "supported": int(counts.get("Supported", 0)),
        "contradicted": int(counts.get("Contradicted", 0)),
        "unverifiable": int(counts.get("Not enough info", 0)),
    }
    return jsonify({
        "page": title,
        "answer": answer,
        "claims": df.to_dict("records"),
        "summary": summary,
        "from_cache": from_cache,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
