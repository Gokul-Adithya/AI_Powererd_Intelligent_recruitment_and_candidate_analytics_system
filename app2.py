# ============================================================
# AI-POWERED RECRUITMENT SYSTEM - STYLE 2 (Dark Purple + Blue)
# UPDATED: RoBERTa model, expected skills, dropdown sections,
#          job role suggestions, improved CSV & report
# ============================================================

import streamlit as st
import numpy as np, torch, re, spacy
import PyPDF2, docx, matplotlib.pyplot as plt
import seaborn as sns, pandas as pd, io, datetime, warnings
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_curve, auc)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, PageBreak)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="HireRight AI — Futuristic", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
    .stApp { background: #0f0c29; background-image: radial-gradient(ellipse at top, #1a1040 0%, #0f0c29 60%); }
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.02) !important; border-right: 1px solid rgba(255,255,255,0.05) !important; }
    [data-testid="stSidebar"] * { color: #aaa !important; }
    h1, h2, h3 { background: linear-gradient(135deg, #667eea, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton > button { background: linear-gradient(135deg, #667eea, #764ba2); color: white !important; border: none; border-radius: 12px; padding: 12px 28px; font-size: 15px; font-weight: 700; width: 100%; transition: all 0.3s; }
    .stButton > button:hover { background: linear-gradient(135deg, #f093fb, #667eea); transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.4); }
    .glass-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 20px; margin: 10px 0; transition: transform 0.2s, border-color 0.2s; }
    .glass-card:hover { transform: translateY(-3px); border-color: rgba(102,126,234,0.4); }
    .shortlisted  { border-left: 4px solid #4ade80 !important; }
    .under-review { border-left: 4px solid #fbbf24 !important; }
    .rejected     { border-left: 4px solid #f87171 !important; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
    .badge-green  { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .badge-orange { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .badge-red    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    .skill-tag          { display: inline-block; background: rgba(102,126,234,0.1); color: #a78bfa; padding: 3px 10px; border-radius: 20px; font-size: 12px; margin: 2px; border: 1px solid rgba(102,126,234,0.2); }
    .skill-tag-expected { display: inline-block; background: rgba(74,222,128,0.1);  color: #4ade80; padding: 3px 10px; border-radius: 20px; font-size: 12px; margin: 2px; border: 1px solid rgba(74,222,128,0.2); }
    .skill-tag-missing  { display: inline-block; background: rgba(248,113,113,0.1); color: #f87171; padding: 3px 10px; border-radius: 20px; font-size: 12px; margin: 2px; border: 1px solid rgba(248,113,113,0.2); }
    .section-header { background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)); border: 1px solid rgba(102,126,234,0.3); color: #a78bfa !important; padding: 12px 20px; border-radius: 12px; font-size: 16px; font-weight: 700; margin: 16px 0 12px 0; }
    .stTextArea label { color: #667eea !important; font-weight: 700 !important; font-size: 14px !important; }
    .stSelectbox label { color: #667eea !important; font-weight: 700 !important; font-size: 14px !important; }
    .stSelectbox > div > div { background: #0f0c29 !important; border: 1px solid rgba(167,139,250,0.4) !important; border-radius: 12px !important; color: #a78bfa !important; }
    .stSelectbox > div > div > div { color: #a78bfa !important; }
    .stSelectbox svg { fill: #a78bfa !important; }
    .stTextArea textarea { background: #0f0c29 !important; border: 1.5px solid rgba(102,126,234,0.5) !important; border-radius: 12px !important; color: #f5c518 !important; font-size: 14px !important; }
    .stTextArea textarea::placeholder { color: rgba(245,197,24,0.35) !important; }
    [data-testid="stFileUploader"] { background: rgba(10,8,40,0.6) !important; border: 2px dashed rgba(102,126,234,0.55) !important; border-radius: 14px !important; padding: 10px !important; }
    [data-testid="stFileUploader"] * { color: #a78bfa !important; }
    [data-testid="stFileUploaderDropzoneInput"] + div button,
    [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 700 !important;
    }
    [data-testid="stFileUploaderDropzone"] { background: transparent !important; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
    [data-testid='stExpander'] { border-radius: 12px !important; margin-bottom: 4px !important; }
    [data-testid='stExpander'] summary { font-weight: 700 !important; font-size: 14px !important; }
    [data-testid='stExpander']:nth-of-type(5n+1) summary p, [data-testid='stExpander']:nth-of-type(5n+1) summary { color: #667eea !important; }
    [data-testid='stExpander']:nth-of-type(5n+1) { border: 1px solid rgba(102,126,234,0.35) !important; }
    [data-testid='stExpander']:nth-of-type(5n+2) summary p, [data-testid='stExpander']:nth-of-type(5n+2) summary { color: #fbbf24 !important; }
    [data-testid='stExpander']:nth-of-type(5n+2) { border: 1px solid rgba(251,191,36,0.35) !important; }
    [data-testid='stExpander']:nth-of-type(5n+3) summary p, [data-testid='stExpander']:nth-of-type(5n+3) summary { color: #f093fb !important; }
    [data-testid='stExpander']:nth-of-type(5n+3) { border: 1px solid rgba(240,147,251,0.35) !important; }
    [data-testid='stExpander']:nth-of-type(5n+4) summary p, [data-testid='stExpander']:nth-of-type(5n+4) summary { color: #60a5fa !important; }
    [data-testid='stExpander']:nth-of-type(5n+4) { border: 1px solid rgba(96,165,250,0.35) !important; }
    [data-testid='stExpander']:nth-of-type(5n+5) summary p, [data-testid='stExpander']:nth-of-type(5n+5) summary { color: #4ade80 !important; }
    [data-testid='stExpander']:nth-of-type(5n+5) { border: 1px solid rgba(74,222,128,0.35) !important; }
    .stat-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(102,126,234,0.15); border-radius: 16px; padding: 20px; text-align: center; transition: all 0.3s; }
    .stat-card:hover { border-color: rgba(102,126,234,0.5); box-shadow: 0 0 20px rgba(102,126,234,0.15); }
    .stat-val { font-size: 32px; font-weight: 800; background: linear-gradient(135deg, #667eea, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stat-sub { font-size: 11px; color: #444; margin-top: 2px; }
    .parse-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(102,126,234,0.25); border-left: 5px solid #667eea; border-radius: 16px; padding: 24px; margin: 12px 0; }
    .role-suggest-card { background: rgba(102,126,234,0.07); border: 1px solid rgba(102,126,234,0.25); border-radius: 12px; padding: 14px 18px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── MODEL LOAD (all-MiniLM-L6-v2 — fast, lightweight, production-ready) ──────
# 420MB, best-in-class sentence similarity — outperforms RoBERTa for semantic matching
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

@st.cache_resource
def load_models():
    nlp       = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return nlp, tokenizer, model

# ── TEXT EXTRACTION ─────────────────────────────────────────────────────────
def extract_text_from_pdf(file):
    """Smart extractor — handles single-column and multi-column PDF layouts."""
    from collections import defaultdict
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words:
                    continue
                width = page.width
                left_words  = [w for w in words if w['x0'] < width * 0.45]
                right_words = [w for w in words if w['x0'] >= width * 0.45]
                left_ratio  = len(left_words) / max(len(words), 1)
                if 0.08 < left_ratio < 0.6 and len(left_words) > 5:
                    # Multi-column layout — reconstruct each column separately
                    def col_to_text(word_list, tol=3):
                        rows = defaultdict(list)
                        for w in word_list:
                            top = round(w['top'] / tol) * tol
                            rows[top].append(w)
                        lines = []
                        for top in sorted(rows.keys()):
                            row_words = sorted(rows[top], key=lambda w: w['x0'])
                            lines.append(" ".join(w['text'] for w in row_words))
                        return "\n".join(lines)
                    page_text = col_to_text(right_words) + "\n" + col_to_text(left_words)
                else:
                    page_text = page.extract_text() or ""
                text += page_text + "\n"
        if text.strip(): return text.strip()
    except Exception: pass
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    except Exception: pass
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

# ── ROBERTA EMBEDDING (mean pooling) ─────────────────────────────────────────
def get_embedding(text, tokenizer, model, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                       truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    token_emb   = outputs.last_hidden_state
    attn_mask   = inputs["attention_mask"]
    mask_expand = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (torch.sum(token_emb * mask_expand, 1) /
            torch.clamp(mask_expand.sum(1), min=1e-9)).squeeze().numpy()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# ── NAME EXTRACTION ──────────────────────────────────────────────────────────
SKIP_KEYWORDS = [
    "resume","curriculum","vitae","cv","profile","summary","objective",
    "contact","address","email","phone","mobile","linkedin","github",
    "http","www","career","about","declaration","name","details",
    "information","page","references","skills","projects","education",
    "experience","achievements","certifications","languages","hobbies",
    "developer","dev","engineer","designer","analyst","scientist",
    "manager","architect","consultant","specialist","intern","lead",
    "model","data","software","full stack","frontend","backend",
    "machine learning","ai","ml","web","mobile","cloud","devops",
    "spring","spring boot","java","python","react","node","angular",
    "django","flask","docker","kubernetes","aws","azure","gcp",
    "mongodb","mysql","postgresql","redis","tensorflow","pytorch",
    "hadoop","spark","tableau","power bi","excel","linux","git",
    "javascript","typescript","html","css","rest","api","sql"
]
NAME_PREFIXES = ["mr.","mrs.","ms.","dr.","prof."]

def is_valid_name(line):
    clean = line.strip()
    for pfx in NAME_PREFIXES:
        if clean.lower().startswith(pfx): clean = clean[len(pfx):].strip(); break
    if re.match(r"(?i)^name\s*[:\-]\s*", clean):
        clean = re.sub(r"(?i)^name\s*[:\-]\s*", "", clean).strip()
    words = clean.split()
    return (
        2 <= len(words) <= 4
        and re.match(r"^[A-Za-z\s.\-]+$", clean)
        and not any(k in clean.lower() for k in SKIP_KEYWORDS)
        and all(w[0].isupper() for w in words if w)
        and not any(c in clean for c in ["|","@","/",":","+"])
    ), clean

def extract_name(raw_text, nlp):
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if lines:
        valid, clean = is_valid_name(lines[0])
        if valid: return clean
    doc = nlp(raw_text[:800])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            valid, clean = is_valid_name(ent.text.strip())
            if valid: return clean
    for line in lines[1:8]:
        valid, clean = is_valid_name(line)
        if valid: return clean
    return "Not Found"

# ── CONTACT EXTRACTION ───────────────────────────────────────────────────────
def extract_email(text):
    m = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return m[0] if m else "Not Found"

def extract_phone(text):
    for p in [r"(\+91[\s\-]?)?[6-9]\d{9}", r"\+?[\d\s\-\(\)]{10,15}"]:
        m = re.findall(p, text)
        if m:
            number = ''.join(filter(str.isdigit, m[0]))
            if len(number) >= 10: return m[0] if isinstance(m[0], str) else ''.join(m[0])
    return "Not Found"

def extract_linkedin(text):
    m = re.findall(r"linkedin\.com/in/[a-zA-Z0-9\-]+", text)
    return m[0] if m else "Not Found"

def extract_github(text):
    m = re.findall(r"github\.com/[a-zA-Z0-9\-]+", text)
    return m[0] if m else "Not Found"

# ── SKILLS ───────────────────────────────────────────────────────────────────
SKILLS_DB = [
    "python","java","c++","c#","c","r","scala","kotlin","swift","go",
    "javascript","typescript","matlab","bash","machine learning","deep learning",
    "artificial intelligence","neural networks","natural language processing","nlp",
    "computer vision","reinforcement learning","bert","gpt","transformers","llm",
    "tensorflow","pytorch","keras","scikit-learn","xgboost","huggingface",
    "spacy","nltk","opencv","pandas","numpy","scipy","matplotlib","seaborn","plotly",
    "data analysis","data science","data visualization","feature engineering",
    "sql","mysql","postgresql","mongodb","sqlite","redis",
    "flask","django","fastapi","streamlit","html","css","react","nodejs","rest api",
    "aws","gcp","azure","docker","kubernetes","git","github",
    "linux","power bi","tableau","excel","hadoop","spark","roberta","distilbert"
]

def extract_skills(text):
    found = []
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found.append(skill.title())
    return list(dict.fromkeys(found)) if found else ["Not Found"]

def extract_expected_skills(jd_text):
    """Extract skills expected/required from the job description."""
    return extract_skills(jd_text)

# ── SECTION EXTRACTION ───────────────────────────────────────────────────────
def extract_section(text, section_headers, stop_headers):
    lines = text.split('\n'); collecting = False; content = []
    for line in lines:
        line_lower = line.lower().strip()
        if collecting and any(h in line_lower for h in stop_headers): break
        if any(h in line_lower for h in section_headers): collecting = True; continue
        if collecting and line.strip() and len(line.strip()) > 2: content.append(line.strip())
    return content

# ── RESUME PARSER ────────────────────────────────────────────────────────────
def parse_resume(text, nlp):
    name     = extract_name(text, nlp)
    raw_text = text
    text     = clean_text(text)

    def edu(t):
        degree_keywords = ["b.tech","b.e","b.sc","m.tech","m.sc","phd","bachelor",
                           "master","diploma","cgpa","gpa","university","college","institute"]
        lns = extract_section(raw_text, ["education","academic","qualification"],
                              ["experience","internship","skill","project","certification",
                               "achievement","objective","summary"])
        for tl in raw_text.split('\n'):
            tl = tl.strip()
            if (any(k in tl.lower() for k in degree_keywords)
                    and tl not in lns and 5 < len(tl) < 200
                    and not re.search(r'@|\+91|http|linkedin|github', tl, re.IGNORECASE)):
                lns.append(tl)
        return list(dict.fromkeys(lns))[:6] or ["Not Found"]

    def exp(t):
        lns = extract_section(raw_text, ["experience","internship","employment","work history"],
                              ["education","skill","project","certification",
                               "achievement","objective","summary","declaration"])
        for tl in raw_text.split('\n'):
            tl = tl.strip()
            if (re.search(r'(20\d{2})\s*[-–]\s*(20\d{2}|present)', tl, re.IGNORECASE)
                    and tl not in lns and len(tl) > 5):
                lns.append(tl)
        cleaned = [l for l in list(dict.fromkeys(lns)) if len(l.split()) <= 20]
        return cleaned[:6] or ["Not Found"]

    def proj(t):
        return extract_section(raw_text, ["project","projects"],
                               ["experience","education","skill","certification"])[:6] or ["Not Found"]

    def ach(t):
        lns = extract_section(raw_text, ["achievement","award","honor"],
                              ["education","skill","project","certification"])
        for tl in raw_text.split('\n'):
            tl = tl.strip()
            if (any(k in tl.lower() for k in ["award","winner","rank","1st","merit",
                                               "scholarship","gold","hackathon","published"])
                    and tl not in lns and len(tl) > 5):
                lns.append(tl)
        return list(dict.fromkeys(lns))[:5] or ["Not Found"]

    def cert(t):
        lns = extract_section(raw_text, ["certification","certificate","course"],
                              ["education","skill","project","achievement"])
        for tl in raw_text.split('\n'):
            tl = tl.strip()
            if (any(k in tl.lower() for k in ["udemy","coursera","nptel","google",
                                               "microsoft","aws","ibm","certified"])
                    and tl not in lns and len(tl) > 5):
                lns.append(tl)
        return list(dict.fromkeys(lns))[:5] or ["Not Found"]

    return {"Name": name, "Email": extract_email(text),
            "Phone": extract_phone(text), "LinkedIn": extract_linkedin(text),
            "GitHub": extract_github(text), "Skills": extract_skills(text),
            "Education": edu(text), "Experience": exp(text), "Projects": proj(text),
            "Achievements": ach(text), "Certifications": cert(text)}

# ── JOB ROLE SUGGESTIONS ─────────────────────────────────────────────────────
JOB_ROLES = {
    "Data Analyst":             ["python","sql","pandas","numpy","tableau","power bi","excel","data analysis","data visualization","matplotlib","seaborn"],
    "Web Developer":            ["html","css","javascript","react","nodejs","git","rest api","mongodb","flask","django"],
    "Machine Learning Engineer":["python","machine learning","scikit-learn","tensorflow","pytorch","numpy","pandas","deep learning","keras","xgboost"],
    "Data Scientist":           ["python","machine learning","deep learning","nlp","data analysis","pandas","numpy","sql","tensorflow","pytorch","scikit-learn"],
    "DevOps Engineer":          ["docker","kubernetes","linux","aws","azure","gcp","git","bash"],
    "Backend Developer":        ["java","python","nodejs","flask","django","fastapi","sql","mongodb","redis","rest api","docker"],
    "NLP / AI Engineer":        ["nlp","natural language processing","python","transformers","bert","spacy","nltk","huggingface","pytorch","tensorflow"],
    "Cloud Engineer":           ["aws","azure","gcp","docker","kubernetes","linux","git","bash"],
    "Frontend Developer":       ["html","css","javascript","react","typescript","git","nodejs"],
    "Database Administrator":   ["sql","mysql","postgresql","mongodb","sqlite","redis","data analysis","excel"],
}

def suggest_job_roles(candidate_skills, top_n=3):
    cskills = set(s.lower() for s in candidate_skills)
    scores  = {}
    for role, required in JOB_ROLES.items():
        matched = cskills.intersection(set(r.lower() for r in required))
        if matched:
            scores[role] = {
                "match_pct":      round(len(matched) / len(required) * 100, 1),
                "matched_skills": [s.title() for s in matched],
                "missing_skills": [s.title() for s in set(r.lower() for r in required) - cskills]
            }
    return sorted(scores.items(), key=lambda x: x[1]["match_pct"], reverse=True)[:top_n]

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_pipeline(uploaded_files, job_description, nlp, tokenizer, roberta_model):
    total           = len(uploaded_files)
    expected_skills = extract_expected_skills(job_description)
    results         = []
    progress        = st.progress(0)
    status          = st.empty()
    STAGES          = total + 4

    for i, uploaded_file in enumerate(uploaded_files):
        pct = int((i + 1) / STAGES * 100)
        status.markdown(
            f"<div style='background:rgba(102,126,234,0.07);border:1px solid rgba(102,126,234,0.2);"
            f"border-radius:10px;padding:12px 18px;margin:6px 0;'>"
            f"<div style='font-size:12px;color:#667eea;font-weight:700;margin-bottom:6px;'>🧠 RoBERTa Pipeline</div>"
            f"📄 Parsing & embedding <b style='color:#a78bfa;'>{uploaded_file.name}</b> "
            f"<span style='color:#555;font-size:11px;'>({pct}%)</span></div>",
            unsafe_allow_html=True)
        progress.progress((i + 1) / STAGES)
        if uploaded_file.name.endswith('.pdf'):    text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'): text = extract_text_from_docx(uploaded_file)
        else: continue
        if not text.strip(): continue
        parsed   = parse_resume(text, nlp)
        features = get_embedding(text, tokenizer, roberta_model)
        results.append({'file_name': uploaded_file.name, 'raw_text': text,
                         'parsed': parsed, 'roberta_features': features})
    if not results: return None

    status.markdown(
        "<div style='background:rgba(102,126,234,0.07);border:1px solid rgba(102,126,234,0.2);"
        "border-radius:10px;padding:12px 18px;margin:6px 0;'>"
        "<div style='font-size:12px;color:#667eea;font-weight:700;margin-bottom:6px;'>🧠 RoBERTa Pipeline</div>"
        "📋 Generating <b style='color:#a78bfa;'>Job Description embedding</b> "
        "<span style='color:#555;font-size:11px;'>(85%)</span></div>",
        unsafe_allow_html=True)
    progress.progress(0.85)
    jd_embedding = get_embedding(job_description, tokenizer, roberta_model)
    for r in results:
        sim = cosine_similarity(r['roberta_features'].reshape(1,-1), jd_embedding.reshape(1,-1))[0][0]
        r['similarity_score'] = round(float(sim), 4)
    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    status.markdown(
        "<div style='background:rgba(102,126,234,0.07);border:1px solid rgba(102,126,234,0.2);"
        "border-radius:10px;padding:12px 18px;margin:6px 0;'>"
        "<div style='font-size:12px;color:#667eea;font-weight:700;margin-bottom:6px;'>🧠 RoBERTa Pipeline</div>"
        "🎯 Training <b style='color:#a78bfa;'>ML Classifier</b> on candidate features "
        "<span style='color:#555;font-size:11px;'>(92%)</span></div>",
        unsafe_allow_html=True)
    progress.progress(0.92)

    # ── ML SCORING ───────────────────────────────────────────────────────────────
    # Minimum 5 resumes needed for LR to be reliable.
    # With fewer resumes, similarity alone is used (LR would overfit / give random predictions).
    USE_LR = len(results) >= 5

    if USE_LR:
        X      = np.array([r["roberta_features"] for r in results])
        scores = [r["similarity_score"] for r in results]
        median = np.median(scores)
        y      = np.array([1 if r["similarity_score"] >= median else 0 for r in results])
        if len(set(y)) < 2:
            mid = len(y) // 2; y[:mid] = 1; y[mid:] = 0
        np.random.seed(42); X_aug, y_aug = [], []
        for i in range(len(X)):
            for _ in range(40): X_aug.append(X[i] + np.random.normal(0, 0.01, X[i].shape)); y_aug.append(y[i])
            for _ in range(10): X_aug.append(X[i] + np.random.normal(0, 0.05, X[i].shape)); y_aug.append(1 - y[i])
        X_aug = np.array(X_aug); y_aug = np.array(y_aug)
        X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug)
        scaler   = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
        lr_model = LogisticRegression(max_iter=1000, random_state=42); lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test); y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
        accuracy  = round(accuracy_score(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
        recall    = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob); roc_auc = round(auc(fpr, tpr), 4)
    else:
        # Not enough resumes — use dummy metrics, similarity-only scoring
        accuracy = precision = recall = 0.0
        roc_auc  = 0.0
        fpr      = np.array([0.0, 1.0])
        tpr      = np.array([0.0, 1.0])

    final_results = []
    for r in results:
        sim = r["similarity_score"]
        if USE_LR:
            feat_scaled = scaler.transform(r["roberta_features"].reshape(1,-1))
            prob        = lr_model.predict_proba(feat_scaled)[0][1]
            # Blend: 70% similarity (more reliable) + 30% LR prediction
            final_score = round((sim * 0.7 + float(prob) * 0.3) * 100, 2)
        else:
            # Similarity only — honest and reliable for small datasets
            prob        = sim   # show similarity as prediction too for display
            final_score = round(sim * 100, 2)
        detected    = r["parsed"]["Skills"]
        possessed   = [s for s in detected if s.lower() in [e.lower() for e in expected_skills]]
        missing     = [s for s in expected_skills if s.lower() not in [d.lower() for d in detected]]
        role_suggestions = suggest_job_roles(detected) if final_score < 50 else []
        final_results.append({**r,
            "prediction_prob":   round(float(prob), 4),
            "final_score":       final_score,
            "possessed_skills":  possessed,
            "missing_skills":    missing,
            "role_suggestions":  role_suggestions})
    final_results = sorted(final_results, key=lambda x: x["final_score"], reverse=True)
    progress.progress(1.0)
    status.markdown(
        "<div style='background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.3);"
        "border-radius:10px;padding:12px 18px;margin:6px 0;'>"
        "✅ <b style='color:#4ade80;'>Analysis complete!</b> "
        "<span style='color:#555;font-size:12px;'>All candidates ranked successfully.</span></div>",
        unsafe_allow_html=True)
    return {"final_results": final_results, "jd_embedding": jd_embedding,
            "expected_skills": expected_skills,
            "fpr": fpr, "tpr": tpr,
            "metrics": {"accuracy": accuracy, "precision": precision,
                        "recall": recall, "roc_auc": roc_auc}}

# ── CSV EXPORT ────────────────────────────────────────────────────────────────
def build_csv(final_results):
    rows = []
    for rank, r in enumerate(final_results, 1):
        p = r["parsed"]; score = r["final_score"]
        if score >= 75:   status = "Shortlisted"
        elif score >= 50: status = "Under Review"
        else:             status = "Rejected"
        suggested = "; ".join([role for role, _ in r.get("role_suggestions", [])]) or "N/A"
        rows.append({
            "Rank":                         rank,
            "Name":                         p["Name"],
            "Email":                        p["Email"],
            "Phone":                        p["Phone"],
            "LinkedIn":                     p["LinkedIn"],
            "GitHub":                       p["GitHub"],
            "Similarity %":                 round(r["similarity_score"] * 100, 2),
            "Prediction %":                 round(r["prediction_prob"] * 100, 2),
            "Final Score %":                score,
            "Status":                       status,
            "Detected Skills":              ", ".join(p["Skills"]),
            "Possessed Skills (JD Match)":  ", ".join(r.get("possessed_skills", [])) or "None",
            "Missing Skills":               ", ".join(r.get("missing_skills", []))   or "None",
            "Education":                    " | ".join(p["Education"]),
            "Experience":                   " | ".join(p["Experience"]),
            "Projects":                     " | ".join(p["Projects"]),
            "Certifications":               " | ".join(p["Certifications"]),
            "Achievements":                 " | ".join(p["Achievements"]),
            "Suggested Roles (Rejected)":   suggested,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:52px; filter:drop-shadow(0 0 20px #667eea);'>🤖</div>
        <div style='font-size:22px; font-weight:800; background:linear-gradient(135deg,#667eea,#f093fb);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>HIRE RIGHT AI</div>
        <div style='font-size:10px; color:#555; margin-top:4px; letter-spacing:2px; text-transform:uppercase;'>Intelligence That Hires Right</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["🏠  Home","📤  Upload & Analyze","📊  Dashboard","👤  Candidate Detail","📄  Report"])
    st.markdown("---")
    st.markdown("<div style='font-size:11px; color:#333; text-align:center;'>9 Modules • MPNet Powered<br/>Cosine Similarity + LR Classifier</div>", unsafe_allow_html=True)

PURPLE="#667eea"; PINK="#f093fb"; GREEN="#4ade80"; YELLOW="#fbbf24"; RED="#f87171"

def set_dark_chart(fig, ax_list):
    fig.patch.set_facecolor("#0f0c29")
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor("#0f0c29"); ax.tick_params(colors="#555")
        ax.xaxis.label.set_color("#555"); ax.yaxis.label.set_color("#555")
        ax.title.set_color("#a78bfa")
        for spine in ax.spines.values(): spine.set_edgecolor("#222")

# ════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style='
        background: linear-gradient(160deg, rgba(102,126,234,0.18) 0%, rgba(118,75,162,0.22) 50%, rgba(240,147,251,0.12) 100%);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 24px;
        padding: 70px 60px 60px 60px;
        text-align: center;
        margin-bottom: 36px;
        position: relative;
        box-shadow: 0 0 60px rgba(102,126,234,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    '>
        <div style='position:absolute;top:-40px;left:50%;transform:translateX(-50%);
                    width:220px;height:220px;background:radial-gradient(circle,rgba(102,126,234,0.18) 0%,transparent 70%);
                    border-radius:50%;pointer-events:none;'></div>
        <div style='font-size:90px; margin-bottom:20px; line-height:1; filter:drop-shadow(0 0 30px #667eea) drop-shadow(0 0 60px rgba(102,126,234,0.5)) drop-shadow(0 0 10px #fff); opacity:1;'>🤖</div>
        <h1 style='
            font-size: 62px;
            font-weight: 900;
            margin: 0 0 10px 0;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #a78bfa 0%, #667eea 40%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 20px rgba(102,126,234,0.4));
            line-height: 1.1;
        '>HIRE RIGHT AI</h1>
        <h2 style='
            font-size: 20px;
            font-weight: 500;
            margin: 0 0 18px 0;
            color: #c4b5fd !important;
            -webkit-text-fill-color: #c4b5fd !important;
            letter-spacing: 0.5px;
        '>AI Powered Intelligent Recruitment & Candidate Analytics System</h2>
        <div style='width:80px;height:2px;background:linear-gradient(90deg,transparent,#667eea,#f093fb,transparent);
                    margin:0 auto 18px auto;border-radius:2px;'></div>
        <p style='
            color: rgba(255,255,255,0.25);
            font-size: 12px;
            margin: 0;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        '>Automated Resume Parsing • Advanced Skill Detection • Intelligent Candidate Matching</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📋 How It Works — 5 Simple Steps</div>", unsafe_allow_html=True)
    steps = [
        ("📄", "Step 1", "Upload Resumes",       "Upload single or multiple PDF / DOCX resumes through the Upload & Analyze page."),
        ("📋", "Step 2", "Paste Job Description","Enter the job description you want to match candidates against."),
        ("🧠", "Step 3", "AI Analysis",          "The system parses resumes, detects skills, and generates semantic embeddings using a sentence transformer model."),
        ("📊", "Step 4", "Ranking & Scoring",    "Candidates are ranked using cosine similarity + ML prediction. Shortlisted, Under Review, and Rejected statuses are assigned automatically."),
        ("📄", "Step 5", "View Reports",         "Explore the Dashboard for analytics, Candidate Detail for deep insights, and download the PDF report."),
    ]
    cols = st.columns(5)
    for col, (icon, step, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""<div class='glass-card' style='text-align:center;min-height:200px;'>
                <div style='font-size:32px;'>{icon}</div>
                <div style='font-size:10px;color:#667eea;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:6px 0 2px 0;'>{step}</div>
                <div style='font-size:13px;font-weight:700;color:#a78bfa;margin-bottom:8px;'>{title}</div>
                <div style='font-size:11px;color:#444;line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💡 Why HireRight AI?</div>", unsafe_allow_html=True)
    w1,w2,w3,w4 = st.columns(4)
    for col,(icon,title,desc) in zip([w1,w2,w3,w4],[
        ("⏱️","Saves Time",  "Screens 100 resumes in minutes"),
        ("⚖️","Bias Free",   "AI-driven objective evaluation"),
        ("🎯","Accurate",    "RoBERTa semantic matching"),
        ("📊","Insightful",  "Visual analytics & skill trends")]):
        with col:
            st.markdown(f"<div class='glass-card' style='text-align:center;'><div style='font-size:30px;'>{icon}</div><div style='font-size:13px;font-weight:700;color:#a78bfa;margin:8px 0 4px 0;'>{title}</div><div style='font-size:12px;color:#444;'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Go to **Upload & Analyze** to get started!")

# ════════════════════════════════════════════════════════════
# UPLOAD & ANALYZE
# ════════════════════════════════════════════════════════════
elif page == "📤  Upload & Analyze":
    st.markdown("<div class='section-header'>📤 Upload Resumes & Analyze</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'><h3>📄 Upload Resumes</h3><p style='color:#555;font-size:13px;'>Upload PDF or DOCX resumes for AI-powered analysis.</p></div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Drop resumes here", type=["pdf","docx"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} resume(s) uploaded!")
            for f in uploaded_files:
                icon = "📕" if f.name.endswith(".pdf") else "📘"
                st.markdown(f"<div style='background:rgba(102,126,234,0.08);border:1px solid rgba(102,126,234,0.2);border-radius:8px;padding:8px 14px;margin:4px 0;font-size:13px;color:#a78bfa;'>{icon} {f.name} — {round(f.size/(1024*1024),2)} MB</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>📋 Job Description</h3><p style='color:#555;font-size:13px;'>Paste the job description to match against.</p></div>", unsafe_allow_html=True)
        job_description = st.text_area("Paste Job Description", height=200, placeholder="We are looking for a Data Scientist with Python, ML, NLP...")
        if job_description:
            st.info(f"📝 {len(job_description.split())} words")
            exp_skills = extract_expected_skills(job_description)
            if exp_skills and exp_skills != ["Not Found"]:
                exp_html = "".join([f"<span class='skill-tag-expected'>{s}</span>" for s in exp_skills])
                st.markdown(f"<div style='margin-top:8px;'><div style='font-size:12px;color:#4ade80;font-weight:700;margin-bottom:4px;'>✅ Expected Skills from JD ({len(exp_skills)})</div>{exp_html}</div>", unsafe_allow_html=True)

    # ── PARSE BUTTON ──
    if uploaded_files:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Parse Resumes — Show Extracted Data", use_container_width=True):
            with st.spinner("🤖 Loading AI model..."):
                nlp, tokenizer, roberta_model = load_models()
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📋 Resume Parsing Output</div>", unsafe_allow_html=True)
            jd_expected = extract_expected_skills(job_description) if job_description else []

            for uploaded_file in uploaded_files:
                uploaded_file.seek(0)
                if uploaded_file.name.endswith(".pdf"):    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".docx"): text = extract_text_from_docx(uploaded_file)
                else: continue
                if not text.strip(): st.warning(f"⚠️ Could not extract text from {uploaded_file.name}"); continue
                parsed = parse_resume(text, nlp)

                detected   = parsed['Skills']
                possessed  = [s for s in detected if s.lower() in [e.lower() for e in jd_expected]]
                missing    = [s for s in jd_expected if s.lower() not in [d.lower() for d in detected]]

                det_html  = "".join([f"<span class='skill-tag'>{s}</span>" for s in detected])
                poss_html = "".join([f"<span class='skill-tag-expected'>{s}</span>" for s in possessed]) if possessed else "<span style='color:#555;font-size:12px;'>None matched</span>"
                miss_html = "".join([f"<span class='skill-tag-missing'>{s}</span>" for s in missing])   if missing   else "<span style='color:#4ade80;font-size:12px;'>✅ All present</span>"

                st.markdown(f"""
                <div class='parse-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;'>
                        <div>
                            <div style='font-size:22px; font-weight:800; color:#ccc;'>👤 {parsed['Name']}</div>
                            <div style='font-size:12px; color:#555; margin-top:4px;'>📁 {uploaded_file.name}</div>
                        </div>
                        <span style='background:rgba(102,126,234,0.2); color:#a78bfa; padding:6px 16px;
                                     border-radius:20px; font-size:12px; font-weight:700; border:1px solid rgba(102,126,234,0.3);'>✅ PARSED</span>
                    </div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px;'>
                        <div style='background:rgba(255,255,255,0.02); border-radius:8px; padding:10px;'>
                            <span style='color:#667eea; font-size:11px; font-weight:700;'>📧 EMAIL</span><br/>
                            <span style='color:#aaa; font-size:13px;'>{parsed['Email']}</span>
                        </div>
                        <div style='background:rgba(255,255,255,0.02); border-radius:8px; padding:10px;'>
                            <span style='color:#667eea; font-size:11px; font-weight:700;'>📞 PHONE</span><br/>
                            <span style='color:#aaa; font-size:13px;'>{parsed['Phone']}</span>
                        </div>
                        <div style='background:rgba(255,255,255,0.02); border-radius:8px; padding:10px;'>
                            <span style='color:#667eea; font-size:11px; font-weight:700;'>🔗 LINKEDIN</span><br/>
                            <span style='color:#aaa; font-size:13px;'>{parsed['LinkedIn']}</span>
                        </div>
                        <div style='background:rgba(255,255,255,0.02); border-radius:8px; padding:10px;'>
                            <span style='color:#667eea; font-size:11px; font-weight:700;'>🐙 GITHUB</span><br/>
                            <span style='color:#aaa; font-size:13px;'>{parsed['GitHub']}</span>
                        </div>
                    </div>
                    <div style='margin-bottom:10px;'>
                        <div style='color:#a78bfa; font-weight:700; font-size:13px; margin-bottom:6px;'>🛠 SKILLS DETECTED ({len(detected)})</div>
                        <div>{det_html}</div>
                    </div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                        <div style='background:rgba(74,222,128,0.04); border-radius:8px; padding:10px; border:1px solid rgba(74,222,128,0.15);'>
                            <div style='color:#4ade80; font-weight:700; font-size:12px; margin-bottom:6px;'>✅ POSSESSED SKILLS (JD Match)</div>
                            <div>{poss_html}</div>
                        </div>
                        <div style='background:rgba(248,113,113,0.04); border-radius:8px; padding:10px; border:1px solid rgba(248,113,113,0.15);'>
                            <div style='color:#f87171; font-weight:700; font-size:12px; margin-bottom:6px;'>❌ MISSING SKILLS (from JD)</div>
                            <div>{miss_html}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"🎓 Education — {parsed['Name']}"):
                    for item in parsed['Education']:
                        st.markdown(f"<div style='background:rgba(102,126,234,0.05);border-left:3px solid #667eea;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
                with st.expander(f"💼 Experience — {parsed['Name']}"):
                    for item in parsed['Experience']:
                        st.markdown(f"<div style='background:rgba(251,191,36,0.05);border-left:3px solid #fbbf24;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
                with st.expander(f"💻 Projects — {parsed['Name']}"):
                    for item in parsed['Projects']:
                        st.markdown(f"<div style='background:rgba(240,147,251,0.05);border-left:3px solid #f093fb;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
                with st.expander(f"📜 Certifications — {parsed['Name']}"):
                    for item in parsed['Certifications']:
                        st.markdown(f"<div style='background:rgba(96,165,250,0.05);border-left:3px solid #60a5fa;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
                with st.expander(f"🏆 Achievements — {parsed['Name']}"):
                    for item in parsed['Achievements']:
                        st.markdown(f"<div style='background:rgba(74,222,128,0.05);border-left:3px solid #4ade80;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)

    # ── ANALYZE BUTTON ──
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_files and job_description:
        if st.button("🤖 Analyze & Rank Candidates", use_container_width=True):
            for f in uploaded_files: f.seek(0)
            with st.spinner("Loading AI model..."): nlp, tokenizer, roberta_model = load_models()
            st.markdown("---"); st.markdown("### ⚡ Processing Pipeline")
            result = run_pipeline(uploaded_files, job_description, nlp, tokenizer, roberta_model)
            if result:
                st.session_state["results2"]        = result
                st.session_state["job_description2"] = job_description
                st.success("✅ Done! Go to Dashboard."); st.balloons()
            else: st.error("❌ Could not process resumes.")
    else:
        if not uploaded_files:  st.warning("⚠️ Please upload at least one resume.")
        if not job_description: st.warning("⚠️ Please enter a job description to rank candidates.")

# ════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "📊  Dashboard":
    if "results2" not in st.session_state: st.warning("⚠️ No results. Go to Upload & Analyze first."); st.stop()
    data = st.session_state["results2"]; final_results = data["final_results"]; metrics = data["metrics"]
    st.markdown("<div class='section-header'>📊 Analytics Dashboard</div>", unsafe_allow_html=True)
    shortlisted  = sum(1 for r in final_results if r["final_score"] >= 75)
    under_review = sum(1 for r in final_results if 50 <= r["final_score"] < 75)
    rejected     = sum(1 for r in final_results if r["final_score"] < 50)
    top_score    = max(r["final_score"] for r in final_results)
    m1,m2,m3,m4,m5 = st.columns(5)
    for col,val,label,sub in zip([m1,m2,m3,m4,m5],
        [len(final_results),shortlisted,under_review,rejected,f"{top_score}%"],
        ["👥 Total","✅ Shortlisted","⚠️ Review","❌ Rejected","🏆 Top Score"],
        ["Candidates","Score ≥75%","50–74%","<50%","Best Match"]):
        with col:
            st.markdown(f"<div class='stat-card'><div style='font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;'>{label}</div><div class='stat-val'>{val}</div><div class='stat-sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🏅 Candidate Rankings</div>", unsafe_allow_html=True)
    for rank,r in enumerate(final_results,1):
        p = r["parsed"]; score = r["final_score"]
        if score>=75:   card_class="shortlisted"; badge_cls="badge-green"; status_txt="✅ Shortlisted"; score_color="#4ade80"
        elif score>=50: card_class="under-review"; badge_cls="badge-orange"; status_txt="⚠️ Under Review"; score_color="#fbbf24"
        else:           card_class="rejected";     badge_cls="badge-red";    status_txt="❌ Rejected";    score_color="#f87171"
        skills_html  = "".join([f"<span class='skill-tag'>{s}</span>" for s in p["Skills"][:8]])
        possessed    = r.get("possessed_skills", [])
        missing      = r.get("missing_skills", [])
        poss_html    = "".join([f"<span class='skill-tag-expected'>{s}</span>" for s in possessed]) if possessed else "<span style='color:#555;font-size:12px;'>None matched</span>"
        miss_html    = "".join([f"<span class='skill-tag-missing'>{s}</span>"  for s in missing])   if missing   else "<span style='color:#4ade80;font-size:12px;'>✅ All present</span>"
        st.markdown(f"""<div class='glass-card {card_class}'>
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'>
                <div><span style='font-size:18px;font-weight:800;color:#ccc;'>#{rank} {p['Name']}</span>
                <span class='badge {badge_cls}' style='margin-left:10px;'>{status_txt}</span></div>
                <div style='text-align:right;'><span style='font-size:28px;font-weight:800;color:{score_color};'>{score}%</span>
                <div style='font-size:11px;color:#444;'>Final Score</div></div>
            </div>
            <div style='margin:8px 0;font-size:12px;color:#444;'>📧 {p['Email']} &nbsp;|&nbsp; 📞 {p['Phone']} &nbsp;|&nbsp; 🤖 Sim: {round(r['similarity_score']*100,1)}% &nbsp;|&nbsp; 🧠 Pred: {round(r['prediction_prob']*100,1)}%</div>
            <div style='margin-top:6px;'>{skills_html}</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;'>
                <div style='background:rgba(74,222,128,0.04);border-radius:8px;padding:10px;border:1px solid rgba(74,222,128,0.15);'>
                    <div style='color:#4ade80;font-weight:700;font-size:11px;margin-bottom:6px;'>✅ POSSESSED SKILLS (JD Match)</div>
                    <div>{poss_html}</div>
                </div>
                <div style='background:rgba(248,113,113,0.04);border-radius:8px;padding:10px;border:1px solid rgba(248,113,113,0.15);'>
                    <div style='color:#f87171;font-weight:700;font-size:11px;margin-bottom:6px;'>❌ MISSING SKILLS (from JD)</div>
                    <div>{miss_html}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        if score < 35 and r.get("role_suggestions"):
            st.markdown("<div style='margin:10px 0 6px 0;background:rgba(102,126,234,0.08);border:1px solid rgba(102,126,234,0.3);border-radius:10px;padding:10px 18px;font-size:13px;font-weight:700;color:#a78bfa;'>💡 Suggested Job Roles</div>", unsafe_allow_html=True)
            for role, info in r["role_suggestions"]:
                m_html = "".join([f"<span class='skill-tag-expected'>{s}</span>" for s in info['matched_skills']])
                g_html = "".join([f"<span class='skill-tag-missing'>{s}</span>" for s in info['missing_skills'][:5]])
                st.markdown(f"""<div class='role-suggest-card' style='margin-left:16px;'>
                    <div style='font-size:14px;font-weight:700;color:#a78bfa;'>🎯 {role} — <span style='color:#4ade80;'>{info['match_pct']}% match</span></div>
                    <div style='margin-top:6px;font-size:12px;color:#555;'>Matched: {m_html}</div>
                    <div style='margin-top:4px;font-size:12px;color:#555;'>To Improve: {g_html}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📈 Visual Analytics</div>", unsafe_allow_html=True)
    names        = [r["parsed"]["Name"].split()[0] for r in final_results]
    final_scores = [r["final_score"] for r in final_results]
    sim_scores   = [r["similarity_score"]*100 for r in final_results]
    pred_scores  = [r["prediction_prob"]*100 for r in final_results]
    bar_colors   = [GREEN if s>=75 else YELLOW if s>=50 else RED for s in final_scores]
    all_skills   = []
    for r in final_results: all_skills.extend(r["parsed"]["Skills"])
    ch1,ch2 = st.columns(2)
    with ch1:
        fig,ax = plt.subplots(figsize=(7,4)); set_dark_chart(fig,ax)
        bars = ax.barh(names,final_scores,color=bar_colors,edgecolor="#0f0c29",height=0.5)
        ax.set_xlim(0,115); ax.set_xlabel("Final Score (%)",color="#555")
        ax.set_title("🎯 Candidate Final Scores",color="#a78bfa",fontweight="bold")
        for bar,score in zip(bars,final_scores):
            ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f"{score}%",va="center",fontsize=9,fontweight="bold",color="#ccc")
        ax.grid(axis="x",alpha=0.1,color="#667eea"); st.pyplot(fig); plt.close()
    with ch2:
        fig,ax = plt.subplots(figsize=(7,4)); set_dark_chart(fig,ax)
        x=np.arange(len(names)); width=0.35
        ax.bar(x-width/2,sim_scores,width,label="Similarity",color=PURPLE,alpha=0.85,edgecolor="#0f0c29")
        ax.bar(x+width/2,pred_scores,width,label="Prediction",color=PINK,alpha=0.85,edgecolor="#0f0c29")
        ax.set_xticks(x); ax.set_xticklabels(names,fontsize=9,color="#555")
        ax.set_ylabel("Score (%)",color="#555"); ax.set_title("🤖 Similarity vs Prediction",color="#a78bfa",fontweight="bold")
        ax.legend(fontsize=9,facecolor="#0f0c29",labelcolor="#aaa",edgecolor="#333")
        ax.grid(axis="y",alpha=0.1,color="#667eea"); st.pyplot(fig); plt.close()
    ch3,ch4 = st.columns(2)
    with ch3:
        fig,ax = plt.subplots(figsize=(7,4)); set_dark_chart(fig,ax)
        pie_vals=[v for v in [shortlisted,under_review,rejected] if v>0]
        pie_labels=[l for l,v in zip([f"Shortlisted\n({shortlisted})",f"Under Review\n({under_review})",f"Rejected\n({rejected})"],[shortlisted,under_review,rejected]) if v>0]
        pie_colors=[c for c,v in zip([GREEN,YELLOW,RED],[shortlisted,under_review,rejected]) if v>0]
        wedges,texts,autotexts = ax.pie(pie_vals,labels=pie_labels,colors=pie_colors,autopct="%1.0f%%",startangle=90,wedgeprops={"edgecolor":"#0f0c29","linewidth":2})
        for t in texts: t.set_color("#555")
        for t in autotexts: t.set_color("white"); t.set_fontweight("bold")
        ax.set_title("📊 Status Distribution",color="#a78bfa",fontweight="bold"); st.pyplot(fig); plt.close()
    with ch4:
        fig,ax = plt.subplots(figsize=(7,4)); set_dark_chart(fig,ax)
        top_skills = Counter(all_skills).most_common(10)
        snames=[s[0] for s in top_skills]; svals=[s[1] for s in top_skills]
        colors_bar=[plt.cm.cool(i/len(snames)) for i in range(len(snames))]
        ax.barh(snames[::-1],svals[::-1],color=colors_bar[::-1],edgecolor="#0f0c29")
        ax.set_xlabel("Count",color="#555"); ax.set_title("🛠 Top Skills",color="#a78bfa",fontweight="bold")
        ax.grid(axis="x",alpha=0.1,color="#667eea"); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>🔥 Skill Gap Analysis</div>", unsafe_allow_html=True)
    expected_skills = data.get("expected_skills", ["Python","Machine Learning","Deep Learning","Nlp","Tensorflow","Scikit-Learn","Pandas","Sql","Data Science","Docker"])
    heatmap_data=[]; cnames_short=[]
    for r in final_results:
        cskills=[s.lower() for s in r["parsed"]["Skills"]]
        heatmap_data.append([1 if s.lower() in cskills else 0 for s in expected_skills])
        cnames_short.append(r["parsed"]["Name"].split()[0])
    fig,ax = plt.subplots(figsize=(12,4)); fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("#0f0c29")
    sns.heatmap(np.array(heatmap_data),annot=True,fmt="d",xticklabels=expected_skills,yticklabels=cnames_short,
                cmap="PuBu",linewidths=0.5,linecolor="#0f0c29",cbar=False,ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",fontsize=10,color="#555")
    ax.set_yticklabels(ax.get_yticklabels(),color="#aaa")
    ax.set_title("Skill Gap (1=Has, 0=Missing)",color="#a78bfa",fontweight="bold",pad=10); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>📈 ROC Curve</div>", unsafe_allow_html=True)
    fpr=data["fpr"]; tpr=data["tpr"]
    fig,ax = plt.subplots(figsize=(8,4)); set_dark_chart(fig,ax)
    ax.plot(fpr,tpr,color=PURPLE,lw=2,label=f"ROC Curve (AUC={data['metrics']['roc_auc']})")
    ax.plot([0,1],[0,1],color="#333",lw=1,linestyle="--"); ax.fill_between(fpr,tpr,alpha=0.1,color=PURPLE)
    ax.set_xlabel("False Positive Rate",color="#555"); ax.set_ylabel("True Positive Rate",color="#555")
    ax.set_title("ROC Curve — Suitability Prediction",color="#a78bfa",fontweight="bold")
    ax.legend(loc="lower right",facecolor="#0f0c29",labelcolor="#aaa",edgecolor="#333"); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
# CANDIDATE DETAIL
# ════════════════════════════════════════════════════════════
elif page == "👤  Candidate Detail":
    if "results2" not in st.session_state: st.warning("⚠️ No results. Go to Upload & Analyze first."); st.stop()
    data = st.session_state["results2"]; final_results = data["final_results"]
    st.markdown("<div class='section-header'>👤 Candidate Detail View</div>", unsafe_allow_html=True)
    names    = [r["parsed"]["Name"] for r in final_results]
    selected = st.selectbox("Select Candidate", names)
    r = next(x for x in final_results if x["parsed"]["Name"] == selected)
    p = r["parsed"]; score = r["final_score"]
    if score>=55:   status_txt="✅ Shortlisted"; score_color="#4ade80"; card_class="shortlisted"
    elif score>=35: status_txt="⚠️ Under Review"; score_color="#fbbf24"; card_class="under-review"
    else:           status_txt="❌ Rejected";     score_color="#f87171"; card_class="rejected"

    st.markdown(f"""<div class='glass-card {card_class}'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
                <div style='font-size:26px;font-weight:800;color:#ccc;'>{p['Name']}</div>
                <div style='font-size:13px;color:#444;margin-top:4px;'>📧 {p['Email']} &nbsp;|&nbsp; 📞 {p['Phone']}</div>
                <div style='font-size:13px;color:#444;margin-top:4px;'>🔗 {p['LinkedIn']} &nbsp;|&nbsp; 🐙 {p['GitHub']}</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:44px;font-weight:800;color:{score_color};'>{score}%</div>
                <div style='font-size:14px;font-weight:700;color:{score_color};'>{status_txt}</div>
            </div>
        </div></div>""", unsafe_allow_html=True)

    sc1,sc2,sc3 = st.columns(3)
    for col,val,label in zip([sc1,sc2,sc3],
        [f"{round(r['similarity_score']*100,2)}%",f"{round(r['prediction_prob']*100,2)}%",f"{score}%"],
        ["🤖 RoBERTa Similarity","🧠 LR Prediction","🎯 Final Score"]):
        with col:
            st.markdown(f"<div class='stat-card'><div style='font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;'>{label}</div><div class='stat-val'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🛠 Skills Analysis</div>", unsafe_allow_html=True)
    possessed = r.get("possessed_skills", [])
    missing   = r.get("missing_skills", [])
    poss_html = "".join([f"<span class='skill-tag-expected'>{s}</span>" for s in possessed]) if possessed else "<span style='color:#555;font-size:12px;'>None matched JD</span>"
    miss_html = "".join([f"<span class='skill-tag-missing'>{s}</span>"  for s in missing])   if missing   else "<span style='color:#4ade80;font-size:12px;'>✅ All skills present</span>"
    det_html  = "".join([f"<span class='skill-tag'>{s}</span>" for s in p["Skills"]])
    st.markdown(f"""
    <div style='background:rgba(102,126,234,0.04);border-radius:8px;padding:12px;border:1px solid rgba(102,126,234,0.15);margin-bottom:10px;'>
        <div style='color:#a78bfa;font-weight:700;font-size:12px;margin-bottom:8px;'>🔍 ALL DETECTED SKILLS ({len(p["Skills"])})</div>
        <div>{det_html}</div>
    </div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
        <div style='background:rgba(74,222,128,0.04);border-radius:8px;padding:12px;border:1px solid rgba(74,222,128,0.2);'>
            <div style='color:#4ade80;font-weight:700;font-size:12px;margin-bottom:8px;'>✅ POSSESSED SKILLS (JD Match)</div>
            <div>{poss_html}</div>
        </div>
        <div style='background:rgba(248,113,113,0.04);border-radius:8px;padding:12px;border:1px solid rgba(248,113,113,0.2);'>
            <div style='color:#f87171;font-weight:700;font-size:12px;margin-bottom:8px;'>❌ MISSING SKILLS (from JD)</div>
            <div>{miss_html}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🎓 Education"):
        for item in p["Education"]:
            st.markdown(f"<div style='background:rgba(102,126,234,0.05);border-left:3px solid #667eea;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
    with st.expander("💼 Experience"):
        for item in p["Experience"]:
            st.markdown(f"<div style='background:rgba(251,191,36,0.05);border-left:3px solid #fbbf24;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
    with st.expander("💻 Projects"):
        for item in p["Projects"]:
            st.markdown(f"<div style='background:rgba(240,147,251,0.05);border-left:3px solid #f093fb;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
    with st.expander("📜 Certifications"):
        for item in p["Certifications"]:
            st.markdown(f"<div style='background:rgba(96,165,250,0.05);border-left:3px solid #60a5fa;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)
    with st.expander("🏆 Achievements"):
        for item in p["Achievements"]:
            st.markdown(f"<div style='background:rgba(74,222,128,0.05);border-left:3px solid #4ade80;border-radius:6px;padding:8px 14px;margin:4px 0;font-size:13px;color:#888;'>• {item}</div>", unsafe_allow_html=True)

    if score < 35 and r.get("role_suggestions"):
        st.markdown("<div class='section-header'>💡 Suggested Job Roles</div>", unsafe_allow_html=True)
        for role, info in r["role_suggestions"]:
            m_html = "".join([f"<span class='skill-tag-expected'>{s}</span>" for s in info['matched_skills']])
            g_html = "".join([f"<span class='skill-tag-missing'>{s}</span>" for s in info['missing_skills'][:5]])
            st.markdown(f"""<div class='role-suggest-card'>
                <div style='font-size:15px;font-weight:700;color:#a78bfa;'>🎯 {role}
                    <span style='color:#4ade80;font-size:13px;margin-left:8px;'>{info['match_pct']}% skill match</span>
                </div>
                <div style='margin-top:8px;font-size:12px;color:#555;'>✅ Matched: {m_html}</div>
                <div style='margin-top:4px;font-size:12px;color:#555;'>📚 Skills to Develop: {g_html}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════
elif page == "📄  Report":
    if "results2" not in st.session_state: st.warning("⚠️ No results. Go to Upload & Analyze first."); st.stop()
    data = st.session_state["results2"]; final_results = data["final_results"]; metrics = data["metrics"]
    shortlisted = sum(1 for r in final_results if r["final_score"] >= 75)
    st.markdown("<div class='section-header'>📄 Recruitment Report</div>", unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4)
    for col,val,label in zip([mc1,mc2,mc3,mc4],
        [f"{metrics['accuracy']}%",f"{metrics['precision']}%",f"{metrics['recall']}%",metrics['roc_auc']],
        ["✅ Accuracy","🎯 Precision","🔁 Recall","📈 ROC AUC"]):
        with col:
            st.markdown(f"<div class='stat-card'><div style='font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;'>{label}</div><div class='stat-val'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📋 Candidate Summary</div>", unsafe_allow_html=True)
    table_data = []
    for rank,r in enumerate(final_results,1):
        p=r["parsed"]; score=r["final_score"]
        if score>=55:   status="✅ Shortlisted"
        elif score>=35: status="⚠️ Under Review"
        else:           status="❌ Rejected"
        table_data.append({
            "Rank": rank, "Name": p["Name"], "Email": p["Email"],
            "Similarity %":   round(r["similarity_score"]*100,2),
            "Prediction %":   round(r["prediction_prob"]*100,2),
            "Final Score %":  score, "Status": status,
            "Possessed Skills": ", ".join(r.get("possessed_skills",[])[:4]) or "—",
            "Missing Skills":   ", ".join(r.get("missing_skills",[])[:4])   or "—",
        })
    df = pd.DataFrame(table_data)
    styled_df = df.style.set_properties(**{
        'background-color': '#0f0c29',
        'color': '#a78bfa',
        'border': '1px solid rgba(167,139,250,0.15)',
        'font-size': '13px',
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', 'rgba(167,139,250,0.15)'),
            ('color', '#a78bfa'),
            ('font-weight', '700'),
            ('font-size', '13px'),
            ('border', '1px solid rgba(167,139,250,0.2)'),
            ('text-align', 'center'),
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', 'rgba(167,139,250,0.08)'),
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
        ]},
    ])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # CSV: auto-save silently to disk (backend only — no UI exposure)
    try:
        import os
        csv_dir = "hireright_exports"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"candidates_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_path, "wb") as f:
            f.write(build_csv(final_results))
    except Exception:
        pass  # silent — never surface CSV errors in UI
    st.markdown("<br>", unsafe_allow_html=True)

    # PDF Export
    st.markdown("<div class='section-header'>📄 Download PDF Report</div>", unsafe_allow_html=True)
    if st.button("📄 Generate & Download PDF", use_container_width=True):
        with st.spinner("⚡ Generating PDF..."):
            PUR=rl_colors.HexColor("#667eea"); LPINK=rl_colors.HexColor("#f093fb")
            GRN=rl_colors.HexColor("#4ade80"); ORN=rl_colors.HexColor("#fbbf24")
            RD=rl_colors.HexColor("#f87171");  WHITE=rl_colors.white
            DARK=rl_colors.HexColor("#0f0c29"); LGRAY=rl_colors.HexColor("#1a1040")
            TEAL=rl_colors.HexColor("#22d3ee")
            buffer=io.BytesIO()
            doc=SimpleDocTemplate(buffer,pagesize=letter,
                leftMargin=0.65*inch,rightMargin=0.65*inch,
                topMargin=0.65*inch,bottomMargin=0.6*inch)
            def S(name,**kw): return ParagraphStyle(name,**kw)
            def sp(h=8):      return Spacer(1,h)
            def page_border(canvas,doc):
                canvas.saveState(); w,h=letter
                canvas.setStrokeColor(PUR); canvas.setLineWidth(3); canvas.rect(18,18,w-36,h-36)
                canvas.setStrokeColor(LPINK); canvas.setLineWidth(1); canvas.rect(24,24,w-48,h-48)
                canvas.setFont("Helvetica",8); canvas.setFillColor(rl_colors.HexColor("#555555"))
                canvas.drawCentredString(w/2,30,"HireRight AI — Intelligence That Hires Right.")
                canvas.drawRightString(w-30,30,f"Page {doc.page}"); canvas.restoreState()

            story=[]
            # Cover
            cover=Table([[Paragraph("🤖  HireRight AI",S("cv",fontName="Helvetica-Bold",fontSize=28,textColor=LPINK,alignment=TA_CENTER))]],colWidths=[doc.width])
            cover.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK),("TOPPADDING",(0,0),(-1,-1),35),("BOTTOMPADDING",(0,0),(-1,-1),35)]))
            story.append(cover)
            bar=Table([[""]],colWidths=[doc.width])
            bar.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PUR),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            story.append(bar); story.append(sp(20))
            story.append(Paragraph("AI-Powered Intelligent Recruitment",S("t1",fontName="Helvetica-Bold",fontSize=18,textColor=PUR,alignment=TA_CENTER)))
            story.append(Paragraph("& Candidate Analytics System",S("t2",fontName="Helvetica-Bold",fontSize=16,textColor=PUR,alignment=TA_CENTER)))
            story.append(sp(6))
            story.append(Paragraph("Intelligence That Hires Right.",S("t3",fontName="Helvetica-Oblique",fontSize=12,textColor=LPINK,alignment=TA_CENTER)))
            story.append(Paragraph("Powered by MPNet — Best-in-Class Sentence Similarity",S("t3b",fontName="Helvetica-Oblique",fontSize=10,textColor=rl_colors.HexColor("#667eea"),alignment=TA_CENTER)))
            story.append(HRFlowable(width="100%",thickness=1.5,color=PUR,spaceAfter=16,spaceBefore=10))
            story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y — %I:%M %p')}",S("t4",fontName="Helvetica",fontSize=10,textColor=rl_colors.HexColor("#555555"),alignment=TA_CENTER)))
            story.append(sp(20))
            stats_t=Table([[
                Paragraph(f"<b>{len(final_results)}</b><br/>Candidates",S("s1",fontName="Helvetica",fontSize=12,textColor=WHITE,alignment=TA_CENTER)),
                Paragraph(f"<b>{shortlisted}</b><br/>Shortlisted",S("s2",fontName="Helvetica",fontSize=12,textColor=WHITE,alignment=TA_CENTER)),
                Paragraph(f"<b>{metrics['accuracy']}%</b><br/>Accuracy",S("s3",fontName="Helvetica",fontSize=12,textColor=WHITE,alignment=TA_CENTER)),
                Paragraph(f"<b>{metrics['roc_auc']}</b><br/>ROC AUC",S("s4",fontName="Helvetica",fontSize=12,textColor=WHITE,alignment=TA_CENTER)),
            ]],colWidths=[doc.width/4]*4)
            stats_t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(0,-1),PUR),("BACKGROUND",(1,0),(1,-1),GRN),
                ("BACKGROUND",(2,0),(2,-1),LPINK),("BACKGROUND",(3,0),(3,-1),rl_colors.HexColor("#a855f7")),
                ("TOPPADDING",(0,0),(-1,-1),14),("BOTTOMPADDING",(0,0),(-1,-1),14),("LINEAFTER",(0,0),(2,-1),1,WHITE)]))
            story.append(stats_t); story.append(PageBreak())

            # Candidate Rankings
            story.append(Paragraph("Candidate Rankings",S("h1",fontName="Helvetica-Bold",fontSize=14,textColor=PUR,spaceAfter=8)))
            story.append(HRFlowable(width="100%",thickness=1.5,color=PUR,spaceAfter=10))
            tdata=[[Paragraph(f"<b>{h}</b>",S(f"th{i}",fontName="Helvetica-Bold",fontSize=9,textColor=WHITE,alignment=TA_CENTER))
                    for i,h in enumerate(["Rank","Name","Sim %","Pred %","Final","Status"])]]
            for rank,r in enumerate(final_results,1):
                sc=r["final_score"]
                if sc>=75: sc_col=GRN; sc_txt="Shortlisted"
                elif sc>=50: sc_col=ORN; sc_txt="Under Review"
                else: sc_col=RD; sc_txt="Rejected"
                tdata.append([
                    Paragraph(str(rank),S(f"ra{rank}",fontName="Helvetica-Bold",fontSize=9,alignment=TA_CENTER)),
                    Paragraph(r["parsed"]["Name"],S(f"rb{rank}",fontName="Helvetica",fontSize=9)),
                    Paragraph(f"{round(r['similarity_score']*100,1)}%",S(f"rc{rank}",fontName="Helvetica",fontSize=9,alignment=TA_CENTER)),
                    Paragraph(f"{round(r['prediction_prob']*100,1)}%",S(f"rd{rank}",fontName="Helvetica",fontSize=9,alignment=TA_CENTER)),
                    Paragraph(f"{sc}%",S(f"re{rank}",fontName="Helvetica-Bold",fontSize=10,alignment=TA_CENTER)),
                    Paragraph(sc_txt,S(f"rf{rank}",fontName="Helvetica-Bold",fontSize=9,textColor=sc_col,alignment=TA_CENTER))])
            rt=Table(tdata,colWidths=[0.5*inch,1.8*inch,1*inch,1*inch,1*inch,1.2*inch])
            rt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),PUR),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGRAY]),
                ("GRID",(0,0),(-1,-1),0.5,rl_colors.HexColor("#333")),
                ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
                ("LEFTPADDING",(0,0),(-1,-1),7),("BOX",(0,0),(-1,-1),1.5,PUR)]))
            story.append(rt); story.append(sp(20))

            # Per-candidate skill detail
            story.append(HRFlowable(width="100%",thickness=1,color=PUR,spaceAfter=10,spaceBefore=6))
            story.append(Paragraph("Skill Analysis per Candidate",S("h2",fontName="Helvetica-Bold",fontSize=13,textColor=PUR,spaceAfter=6)))
            for rank,r in enumerate(final_results,1):
                p=r["parsed"]; sc=r["final_score"]
                if sc>=75: status_str="Shortlisted"; sc_col=GRN
                elif sc>=50: status_str="Under Review"; sc_col=ORN
                else: status_str="Rejected"; sc_col=RD
                story.append(sp(6))
                name_row=Table([[
                    Paragraph(f"<b>#{rank} — {p['Name']}</b>",S(f"nr{rank}",fontName="Helvetica-Bold",fontSize=11,textColor=WHITE)),
                    Paragraph(f"<b>{status_str} | {sc}%</b>",S(f"ns{rank}",fontName="Helvetica-Bold",fontSize=11,textColor=sc_col,alignment=TA_CENTER))
                ]],colWidths=[doc.width*0.65, doc.width*0.35])
                name_row.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK),
                    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
                    ("LEFTPADDING",(0,0),(0,-1),10)]))
                story.append(name_row)
                possessed_str = ", ".join(r.get("possessed_skills",[])[:6]) or "None"
                missing_str   = ", ".join(r.get("missing_skills",[])[:6])   or "None"
                skill_rows = [
                    [Paragraph("<b>Possessed Skills (JD Match):</b>",S(f"pk{rank}",fontName="Helvetica-Bold",fontSize=8,textColor=GRN)),
                     Paragraph(possessed_str,S(f"pv{rank}",fontName="Helvetica",fontSize=8,textColor=rl_colors.HexColor("#aaaaaa")))],
                    [Paragraph("<b>Missing Skills:</b>",S(f"mk{rank}",fontName="Helvetica-Bold",fontSize=8,textColor=RD)),
                     Paragraph(missing_str,S(f"mv{rank}",fontName="Helvetica",fontSize=8,textColor=rl_colors.HexColor("#aaaaaa")))],
                ]
                if sc < 50 and r.get("role_suggestions"):
                    roles_str = " | ".join([f"{role} ({info['match_pct']}%)" for role,info in r["role_suggestions"]])
                    skill_rows.append([
                        Paragraph("<b>Suggested Roles:</b>",S(f"rk{rank}",fontName="Helvetica-Bold",fontSize=8,textColor=TEAL)),
                        Paragraph(roles_str,S(f"rv{rank}",fontName="Helvetica",fontSize=8,textColor=rl_colors.HexColor("#aaaaaa")))
                    ])
                sd_table=Table(skill_rows,colWidths=[1.6*inch, doc.width-1.6*inch])
                sd_table.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,-1),LGRAY),
                    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                    ("LEFTPADDING",(0,0),(-1,-1),8),
                    ("LINEBELOW",(0,0),(-1,-2),0.3,rl_colors.HexColor("#333"))]))
                story.append(sd_table)

            doc.build(story,onFirstPage=page_border,onLaterPages=page_border)
            buffer.seek(0)
        st.download_button(
            label="📥 Download Recruitment_Report_v3.pdf",
            data=buffer,
            file_name=f"Recruitment_Report_v3_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True)
        st.success("✅ PDF ready!")
