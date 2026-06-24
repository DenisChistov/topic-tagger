import json
import os

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_model_repo():
    try:
        secret = st.secrets.get("MODEL_REPO")
    except Exception:
        secret = None
    return os.environ.get("MODEL_REPO") or secret or "De4u/arxiv-topic-tagger"


MODEL_REPO = _resolve_model_repo()
SEQ_LIMIT = 256
COVERAGE = 0.95

READABLE = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CV": "Computer Vision & Pattern Recognition",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.IR": "Information Retrieval",
    "cs.RO": "Robotics",
    "cs.LO": "Logic in Computer Science",
    "cs.SI": "Social & Information Networks",
    "cs.DS": "Data Structures & Algorithms",
    "cs.SD": "Sound",
    "cs.CR": "Cryptography & Security",
    "cs.DB": "Databases",
    "cs.GT": "Computer Science & Game Theory",
    "cs.IT": "Information Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.DC": "Distributed, Parallel & Cluster Computing",
    "cs.CY": "Computers & Society",
    "cs.CE": "Computational Engineering",
    "cs.MM": "Multimedia",
    "cs.SE": "Software Engineering",
    "cs.MA": "Multiagent Systems",
    "cs.NI": "Networking & Internet Architecture",
    "math.OC": "Optimization & Control",
    "math.ST": "Statistics Theory",
    "stat.ML": "Machine Learning (Statistics)",
    "stat.ME": "Methodology (Statistics)",
    "q-bio.NC": "Neurons & Cognition",
    "q-bio.QM": "Quantitative Methods (Biology)",
    "cmp-lg": "Computational Linguistics (legacy)",
}

st.set_page_config(page_title="arXiv Topic Tagger", page_icon="📄", layout="wide")

MINIMAL_CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}
.stApp {background: #f8fafc;}
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem;}
.app-header {margin-bottom: 1rem;}
.app-title {font-size: 2rem; font-weight: 700; letter-spacing: -0.01em; margin-bottom: .35rem; color: #0f172a;}
.app-sub {color: #475569; margin-bottom: 0; font-size: 1rem; line-height: 1.55;}
.panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1rem .7rem;
}
.result-head {font-weight: 650; margin: 0 0 0.8rem; color: #0f172a;}
.topic-row {margin: 0.8rem 0;}
.topic-head {display: flex; justify-content: space-between; align-items: baseline; font-size: 0.96rem; margin-bottom: 6px;}
.topic-name {color: #0f172a; line-height: 1.3;}
.topic-tag {color: #64748b; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.82rem;}
.topic-pct {color: #0f172a; font-variant-numeric: tabular-nums; font-weight: 600;}
.bar-track {height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden;}
.bar-fill {height: 8px; background: #2563eb; border-radius: 999px;}
.stTextInput label, .stTextArea label {color: #0f172a; font-weight: 600;}
.stTextInput input, .stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #0f172a !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {color: #94a3b8 !important;}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    border: 0;
    background: #2563eb;
    color: #ffffff;
    font-weight: 600;
    padding: 0.6rem 1rem;
}
.stButton>button:hover {background: #1d4ed8;}
@media (max-width: 900px) {
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
    .app-title {font-size: 1.55rem;}
    .panel {padding: .9rem .85rem .65rem;}
}
</style>
"""
st.markdown(MINIMAL_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Загружаю модель…")
def get_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    mdl.eval()
    map_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_map.json")
    with open(map_path, encoding="utf-8") as fh:
        mapping = {int(k): v for k, v in json.load(fh).items()}
    return tok, mdl, mapping


def nice_name(tag):
    return READABLE.get(tag, tag)


def rank_topics(title, abstract, coverage=COVERAGE):
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    text = title if not abstract else f"{title}. {abstract}"

    tok, mdl, mapping = get_pipeline()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LIMIT)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[0]

    order = torch.argsort(probs, descending=True).tolist()
    chosen, acc = [], 0.0
    for j in order:
        chosen.append((mapping[j], probs[j].item()))
        acc += probs[j].item()
        if acc >= coverage:
            break
    return chosen


st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown('<div class="app-title">Классификатор тематик arXiv</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Введите название статьи и, при желании, аннотацию. Модель определит наиболее вероятные рубрики arXiv.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    title = st.text_input(
        "Название статьи",
        placeholder="Например, Deep Residual Learning for Image Recognition",
    )
    abstract = st.text_area(
        "Аннотация",
        height=220,
        placeholder="Вставьте аннотацию или оставьте поле пустым, чтобы классифицировать только по названию.",
    )
    submit = st.button("Определить тематику", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if submit:
        if not title.strip() and not abstract.strip():
            st.warning("Введите хотя бы название статьи.")
        else:
            try:
                results = rank_topics(title, abstract)
                st.markdown(
                    f'<div class="result-head">Вероятные рубрики · {len(results)} шт.</div>',
                    unsafe_allow_html=True,
                )
                for tag, p in results:
                    st.markdown(
                        '<div class="topic-row">'
                        '<div class="topic-head">'
                        f'<span class="topic-name">{nice_name(tag)} '
                        f'<span class="topic-tag">{tag}</span></span>'
                        f'<span class="topic-pct">{p * 100:.1f}%</span>'
                        "</div>"
                        f'<div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%"></div></div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as err:
                st.error(f"Не удалось выполнить классификацию: {err}")
    else:
        st.markdown('<div class="result-head">Результаты появятся здесь</div>', unsafe_allow_html=True)
        st.info("Заполните поля слева и нажмите кнопку, чтобы получить рубрики.")
    st.markdown("</div>", unsafe_allow_html=True)
import json
import os

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def _resolve_model_repo():
    # На Streamlit Cloud имя модели можно переопределить через Secrets (MODEL_REPO).
    try:
        secret = st.secrets.get("MODEL_REPO")
    except Exception:
        secret = None
    return os.environ.get("MODEL_REPO") or secret or "De4u/arxiv-topic-tagger"


MODEL_REPO = _resolve_model_repo()
SEQ_LIMIT = 256
COVERAGE = 0.95

# Человекочитаемые названия рубрик arXiv.
READABLE = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CV": "Computer Vision & Pattern Recognition",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.IR": "Information Retrieval",
    "cs.RO": "Robotics",
    "cs.LO": "Logic in Computer Science",
    "cs.SI": "Social & Information Networks",
    "cs.DS": "Data Structures & Algorithms",
    "cs.SD": "Sound",
    "cs.CR": "Cryptography & Security",
    "cs.DB": "Databases",
    "cs.GT": "Computer Science & Game Theory",
    "cs.IT": "Information Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.DC": "Distributed, Parallel & Cluster Computing",
    "cs.CY": "Computers & Society",
    "cs.CE": "Computational Engineering",
    "cs.MM": "Multimedia",
    "cs.SE": "Software Engineering",
    "cs.MA": "Multiagent Systems",
    "cs.NI": "Networking & Internet Architecture",
    "math.OC": "Optimization & Control",
    "math.ST": "Statistics Theory",
    "stat.ML": "Machine Learning (Statistics)",
    "stat.ME": "Methodology (Statistics)",
    "q-bio.NC": "Neurons & Cognition",
    "q-bio.QM": "Quantitative Methods (Biology)",
    "cmp-lg": "Computational Linguistics (legacy)",
}

st.set_page_config(page_title="arXiv Topic Tagger", page_icon="\U0001F4C4", layout="centered")

# Минималистичная тема: один тёмный акцент, тонкие линии, без градиентов.
MINIMAL_CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {max-width: 740px; padding-top: 2.6rem;}
.app-title {font-size: 1.85rem; font-weight: 650; letter-spacing: -0.01em; margin-bottom: .2rem;}
.app-sub {color: #6b7280; margin-bottom: 1.7rem; font-size: 0.97rem; line-height: 1.5;}
.result-head {font-weight: 600; margin: 0.4rem 0 0.7rem;}
.topic-row {margin: 0.6rem 0;}
.topic-head {display: flex; justify-content: space-between; align-items: baseline;
             font-size: 0.95rem; margin-bottom: 5px;}
.topic-name {color: #111827;}
.topic-tag {color: #9ca3af; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.82rem;}
.topic-pct {color: #111827; font-variant-numeric: tabular-nums;}
.bar-track {height: 6px; background: #eef0f3; border-radius: 999px; overflow: hidden;}
.bar-fill {height: 6px; background: #111827; border-radius: 999px;}
.stTextInput input, .stTextArea textarea {border-radius: 8px;}
hr {margin: 1.4rem 0; border: none; border-top: 1px solid #ececf0;}
</style>
"""
st.markdown(MINIMAL_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Загружаю модель…")
def get_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    mdl.eval()
    map_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_map.json")
    with open(map_path, encoding="utf-8") as fh:
        mapping = {int(k): v for k, v in json.load(fh).items()}
    return tok, mdl, mapping


def nice_name(tag):
    return READABLE.get(tag, tag)


def rank_topics(title, abstract, coverage=COVERAGE):
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    text = title if not abstract else f"{title}. {abstract}"

    tok, mdl, mapping = get_pipeline()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LIMIT)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[0]

    order = torch.argsort(probs, descending=True).tolist()
    chosen, acc = [], 0.0
    for j in order:
        chosen.append((mapping[j], probs[j].item()))
        acc += probs[j].item()
        if acc >= coverage:
            break
    return chosen


st.markdown('<div class="app-title">Классификатор тематик arXiv</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Введите название статьи и, по желанию, аннотацию. '.</div>',
    unsafe_allow_html=True,
)

title = st.text_input(
    "Название статьи",
    placeholder="например, Deep Residual Learning for Image Recognition",
)
abstract = st.text_area(
    "Аннотация (По желанию)",
    height=170,
    placeholder="Вставьте аннотацию или оставьте поле пустым, чтобы классифицировать только по названию.",
)

if st.button("Определить тематику", type="primary"):
    if not title.strip() and not abstract.strip():
        st.warning("Введите хотя бы название статьи.")
    else:
        try:
            results = rank_topics(title, abstract)
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-head">Вероятные рубрики · {len(results)} шт.</div>',
                unsafe_allow_html=True,
            )
            for tag, p in results:
                st.markdown(
                    '<div class="topic-row">'
                    '<div class="topic-head">'
                    f'<span class="topic-name">{nice_name(tag)} '
                    f'<span class="topic-tag">{tag}</span></span>'
                    f'<span class="topic-pct">{p * 100:.1f}%</span>'
                    '</div>'
                    f'<div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%"></div></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        except Exception as err:
            st.error(f"Не удалось выполнить классификацию: {err}")
