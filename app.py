import os
import string
import joblib
import nltk
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from streamlit_option_menu import option_menu
from ultralytics import YOLO
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
load_dotenv()

st.set_page_config(page_title="ShoreSense", page_icon="🐋", layout="wide")

DISCLAIMER = "Educational project only. All documents are synthetic educational documents, not official rescue guidance. Contact local wildlife authorities in a real emergency."


# ---------- Load saved models once ----------
if "loaded" not in st.session_state:
    st.session_state.yolo = YOLO("models/best.pt")
    st.session_state.tfidf = joblib.load("models/tfidf.pkl")
    st.session_state.nlp_models = joblib.load("models/nlp_models.pkl")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.session_state.loaded = True

yolo = st.session_state.yolo
tfidf = st.session_state.tfidf
nlp_models = st.session_state.nlp_models
db = st.session_state.db

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

prompt = ChatPromptTemplate.from_template("""You are ShoreSense, an educational assistant for marine animal stranding guidance.

Rules:
1. Answer ONLY from the context below.
2. Cite the file name of every source, like [file_name.pdf].
3. If documents from different regions give different guidance, state the difference by region.
4. If the answer is not in the context, reply exactly: The documents do not cover this. Please contact your local wildlife authorities.
5. These are synthetic educational documents, not official rescue guidance.

Context:
{context}

Question: {question}

Answer:""")


# ---------- Functions ----------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def detect_species(image):
    result = yolo.predict(image, conf=0.25)[0]
    if len(result.boxes) == 0:
        return None, None, None
    i = int(result.boxes.conf.argmax())
    species = yolo.names[int(result.boxes.cls[i])]
    confidence = float(result.boxes.conf[i])
    plotted = result.plot()[:, :, ::-1]
    return species, confidence, plotted


def extract_report(text):
    vector = tfidf.transform([clean_text(text)])
    report = {}
    for target in ["species", "condition", "gear", "urgency"]:
        report[target] = nlp_models[target].predict(vector)[0]
    return report


def answer_question(question, species=None, region=None):
    if not os.getenv("GROQ_API_KEY"):
        return "GROQ_API_KEY is missing. Add it to your .env file.", []

    query = question
    if species:
        query = query + " " + species
    if region:
        query = query + " " + region

    docs = db.similarity_search(query, k=4)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

    try:
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        response = (prompt | llm).invoke({"context": context, "question": query})
        return response.content, docs
    except Exception as e:
        return "API error: " + str(e), docs


def show_urgency(urgency):
    if urgency == "high":
        st.error("Urgency: HIGH")
    elif urgency == "medium":
        st.warning("Urgency: MEDIUM")
    else:
        st.success("Urgency: LOW")


def same_species(a, b):
    a = str(a).lower().replace("_", " ").strip()
    b = str(b).lower().replace("_", " ").strip()
    return a == b


def show_sources(docs):
    if docs:
        st.write("Sources:", ", ".join(sorted(set(d.metadata["source"] for d in docs))))
        with st.expander("Retrieved chunks"):
            for d in docs:
                st.markdown("**" + d.metadata["source"] + "**")
                st.write(d.page_content)
                st.divider()


# ---------- Sidebar menu ----------
with st.sidebar:
    page = option_menu(
        "ShoreSense",
        ["Home", "Species Detection", "Report Understanding", "Guidance Q&A", "Full Triage"],
        icons=["house", "camera", "file-text", "chat-dots", "clipboard-check"],
        default_index=0,
    )

st.warning(DISCLAIMER)


# ---------- Home ----------
if page == "Home":
    st.title("🐋 ShoreSense")
    st.subheader("Marine Stranding & Bycatch Triage")
    st.write("Upload a photo and describe what you see. ShoreSense detects the species, reads your report and finds guidance from the documents.")

    col1, col2, col3 = st.columns(3)
    col1.info("1. Detect\n\nYOLOv8 finds the species in the photo.")
    col2.info("2. Understand\n\nNLP reads species, condition, gear and urgency.")
    col3.info("3. Guide\n\nRAG finds cited guidance from the documents.")


# ---------- Species Detection ----------
elif page == "Species Detection":
    st.title("Species Detection")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file).convert("RGB")
        species, confidence, plotted = detect_species(image)

        if species is None:
            st.image(image)
            st.error("No animal detected in this image.")
        else:
            st.image(plotted)
            st.success(f"Species: {species} | Confidence: {confidence:.2f}")


# ---------- Report Understanding ----------
elif page == "Report Understanding":
    st.title("Report Understanding")
    text = st.text_area("Describe what you see")

    if st.button("Report"):
        if text.strip() == "":
            st.error("Please type a description.")
        else:
            report = extract_report(text)
            table = pd.DataFrame({
                "Field": ["Species", "Condition", "Gear"],
                "Value": [report["species"], report["condition"], report["gear"]],
            })
            st.table(table)
            show_urgency(report["urgency"])


# ---------- Guidance Q&A ----------
elif page == "Guidance Q&A":
    st.title("Guidance Q&A")
    question = st.text_input("Ask a question")

    col1, col2 = st.columns(2)
    species = col1.selectbox("Species (optional)", ["Any"] + list(yolo.names.values()))
    region = col2.selectbox("Region (optional)", ["Any", "United States", "India"])

    if st.button("Get Answer"):
        if question.strip() == "":
            st.error("Please type a question.")
        else:
            sp = None if species == "Any" else species
            rg = None if region == "Any" else region
            answer, docs = answer_question(question, sp, rg)

            if "do not cover this" in answer.lower():
                st.warning("The documents do not cover this question.")
            st.write(answer)
            show_sources(docs)


# ---------- Full Triage ----------
elif page == "Full Triage":
    st.title("Full Triage")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    text = st.text_area("Describe what you see")

    if st.button("Run Triage"):
        if file is None:
            st.error("Please upload an image.")
        elif text.strip() == "":
            st.error("Please type a description.")
        else:
            image = Image.open(file).convert("RGB")
            species, confidence, plotted = detect_species(image)
            report = extract_report(text)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Image")
                if species is None:
                    st.image(image)
                    st.error("No animal detected in the image.")
                else:
                    st.image(plotted)
                    st.success(f"Species: {species} | Confidence: {confidence:.2f}")

            with col2:
                st.subheader("Report")
                table = pd.DataFrame({
                    "Field": ["Species", "Condition", "Gear"],
                    "Value": [report["species"], report["condition"], report["gear"]],
                })
                st.table(table)
                show_urgency(report["urgency"])

            if species is not None and report["species"] != "unknown" and not same_species(species, report["species"]):
                st.warning(f"Species mismatch: image shows '{species}' but the text says '{report['species']}'.")

            final_species = species if species is not None else report["species"]
            question = f"What should I do about a {report['condition']} {final_species} involving {report['gear']}? Urgency is {report['urgency']}."

            st.subheader("Guidance")
            answer, docs = answer_question(question, final_species)
            st.write(answer)
            show_sources(docs)
