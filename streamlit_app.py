import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
from docx import Document
from io import BytesIO
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="HireSense AI", layout="wide")
st.title("🚀 HireSense AI – Recruiter Intelligence Platform")

# ---------------------------------------------------
# LOAD EMBEDDING MODEL (cached)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------------------------
# LOAD OPENAI CLIENT
# ---------------------------------------------------
def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except:
        return None

client = get_openai_client()

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def parse_resume(file):
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text

    elif filename.endswith(".docx"):
        doc = Document(BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])

    return ""


def get_embedding(text):
    return model.encode(text)


def compute_semantic_similarity(job_emb, resume_emb):
    return cosine_similarity([job_emb], [resume_emb])[0][0]


def extract_skills(text):
    skills_db = [
        "python", "machine learning", "deep learning",
        "sql", "docker", "kubernetes", "nlp", "aws"
    ]
    text_lower = text.lower()
    return set([skill for skill in skills_db if skill in text_lower])


def skill_overlap(job_text, resume_text):
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)

    if not job_skills:
        return 0

    return len(job_skills.intersection(resume_skills)) / len(job_skills)


def experience_score(resume_text):
    matches = re.findall(r"(\d+)\+?\s+years", resume_text.lower())
    if matches:
        years = max([int(y) for y in matches])
        return min(years / 10, 1.0)
    return 0


def final_score(semantic, skill, experience):
    return (
        0.6 * semantic +
        0.25 * skill +
        0.15 * experience
    )


def generate_summary(job_description, resume_text):
    if client is None:
        return "LLM summary unavailable (no API key configured)."

    prompt = f"""
You are an expert AI recruiter assistant.

Job Description:
{job_description}

Candidate Resume:
{resume_text}

Write a concise professional summary (3-4 lines)
highlighting strengths, relevance, and recommendation tone.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"


# ---------------------------------------------------
# UI INPUTS
# ---------------------------------------------------

job_description = st.text_area(
    "📄 Paste Job Description",
    height=200
)

uploaded_files = st.file_uploader(
    "📎 Upload Resumes (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# RANKING LOGIC
# ---------------------------------------------------

if st.button("🚀 Rank Candidates"):

    if not job_description or not uploaded_files:
        st.warning("Please provide job description and upload resumes.")
    else:
        with st.spinner("Analyzing candidates..."):

            results = []
            resume_embeddings = []

            job_embedding = get_embedding(job_description)

            for file in uploaded_files:
                text = parse_resume(file)

                if not text:
                    continue

                resume_embedding = get_embedding(text)
                resume_embeddings.append(resume_embedding)

                semantic = compute_semantic_similarity(
                    job_embedding, resume_embedding
                )

                skill = skill_overlap(job_description, text)
                experience = experience_score(text)
                score = final_score(semantic, skill, experience)

                summary = generate_summary(job_description, text)

                results.append({
                    "filename": file.name,
                    "semantic_score": round(float(semantic), 3),
                    "skill_score": round(float(skill), 3),
                    "experience_score": round(float(experience), 3),
                    "final_score": round(float(score), 3),
                    "summary": summary
                })

            # Clustering
            if len(resume_embeddings) >= 2:
                kmeans = KMeans(
                    n_clusters=min(3, len(resume_embeddings)),
                    random_state=42,
                    n_init=10
                )
                clusters = kmeans.fit_predict(resume_embeddings)

                for i, cluster_id in enumerate(clusters):
                    results[i]["cluster"] = int(cluster_id)
            else:
                for r in results:
                    r["cluster"] = 0

            df = pd.DataFrame(results)
            df = df.sort_values(
                by="final_score",
                ascending=False
            ).reset_index(drop=True)

        st.success("Ranking Complete!")

        # ---------------------------------------------------
        # RESULTS TABLE
        # ---------------------------------------------------
        st.subheader("🏆 Ranked Candidates")
        st.dataframe(df.drop(columns=["summary"]), use_container_width=True)

        # ---------------------------------------------------
        # LLM SUMMARIES
        # ---------------------------------------------------
        st.subheader("🧠 AI Candidate Insights")

        for _, row in df.iterrows():
            with st.expander(
                f"{row['filename']} — Score: {row['final_score']}"
            ):
                st.write(row["summary"])

        # ---------------------------------------------------
        # SKILL DISTRIBUTION
        # ---------------------------------------------------
        st.subheader("📊 Skill Match Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["skill_score"], bins=5)
        ax.set_xlabel("Skill Score")
        ax.set_ylabel("Number of Candidates")
        st.pyplot(fig)

        # ---------------------------------------------------
        # CLUSTER DISTRIBUTION
        # ---------------------------------------------------
        st.subheader("🔍 Candidate Clusters")

        if "cluster" in df.columns:
            st.bar_chart(df["cluster"].value_counts())

        # ---------------------------------------------------
        # PCA VISUALIZATION
        # ---------------------------------------------------
        if len(df) >= 2:
            st.subheader("🧭 Talent Landscape (PCA Projection)")

            features = df[[
                "semantic_score",
                "skill_score",
                "experience_score"
            ]].values

            pca = PCA(n_components=2)
            components = pca.fit_transform(features)

            fig, ax = plt.subplots()
            ax.scatter(components[:, 0], components[:, 1])
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            st.pyplot(fig)

        # ---------------------------------------------------
        # DOWNLOAD CSV
        # ---------------------------------------------------
        st.subheader("⬇ Download Results")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Ranked Candidates CSV",
            csv,
            "ranked_candidates.csv",
            "text/csv"
        )
