import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

BACKEND_URL = "http://127.0.0.1:8200/api/rank"



st.set_page_config(page_title="HireSense AI", layout="wide")

st.title("HireSense AI – Recruiter Intelligence Dashboard")

st.markdown("### 📌 Upload resumes and rank candidates using AI")

# --- Job Description ---
job_description = st.text_area(
    "Paste Job Description Here",
    height=200
)

# --- Resume Upload ---
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# --- Rank Button ---
if st.button("🚀 Rank Candidates"):

    if not job_description or not uploaded_files:
        st.warning("Please provide job description and upload resumes.")
    else:
        with st.spinner("Analyzing candidates with AI..."):

            files = [
                ("resumes", (file.name, file, file.type))
                for file in uploaded_files
            ]

            response = requests.post(
                BACKEND_URL,
                data={"job_description": job_description},
                files=files
            )

            if response.status_code != 200:
                st.error("Backend error. Please check API.")
                st.stop()

            data = response.json()
            df = pd.DataFrame(data["ranked_candidates"])

        st.success("Ranking Complete!")

        # ===============================
        # Ranked Table
        # ===============================
        st.subheader("🏆 Ranked Candidates")
        st.dataframe(df.drop(columns=["summary"]), use_container_width=True)

        # ===============================
        # LLM Summaries
        # ===============================
        st.subheader("🧠 AI Candidate Insights")

        for _, row in df.iterrows():
            with st.expander(f"{row['filename']} — Score: {row['final_score']}"):
                st.write(row["summary"])

        # ===============================
        # Skill Match Distribution
        # ===============================
        st.subheader("📊 Skill Match Distribution")

        if "skill_score" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["skill_score"], bins=5)
            ax.set_xlabel("Skill Match Score")
            ax.set_ylabel("Number of Candidates")
            st.pyplot(fig)

        # ===============================
        # Cluster Distribution
        # ===============================
        st.subheader("🔍 Candidate Clusters")

        if "cluster" in df.columns:
            cluster_counts = df["cluster"].value_counts()
            st.bar_chart(cluster_counts)

        # ===============================
        # Optional: PCA Cluster Visualization
        # (Simulated 2D projection using scores)
        # ===============================
        st.subheader("🧭 Talent Landscape (PCA Projection)")

        if len(df) >= 2:

            features = df[[
                "semantic_score",
                "skill_score",
                "experience_score"
            ]].values

            pca = PCA(n_components=2)
            components = pca.fit_transform(features)

            fig, ax = plt.subplots()
            scatter = ax.scatter(
                components[:, 0],
                components[:, 1]
            )

            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("Candidate Clusters Projection")

            st.pyplot(fig)

        # ===============================
        # CSV Export
        # ===============================
        st.subheader("⬇ Download Results")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Ranked Candidates CSV",
            data=csv,
            file_name="ranked_candidates.csv",
            mime="text/csv"
        )
