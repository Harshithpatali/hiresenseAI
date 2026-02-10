# hiresenseAI
# 🚀 HireSense AI – Recruiter Intelligence Platform

HireSense AI is a production-grade Machine Learning system that ranks resumes using transformer-based semantic embeddings, hybrid scoring logic, clustering, and AI-generated recruiter insights.

This project demonstrates end-to-end ML engineering including NLP pipelines, inference APIs, unsupervised learning, LLM integration, visualization dashboards, and Dockerized deployment.

---

## 🧠 Overview

HireSense AI allows recruiters to:

- Upload multiple resumes (PDF/DOCX)
- Provide a job description
- Automatically rank candidates using AI
- View AI-generated recruiter summaries
- Analyze skill distribution
- Explore candidate clusters
- Download ranked results as CSV

The system combines semantic similarity, rule-based scoring, and unsupervised learning to produce intelligent candidate rankings.

---

## 🏗 System Architecture

Streamlit Dashboard
↓
FastAPI Backend
↓
SentenceTransformer Embeddings
↓
Hybrid Scoring Engine
↓
LLM Summary Layer
↓
Clustering (KMeans) + PCA Visualization


---

## 🛠 Tech Stack

### 🔹 Machine Learning & NLP
- Sentence Transformers (all-MiniLM-L6-v2)
- Scikit-learn (KMeans, PCA)
- Cosine Similarity
- Custom Hybrid Ranking Formula

### 🔹 Backend
- FastAPI
- Uvicorn
- Python-dotenv
- OpenAI API (or local LLM alternative)

### 🔹 Frontend
- Streamlit
- Pandas
- Matplotlib

### 🔹 DevOps & Deployment
- Docker
- Docker Compose
- Environment Variables (.env)
- Modular Project Structure

---

## ⚙️ Ranking Logic

Final Score:

- 60% Semantic Similarity
- 25% Skill Overlap
- 15% Experience Score

Additional Enhancements:
- LLM-generated candidate summaries
- KMeans clustering of candidates
- PCA-based talent landscape visualization

---

## 📁 Project Structure

```text
hiresense-demo/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── ranking.py
│   │   ├── embedding.py
│   │   ├── resume_parser.py
│   │   ├── scoring.py
│   │   └── llm_summary.py
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml
├── .gitignore
└── README.md


##🚀 How It Works

Recruiter uploads resumes and provides job description.

Backend generates embeddings for job description and resumes.

Hybrid scoring calculates candidate ranking.

LLM generates recruiter-friendly summary.

Candidates are clustered using KMeans.

Dashboard visualizes analytics and rankings.

🐳 Run with Docker
Build and Start
docker-compose up --build

Access Applications

Frontend Dashboard:

http://localhost:8501


Backend API Docs:

http://localhost:8200/docs

🔐 Environment Setup

Create a .env file inside backend/:

OPENAI_API_KEY=your_api_key_here


Ensure .env is included in .gitignore.

📊 Features Demonstrated

Transformer-based resume ranking

Hybrid ML scoring system

Unsupervised candidate clustering

AI-powered recruiter summaries

API-based ML architecture

Dockerized full-stack deployment

Interactive analytics dashboard

🎯 What This Project Shows

Production-style ML system design

NLP engineering with embeddings

Hybrid model + rule-based scoring integration

LLM integration in real workflows

End-to-end deployment capability

Full-stack ML engineering skills

🔮 Future Improvements

Persistent vector database (FAISS disk index)

PostgreSQL integration

Bias detection module

Model monitoring & logging

Kubernetes deployment

Multi-tenant SaaS architecture

👨‍💻 Author

Built as a production-style ML Engineering portfolio project demonstrating real-world system architecture and deployment practices.


