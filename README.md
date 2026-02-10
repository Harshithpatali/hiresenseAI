# 🚀 HireSense AI – Recruiter Intelligence Platform

HireSense AI is a production-style Machine Learning system that ranks resumes using transformer-based semantic embeddings, hybrid scoring logic, clustering, and AI-generated recruiter insights.

This project demonstrates end-to-end ML engineering including NLP pipelines, model inference APIs, unsupervised learning, LLM integration, interactive analytics dashboards, and Dockerized deployment.

---

## 🧠 Key Features

### 🔹 AI Resume Ranking
- SentenceTransformer embeddings (MiniLM)
- Semantic similarity scoring
- Skill overlap scoring
- Experience-based weighting
- Hybrid final ranking model

### 🔹 AI Recruiter Insights
- LLM-powered candidate summaries
- Context-aware job matching explanation
- Professional hiring recommendation tone
- Secure environment-based API key management

### 🔹 Talent Analytics
- Skill score distribution visualization
- Candidate clustering using KMeans
- PCA-based talent landscape projection
- Downloadable ranked results (CSV)

### 🔹 Production Architecture
- FastAPI backend (REST inference API)
- Streamlit recruiter dashboard
- Dockerized multi-container setup
- Modular and scalable project structure

---

## 🏗 System Architecture

Streamlit Dashboard
↓
FastAPI Backend
↓
Embedding Model (SentenceTransformer)
↓
Hybrid Scoring Engine
↓
LLM Summary Layer
↓
Clustering + Visualization


---

## 🛠 Tech Stack

### 🔹 Machine Learning & NLP
- Sentence Transformers (MiniLM)
- Scikit-learn (KMeans, PCA)
- Cosine Similarity
- Custom Hybrid Scoring Logic

### 🔹 Backend
- FastAPI
- Uvicorn
- Python-dotenv
- OpenAI API (or Local LLM)

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

## 📁 Project Structure

hiresense-demo/
│
├── backend/
│ ├── app/
│ │ ├── main.py
│ │ ├── ranking.py
│ │ ├── embedding.py
│ │ ├── resume_parser.py
│ │ ├── scoring.py
│ │ └── llm_summary.py
│ ├── requirements.txt
│ └── Dockerfile
│
├── frontend/
│ ├── app.py
│ ├── requirements.txt
│ └── Dockerfile
│
├── docker-compose.yml
├── .gitignore
└── README.md


---

## 🚀 How It Works

1. Recruiter uploads resumes and provides job description.
2. Backend generates embeddings for resumes and job description.
3. Hybrid scoring is applied:
   - 60% semantic similarity
   - 25% skill overlap
   - 15% experience score
4. Candidates are clustered using KMeans.
5. LLM generates professional recruiter-friendly summaries.
6. Dashboard displays rankings, analytics, and downloadable results.

---

## 🐳 Run with Docker

### Build and Start Containers

```bash
docker-compose up --build
Access Applications
Frontend Dashboard:

http://localhost:8501
Backend API Docs:

http://localhost:8200/docs
🔐 Environment Variables
Create a .env file inside backend/:

OPENAI_API_KEY=your_api_key_here
Make sure .env is included in .gitignore.

📊 Example Output
Ranked candidates table

AI-generated candidate insights

Skill match histogram

Cluster distribution chart

PCA talent projection

CSV export functionality

🎯 What This Project Demonstrates
Production-grade NLP pipeline design

Hybrid ML ranking systems

Embedding-based semantic search

Unsupervised candidate clustering

LLM integration in real workflows

API-driven ML system architecture

Containerized ML deployment

🔮 Future Enhancements
Persistent FAISS vector index

PostgreSQL database integration

Bias detection module

Model monitoring & logging

Kubernetes deployment

Multi-tenant SaaS version

👨‍💻 Author
Built as a production-style ML Engineering portfolio project demonstrating real-world system design and deployment practices.


---

Now run:

```bash
git add README.md
git commit -m "Add professional README"
git push
