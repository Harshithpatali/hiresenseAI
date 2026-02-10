from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import numpy as np
from sklearn.cluster import KMeans

from app.resume_parser import parse_resume
from app.embedding import get_embedding
from app.scoring import (
    compute_semantic_similarity,
    skill_overlap,
    experience_score,
    final_score
)
from app.llm_summary import generate_summary

router = APIRouter()


@router.post("/rank")
async def rank_resumes(
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...)
):
    results = []
    resume_embeddings = []

    # Step 1: Embed job description once
    job_embedding = get_embedding(job_description)

    # Step 2: Process each resume
    for resume in resumes:

        text = parse_resume(resume)

        if not text:
            continue

        resume_embedding = get_embedding(text)
        resume_embeddings.append(resume_embedding)

        # --- Scoring ---
        semantic = compute_semantic_similarity(
            job_embedding,
            resume_embedding
        )

        skill = skill_overlap(job_description, text)
        experience = experience_score(text)

        score = final_score(semantic, skill, experience)

        # --- LLM Summary ---
        summary = generate_summary(job_description, text)

        results.append({
            "filename": resume.filename,
            "semantic_score": round(float(semantic), 3),
            "skill_score": round(float(skill), 3),
            "experience_score": round(float(experience), 3),
            "final_score": round(float(score), 3),
            "summary": summary
        })

    # Step 3: Candidate Clustering
    if len(resume_embeddings) >= 2:
        embeddings_array = np.array(resume_embeddings)

        n_clusters = min(3, len(resume_embeddings))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        for i, cluster_id in enumerate(cluster_labels):
            results[i]["cluster"] = int(cluster_id)

    else:
        for result in results:
            result["cluster"] = 0

    # Step 4: Sort by final score
    results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return {
        "total_candidates": len(results),
        "ranked_candidates": results
    }
