import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_semantic_similarity(job_emb, resume_emb):
    return cosine_similarity([job_emb], [resume_emb])[0][0]

def extract_skills(text):
    skills_db = ["python", "machine learning", "deep learning", 
                 "sql", "docker", "kubernetes", "nlp", "aws"]

    text_lower = text.lower()
    found = [skill for skill in skills_db if skill in text_lower]
    return set(found)

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
