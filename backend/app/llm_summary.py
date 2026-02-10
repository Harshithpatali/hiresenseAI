import os
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment.")

client = OpenAI(api_key=API_KEY)


# --------------------------------------------------
# Generate LLM Candidate Summary
# --------------------------------------------------
def generate_summary(job_description: str, resume_text: str) -> str:
    """
    Generates a short recruiter-friendly AI summary
    comparing resume against job description.
    """

    if not API_KEY:
        return "ERROR: OpenAI API key not configured."

    prompt = f"""
You are an expert AI recruiter assistant.

Job Description:
{job_description}

Candidate Resume:
{resume_text}

Write a concise professional summary (3-4 lines) covering:
- Key strengths
- Relevance to the job
- Notable skills
- Overall hiring recommendation tone

Keep it professional and objective.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional recruitment analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.4
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        # Return real error for debugging (temporary)
        return f"ERROR generating summary: {str(e)}"
