import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constant default Job Descriptions for the 4 core job roles in the dataset
DEFAULT_JOB_DESCRIPTIONS = {
    "AI Researcher": (
        "Conduct research in artificial intelligence, machine learning, and deep learning. "
        "Develop novel algorithms and models using PyTorch, TensorFlow, and natural language processing (NLP). "
        "Prototype new AI applications and optimize models for research and production environments."
    ),
    "Data Scientist": (
        "Extract insights from structured and unstructured data. Build predictive models, machine "
        "learning pipelines, and perform statistical and data analysis. Highly skilled in Python, "
        "SQL, machine learning, deep learning, and translating business problems into data solutions."
    ),
    "Cybersecurity Analyst": (
        "Monitor, detect, and protect computer networks and systems from threats and vulnerabilities. "
        "Perform ethical hacking, system audits, vulnerability assessments, penetration testing, and "
        "incident response. Expert in Linux security, networking protocols, and cyber threat mitigation."
    ),
    "Software Engineer": (
        "Design, develop, and maintain high-performance software applications. Write clean, efficient, "
        "and testable code in languages like Java, C++, Python, or JavaScript. Experience with web frameworks "
        "like React, database systems (SQL), and standard software architecture patterns."
    )
}

# Constant default required skills for each role for hidden talent mapping
DEFAULT_REQUIRED_SKILLS = {
    "AI Researcher": ["Python", "Pytorch", "TensorFlow", "NLP", "Deep Learning", "Machine Learning"],
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Deep Learning"],
    "Cybersecurity Analyst": ["Cybersecurity", "Ethical Hacking", "Networking", "Linux"],
    "Software Engineer": ["SQL", "React", "Java", "C++"]
}

def clean_skill(skill_name):
    """Clean and standardize a skill name."""
    if not isinstance(skill_name, str):
        return ""
    # Strip spaces, standard casing, etc.
    return skill_name.strip()

def parse_skills_list(skills_field):
    """Parse comma separated skills field into clean list."""
    if pd.isna(skills_field) or not isinstance(skills_field, str):
        return []
    return [clean_skill(s) for s in skills_field.split(",") if clean_skill(s)]

def min_max_normalize(series):
    """Perform Min-Max normalization to scale series values between 0 and 100."""
    if series.empty:
        return series
    val_min = series.min()
    val_max = series.max()
    if val_max == val_min:
        return pd.Series(100.0, index=series.index)
    return 100.0 * (series - val_min) / (val_max - val_min)

def get_env_var(name, default=None):
    """Get environment variable safely."""
    return os.environ.get(name, default)

def load_sentence_transformer(model_name: str):
    """
    Loads SentenceTransformer model with automatic fallback to offline mode
    if network requests to Hugging Face fail. Useful for environments with intermittent connectivity.
    """
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"\nNetwork warning: Connection to Hugging Face failed while loading model '{model_name}': {e}")
        print("Self-healing: Attempting to load from local cache in offline mode...")
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            return SentenceTransformer(model_name)
        except Exception as inner_e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer '{model_name}' in both online and offline modes. "
                "Verify your internet connection and that the model has been downloaded at least once."
            ) from inner_e

