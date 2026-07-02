import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticMatcher:
    """
    Computes semantic similarity between candidate profiles and job descriptions.
    Uses Sentence Transformers to encode text and calculate cosine similarity.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Lazy loading of model to optimize imports
        self.model = SentenceTransformer(model_name)

    def build_profile_text(self, row: pd.Series) -> str:
        """
        Builds a comprehensive, natural language candidate profile representation
        suitable for semantic embedding.
        """
        skills = row.get("Skills", "")
        experience = row.get("Experience (Years)", 0)
        education = row.get("Education", "Not Specified")
        
        certs = row.get("Certifications", "")
        if pd.isna(certs) or str(certs).lower() in ["nan", "none", ""]:
            certs = "No formal certifications listed"
            
        projects = row.get("Projects Count", 0)
        role = row.get("Job Role", "Professional")
        
        profile_text = (
            f"Candidate is a {role} specializing in {skills}. "
            f"They have {experience} years of professional experience and "
            f"hold a degree in {education}. "
            f"Certifications: {certs}. "
            f"They have completed {projects} projects."
        )
        return profile_text

    def get_embeddings(self, texts: list) -> np.ndarray:
        """Generate Sentence Transformer embeddings for a list of texts."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def score_candidates(self, df: pd.DataFrame, job_description: str) -> pd.Series:
        """
        Generate similarity scores for all candidates against a job description.
        Returns a pandas Series of scores between 0 and 100.
        """
        if df.empty:
            return pd.Series([], dtype=float)
            
        # Build profiles
        profile_texts = df.apply(self.build_profile_text, axis=1).tolist()
        
        # Generate embeddings
        profile_embeddings = self.get_embeddings(profile_texts)
        job_embedding = self.get_embeddings([job_description])[0]
        
        # Compute cosine similarity
        similarities = util.cos_sim(profile_embeddings, job_embedding).cpu().numpy().flatten()
        
        # Scale to 0-100 range
        scores = np.clip(similarities, -1.0, 1.0)
        # Shift range from [-1, 1] to [0, 100] (specifically, cosine similarities for positive match will typically be in 0 to 1)
        # Reciprocal/normalized score: we will represent it as percentage
        scores = scores * 100.0
        # Clip negative similarities to 0
        scores = np.maximum(scores, 0.0)
        
        return pd.Series(scores, index=df.index)

if __name__ == "__main__":
    # Quick visual check of text building
    matcher = SemanticMatcher()
    sample_row = pd.Series({
        "Skills": "TensorFlow, NLP, Pytorch",
        "Experience (Years)": 10,
        "Education": "B.Sc",
        "Certifications": "AWS Certified",
        "Projects Count": 8,
        "Job Role": "AI Researcher"
    })
    print(matcher.build_profile_text(sample_row))
