import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from src.utils import parse_skills_list

class GrowthPotentialScorer:
    """
    Computes a candidate's Growth Potential Score (0-100).
    The score is independent of current job role fit and represents long-term capability.
    """
    def __init__(self, model: SentenceTransformer = None):
        if model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.model = model

        # Certification quality map
        self.cert_scores = {
            "deep learning specialization": 95,
            "google ml": 90,
            "aws certified": 85
        }

    def compute_cert_quality(self, certs_str: str) -> float:
        """Rate the quality/difficulty of candidate certifications."""
        if pd.isna(certs_str) or not isinstance(certs_str, str) or certs_str.strip() == "":
            return 0.0
            
        certs = [c.strip().lower() for c in certs_str.split(",") if c.strip()]
        if not certs:
            return 0.0
            
        # Get score for each cert
        scores = [self.cert_scores.get(c, 50) for c in certs] # Default unknown certs to 50
        
        # Max certification score + bonus for multiple certifications (+5 points per extra, capped at 100)
        max_score = max(scores)
        bonus = (len(certs) - 1) * 5.0
        return min(max_score + bonus, 100.0)

    def compute_technical_breadth(self, skills: list[str], skill_embeddings: Dict[str, np.ndarray]) -> float:
        """
        Computes technical breadth as the average pairwise distance (1 - similarity) 
        between candidate's skill embeddings. High distance indicates skills covering multiple domains.
        Uses cached skill embeddings for high performance.
        """
        if not skills:
            return 0.0
        if len(skills) == 1:
            return 30.0 # Base score for having a specialized skill
            
        # Get unique clean skills
        unique_skills = list(set([s.strip() for s in skills if s.strip()]))
        if len(unique_skills) <= 1:
            return 30.0

        try:
            # Retrieve pre-computed embeddings from cache
            embeddings = np.array([skill_embeddings[s] for s in unique_skills if s in skill_embeddings])
            if len(embeddings) <= 1:
                return 30.0
                
            # Compute pairwise cosine similarities
            sims = util.cos_sim(embeddings, embeddings).cpu().numpy()
            
            # Exclude diagonal (similarity with self which is 1.0)
            n = len(embeddings)
            triu_indices = np.triu_indices(n, k=1)
            pairwise_sims = sims[triu_indices]
            
            # Average similarity
            avg_sim = float(np.mean(pairwise_sims))
            
            # Breadth is distance (1 - similarity)
            # Clip between 0 and 1, then scale to 0-100
            breadth_val = max(0.0, 1.0 - avg_sim)
            breadth_score = breadth_val * 100.0
            
            # Combine with base score to reward higher skill count
            final_breadth = 30.0 + (breadth_score * 0.7)
            return min(final_breadth, 100.0)
        except Exception:
            # Fallback in case of failure
            return 50.0

    def score_candidates(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Growth Potential Score for all candidates.
        Returns a pandas Series of scores between 0 and 100.
        """
        if df.empty:
            return pd.Series([], dtype=float)

        # Performance Optimization: Pre-encode all unique skills in the dataset in a single batch
        all_unique_skills = set()
        for _, row in df.iterrows():
            skills = parse_skills_list(row.get("Skills", ""))
            all_unique_skills.update(skills)
            
        unique_skills_list = list(all_unique_skills)
        if unique_skills_list:
            embeddings_list = self.model.encode(unique_skills_list, show_progress_bar=False, convert_to_numpy=True)
            skill_embeddings = {skill: emb for skill, emb in zip(unique_skills_list, embeddings_list)}
        else:
            skill_embeddings = {}

        growth_scores = []
        for _, row in df.iterrows():
            skills = parse_skills_list(row.get("Skills", ""))
            experience = float(row.get("Experience (Years)", 0.0))
            projects = float(row.get("Projects Count", 0.0))
            cert_count = float(row.get("cert_count", 0.0))
            
            # 1. Project Complexity Score (min-max of projects count)
            proj_complexity = min(projects * 10.0, 100.0)
            
            # 2. Skill Diversity Score (number of unique skills)
            skill_diversity = min(len(skills) * 25.0, 100.0)
            
            # 3. Certification Quality
            cert_quality = self.compute_cert_quality(row.get("Certifications", ""))
            
            # 4. Learning Progression (Velocity of acquiring certifications and projects over time)
            denom = 1.0 + (experience * 0.1)
            velocity = (cert_count * 30.0 + projects * 7.0) / denom
            learning_progression = min(velocity, 100.0)
            
            # 5. Technical Breadth (Uses pre-computed skill embeddings cache)
            tech_breadth = self.compute_technical_breadth(skills, skill_embeddings)
            
            # Combine components with equal weight (20% each)
            total_score = (
                proj_complexity * 0.20 +
                skill_diversity * 0.20 +
                cert_quality * 0.20 +
                learning_progression * 0.20 +
                tech_breadth * 0.20
            )
            
            growth_scores.append(round(total_score, 2))
            
        return pd.Series(growth_scores, index=df.index)

if __name__ == "__main__":
    scorer = GrowthPotentialScorer()
    sample_df = pd.DataFrame([
        {
            "Skills": "TensorFlow, NLP, Pytorch",
            "Experience (Years)": 1,
            "Projects Count": 8,
            "cert_count": 1,
            "Certifications": "Google ML"
        }
    ])
    print(scorer.score_candidates(sample_df))
