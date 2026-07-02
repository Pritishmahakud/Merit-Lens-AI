import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer, util
from src.utils import parse_skills_list

# Technology Context Map to enrich raw skill names with domain metadata before embedding.
TECH_CONTEXTS = {
    "flask": "Flask Python web microframework backend development",
    "fastapi": "FastAPI Python web API framework backend development",
    "django": "Django Python web framework backend development",
    "pytorch": "PyTorch deep learning framework machine learning AI library",
    "tensorflow": "TensorFlow deep learning framework machine learning AI library",
    "keras": "Keras deep learning framework neural networks AI library",
    "react": "React frontend web JavaScript library UI interface",
    "vue": "Vue frontend web JavaScript framework UI interface",
    "angular": "Angular frontend web TypeScript framework UI interface",
    "aws": "AWS Amazon Web Services cloud platform infrastructure compute",
    "azure": "Azure Microsoft cloud platform infrastructure compute",
    "gcp": "GCP Google Cloud Platform infrastructure compute",
    "sql": "SQL database query language relational data management",
    "mongodb": "MongoDB NoSQL database document store database",
    "python": "Python programming language software development scripting",
    "java": "Java programming language enterprise application development",
    "c++": "C++ programming language systems software development",
    "ethical hacking": "Ethical Hacking penetration testing cybersecurity security audit",
    "cybersecurity": "Cybersecurity network security threat protection",
    "linux": "Linux operating system systems administration CLI",
    "networking": "Networking computer networks protocols security routers",
    "nlp": "NLP Natural Language Processing text mining computational linguistics"
}

class HiddenTalentDetector:
    """
    Detects transferable and related skills using Sentence Transformers
    and computes a Transferable Skill Score based on semantic overlap.
    """
    def __init__(self, model: SentenceTransformer = None):
        if model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.model = model

    def _get_contextualized_skill(self, skill_name: str) -> str:
        """Returns skill name enriched with contextual details for better embedding representation."""
        clean_name = skill_name.strip().lower()
        if clean_name in TECH_CONTEXTS:
            return TECH_CONTEXTS[clean_name]
        return f"Software development skill and technology: {skill_name}"

    def analyze_skills(
        self, candidate_skills: List[str], required_skills: List[str],
        transfer_threshold: float = 0.50, related_threshold: float = 0.35,
        skill_embeddings: Dict[str, np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Classifies candidate skills against required skills into exact, transferable, related, and missing.
        Uses cached skill embeddings if provided for high performance.
        """
        if not required_skills:
            return {
                "exact_matches": [],
                "transferable_skills": {},
                "related_skills": {},
                "missing_skills": [],
                "score": 100.0
            }

        if not candidate_skills:
            return {
                "exact_matches": [],
                "transferable_skills": {},
                "related_skills": {},
                "missing_skills": required_skills.copy(),
                "score": 0.0
            }

        # Clear duplicates and standardize lists
        req_clean = list(set([r.strip() for r in required_skills if r.strip()]))
        cand_clean = list(set([c.strip() for c in candidate_skills if c.strip()]))

        if skill_embeddings is not None:
            # Retrieve from cache
            req_embeddings = np.array([skill_embeddings[r] for r in req_clean if r in skill_embeddings])
            cand_embeddings = np.array([skill_embeddings[c] for c in cand_clean if c in skill_embeddings])
        else:
            # Embed on the fly
            req_contextual = [self._get_contextualized_skill(r) for r in req_clean]
            cand_contextual = [self._get_contextualized_skill(c) for c in cand_clean]
            req_embeddings = self.model.encode(req_contextual, show_progress_bar=False, convert_to_numpy=True)
            cand_embeddings = self.model.encode(cand_contextual, show_progress_bar=False, convert_to_numpy=True)

        if len(req_embeddings) == 0 or len(cand_embeddings) == 0:
            return {
                "exact_matches": [],
                "transferable_skills": {},
                "related_skills": {},
                "missing_skills": req_clean,
                "score": 0.0
            }

        # Compute pairwise cosine similarity matrix
        sim_matrix = util.cos_sim(req_embeddings, cand_embeddings).cpu().numpy()

        exact_matches = []
        transferable_skills = {}
        related_skills = {}
        missing_skills = []

        total_points = 0.0

        for idx, req_skill in enumerate(req_clean):
            # Check for exact string matches first
            exact_found = False
            for cand_skill in cand_clean:
                if req_skill.lower() == cand_skill.lower():
                    exact_matches.append(req_skill)
                    total_points += 1.0
                    exact_found = True
                    break
            
            if exact_found:
                continue

            # Find best semantic match
            best_cand_idx = np.argmax(sim_matrix[idx])
            best_sim = float(sim_matrix[idx][best_cand_idx])
            best_cand_skill = cand_clean[best_cand_idx]

            if best_sim >= 0.95:
                exact_matches.append(req_skill)
                total_points += 1.0
            elif best_sim >= transfer_threshold:
                transferable_skills[req_skill] = {
                    "matched_skill": best_cand_skill,
                    "similarity": round(best_sim, 2)
                }
                total_points += 0.7
            elif best_sim >= related_threshold:
                related_skills[req_skill] = {
                    "matched_skill": best_cand_skill,
                    "similarity": round(best_sim, 2)
                }
                total_points += 0.4
            else:
                missing_skills.append(req_skill)
                total_points += 0.0

        score = (total_points / len(req_clean)) * 100.0

        return {
            "exact_matches": exact_matches,
            "transferable_skills": transferable_skills,
            "related_skills": related_skills,
            "missing_skills": missing_skills,
            "score": round(score, 2)
        }

    def process_dataframe(
        self, df: pd.DataFrame, required_skills: List[str]
    ) -> Tuple[pd.Series, List[Dict[str, Any]]]:
        """
        Process candidate dataframe. Pre-encodes all unique skills in the dataset
        for O(1) loop executions.
        """
        if df.empty:
            return pd.Series([], dtype=float), []

        # Performance Optimization: Pre-encode all unique skills in the dataset in a single batch
        all_unique_skills = set(required_skills)
        for _, row in df.iterrows():
            cand_skills = parse_skills_list(row.get("Skills", ""))
            all_unique_skills.update(cand_skills)
            
        unique_skills_list = list(all_unique_skills)
        if unique_skills_list:
            contextualized_skills = [self._get_contextualized_skill(s) for s in unique_skills_list]
            embeddings_list = self.model.encode(contextualized_skills, show_progress_bar=False, convert_to_numpy=True)
            skill_embeddings = {skill: emb for skill, emb in zip(unique_skills_list, embeddings_list)}
        else:
            skill_embeddings = {}

        scores = []
        details_list = []
        
        for _, row in df.iterrows():
            cand_skills = parse_skills_list(row.get("Skills", ""))
            analysis = self.analyze_skills(cand_skills, required_skills, skill_embeddings=skill_embeddings)
            scores.append(analysis["score"])
            details_list.append(analysis)
            
        return pd.Series(scores, index=df.index), details_list

if __name__ == "__main__":
    detector = HiddenTalentDetector()
    req = ["FastAPI", "PyTorch", "AWS", "React"]
    cand = ["Flask", "TensorFlow", "Azure", "Vue"]
    analysis = detector.analyze_skills(cand, req)
    print("Analysis Output:")
    import json
    print(json.dumps(analysis, indent=2))
