import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

class FeatureEngineer:
    """
    Performs domain matching and creates engineered numerical features
    to enrich candidate profiles before ranking.
    """
    def __init__(self, model: SentenceTransformer = None):
        if model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.model = model

    def compute_domain_match_score(
        self, df: pd.DataFrame, target_role: str, target_required_skills: list[str]
    ) -> pd.Series:
        """
        Calculates a Domain Match Score (0-100) based on:
        1. Semantic similarity of the candidate's current Job Role to target Job Role.
        2. Semantic similarity of the candidate's skills to the target required skills.
        Uses batched encoding for high performance.
        """
        if df.empty:
            return pd.Series([], dtype=float)

        # 1. Embed target role and required skills
        target_role_emb = self.model.encode([target_role], show_progress_bar=False, convert_to_numpy=True)[0]
        
        target_skills_str = ", ".join(target_required_skills)
        target_skills_emb = self.model.encode([target_skills_str], show_progress_bar=False, convert_to_numpy=True)[0]

        # Performance Optimization: Batch encode unique candidate roles and skills strings
        unique_roles = [r for r in df["Job Role"].dropna().unique().tolist() if str(r).strip()]
        if unique_roles:
            role_embs = self.model.encode(unique_roles, show_progress_bar=False, convert_to_numpy=True)
            role_emb_dict = {role: emb for role, emb in zip(unique_roles, role_embs)}
        else:
            role_emb_dict = {}

        unique_skills_strs = [s for s in df["Skills"].dropna().unique().tolist() if str(s).strip()]
        if unique_skills_strs:
            skills_embs = self.model.encode(unique_skills_strs, show_progress_bar=False, convert_to_numpy=True)
            skills_emb_dict = {skills: emb for skills, emb in zip(unique_skills_strs, skills_embs)}
        else:
            skills_emb_dict = {}

        domain_scores = []

        for _, row in df.iterrows():
            candidate_role = row.get("Job Role", "Professional")
            candidate_skills_str = row.get("Skills", "")
            
            # Role Similarity
            if str(candidate_role).strip().lower() == target_role.strip().lower():
                role_sim = 1.0
            else:
                cand_role = str(candidate_role).strip()
                if cand_role in role_emb_dict:
                    cand_role_emb = role_emb_dict[cand_role]
                    role_sim = float(util.cos_sim(cand_role_emb, target_role_emb).cpu().numpy()[0][0])
                    role_sim = max(0.0, role_sim)
                else:
                    role_sim = 0.0
                
            # Skills Domain Similarity
            cand_skills = str(candidate_skills_str).strip()
            if not cand_skills or cand_skills not in skills_emb_dict:
                skills_sim = 0.0
            else:
                cand_skills_emb = skills_emb_dict[cand_skills]
                skills_sim = float(util.cos_sim(cand_skills_emb, target_skills_emb).cpu().numpy()[0][0])
                skills_sim = max(0.0, skills_sim)

            # Composite domain match score (50% role similarity + 50% skill domain similarity)
            composite_score = (role_sim * 0.5 + skills_sim * 0.5) * 100.0
            domain_scores.append(round(composite_score, 2))

        return pd.Series(domain_scores, index=df.index)

    def engineer_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates meaningful engineered numerical features:
        - experience_to_projects_ratio
        - skills_per_year_experience
        - certifications_per_year_experience
        - salary_to_experience_ratio
        - adjusted_experience_score
        """
        engineered_df = df.copy()
        
        exp = engineered_df['experience_score'].astype(float)
        proj = engineered_df['project_score'].astype(float)
        skills_cnt = engineered_df['skill_count'].astype(float)
        certs_cnt = engineered_df['cert_count'].astype(float)
        edu = engineered_df['education_score'].astype(float)
        
        salary_col = 'Salary Expectation ($)' if 'Salary Expectation ($)' in engineered_df.columns else 'Salary Expectation'
        salary = engineered_df[salary_col].astype(float)

        engineered_df['experience_to_projects_ratio'] = exp / (proj + 1.0)
        engineered_df['skills_per_year_experience'] = skills_cnt / (exp + 1.0)
        engineered_df['certifications_per_year_experience'] = certs_cnt / (exp + 1.0)
        engineered_df['salary_to_experience_ratio'] = salary / (exp + 1.0)
        engineered_df['adjusted_experience_score'] = exp + (edu * 0.5)

        return engineered_df

    def process(
        self, df: pd.DataFrame, target_role: str, target_required_skills: list[str]
    ) -> pd.DataFrame:
        """Runs the entire feature engineering pipeline on a DataFrame."""
        df_out = df.copy()
        df_out['domain_match_score'] = self.compute_domain_match_score(df_out, target_role, target_required_skills)
        df_out['domain_match'] = df_out['domain_match_score']
        df_out = self.engineer_numerical_features(df_out)
        return df_out

if __name__ == "__main__":
    fe = FeatureEngineer()
    test_df = pd.DataFrame([
        {
            "Skills": "TensorFlow, NLP, Pytorch",
            "Experience (Years)": 10,
            "Education": "B.Sc",
            "Certifications": "AWS Certified",
            "Projects Count": 8,
            "Job Role": "AI Researcher",
            "Salary Expectation ($)": 104895,
            "skill_count": 3,
            "experience_score": 10,
            "cert_count": 1,
            "education_score": 1,
            "project_score": 8
        }
    ])
    processed = fe.process(test_df, "AI Researcher", ["Python", "Pytorch", "TensorFlow"])
    print(processed.columns)
