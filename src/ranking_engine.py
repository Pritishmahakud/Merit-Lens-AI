import pandas as pd
import numpy as np
from src.utils import min_max_normalize

class HybridRankingEngine:
    """
    Combines individual candidate scores using a configurable, weighted formula:
    - Semantic Similarity: 35%
    - Skill Match (Exact): 20%
    - Experience: 15%
    - Hidden Talent: 10%
    - Growth Potential: 10%
    - Project Relevance: 5%
    - LLM Evaluation: 5%
    Normalizes all metrics before combination and produces overall scores and ranks.
    """
    def __init__(self, weights: dict = None):
        self.default_weights = {
            "semantic_similarity": 0.35,
            "skill_match": 0.20,
            "experience": 0.15,
            "hidden_talent": 0.10,
            "growth_potential": 0.10,
            "project_relevance": 0.05,
            "llm_evaluation": 0.05
        }
        self.weights = weights if weights else self.default_weights
        
        # Verify weights sum to 1.0 (approximately)
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            # Normalize weights to sum to 1.0
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def compute_hybrid_score(self, row: pd.Series) -> float:
        """Calculate weighted score for a single row after individual components are normalized."""
        score = (
            row["semantic_similarity_norm"] * self.weights.get("semantic_similarity", 0.35) +
            row["skill_match_norm"] * self.weights.get("skill_match", 0.20) +
            row["experience_norm"] * self.weights.get("experience", 0.15) +
            row["hidden_talent_norm"] * self.weights.get("hidden_talent", 0.10) +
            row["growth_potential_norm"] * self.weights.get("growth_potential", 0.10) +
            row["project_relevance_norm"] * self.weights.get("project_relevance", 0.05) +
            row["llm_evaluation_norm"] * self.weights.get("llm_evaluation", 0.05)
        )
        return round(score, 2)

    def rank_candidates(
        self, df: pd.DataFrame, 
        semantic_scores: pd.Series, 
        skill_match_scores: pd.Series, # exact skill match
        hidden_talent_scores: pd.Series, # transferable/related skill score
        growth_scores: pd.Series,
        project_relevance_scores: pd.Series,
        llm_evaluation_scores: pd.Series = None
    ) -> pd.DataFrame:
        """
        Combines all candidate scores, normalizes them, and computes overall_score and candidate_rank.
        If llm_evaluation_scores is None, a placeholder is used, which can be refined after top-N ranking.
        """
        ranked_df = df.copy()
        
        # Add raw scores to DataFrame
        ranked_df["semantic_similarity"] = semantic_scores
        ranked_df["skill_match"] = skill_match_scores
        ranked_df["hidden_talent"] = hidden_talent_scores
        ranked_df["growth_potential"] = growth_scores
        ranked_df["project_relevance"] = project_relevance_scores
        
        if llm_evaluation_scores is not None:
            ranked_df["llm_evaluation"] = llm_evaluation_scores
        else:
            # If not provided, initialize with a heuristic (e.g., matching semantic similarity)
            ranked_df["llm_evaluation"] = semantic_scores.copy()

        # Min-Max Normalize all columns to 0-100 range for fair scaling
        ranked_df["semantic_similarity_norm"] = min_max_normalize(ranked_df["semantic_similarity"])
        ranked_df["skill_match_norm"] = min_max_normalize(ranked_df["skill_match"])
        # Experience score is already defined in preprocessing, let's normalize it
        ranked_df["experience_norm"] = min_max_normalize(ranked_df["experience_score"])
        ranked_df["hidden_talent_norm"] = min_max_normalize(ranked_df["hidden_talent"])
        ranked_df["growth_potential_norm"] = min_max_normalize(ranked_df["growth_potential"])
        ranked_df["project_relevance_norm"] = min_max_normalize(ranked_df["project_relevance"])
        ranked_df["llm_evaluation_norm"] = min_max_normalize(ranked_df["llm_evaluation"])

        # Compute overall score
        ranked_df["overall_score"] = ranked_df.apply(self.compute_hybrid_score, axis=1)
        
        # Sort and assign candidate_rank
        # Higher score -> better rank (rank 1 is best)
        ranked_df = ranked_df.sort_values(by="overall_score", ascending=False)
        ranked_df["candidate_rank"] = range(1, len(ranked_df) + 1)
        
        return ranked_df

    def rank_candidates_two_pass(
        self, df: pd.DataFrame,
        semantic_scores: pd.Series,
        skill_match_scores: pd.Series,
        hidden_talent_scores: pd.Series,
        growth_scores: pd.Series,
        project_relevance_scores: pd.Series,
        llm_explainer,  # LLMExplainer instance
        top_n: int = 20
    ) -> tuple[pd.DataFrame, list]:
        """
        Executes a two-pass ranking pipeline:
        Pass 1: Ranks candidates using non-LLM weights (95% of total score) to identify top-N.
        Pass 2: Runs LLM Explainer on top-N candidates to generate rich feedback and actual LLM evaluation scores.
        Merges scores back and produces the final ranked candidate dataframe.
        """
        # Step 1: Run Pass 1 (without actual LLM score, using placeholder)
        initial_ranked = self.rank_candidates(
            df, semantic_scores, skill_match_scores, 
            hidden_talent_scores, growth_scores, project_relevance_scores,
            llm_evaluation_scores=None
        )
        
        # Get top-N candidates
        top_candidates = initial_ranked.head(top_n).copy()
        
        print(f"Executing Pass 2: Running LLM Explainer on top {top_n} candidates...")
        
        # Step 2: Generate LLM explanations and actual qualitative LLM scores for top-N
        llm_results = llm_explainer.explain_candidates(top_candidates)
        
        # Map LLM scores back to full dataframe
        llm_scores_map = {}
        for res in llm_results:
            resume_id = res["resume_id"]
            llm_scores_map[resume_id] = res["current_fit_score"]
            
        # For candidates outside top-N, set LLM evaluation to their normalized semantic_similarity
        full_llm_scores = []
        for idx, row in initial_ranked.iterrows():
            res_id = row["Resume_ID"]
            if res_id in llm_scores_map:
                full_llm_scores.append(llm_scores_map[res_id])
            else:
                # Heuristic fallback for non-top candidates (based on semantic similarity)
                full_llm_scores.append(float(row["semantic_similarity"]))
                
        # Re-run ranking with actual LLM scores
        final_ranked = self.rank_candidates(
            df, semantic_scores, skill_match_scores,
            hidden_talent_scores, growth_scores, project_relevance_scores,
            llm_evaluation_scores=pd.Series(full_llm_scores, index=initial_ranked.index)
        )
        
        return final_ranked, llm_results
