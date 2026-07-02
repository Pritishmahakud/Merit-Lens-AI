import os
import argparse
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import our modular classes
from src.preprocessing import run_preprocessing
from src.utils import DEFAULT_JOB_DESCRIPTIONS, DEFAULT_REQUIRED_SKILLS, parse_skills_list, load_sentence_transformer
from src.semantic_match import SemanticMatcher
from src.hidden_talent import HiddenTalentDetector
from src.growth_score import GrowthPotentialScorer
from src.feature_engineering import FeatureEngineer
from src.ranking_engine import HybridRankingEngine
from src.llm_explainer import LLMExplainer

def run_pipeline(
    job_role: str,
    job_description: str = None,
    required_skills: list = None,
    llm_top_n: int = 20,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    raw_data_path: str = "AI_Resume_Screening.csv",
    processed_data_path: str = "processed_resume_data.csv",
    output_ranked_path: str = "ranked_candidates.csv",
    output_explanations_path: str = "candidate_explanations.json"
):
    print("=" * 60)
    print("STARTING MERIT LENS AI RANKING PIPELINE")
    print("=" * 60)
    
    # 1. Resolve Target Job Configuration
    if job_role not in DEFAULT_JOB_DESCRIPTIONS:
        print(f"Warning: Job role '{job_role}' is not in pre-defined roles.")
        if not job_description:
            raise ValueError(f"Job Description is required for custom role '{job_role}'.")
        if not required_skills:
            raise ValueError(f"Required skills list is required for custom role '{job_role}'.")
    else:
        if not job_description:
            job_description = DEFAULT_JOB_DESCRIPTIONS[job_role]
        if not required_skills:
            required_skills = DEFAULT_REQUIRED_SKILLS[job_role]
            
    print(f"Target Job Role: {job_role}")
    print(f"Required Skills: {required_skills}")
    print(f"Job Description: {job_description}\n")

    # 2. Check and run preprocessing
    if not os.path.exists(processed_data_path):
        print(f"Processed file '{processed_data_path}' not found. Running Preprocessing stage...")
        run_preprocessing(raw_data_path, processed_data_path)
    else:
        print(f"Loading pre-processed resume data from '{processed_data_path}'...")
        
    df = pd.read_csv(processed_data_path)
    print(f"Loaded {len(df)} candidate records.")

    # 3. Load Embedding Model (Shared instance to optimize memory and performance)
    print(f"Loading Sentence Transformer model '{model_name}'...")
    shared_transformer = load_sentence_transformer(model_name)

    
    # 4. Instantiate modules
    semantic_matcher = SemanticMatcher(model_name=model_name)
    semantic_matcher.model = shared_transformer # Reuse model
    
    talent_detector = HiddenTalentDetector(model=shared_transformer)
    growth_scorer = GrowthPotentialScorer(model=shared_transformer)
    feature_engineer = FeatureEngineer(model=shared_transformer)
    
    llm_explainer = LLMExplainer(
        target_role=job_role,
        target_description=job_description,
        required_skills=required_skills
    )
    
    ranking_engine = HybridRankingEngine()

    # 5. Compute Component Scores
    print("\nComputing Semantic Match scores...")
    semantic_scores = semantic_matcher.score_candidates(df, job_description)
    
    print("Computing Transferable and Hidden Skill scores...")
    # Hidden talent scores is the Transferable Skill Score
    hidden_talent_scores, details_list = talent_detector.process_dataframe(df, required_skills)
    
    # Skill match scores is the Exact Match percentage
    exact_match_scores = pd.Series([
        (len(d["exact_matches"]) / len(required_skills)) * 100.0 if required_skills else 100.0
        for d in details_list
    ], index=df.index)
    
    # Add cert_count to df if not already computed or loaded
    # (should be there in processed_resume_data.csv)
    if "cert_count" not in df.columns:
        df["cert_count"] = df["Certifications"].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).lower() != 'nan' else 0
        )
        
    print("Computing Growth Potential scores...")
    growth_scores = growth_scorer.score_candidates(df)
    
    print("Running Feature Engineering and Domain Match Analysis...")
    # Feature engineering returns df with domain_match_score and numerical features
    df_engineered = feature_engineer.process(df, job_role, required_skills)
    
    # Calculate Project Relevance: Project score * Domain Match Score (scaled 0-100)
    project_relevance_scores = df_engineered["project_score"] * df_engineered["domain_match_score"] / 10.0
    project_relevance_scores = project_relevance_scores.fillna(0.0)

    # 6. Execute Two-Pass Ranking and LLM Explanation
    print("\nExecuting Hybrid Ranking Engine...")
    final_ranked, llm_explanations = ranking_engine.rank_candidates_two_pass(
        df_engineered,
        semantic_scores=semantic_scores,
        skill_match_scores=exact_match_scores,
        hidden_talent_scores=hidden_talent_scores,
        growth_scores=growth_scores,
        project_relevance_scores=project_relevance_scores,
        llm_explainer=llm_explainer,
        top_n=llm_top_n
    )

    # 7. Save Output Files
    print(f"\nSaving ranked candidates to '{output_ranked_path}'...")
    final_ranked.to_csv(output_ranked_path, index=False)
    
    print(f"Saving candidate evaluations/explanations to '{output_explanations_path}'...")
    with open(output_explanations_path, "w") as f:
        json.dump(llm_explanations, f, indent=4)
        
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    
    # Display top 5 candidates
    print("\nTOP 5 CANDIDATES:")
    cols_to_show = ["Resume_ID", "Skills", "Experience (Years)", "Education", "overall_score", "candidate_rank"]
    print(final_ranked[cols_to_show].head().to_string(index=False))
    print("=" * 60)
    
    return final_ranked, llm_explanations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merit Lens AI Candidate Ranking Pipeline")
    parser.add_argument(
        "--job-role", 
        type=str, 
        default="Data Scientist",
        choices=["AI Researcher", "Data Scientist", "Cybersecurity Analyst", "Software Engineer"],
        help="Target job role for candidate matching"
    )
    parser.add_argument(
        "--job-desc",
        type=str,
        default=None,
        help="Optional custom job description text"
    )
    parser.add_argument(
        "--req-skills",
        type=str,
        default=None,
        help="Optional comma-separated list of required skills"
    )
    parser.add_argument(
        "--llm-top-n",
        type=int,
        default=20,
        help="Number of top candidates to explain using LLM"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name to use"
    )
    
    args = parser.parse_args()
    
    req_skills_list = None
    if args.req_skills:
        req_skills_list = [s.strip() for s in args.req_skills.split(",") if s.strip()]
        
    run_pipeline(
        job_role=args.job_role,
        job_description=args.job_desc,
        required_skills=req_skills_list,
        llm_top_n=args.llm_top_n,
        model_name=args.model_name
    )
