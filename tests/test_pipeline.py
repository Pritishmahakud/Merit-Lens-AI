import unittest
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocessing import preprocess_dataframe
from src.semantic_match import SemanticMatcher
from src.hidden_talent import HiddenTalentDetector
from src.growth_score import GrowthPotentialScorer
from src.feature_engineering import FeatureEngineer
from src.ranking_engine import HybridRankingEngine
from src.llm_explainer import LLMExplainer
from src.utils import load_sentence_transformer

class TestMeritLensPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the default lightweight model for test validation
        cls.shared_model = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        cls.target_role = "AI Researcher"
        cls.target_desc = "Develop deep learning algorithms using PyTorch and TensorFlow."
        cls.required_skills = ["Python", "PyTorch", "TensorFlow", "NLP"]

    def setUp(self):
        # Create a mock raw dataset representing one top candidate and one weak candidate
        self.mock_raw_df = pd.DataFrame([
            {
                "Resume_ID": 1,
                "Name": "Ashley Ali",
                "Skills": "TensorFlow, NLP, Pytorch",
                "Experience (Years)": 10,
                "Education": "PhD",
                "Certifications": "Deep Learning Specialization, Google ML",
                "Job Role": "AI Researcher",
                "Recruiter Decision": "Hire",
                "Salary Expectation ($)": 100000,
                "Projects Count": 8,
                "AI Score (0-100)": 100
            },
            {
                "Resume_ID": 2,
                "Name": "John Doe",
                "Skills": "SQL",
                "Experience (Years)": 1,
                "Education": "B.Sc",
                "Certifications": "",
                "Job Role": "Software Engineer",
                "Recruiter Decision": "Reject",
                "Salary Expectation ($)": 50000,
                "Projects Count": 1,
                "AI Score (0-100)": 30
            }
        ])

    def test_preprocessing(self):
        """Verify preprocessing anonymizes names and calculates expected score attributes."""
        processed_df = preprocess_dataframe(self.mock_raw_df)
        
        # Check that Name is removed
        self.assertNotIn("Name", processed_df.columns)
        self.assertNotIn("Gender", processed_df.columns)
        self.assertNotIn("College Tier", processed_df.columns)
        
        # Check engineered features
        self.assertIn("skill_count", processed_df.columns)
        self.assertEqual(processed_df.loc[0, "skill_count"], 3)
        self.assertEqual(processed_df.loc[1, "skill_count"], 1)
        self.assertEqual(processed_df.loc[0, "education_score"], 5) # PhD -> 5
        self.assertEqual(processed_df.loc[0, "cert_count"], 2)

    def test_semantic_matching(self):
        """Verify semantic similarity calculation scales and calculates correct output."""
        matcher = SemanticMatcher(model_name="sentence-transformers/all-MiniLM-L6-v2")
        matcher.model = self.shared_model
        
        processed_df = preprocess_dataframe(self.mock_raw_df)
        scores = matcher.score_candidates(processed_df, self.target_desc)
        
        self.assertEqual(len(scores), 2)
        # Candidate 1 should have higher semantic similarity than Candidate 2
        self.assertGreater(scores.iloc[0], scores.iloc[1])
        # Verify scores are in 0-100 range
        for score in scores:
            self.assertTrue(0.0 <= score <= 100.0)

    def test_hidden_talent_detection(self):
        """Verify transferable skill mapping successfully matches concept pairs (e.g. Flask -> FastAPI)."""
        detector = HiddenTalentDetector(model=self.shared_model)
        
        # Flask and FastAPI should show transferable similarity
        analysis = detector.analyze_skills(["Flask"], ["FastAPI"])
        self.assertIn("FastAPI", analysis["transferable_skills"])
        self.assertEqual(analysis["transferable_skills"]["FastAPI"]["matched_skill"], "Flask")
        self.assertGreater(analysis["transferable_skills"]["FastAPI"]["similarity"], 0.50)
        
        # Compute scoring for mock candidates
        processed_df = preprocess_dataframe(self.mock_raw_df)
        scores, details = detector.process_dataframe(processed_df, self.required_skills)
        
        self.assertEqual(len(scores), 2)
        # Candidate 1 (skills: TensorFlow, NLP, Pytorch) should score high against required_skills (Python, PyTorch, TensorFlow, NLP)
        self.assertGreater(scores.iloc[0], scores.iloc[1])

    def test_growth_scorer(self):
        """Verify growth potential score calculations remain within 0-100 bounds."""
        scorer = GrowthPotentialScorer(model=self.shared_model)
        
        processed_df = preprocess_dataframe(self.mock_raw_df)
        scores = scorer.score_candidates(processed_df)
        
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertTrue(0.0 <= score <= 100.0)
        # Candidate 1 (PhD, 8 projects, 2 certs) should have a higher growth score than Candidate 2
        self.assertGreater(scores.iloc[0], scores.iloc[1])

    def test_feature_engineering(self):
        """Verify that numerical features are computed correctly without division by zero."""
        fe = FeatureEngineer(model=self.shared_model)
        processed_df = preprocess_dataframe(self.mock_raw_df)
        
        engineered_df = fe.process(processed_df, self.target_role, self.required_skills)
        
        self.assertIn("domain_match_score", engineered_df.columns)
        self.assertIn("experience_to_projects_ratio", engineered_df.columns)
        self.assertIn("skills_per_year_experience", engineered_df.columns)
        self.assertIn("certifications_per_year_experience", engineered_df.columns)
        self.assertIn("salary_to_experience_ratio", engineered_df.columns)
        self.assertIn("adjusted_experience_score", engineered_df.columns)
        
        # Candidate 1 is an AI Researcher, which matches the target role, score should be high
        self.assertGreater(engineered_df.loc[0, "domain_match_score"], 80)

    def test_ranking_engine(self):
        """Verify hybrid ranking assigns rank 1 to the best candidate."""
        ranking_engine = HybridRankingEngine()
        
        # Mock series scores
        semantic_scores = pd.Series([90.0, 30.0], index=[0, 1])
        skill_match_scores = pd.Series([75.0, 0.0], index=[0, 1])
        hidden_talent_scores = pd.Series([85.0, 10.0], index=[0, 1])
        growth_scores = pd.Series([95.0, 20.0], index=[0, 1])
        project_relevance_scores = pd.Series([80.0, 10.0], index=[0, 1])
        llm_scores = pd.Series([90.0, 30.0], index=[0, 1])
        
        processed_df = preprocess_dataframe(self.mock_raw_df)
        
        ranked_df = ranking_engine.rank_candidates(
            processed_df, semantic_scores, skill_match_scores,
            hidden_talent_scores, growth_scores, project_relevance_scores,
            llm_evaluation_scores=llm_scores
        )
        
        self.assertEqual(len(ranked_df), 2)
        # Candidate 1 (index 0) must be ranked 1st
        self.assertEqual(ranked_df.iloc[0]["Resume_ID"], 1)
        self.assertEqual(ranked_df.iloc[0]["candidate_rank"], 1)
        
        # Verify rank 2
        self.assertEqual(ranked_df.iloc[1]["Resume_ID"], 2)
        self.assertEqual(ranked_df.iloc[1]["candidate_rank"], 2)

if __name__ == "__main__":
    unittest.main()
