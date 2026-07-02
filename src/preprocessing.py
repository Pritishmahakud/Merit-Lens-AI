import pandas as pd
import random

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the notebook preprocessing logic on the input DataFrame.
    Includes bias-neutral steps (simulating and dropping Name, Gender, College Tier)
    and calculating standard scoring metrics.
    """
    processed_df = df.copy()
    
    # Simulate Bias Variables (Gender and College Tier) for demonstration purposes
    random.seed(42)  # Set seed for reproducibility of random assignments
    genders = ['Male', 'Female']
    tiers = ['Tier-1', 'Tier-2', 'Tier-3']
    processed_df['Gender'] = [random.choice(genders) for _ in range(len(processed_df))]
    processed_df['College Tier'] = [random.choice(tiers) for _ in range(len(processed_df))]
    
    # Anonymization / Bias Masking: Drop Name, Gender, and College Tier
    processed_df = processed_df.drop(columns=['Name', 'Gender', 'College Tier'], errors='ignore')
    
    # Basic Feature Engineering from notebook
    # 1. Skill Count (comma separated)
    processed_df['skill_count'] = processed_df['Skills'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )
    
    # 2. Experience Score (mapping years of experience directly)
    processed_df['experience_score'] = processed_df['Experience (Years)'].fillna(0)
    
    # 3. Certification Count (comma separated, handling nan as 0)
    processed_df['cert_count'] = processed_df['Certifications'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).lower() != 'nan' else 0
    )
    
    # 4. Education Score Mapping
    edu_map = {
        'b.tech': 3,
        'mba': 4,
        'phd': 5
    }
    def get_edu_score(x):
        if pd.isna(x):
            return 1
        return edu_map.get(str(x).lower(), 1)
        
    processed_df['education_score'] = processed_df['Education'].apply(get_edu_score)
    
    # 5. Project Score (mapping Projects Count directly)
    processed_df['project_score'] = processed_df['Projects Count'].fillna(0)
    
    # 6. Domain Match (Notebook simply copies Job Role)
    processed_df['domain_match'] = processed_df['Job Role']
    
    return processed_df

def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """Load raw data, perform preprocessing, and save to output CSV."""
    df = pd.read_csv(input_path)
    processed_df = preprocess_dataframe(df)
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed {len(processed_df)} resumes and saved to {output_path}")
    return processed_df

if __name__ == "__main__":
    # Test script if run independently
    run_preprocessing("AI_Resume_Screening.csv", "processed_resume_data.csv")
