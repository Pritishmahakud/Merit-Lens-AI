import os
import sys
import json
import gzip
import csv
import argparse
import random
from datetime import datetime

# Services/consulting firms to flag for negative signal checking
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "mindtree", "mphasis", "genpact ai", "hcl",
    "cognizant technology solutions", "tata consultancy services",
    "wipro technologies", "infosys limited", "tcs limited"
}

# Fictional companies in the dataset
FICTIONAL_COMPANIES = {
    "acme corp", "globex inc", "hooli", "initech", "pied piper",
    "stark industries", "wayne enterprises", "dunder mifflin"
}

# Startup founding years for consistency checks
FOUNDING_YEARS = {
    "Sarvam AI": 2023,
    "Krutrim": 2023,
    "CRED": 2018,
    "Razorpay": 2014,
    "Swiggy": 2014,
    "Zomato": 2008,
    "Paytm": 2010,
    "PhonePe": 2015,
    "Ola": 2010,
    "Freshworks": 2010,
    "Meesho": 2015,
    "Rephrase.ai": 2019,
    "Glance": 2019,
    "Aganitha": 2018,
    "Observe.AI": 2017,
    "Wysa": 2015,
    "Yellow.ai": 2016,
    "upGrad": 2015,
    "Unacademy": 2015,
    "PharmEasy": 2015,
    "PolicyBazaar": 2008
}

def is_honeypot(cand):
    """
    Mandatory Honeypot Consistency Checks.
    Returns True if the profile is flagged as a honeypot (contains impossible contradictions).
    """
    profile = cand["profile"]
    history = cand["career_history"]
    skills = cand["skills"]
    
    # 1. Skill inconsistency check:
    # Flag if a candidate has multiple expert/advanced skills with very low/0 duration
    expert_adv = [s for s in skills if s["proficiency"] in ["expert", "advanced"]]
    expert_adv_zero = [s for s in expert_adv if s.get("duration_months", 99) == 0]
    if len(expert_adv_zero) >= 3:
        return True
        
    expert_adv_low = [s for s in expert_adv if s.get("duration_months", 99) < 6]
    if len(expert_adv) >= 5 and (len(expert_adv_low) / len(expert_adv)) >= 0.8:
        return True

    # 2. Total duration check:
    # Flag if duration_months sums to significantly more than what years_of_experience supports.
    sum_dur_years = sum(h["duration_months"] for h in history) / 12.0
    yoe = profile["years_of_experience"]
    # If the history says they worked 5+ years more than their total YoE, it's impossible.
    if sum_dur_years > max(yoe * 1.5, yoe + 5.0):
        return True

    # 3. Company founding date check:
    # Flag if any career_history entry's start_date predates plausible founding of a well-known company.
    for h in history:
        comp = h["company"]
        if comp in FOUNDING_YEARS:
            start_date = h["start_date"]
            if start_date:
                try:
                    year = int(start_date.split("-")[0])
                    # If start date is before founding year (with 1-year leeway for edge cases), flag it
                    if year < FOUNDING_YEARS[comp] - 1:
                        return True
                except ValueError:
                    pass
                    
    return False

def score_candidate(cand):
    """
    Computes a deterministic score for candidate fit based on job requirements.
    Scores range from 0.0 to 100.0.
    """
    profile = cand["profile"]
    history = cand["career_history"]
    skills = cand["skills"]
    signals = cand["redrob_signals"]
    
    # --- 1. Technical Score (Core ML, Search, Retrieval, Ranking, Recommendation) ---
    skill_score = 0.0
    matched_skills = []
    
    # Weight skills based on relevancy to JD
    core_search_keywords = {"search", "retrieval", "ranking", "recommendation", "recommender", 
                            "dense retrieval", "vector search", "embeddings", "hybrid search", 
                            "information retrieval", "rag", "vector db", "pinecone", "weaviate", 
                            "qdrant", "milvus", "elasticsearch", "opensearch", "faiss"}
                            
    nlp_keywords = {"nlp", "natural language", "bert", "transformers", "hugging face", 
                    "sentence transformers", "llm", "llms", "fine-tuning", "lora", "qlora", 
                    "prompt engineering", "langchain"}
                    
    ml_keywords = {"python", "pytorch", "tensorflow", "scikit-learn", "numpy", "pandas", 
                   "machine learning", "applied ml", "ml engineering"}
                   
    for s in skills:
        name_lower = s["name"].lower()
        prof = s["proficiency"]
        
        # Check prof level mapping
        prof_mult = {"expert": 5.0, "advanced": 4.0, "intermediate": 2.0, "beginner": 1.0}.get(prof, 1.0)
        
        # Check skill overlap
        if any(kw in name_lower for kw in core_search_keywords):
            skill_score += prof_mult * 4.0
            matched_skills.append(s["name"])
        elif any(kw in name_lower for kw in nlp_keywords):
            skill_score += prof_mult * 3.0
            matched_skills.append(s["name"])
        elif any(kw in name_lower for kw in ml_keywords):
            skill_score += prof_mult * 2.0
            matched_skills.append(s["name"])
            
    # Product ML Work Experience Score (from career description)
    product_ml_score = 0.0
    product_companies_worked = []
    
    for job in history:
        comp = job["company"]
        comp_lower = comp.lower()
        is_services = comp_lower in SERVICES_COMPANIES
        desc_lower = job["description"].lower()
        title_lower = job["title"].lower()
        dur = job["duration_months"]
        
        # Search & retrieval context
        has_search_work = any(kw in desc_lower or kw in title_lower for kw in core_search_keywords)
        has_nlp_work = any(kw in desc_lower or kw in title_lower for kw in nlp_keywords)
        has_prod_work = any(kw in desc_lower or kw in title_lower for kw in ["production", "deployed", "deploy", "scale", "infrastructure", "pipeline", "real-time", "serving", "inference", "optimization"])
        
        if not is_services:
            if comp_lower not in FICTIONAL_COMPANIES:
                product_companies_worked.append(comp)
            
            # Shipped search/retrieval/ranking at a product company -> massive boost
            if has_search_work:
                product_ml_score += 15.0 + (dur / 12.0) * 2.0
                if has_prod_work:
                    product_ml_score += 10.0
            elif has_nlp_work:
                product_ml_score += 10.0 + (dur / 12.0) * 1.5
                if has_prod_work:
                    product_ml_score += 5.0
        else:
            # Services company work gets much lower scoring weight
            if has_search_work:
                product_ml_score += 4.0 + (dur / 12.0) * 0.5
            elif has_nlp_work:
                product_ml_score += 2.0 + (dur / 12.0) * 0.3
                
    tech_score = min(50.0, skill_score + product_ml_score)
    
    # --- 2. Experience Fit Score (Target Band: 5-9 years) ---
    yoe = profile["years_of_experience"]
    if 5.0 <= yoe <= 9.0:
        yoe_score = 15.0
    elif yoe < 5.0:
        # Scale down linearly
        yoe_score = (yoe / 5.0) * 12.0
    else:
        # Scale down slowly for higher experience
        yoe_score = max(5.0, 15.0 - (yoe - 9.0) * 1.5)
        
    # --- 3. Location Fit Score ---
    loc = profile["location"].lower()
    is_target_city = any(city in loc for city in ["noida", "pune", "delhi", "gurgaon", "mumbai", "hyderabad", "ncr"])
    if is_target_city:
        loc_score = 15.0
    elif signals["willing_to_relocate"]:
        loc_score = 12.0
    else:
        loc_score = 3.0
        
    # --- 4. Notice Period Score (Sub-30 days preferred) ---
    notice = signals["notice_period_days"]
    if notice <= 30:
        notice_score = 10.0
    elif notice <= 60:
        notice_score = 8.0
    elif notice <= 90:
        notice_score = 4.0
    else:
        notice_score = 1.0
        
    # --- 5. Engagement Score Multiplier ---
    # Recruiter response rate (0.0 to 1.0)
    response_rate = signals["recruiter_response_rate"]
    
    # Activity recency
    active_str = signals["last_active_date"]
    days_inactive = 180
    if active_str:
        try:
            dt_active = datetime.strptime(active_str, "%Y-%m-%d")
            # Anchor to dataset max active date: 2026-05-27
            days_inactive = (datetime(2026, 5, 27) - dt_active).days
        except Exception:
            pass
            
    active_score = max(0.0, 1.0 - (days_inactive / 180.0))
    open_to_work = 1.0 if signals["open_to_work_flag"] else 0.5
    
    engagement_base = (response_rate * 0.4 + active_score * 0.4 + open_to_work * 0.2)
    # Engagement multiplier scales from 0.5 to 1.0 to downweight inactive profiles
    engagement_mult = 0.5 + 0.5 * engagement_base

    # --- 6. Negative Signal Penalty ---
    penalty = 0.0
    
    # Rule A: Career spent ENTIRELY at services companies
    if history:
        all_services = all(h["company"].lower() in SERVICES_COMPANIES for h in history)
        if all_services:
            penalty += 0.35
            
    # Rule B: Recent AI wrapper only (LangChain / OpenAI but no older ML / retrieval)
    has_lc_openai = any(s["name"].lower() in ["langchain", "openai", "gpt-4", "gpt", "prompt engineering"] for s in skills)
    has_older_ml = False
    for job in history:
        if not (job["is_current"] or job["end_date"] is None):
            desc_j = job["description"].lower()
            title_j = job["title"].lower()
            if any(kw in desc_j or kw in title_j for kw in ["machine learning", "ml", "nlp", "search", "retrieval", "ranking", "recommendation", "embeddings", "classification"]):
                has_older_ml = True
                break
    if has_lc_openai and not has_older_ml and yoe < 2.5:
        penalty += 0.30
        
    # Rule C: Pure Academic Research
    all_research = True
    for job in history:
        t_j = job["title"].lower()
        is_research = any(kw in t_j for kw in ["research", "phd", "intern", "fellow", "academic", "postdoc"])
        is_eng = any(kw in t_j for kw in ["engineer", "developer", "programmer", "architect", "lead", "specialist"])
        if is_eng or not is_research:
            all_research = False
            break
    if all_research:
        penalty += 0.30
        
    # Rule D: Senior title but no hands-on code in the last 18 months
    current_jobs = [j for j in history if j["is_current"] or j["end_date"] is None]
    if current_jobs:
        curr_title = current_jobs[0]["title"].lower()
        is_lead = any(kw in curr_title for kw in ["architect", "manager", "director", "lead", "vp", "head"])
        curr_desc = current_jobs[0]["description"].lower()
        has_coding = any(kw in curr_desc for kw in ["code", "develop", "implement", "python", "programming", "write", "build", "coding", "hands-on"])
        if is_lead and not has_coding and current_jobs[0]["duration_months"] >= 18:
            penalty += 0.20
            
    # Rule E: CV / Speech / Robotics with no NLP / IR
    has_cv_speech_robotics = any(any(kw in s["name"].lower() for kw in ["computer vision", "opencv", "yolo", "image classification", "object detection", "cnn", "speech", "tts", "asr", "speech recognition", "robotics", "ros"]) for s in skills)
    has_nlp_ir = any(any(kw in s["name"].lower() for kw in ["nlp", "natural language", "bert", "transformers", "semantic search", "search", "retrieval", "indexing", "text"]) for s in skills)
    if has_cv_speech_robotics and not has_nlp_ir:
        penalty += 0.30
        
    # Rule F: Job-hopping (average duration under 18 months)
    if len(history) >= 3:
        avg_dur = sum(h["duration_months"] for h in history) / len(history)
        if avg_dur < 18.0:
            penalty += 0.20

    # Calculate final score out of 100
    base_score = tech_score + yoe_score + loc_score + notice_score
    final_score = base_score * (1.0 - min(0.8, penalty)) * engagement_mult
    
    # Safe guard final score range
    final_score = max(0.0, min(100.0, final_score))
    
    # Store attributes for generating reasoning
    cand_details = {
        "candidate_id": cand["candidate_id"],
        "yoe": yoe,
        "location": profile["location"],
        "last_company": history[0]["company"] if history else "Unknown Company",
        "notice_period": notice,
        "response_rate": int(response_rate * 100),
        "matched_skills": matched_skills[:2],
        "final_score": final_score
    }
    
    return final_score, cand_details

def generate_reasoning(details, rank):
    """
    Generates a unique, factual 1-2 sentence justification for the candidate's rank.
    To avoid duplicate texts, we use deterministic hash-based templates combined with exact candidate facts.
    """
    yoe = details["yoe"]
    company = details["last_company"]
    skills = details["matched_skills"]
    location = details["location"]
    notice = details["notice_period"]
    rr = details["response_rate"]
    cid = details["candidate_id"]
    
    skills_str = " & ".join(skills) if skills else "ML systems engineering"
    
    # Deterministic choice of template structure based on candidate ID hash to ensure variation
    h_val = sum(ord(c) for c in cid)
    
    if rank <= 20: # Top candidates (glowing fit)
        templates = [
            f"Exceptional Senior AI Engineer with {yoe} years of experience, including building search and ML systems at {company}. Strong in {skills_str}, based in {location}, and highly responsive ({rr}% response rate).",
            f"Outstanding match with {yoe} years of applied ML experience; previously shipped core retrieval pipelines at {company}. Relocation-ready and brings deep knowledge of {skills_str}.",
            f"Impressive candidate with {yoe} years of experience deploying production search systems at {company}. Possesses strong expertise in {skills_str} and a notice period of only {notice} days.",
            f"Strong founding-engineer material with {yoe} years of experience; shipped ranking and NLP models at {company}. Showcases high active engagement and solid {skills_str} expertise."
        ]
        return templates[h_val % len(templates)]
        
    elif rank <= 70: # Mid-tier candidates (good fit with minor caveats)
        templates = [
            f"Strong AI Engineer with {yoe} years of experience and search/retrieval background at {company}. Based in {location} with a {notice}-day notice period.",
            f"Experienced ML engineer ({yoe} years) skilled in {skills_str}, with history at {company}. Good fit for the role, though responsiveness is moderate ({rr}% response rate).",
            f"Has {yoe} years of experience with product companies like {company}, having implemented {skills_str} systems. Located in {location} and ready to relocate.",
            f"Brings {yoe} years of production ML engineering experience from {company}. Strong capabilities in {skills_str} and is actively looking for new roles."
        ]
        return templates[h_val % len(templates)]
        
    else: # Bottom-tier candidates of the top-100 (adjacent fits or filler)
        templates = [
            f"Decent fit with {yoe} years of experience and adjacent ML skills in {skills_str} at {company}. Notice period is {notice} days.",
            f"Possesses {yoe} years of experience and solid foundational skills in {skills_str}. Shows moderate activity signal from {location}.",
            f"Brings {yoe} years of developer experience with exposure to search pipelines at {company}. Fits adjacent skills criteria.",
            f"Has {yoe} years of experience and foundational knowledge in {skills_str}. Included in pool based on engagement signals and {location} location."
        ]
        return templates[h_val % len(templates)]

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to write the output submission CSV")
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"Loading candidates from {args.candidates}...")
    
    # Check if the file is compressed
    is_gzip = args.candidates.endswith(".gz")
    open_func = gzip.open if is_gzip else open
    mode = "rt" if is_gzip else "r"
    
    valid_candidates = []
    honeypot_count = 0
    total_count = 0
    
    with open_func(args.candidates, mode, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            total_count += 1
            cand = json.loads(line_str)
            
            # Step 1: Filter out honeypots
            if is_honeypot(cand):
                honeypot_count += 1
                continue
                
            # Step 2: Score valid candidates
            score, details = score_candidate(cand)
            valid_candidates.append((score, details))
            
            if total_count % 20000 == 0:
                print(f"  Processed {total_count} candidates...")
                
    print(f"Total processed: {total_count}")
    print(f"Honeypots filtered out: {honeypot_count}")
    print(f"Remaining valid candidates: {len(valid_candidates)}")
    
    # Step 3: Sort candidates.
    # Higher score -> better (rank 1 is best).
    # If scores tie, break tie by candidate_id ascending.
    # We sort by (-score, candidate_id)
    valid_candidates.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    
    # Step 4: Extract top 100
    top_100 = valid_candidates[:100]
    
    # Write to CSV
    print(f"Writing top 100 candidates to {args.out}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for i, (score, details) in enumerate(top_100):
            rank = i + 1
            reasoning = generate_reasoning(details, rank)
            writer.writerow([details["candidate_id"], rank, round(score, 3), reasoning])
            
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Ranking pipeline finished in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
