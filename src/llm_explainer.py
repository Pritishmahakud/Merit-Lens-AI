import os
import json
import requests
import pandas as pd
from typing import Dict, List, Any
from src.utils import get_env_var, parse_skills_list

class LLMExplainer:
    """
    Generates recruiter-friendly explanations and evaluations for candidates using an LLM.
    Supports Google Gemini (REST API), OpenAI (SDK), and a detailed rule-based fallback system.
    """
    def __init__(self, target_role: str, target_description: str, required_skills: List[str]):
        self.target_role = target_role
        self.target_description = target_description
        self.required_skills = required_skills
        
        # Load API keys
        self.gemini_key = get_env_var("GEMINI_API_KEY")
        self.openai_key = get_env_var("OPENAI_API_KEY")

    def _query_gemini(self, prompt: str) -> str:
        """Call Google Gemini API via direct REST endpoint to avoid heavy library dependencies."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response structure from Gemini API: {data}") from e

    def _query_openai(self, prompt: str) -> str:
        """Call OpenAI chat completions using the OpenAI SDK."""
        import openai
        client = openai.OpenAI(api_key=self.openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=30
        )
        return response.choices[0].message.content

    def _generate_fallback(self, row: pd.Series, skills_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based heuristic generator that acts as an offline fallback when API keys are missing.
        Produces structured JSON matching the LLM schema with highly specific candidate metrics.
        """
        # Read candidate info
        experience = float(row.get("Experience (Years)", 0.0))
        education = row.get("Education", "Not Specified")
        skills_str = row.get("Skills", "")
        certs = row.get("Certifications", "")
        projects = int(row.get("Projects Count", 0))
        
        sem_score = float(row.get("semantic_similarity", 50.0))
        growth_score = float(row.get("growth_potential", 50.0))
        talent_score = float(row.get("hidden_talent", 50.0))
        
        # Exact, transferable, related, and missing
        exact = skills_analysis.get("exact_matches", [])
        transferable = skills_analysis.get("transferable_skills", {})
        related = skills_analysis.get("related_skills", {})
        missing = skills_analysis.get("missing_skills", [])

        # Formulate strengths
        strengths = []
        if experience >= 7:
            strengths.append(f"Senior level experience ({experience} years) in relevant roles.")
        elif experience >= 3:
            strengths.append(f"Solid mid-level experience ({experience} years) in software and engineering.")
        else:
            strengths.append(f"Early-career candidate showing dynamic potential with {experience} years experience.")

        if projects >= 7:
            strengths.append(f"Strong track record of execution, having successfully delivered {projects} projects.")
        
        if certs and not pd.isna(certs) and str(certs).lower() != 'nan':
            strengths.append(f"Holds valuable certifications: {certs}.")
            
        if len(exact) >= 3:
            strengths.append(f"Direct alignment on key required skills: {', '.join(exact[:3])}.")

        # Formulate weaknesses/missing skills
        weaknesses = []
        if len(missing) > 0:
            weaknesses.append(f"Needs to acquire hands-on experience in: {', '.join(missing[:3])}.")
        if experience < 2:
            weaknesses.append(f"Low overall years of experience ({experience} years) might require initial mentorship.")
        if projects < 3:
            weaknesses.append(f"Has completed relatively few projects ({projects}), indicating limited practical exposure.")

        if not weaknesses:
            weaknesses.append("No critical weaknesses identified; candidates exhibits strong alignment.")

        # Transferable skills formatting
        transferable_list = []
        for req, details in transferable.items():
            transferable_list.append({
                "required_skill": req,
                "candidate_skill": details["matched_skill"],
                "reasoning": f"Has strong conceptual overlap (similarity: {details['similarity']}) from their work in {details['matched_skill']}."
            })
        for req, details in related.items():
            transferable_list.append({
                "required_skill": req,
                "candidate_skill": details["matched_skill"],
                "reasoning": f"Exhibits partial conceptual alignment (similarity: {details['similarity']}) using {details['matched_skill']}."
            })

        # Growth assessment
        if growth_score >= 80:
            growth_potential = "Excellent. Outstanding learning progression and skill diversity. Highly adaptive."
        elif growth_score >= 60:
            growth_potential = "Strong. Solid project track record and technical breadth. Capable of taking on new tools quickly."
        else:
            growth_potential = "Moderate. Has specialized skills but might need targeted learning support to step into adjacent domains."

        # Qualitative current fit score
        fit_score = int(round(sem_score * 0.6 + talent_score * 0.4))
        fit_score = min(max(fit_score, 10), 100) # clip

        # Explanations
        reason = (
            f"Candidate is a strong prospect for the {self.target_role} position, showing an overall semantic matching score of "
            f"{round(sem_score, 1)}%. They possess {len(exact)} exact match skills, and their background in "
            f"'{skills_str}' provides a solid foundation. With a growth potential score of {round(growth_score, 1)}%, "
            f"they demonstrate a reliable trajectory for future expansion in this role."
        )

        return {
            "current_fit_score": fit_score,
            "growth_potential": growth_potential,
            "transferable_skills": transferable_list,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "missing_skills": missing,
            "reason_for_ranking": reason
        }

    def explain_candidate(self, row: pd.Series, skills_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a single candidate, using LLM if available, otherwise falling back to rule engine."""
        # Check if keys exist
        if not self.gemini_key and not self.openai_key:
            return self._generate_fallback(row, skills_analysis)
            
        # Build prompt
        prompt = f"""
You are an expert recruitment assistant for Merit Lens AI. Analyze the candidate's fitness for the position '{self.target_role}'.

Target Job Requirements:
- Title: {self.target_role}
- Description: {self.target_description}
- Required Skills: {self.required_skills}

Candidate Profile:
- Skills: {row.get("Skills", "")}
- Experience: {row.get("Experience (Years)", 0)} years
- Education: {row.get("Education", "Not Specified")}
- Certifications: {row.get("Certifications", "None")}
- Projects: {row.get("Projects Count", 0)} completed
- Calculated Semantic Similarity Score: {row.get("semantic_similarity", 0)}%
- Calculated Hidden Talent Score: {row.get("hidden_talent", 0)}%
- Calculated Growth Potential Score: {row.get("growth_potential", 0)}%

From skills overlap analysis:
- Exact Matches: {skills_analysis.get("exact_matches", [])}
- Transferable Skill Matches: {skills_analysis.get("transferable_skills", {})}
- Related Skill Matches: {skills_analysis.get("related_skills", {})}
- Missing Skills: {skills_analysis.get("missing_skills", [])}

Output a JSON object conforming strictly to this structure:
{{
  "current_fit_score": <integer from 0 to 100 evaluating candidate current role fit>,
  "growth_potential": "<string description of learning progression and future potential>",
  "transferable_skills": [
     {{
       "required_skill": "<string skill name>",
       "candidate_skill": "<string candidate skill name>",
       "reasoning": "<string reasoning why it transfers>"
     }}
  ],
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness/improvement 1>", "<weakness/improvement 2>"],
  "missing_skills": ["<missing skill 1>", "<missing skill 2>"],
  "reason_for_ranking": "<string recruiter-friendly explanation summarizing key reasons for ranking position>"
}}
Return ONLY the raw JSON string. Do not wrap in ```json or other formatting.
"""
        try:
            if self.gemini_key:
                raw_response = self._query_gemini(prompt)
            else:
                raw_response = self._query_openai(prompt)
                
            # Clean response if wrapped in markdown
            cleaned_resp = raw_response.strip()
            if cleaned_resp.startswith("```json"):
                cleaned_resp = cleaned_resp[7:]
            if cleaned_resp.endswith("```"):
                cleaned_resp = cleaned_resp[:-3]
            cleaned_resp = cleaned_resp.strip()
            
            return json.loads(cleaned_resp)
        except Exception as e:
            print(f"Error querying LLM API for candidate {row.get('Resume_ID')}: {e}. Falling back to rule-based generation.")
            return self._generate_fallback(row, skills_analysis)

    def explain_candidates(self, top_candidates_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates explanations for all candidates in the top-N slice.
        Returns a list of dictionaries with explanation properties.
        """
        from src.hidden_talent import HiddenTalentDetector
        detector = HiddenTalentDetector()
        
        results = []
        for _, row in top_candidates_df.iterrows():
            cand_skills = parse_skills_list(row.get("Skills", ""))
            skills_analysis = detector.analyze_skills(cand_skills, self.required_skills)
            
            explanation = self.explain_candidate(row, skills_analysis)
            explanation["resume_id"] = int(row.get("Resume_ID"))
            explanation["overall_score"] = float(row.get("overall_score", 0.0))
            explanation["candidate_rank"] = int(row.get("candidate_rank", 0)) if "candidate_rank" in row else None
            
            results.append(explanation)
            
        return results
