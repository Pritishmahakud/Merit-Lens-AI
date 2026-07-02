import os
# Configure CPU thread limits and memory constraints BEFORE importing any ML packages
# This prevents RAM spikes and OOM (Out Of Memory) crashes on cloud servers like Render Free Tier.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import urllib.parse
import pandas as pd
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("PORT", 8000))

class RecruiterDashboardHandler(BaseHTTPRequestHandler):
    """
    Lightweight API and static asset handler for the Merit Lens AI Recruiter Web Dashboard.
    """
    def log_message(self, format, *args):
        # Override to suppress default console spam from requests
        pass

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        # Parse query parameters if any
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/" or path == "/index.html":
            self._serve_static_file("index.html", "text/html")
        elif path.startswith("/static/"):
            # Resolve static asset type
            file_path = path.lstrip("/")
            content_type = "text/plain"
            if file_path.endswith(".css"):
                content_type = "text/css"
            elif file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".png"):
                content_type = "image/png"
            elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                content_type = "image/jpeg"
            self._serve_static_file(file_path, content_type)
        elif path == "/api/candidates":
            self._handle_get_candidates()
        elif path == "/api/explanations":
            self._handle_get_explanations()
        elif path == "/api/download":
            self._handle_get_download()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/rank":
            self._handle_post_rank()
        elif path == "/api/upload":
            self._handle_post_upload()
        else:
            self.send_error(404, "Endpoint Not Found")

    def _serve_static_file(self, rel_path: str, content_type: str):
        if not os.path.exists(rel_path):
            self.send_error(404, f"File {rel_path} Not Found")
            return
        
        try:
            with open(rel_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal server error: {e}")

    def _handle_get_candidates(self):
        csv_path = "ranked_candidates.csv"
        if not os.path.exists(csv_path):
            # If pipeline hasn't been run yet, return empty list
            self._send_json([])
            return

        try:
            df = pd.read_csv(csv_path)
            # Replace NaNs with None/null for valid JSON rendering
            df = df.replace({np.nan: None})
            candidates = df.to_dict(orient="records")
            self._send_json(candidates)
        except Exception as e:
            self.send_response(500)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(f"Error loading candidates: {e}".encode("utf-8"))

    def _handle_get_explanations(self):
        json_path = "candidate_explanations.json"
        if not os.path.exists(json_path):
            self._send_json([])
            return

        try:
            with open(json_path, "r") as f:
                explanations = json.load(f)
            self._send_json(explanations)
        except Exception as e:
            self.send_response(500)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(f"Error loading explanations: {e}".encode("utf-8"))

    def _handle_get_download(self):
        csv_path = "processed_resume_data.csv"
        if not os.path.exists(csv_path):
            csv_path = "AI_Resume_Screening.csv"
            
        if not os.path.exists(csv_path):
            self.send_error(404, "No candidate pool dataset found to download")
            return
            
        try:
            with open(csv_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=candidate_pool.csv")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(f"Error downloading dataset: {e}".encode("utf-8"))

    def _handle_post_rank(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            params = json.loads(post_data.decode("utf-8"))
            job_role = params.get("job_role", "Data Scientist")
            job_desc = params.get("job_desc", None)
            
            req_skills = params.get("req_skills", None)
            req_skills_list = None
            if req_skills:
                req_skills_list = [s.strip() for s in req_skills.split(",") if s.strip()]

            print(f"\n[Web Server] Triggering ranking pipeline for role: {job_role}")
            
            # Dynamic import here to keep server boot footprint tiny (prevents OOM on startup)
            from run_pipeline import run_pipeline
            
            # Execute the ranking pipeline programmatically
            final_df, explanations = run_pipeline(
                job_role=job_role,
                job_description=job_desc,
                required_skills=req_skills_list
            )
            
            # Replace NaNs for JSON compatibility
            final_df = final_df.replace({np.nan: None})
            candidates = final_df.to_dict(orient="records")
            
            response_payload = {
                "success": True,
                "candidates": candidates,
                "explanations": explanations
            }
            self._send_json(response_payload)
            
            # Trigger garbage collection immediately to release model RAM
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"[Web Server] Pipeline execution failed: {e}")
            self.send_response(500)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

    def _handle_post_upload(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            print("[Web Server] New candidate pool upload received. Overwriting raw pool...")
            # Save the uploaded CSV to the raw candidates path
            with open("AI_Resume_Screening.csv", "wb") as f:
                f.write(post_data)
                
            # Run the pre-processing stage to clean and build features
            from src.preprocessing import run_preprocessing
            run_preprocessing("AI_Resume_Screening.csv", "processed_resume_data.csv")
            
            # Delete any previous ranked output files to force a recalculation
            if os.path.exists("ranked_candidates.csv"):
                os.remove("ranked_candidates.csv")
            if os.path.exists("candidate_explanations.json"):
                os.remove("candidate_explanations.json")
                
            self._send_json({"success": True})
        except Exception as e:
            print(f"[Web Server] Candidate upload or pre-processing failed: {e}")
            self.send_response(500)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

    def _send_json(self, data: any):
        try:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_error(500, f"Error encoding JSON: {e}")

def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, RecruiterDashboardHandler)
    print("=" * 60)
    print(f"MERIT LENS AI DASHBOARD SERVER ACTIVE AT http://localhost:{PORT}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Web Server] Server shutting down...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
