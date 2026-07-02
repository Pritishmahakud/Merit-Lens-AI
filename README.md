# Merit Lens AI - Bias-Neutral Candidate Ranking Portal

Merit Lens AI is an end-to-end recruitment matching engine designed to evaluate, rank, and explain candidate fit for target job roles using a hybrid combination of Semantic Search (SentenceTransformers), Transferable Skills Analysis, and Growth Potential scoring.

The project features a modular Python backend pipeline and an interactive, glassmorphic Recruiter Web Dashboard.

---

## 🚀 Key Features

*   **Semantic Matching Engine**: Utilizes SentenceTransformers (`all-MiniLM-L6-v2`) to perform cosine similarity matching on dense embeddings of resumes against job requirements.
*   **Transferable Skills Analysis**: Detects adjacent, non-exact skills matching job descriptions (e.g. mapping Flask experience to a FastAPI requirements) using pairwise embedding similarity.
*   **Career Growth Potential Scorer**: Evaluates a candidate's development trajectory by measuring project velocity, certification density/difficulty, experience age, and technical breadth.
*   **Performance Optimization**: Features unique-values caching reducing candidate processing times by **90%** (under 12 seconds for 1,000 profiles).
*   **Recruiter Portal Dashboard**: Fully responsive dark-mode portal featuring:
    *   **Anonymized Profiles**: Raw `Resume_ID`s mapped deterministically to names (e.g. *Arjun Sharma*, *Diego Tanaka*).
    *   **Pagination ("Load More")**: Smooth, fast data rendering in blocks of 10 to prevent browser freezing.
    *   **Interactive Details Drawer**: Highlights match scores, evaluation reasoning, transferable skill graphs, and developer strengths/weaknesses.
    *   **Custom Pool Upload/Download**: Supports uploading custom recruiter pools in CSV format and downloading the current active candidate list directly from the UI.

---

## 📁 Repository Structure

```text
├── src/                      # Backend python modules
│   ├── utils.py              # File loading, normalizations, and self-healing ML models loader
│   ├── preprocessing.py      # Cleans raw CSV candidate profiles
│   ├── semantic_match.py     # Generates embeddings and calculates cosine similarity
│   ├── hidden_talent.py      # Scans and maps transferable skills overlap
│   ├── growth_score.py       # Computes growth potential, technical breadth, and project velocities
│   ├── feature_engineering.py# Evaluates domain matching & numerical features
│   └── ranking_engine.py     # Orchestrator running the hybrid calculations & LLM explainer
├── static/                   # Web interface assets
│   ├── style.css             # Premium glassmorphic styling
│   └── app.js                # Frontend controllers, event handlers, and rendering logic
├── tests/                    # Integration & automated test suite
│   └── test_pipeline.py      # Run with: python -m unittest tests/test_pipeline.py
├── notebooks/                # Experimental work
│   └── Untitled0.ipynb       # Playground notebook
├── index.html                # Main recruiter dashboard HTML structure
├── app.py                    # Lightweight Python http.server API and file serving backend
├── run_pipeline.py           # Command line orchestrator for python run tasks
├── requirements.txt          # Python ecosystem package dependencies
└── .gitignore                # Rules to exclude heavy data, caches, and secrets from Git
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Merit_Lens.git
    cd Merit_Lens
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Running the Portal

To start the Recruiter Dashboard Web Server, run:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

### Steps to Evaluate Candidates:
1.  **Upload Pool**: (Optional) Use the **"Upload Candidate CSV"** button in the sidebar to upload your own custom candidate pool.
2.  **Select Job Role**: Pick a target job role from the dropdown menu (e.g. *Data Scientist*, *Software Engineer*) or enter a custom job description and required skills.
3.  **Evaluate**: Click the **"Run Ranking Engine"** button to start the ML scoring pipeline.
4.  **Explore**: Scroll through candidate profiles, search by skills/education, download the pool, and click **"Explain Fit"** on any candidate to inspect their detailed fit diagnostics.
