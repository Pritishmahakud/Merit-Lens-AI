// Merit Lens AI Recruiter Portal Front-end Logic

document.addEventListener("DOMContentLoaded", () => {
    // Deterministic Candidate Name Generator lists
    const FIRST_NAMES = ["Arjun", "Sarah", "Amit", "Elena", "Michael", "Sophia", "David", "Priya", "Carlos", "Yuki", "Fatima", "James", "Emily", "Raj", "Chloe", "Daniel", "Amara", "Alexander", "Zainab", "Gabriel", "Rohan", "Maya", "Tariq", "Li", "Aisha", "Diego", "Nina", "Marcus", "Kanya", "Sanjay", "Tara"];
    const LAST_NAMES = ["Sharma", "Smith", "Patel", "Rostova", "Johnson", "Chen", "Okoye", "Nair", "Garcia", "Tanaka", "Ali", "Davis", "Wong", "Mehta", "Dupont", "Kim", "Silva", "Taylor", "Khan", "Kovalev", "Reddy", "O'Connor", "Kumar", "Zhao", "Gomez", "Ivanov", "Diallo", "Muller", "Singh", "Das"];
    
    function getDeterministicName(resumeId) {
        const id = Number(resumeId) || 0;
        const firstIdx = id % FIRST_NAMES.length;
        const lastIdx = (id * 7) % LAST_NAMES.length;
        return `${FIRST_NAMES[firstIdx]} ${LAST_NAMES[lastIdx]}`;
    }

    // DOM Elements Cache
    const pipelineForm = document.getElementById("pipeline-form");
    const jobRoleSelect = document.getElementById("job-role");
    const jobDescTextarea = document.getElementById("job-desc");
    const reqSkillsInput = document.getElementById("req-skills");
    const submitBtn = document.getElementById("submit-btn");
    
    const candidatesTbody = document.getElementById("candidates-tbody");
    const searchInput = document.getElementById("search-input");
    const refreshBtn = document.getElementById("refresh-btn");
    const loadingOverlay = document.getElementById("loading-overlay");
    
    const explanationDrawer = document.getElementById("explanation-drawer");
    const closeDrawerBtn = document.getElementById("close-drawer-btn");
    const drawerContent = document.getElementById("drawer-content");

    const loadMoreContainer = document.getElementById("load-more-container");
    const loadMoreBtn = document.getElementById("load-more-btn");

    // Upload & Download DOM Elements
    const downloadBtn = document.getElementById("download-btn");
    const csvUploadInput = document.getElementById("csv-upload");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadStatus = document.getElementById("upload-status");
    
    // Overview Metrics Slots
    const totalCandidatesEl = document.getElementById("metric-total-candidates");
    const topScoreEl = document.getElementById("metric-top-score");
    const avgGrowthEl = document.getElementById("metric-avg-growth");

    // Local state cache
    let candidatesList = [];
    let filteredList = [];
    let explanationsDict = {};
    let displayedCount = 10;

    // Initial load
    initWorkspace();

    // Event Listeners
    pipelineForm.addEventListener("submit", handlePipelineRun);
    refreshBtn.addEventListener("click", initWorkspace);
    searchInput.addEventListener("input", handleSearchFilter);
    closeDrawerBtn.addEventListener("click", closeDrawer);
    loadMoreBtn.addEventListener("click", handleLoadMore);

    downloadBtn.addEventListener("click", handleDownloadPool);
    uploadBtn.addEventListener("click", () => csvUploadInput.click());
    csvUploadInput.addEventListener("change", handleUploadPool);
    
    // Close drawer when clicking escape key
    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeDrawer();
    });

    /**
     * Initializes the dashboard by fetching data from local APIs.
     */
    async function initWorkspace() {
        showTbodyLoading("Fetching data from workspace...");
        displayedCount = 10;
        try {
            // Fetch candidates list
            const candidatesRes = await fetch("/api/candidates");
            const candidates = await candidatesRes.json();
            
            // Fetch explanations
            const explanationsRes = await fetch("/api/explanations");
            const explanations = await explanationsRes.json();
            
            candidatesList = candidates || [];
            filteredList = candidatesList;
            
            // Cache explanations by resume_id for O(1) lookups
            explanationsDict = {};
            if (Array.isArray(explanations)) {
                explanations.forEach(exp => {
                    explanationsDict[exp.resume_id] = exp;
                });
            }

            renderDashboard(candidatesList);
        } catch (error) {
            console.error("Error initializing workspace:", error);
            showTbodyError(`Failed to load workspace data: ${error.message}. Trigger the ranking engine to generate files.`);
        }
    }

    /**
     * Handles trigger pipeline form submission
     */
    async function handlePipelineRun(e) {
        e.preventDefault();
        
        const payload = {
            job_role: jobRoleSelect.value,
            job_desc: jobDescTextarea.value.trim() || null,
            req_skills: reqSkillsInput.value.trim() || null
        };
        
        // Show loading modal overlay
        loadingOverlay.classList.add("active");
        displayedCount = 10;
        
        try {
            const res = await fetch("/api/rank", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || "Execution failed.");
            }
            
            const data = await res.json();
            
            if (data.success) {
                candidatesList = data.candidates || [];
                filteredList = candidatesList;
                
                // Cache explanations
                explanationsDict = {};
                if (Array.isArray(data.explanations)) {
                    data.explanations.forEach(exp => {
                        explanationsDict[exp.resume_id] = exp;
                    });
                }
                
                renderDashboard(candidatesList);
            } else {
                throw new Error(data.error || "Engine failed to produce valid outputs.");
            }
        } catch (error) {
            console.error("Pipeline trigger failed:", error);
            alert(`Ranking Pipeline failed: ${error.message}`);
        } finally {
            // Hide loading modal overlay
            loadingOverlay.classList.remove("active");
        }
    }

    /**
     * Client-side search filters candidate table list
     */
    function handleSearchFilter() {
        const query = searchInput.value.toLowerCase().trim();
        displayedCount = 10;
        
        if (!query) {
            filteredList = candidatesList;
            renderDashboard(candidatesList, false); // Render full list, do not update metrics
            return;
        }

        filteredList = candidatesList.filter(c => {
            const name = getDeterministicName(c.Resume_ID).toLowerCase();
            const idMatch = String(c.Resume_ID).includes(query);
            const nameMatch = name.includes(query);
            const skillMatch = c.Skills && c.Skills.toLowerCase().includes(query);
            const eduMatch = c.Education && c.Education.toLowerCase().includes(query);
            const roleMatch = c["Job Role"] && c["Job Role"].toLowerCase().includes(query);
            return idMatch || nameMatch || skillMatch || eduMatch || roleMatch;
        });

        renderCandidatesList(filteredList);
    }

    /**
     * Handles load more button click to display additional candidates
     */
    function handleLoadMore() {
        displayedCount += 10;
        renderCandidatesList(filteredList);
    }

    /**
     * Handles downloading the current dataset file
     */
    function handleDownloadPool() {
        window.location.href = "/api/download";
    }

    /**
     * Handles custom CSV dataset upload
     */
    async function handleUploadPool() {
        const file = csvUploadInput.files[0];
        if (!file) return;

        uploadStatus.innerText = "Reading file...";
        uploadStatus.style.color = "var(--text-secondary)";

        try {
            const text = await file.text();
            uploadStatus.innerText = "Processing pool...";
            uploadStatus.style.color = "var(--accent-indigo)";

            const res = await fetch("/api/upload", {
                method: "POST",
                headers: { "Content-Type": "text/csv" },
                body: text
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || "Upload processing failed.");
            }

            const data = await res.json();
            if (data.success) {
                uploadStatus.innerText = "Upload complete!";
                uploadStatus.style.color = "var(--accent-green)";
                alert("Custom candidate pool uploaded and pre-processed successfully!\nClick 'Run Ranking Engine' to rank this new pool.");
                csvUploadInput.value = "";
                setTimeout(() => { uploadStatus.innerText = ""; }, 3000);
                initWorkspace();
            } else {
                throw new Error(data.error || "Dataset preprocessing failed.");
            }
        } catch (error) {
            console.error("Upload failed:", error);
            uploadStatus.innerText = `Error: ${error.message}`;
            uploadStatus.style.color = "var(--accent-red)";
            alert(`Dataset upload failed: ${error.message}`);
            csvUploadInput.value = "";
        }
    }

    /**
     * Renders candidate lists and updates status metrics
     */
    function renderDashboard(candidates, updateMetrics = true) {
        renderCandidatesList(candidates);
        if (updateMetrics) {
            updateOverviewMetrics(candidates);
        }
    }

    function renderCandidatesList(candidates) {
        candidatesTbody.innerHTML = "";
        
        if (candidates.length === 0) {
            candidatesTbody.innerHTML = `
                <tr>
                    <td colspan="7" class="no-results-state">
                        <i class="fa-solid fa-folder-open"></i> No ranked candidates found. Trigger the ranking engine to generate files.
                    </td>
                </tr>
            `;
            loadMoreContainer.style.display = "none";
            return;
        }

        // Slice list to display incremental segments
        const visibleSlice = candidates.slice(0, displayedCount);

        visibleSlice.forEach(c => {
            const tr = document.createElement("tr");
            
            // Format scores
            const overallScore = typeof c.overall_score === 'number' ? c.overall_score.toFixed(1) : '-';
            const growthScore = typeof c.growth_potential === 'number' ? c.growth_potential.toFixed(1) : '0';
            const candidateName = getDeterministicName(c.Resume_ID);
            
            // Build row content
            tr.innerHTML = `
                <td><span class="rank-badge">${c.candidate_rank || '-'}</span></td>
                <td>
                    <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-primary);">${candidateName}</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.05rem;">ID: #${c.Resume_ID}</div>
                </td>
                <td>${c.Education || 'Not Specified'}</td>
                <td>${c["Experience (Years)"] || 0} years</td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.85rem; font-weight: 500; width: 30px;">${Math.round(growthScore)}</span>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: ${growthScore}%"></div>
                        </div>
                    </div>
                </td>
                <td><span class="score-badge">${overallScore}</span></td>
                <td>
                    <button class="explain-btn" data-id="${c.Resume_ID}">
                        <i class="fa-solid fa-circle-info"></i> Explain Fit
                    </button>
                </td>
            `;
            
            // Attach explain button event click
            const explainBtn = tr.querySelector(".explain-btn");
            explainBtn.addEventListener("click", () => openDrawer(c.Resume_ID));

            candidatesTbody.appendChild(tr);
        });

        // Toggle visibility of the Load More button
        if (candidates.length > displayedCount) {
            loadMoreContainer.style.display = "flex";
        } else {
            loadMoreContainer.style.display = "none";
        }
    }

    function updateOverviewMetrics(candidates) {
        if (candidates.length === 0) {
            totalCandidatesEl.innerText = "0";
            topScoreEl.innerText = "-";
            avgGrowthEl.innerText = "-";
            return;
        }

        const totalCount = candidates.length;
        
        // Find top score
        const scores = candidates.map(c => c.overall_score).filter(s => typeof s === 'number');
        const topScore = scores.length > 0 ? Math.max(...scores).toFixed(1) : '-';
        
        // Find average growth score
        const growthScores = candidates.map(c => c.growth_potential).filter(s => typeof s === 'number');
        const avgGrowth = growthScores.length > 0 ? (growthScores.reduce((a, b) => a + b, 0) / growthScores.length).toFixed(1) : '-';

        totalCandidatesEl.innerText = totalCount;
        topScoreEl.innerText = topScore;
        avgGrowthEl.innerText = `${avgGrowth}%`;
    }

    /**
     * Opens Explanations Details Drawer
     */
    function openDrawer(resumeId) {
        const candidate = candidatesList.find(c => Number(c.Resume_ID) === Number(resumeId));
        if (!candidate) return;

        const explanation = explanationsDict[resumeId];
        renderExplanationDetails(candidate, explanation);
        explanationDrawer.classList.add("active");
    }

    function closeDrawer() {
        explanationDrawer.classList.remove("active");
    }

    function renderExplanationDetails(c, exp) {
        if (!exp) {
            renderMissingExplanationUI(c);
            return;
        }

        const overallScore = typeof c.overall_score === 'number' ? c.overall_score.toFixed(1) : '-';
        const rank = c.candidate_rank ? `#${c.candidate_rank}` : 'Unranked';
        const candidateName = getDeterministicName(c.Resume_ID);
        
        // Renders badged lists
        const exactMatchesHtml = renderBadges(exp.exact_matches, 'exact', '<i class="fa-solid fa-check"></i>');
        
        // Transferable mappings
        let transferableHtml = "";
        if (exp.transferable_skills && exp.transferable_skills.length > 0) {
            transferableHtml = exp.transferable_skills.map(skill => 
                '<div class="skill-category-block" style="margin-bottom: 0.50rem;">' +
                    '<div style="font-size: 0.85rem; font-weight: 500;">' +
                        '<span class="tech-badge transfer">' + skill.candidate_skill + '</span>' +
                        '<i class="fa-solid fa-arrow-right-long" style="margin: 0 0.25rem; font-size: 0.75rem; color: var(--text-secondary);"></i>' +
                        '<span class="tech-badge exact" style="background-color: transparent; border-style: dashed;">' + skill.required_skill + '</span>' +
                    '</div>' +
                    '<small style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.1rem; display: block; line-height: 1.3;">' +
                        skill.reasoning +
                    '</small>' +
                '</div>'
            ).join("");
        } else {
            transferableHtml = '<span style="font-size: 0.85rem; color: var(--text-secondary);">No active skill transfer required.</span>';
        }

        const missingMatchesHtml = renderBadges(exp.missing_skills, 'missing', '<i class="fa-solid fa-xmark"></i>');

        // Strengths & Weaknesses
        const strengthsHtml = exp.strengths && exp.strengths.length > 0 
            ? exp.strengths.map(s => '<li><i class="fa-solid fa-circle-check"></i> ' + s + '</li>').join("")
            : '<li><i class="fa-solid fa-circle-check"></i> Found direct alignment across standard role attributes.</li>';

        const weaknessesHtml = exp.weaknesses && exp.weaknesses.length > 0
            ? exp.weaknesses.map(w => '<li><i class="fa-solid fa-circle-exclamation"></i> ' + w + '</li>').join("")
            : '<li><i class="fa-solid fa-circle-exclamation"></i> No critical alignment issues identified.</li>';

        drawerContent.innerHTML = `
            <!-- Top Card -->
            <div class="drawer-card-header">
                <h3 style="font-size: 1.35rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.1rem;">${candidateName}</h3>
                <span style="font-size: 0.8rem; color: var(--text-secondary); display: block; margin-bottom: 0.65rem;">Candidate Profile #${c.Resume_ID}</span>
                <div class="score-value">${overallScore}</div>
                <div class="rank-text">Ranked ${rank} in Pool</div>
            </div>

            <!-- Reasoning -->
            <div class="drawer-section">
                <h4 class="drawer-section-title">Evaluation Summary</h4>
                <p class="reasoning-text">${exp.reason_for_ranking}</p>
            </div>

            <!-- Scores Comparison -->
            <div class="drawer-scores-comparison">
                <div class="comparison-box fit">
                    <span>Role Fit Score</span>
                    <h5>${exp.current_fit_score || '-'}%</h5>
                </div>
                <div class="comparison-box growth">
                    <span>Growth Score</span>
                    <h5>${Math.round(c.growth_potential)}%</h5>
                </div>
            </div>

            <!-- Growth Potential Details -->
            <div class="drawer-section">
                <h4 class="drawer-section-title">Growth Analysis</h4>
                <p style="font-size: 0.9rem; line-height: 1.4; color: var(--text-secondary);">${exp.growth_potential}</p>
            </div>

            <!-- Skills Categorization -->
            <div class="drawer-section">
                <h4 class="drawer-section-title">Skill Overlap Analysis</h4>
                <div class="skills-grid">
                    <div class="skill-category-block">
                        <span class="skill-cat-title exact"><i class="fa-solid fa-circle-check"></i> Exact Match Skills</span>
                        <div class="badge-container">${exactMatchesHtml}</div>
                    </div>
                    
                    <div class="skill-category-block" style="border-top: 1px solid var(--border-color); padding-top: 0.75rem;">
                        <span class="skill-cat-title transfer"><i class="fa-solid fa-shuffle"></i> Transferable Skills</span>
                        <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.25rem;">
                            ${transferableHtml}
                        </div>
                    </div>
                    
                    <div class="skill-category-block" style="border-top: 1px solid var(--border-color); padding-top: 0.75rem;">
                        <span class="skill-cat-title missing"><i class="fa-solid fa-triangle-exclamation"></i> Missing Required Skills</span>
                        <div class="badge-container">${missingMatchesHtml}</div>
                    </div>
                </div>
            </div>

            <!-- Strengths & Weaknesses -->
            <div class="drawer-section">
                <h4 class="drawer-section-title">Recruiter Quick Facts</h4>
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">Core Strengths</span>
                        <ul class="bullets-list strengths" style="margin-top: 0.25rem;">
                            ${strengthsHtml}
                        </ul>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">Development Areas</span>
                        <ul class="bullets-list weaknesses" style="margin-top: 0.25rem;">
                            ${weaknessesHtml}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Renders a basic, programmatic fallback if explanations are not cached for the candidate (outside top-N pool)
     */
    function renderMissingExplanationUI(c) {
        const overallScore = typeof c.overall_score === 'number' ? c.overall_score.toFixed(1) : '-';
        const rank = c.candidate_rank ? `#${c.candidate_rank}` : 'Unranked';
        const candidateName = getDeterministicName(c.Resume_ID);
        
        drawerContent.innerHTML = `
            <div class="drawer-card-header">
                <h3 style="font-size: 1.35rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.1rem;">${candidateName}</h3>
                <span style="font-size: 0.8rem; color: var(--text-secondary); display: block; margin-bottom: 0.65rem;">Candidate Profile #${c.Resume_ID}</span>
                <div class="score-value">${overallScore}</div>
                <div class="rank-text">Ranked ${rank} in Pool</div>
            </div>
            
            <div class="drawer-section">
                <h4 class="drawer-section-title">General Summary</h4>
                <p class="reasoning-text">
                    This candidate has an overall matching score of ${overallScore}%, driven by a years of experience score of ${c.experience_score} 
                    and a growth potential score of ${c.growth_potential.toFixed(1)}%. Detail-level qualitative evaluations are reserved for 
                    top tier prospects.
                </p>
            </div>

            <div class="drawer-scores-comparison" style="margin-top: 1rem;">
                <div class="comparison-box fit">
                    <span>Base Domain Match</span>
                    <h5>${c.domain_match ? c.domain_match.toFixed(1) : '-'}%</h5>
                </div>
                <div class="comparison-box growth">
                    <span>Growth Potential</span>
                    <h5>${c.growth_potential.toFixed(1)}%</h5>
                </div>
            </div>

            <div class="drawer-section">
                <h4 class="drawer-section-title">Candidate Details</h4>
                <ul class="bullets-list strengths" style="padding: 0.5rem 0;">
                    <li><i class="fa-solid fa-graduation-cap" style="color: var(--accent-indigo);"></i> <strong>Education Level:</strong> ${c.Education} (Mapped Score: ${c.education_score}/5)</li>
                    <li><i class="fa-solid fa-clock" style="color: var(--accent-indigo);"></i> <strong>Experience Age:</strong> ${c["Experience (Years)"]} Years</li>
                    <li><i class="fa-solid fa-code" style="color: var(--accent-indigo);"></i> <strong>Skills:</strong> ${c.Skills}</li>
                    <li><i class="fa-solid fa-diagram-project" style="color: var(--accent-indigo);"></i> <strong>Project Delivery:</strong> ${c["Projects Count"]} Projects</li>
                </ul>
            </div>
        `;
    }

    function renderBadges(arr, type, icon = "") {
        if (!arr || arr.length === 0) {
            return `<span style="font-size: 0.85rem; color: var(--text-secondary);">None</span>`;
        }
        return arr.map(item => `<span class="tech-badge ${type}">${icon} ${item}</span>`).join(" ");
    }

    function showTbodyLoading(message) {
        candidatesTbody.innerHTML = `
            <tr>
                <td colspan="7" class="loading-state">
                    <i class="fa-solid fa-spinner fa-spin"></i> ${message}
                </td>
            </tr>
        `;
        loadMoreContainer.style.display = "none";
    }

    function showTbodyError(message) {
        candidatesTbody.innerHTML = `
            <tr>
                <td colspan="7" class="no-results-state" style="color: var(--accent-red);">
                    <i class="fa-solid fa-triangle-exclamation"></i> ${message}
                </td>
            </tr>
        `;
        loadMoreContainer.style.display = "none";
    }
});
