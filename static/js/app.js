console.log("app.js loaded"); // DEBUG

document.addEventListener("DOMContentLoaded", () => {
    const rankBtn = document.getElementById("rankBtn");

    if (rankBtn) {
        rankBtn.addEventListener("click", async () => {
            console.log("Rank button clicked");

            const jdText = document.getElementById("jdText").value;
            const jdSkillsRaw = document.getElementById("jdSkills").value;

            if (!jdText) {
                alert("Please enter Job Description");
                return;
            }

            const jdSkills = jdSkillsRaw
                ? jdSkillsRaw.split(",").map(s => s.trim())
                : [];

            try {
                const res = await fetch("/rank", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        jd_text: jdText,
                        jd_skills: jdSkills,
                        top_k: 5
                    })
                });

                const data = await res.json();
                console.log("Rank response:", data);

                const resultsDiv = document.getElementById("results");
                resultsDiv.innerHTML = "";

                if (!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = "<p>No candidates found.</p>";
                    return;
                }

                data.results.forEach(c => {
                    resultsDiv.innerHTML += `
                        <div class="card">
                            <h3>${c.candidate.name || "Unknown"}</h3>
                            <p><b>Final Score:</b> ${c.final_score.toFixed(3)}</p>
                            <p><b>Matched Skills:</b> ${c.skill_matches.join(", ")}</p>
                            <p><b>Missing Skills:</b> ${c.missing_skills.join(", ")}</p>
                        </div>
                    `;
                });

            } catch (err) {
                console.error("Ranking failed:", err);
                alert("Ranking failed. Check console.");
            }
        });
    }
});
