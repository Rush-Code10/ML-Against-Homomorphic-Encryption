const form = document.getElementById("simulation-form");
const runButton = document.getElementById("run-button");
const buttonText = document.querySelector(".button-text");
const buttonLoader = document.querySelector(".button-loader");
const statusContainer = document.getElementById("run-status-container");
const statusNode = document.getElementById("run-status");
const statusTitle = document.getElementById("status-title");
const statusMessage = document.getElementById("status-message");
const summaryNode = document.getElementById("result-summary");
const storyNode = document.getElementById("story-summary");
const metaNode = document.getElementById("experiment-meta");
const modelNode = document.getElementById("model-metrics");
const plotsNode = document.getElementById("plots-grid");

function setStatus(type, title, message) {
    statusContainer.style.display = 'block';
    statusNode.className = `run-status status-${type}`;
    statusTitle.textContent = title;
    statusMessage.textContent = message;
}

function percent(value) {
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function metricCard(title, value, footnote) {
    return `
        <article class="metric-card">
            <p class="eyebrow">${title}</p>
            <div class="metric-value">${value}</div>
            <div class="metric-footnote">${footnote}</div>
        </article>
    `;
}

function renderSummary(result) {
    summaryNode.innerHTML = [
        metricCard("Backend", result.backend, "Execution engine used for encrypted operations"),
        metricCard("Dataset Rows", result.total_rows, `${result.baseline_rows} baseline / ${result.defended_rows} defended`),
        metricCard("RF Baseline", percent(result.accuracy_comparison.rf_baseline), "How accurately metadata identifies operations before defense"),
        metricCard("RF Defended", percent(result.accuracy_comparison.rf_defended), "Classification accuracy after metadata perturbation"),
    ].join("");

    storyNode.innerHTML = `
        <p><strong>Leakage strength:</strong> ${result.story.leakage_strength}. The baseline classifier can separate encrypted workloads using only metadata traces.</p>
        <p><strong>Defense effect:</strong> ${result.story.defense_effectiveness}. Random Forest accuracy drops by ${result.story.rf_reduction} percentage points and Torch drops by ${result.story.torch_reduction} points.</p>
        <p><strong>Interpretation:</strong> the encryption remains intact, but the system-level footprint still leaks workload identity unless metadata is deliberately blurred.</p>
        ${result.story.suspicious_accuracy ? "<p><strong>Sanity note:</strong> near-perfect accuracy can mean the dataset is too clean for a believable attack narrative. Increase workload variation or use the TenSEAL backend for a tougher simulation.</p>" : ""}
    `;

    metaNode.innerHTML = `
        <dt>Samples / op</dt><dd>${result.config.samples_per_operation}</dd>
        <dt>Torch epochs</dt><dd>${result.config.torch_epochs}</dd>
        <dt>Defense noise</dt><dd>${result.config.defense_noise_scale}</dd>
        <dt>Dataset</dt><dd>${result.dataset_path}</dd>
    `;

    const baseline = result.metrics.baseline;
    const defended = result.metrics.defended;
    modelNode.innerHTML = [
        ["Dummy Baseline", baseline.dummy.accuracy, defended.dummy.accuracy, "Chance-like reference point so the attack models are judged against a trivial predictor."],
        ["Random Forest", baseline.random_forest.accuracy, defended.random_forest.accuracy, "Best baseline leakage detector over tabular metadata"],
        ["SVM", baseline.svm.accuracy, defended.svm.accuracy, "Nonlinear margin-based baseline for metadata separation"],
        ["Torch MLP", baseline.torch.accuracy, defended.torch.accuracy, "Neural model over the metadata feature vector"],
    ].map(([name, baselineAcc, defendedAcc, description]) => `
        <article class="model-card">
            <p class="eyebrow">${name}</p>
            <div class="model-score">
                <strong>${percent(baselineAcc)}</strong>
                <span>baseline</span>
            </div>
            <div class="model-score">
                <strong>${percent(defendedAcc)}</strong>
                <span>defended</span>
            </div>
            <p>${description}</p>
        </article>
    `).join("");

    const plotCards = [
        ["Accuracy Comparison", result.plots.accuracy_comparison, "Direct before-vs-after defense comparison across models."],
        ["Feature Importance (Baseline)", result.plots.feature_importance_baseline, "Shows which metadata fields most strongly expose workload identity."],
        ["Feature Importance (Defended)", result.plots.feature_importance_defended, "Feature shifts after noise injection reveal where the defense blunts leakage."],
        ["RF Confusion Matrix (Baseline)", result.plots.confusion_matrix_rf_baseline, "Which encrypted operations remain separable before defense."],
        ["RF Confusion Matrix (Defended)", result.plots.confusion_matrix_rf_defended, "Where the defense creates ambiguity between operations."],
        ["Torch Confusion Matrix (Baseline)", result.plots.confusion_matrix_torch_baseline, "Neural-model view of leakage patterns on the clean metadata."],
        ["Torch Confusion Matrix (Defended)", result.plots.confusion_matrix_torch_defended, "Neural-model errors after metadata perturbation."],
    ];

    plotsNode.innerHTML = plotCards.map(([title, src, description]) => `
        <article class="plot-card">
            <h3>${title}</h3>
            <p>${description}</p>
            <div class="plot-container">
                <iframe
                    src="${src}?v=${Date.now()}"
                    frameborder="0"
                    loading="lazy"
                    onload="this.closest('.plot-container').classList.add('loaded')"
                ></iframe>
            </div>
        </article>
    `).join("");
}

async function runSimulation(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    payload.samples_per_operation = Number(payload.samples_per_operation);
    payload.torch_epochs = Number(payload.torch_epochs);
    payload.defense_noise_scale = Number(payload.defense_noise_scale);

    runButton.disabled = true;
    buttonText.style.display = 'none';
    buttonLoader.style.display = 'flex';

    setStatus('loading', 'Simulation Running',
        'The server is generating encrypted workloads, training models, and writing fresh artifacts. This may take a few minutes...');

    try {
        const response = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "Simulation failed");
        }
        renderSummary(data.result);
        setStatus('good', 'Simulation Complete',
            'The dashboard now reflects the latest run. All artifacts have been generated successfully.');
    } catch (error) {
        setStatus('error', 'Simulation Failed',
            `${error.message}. Try reducing samples or using the mock backend for faster testing.`);
    } finally {
        runButton.disabled = false;
        buttonText.style.display = 'inline';
        buttonLoader.style.display = 'none';
    }
}

form.addEventListener("submit", runSimulation);

if (window.__INITIAL_STATE__) {
    renderSummary(window.__INITIAL_STATE__);
    setStatus('good', 'Previous Results Loaded',
        'Displaying the most recent saved simulation. Run a new simulation to update the results.');
}
