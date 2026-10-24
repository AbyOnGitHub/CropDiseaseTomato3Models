/**
 * TomatoGuard AI - Frontend Dashboard Controller
 * Connects directly to FastAPI backend for genuine multi-model inference.
 */

// Application State
const state = {
    selectedModel: "yolo", // "yolo" | "novel_cnn" | "mobilevit" | "compare_all"
    selectedFile: null,
    isProcessing: false,
    diseaseCatalog: null,
    modelsMeta: null
};

// DOM Element References
const elements = {
    // Model Selection
    modelPills: document.querySelectorAll(".model-pill"),
    metaArchName: document.getElementById("meta-arch-name"),
    metaFramework: document.getElementById("meta-framework"),
    metaRecommendation: document.getElementById("meta-recommendation"),

    // Dropzone & File Input
    dropzone: document.getElementById("dropzone"),
    dropzoneContent: document.getElementById("dropzone-content"),
    fileInput: document.getElementById("file-input"),
    btnBrowse: document.getElementById("btn-browse"),
    previewWrapper: document.getElementById("preview-wrapper"),
    previewImg: document.getElementById("preview-img"),
    previewInfo: document.getElementById("preview-info"),
    btnClearImg: document.getElementById("btn-clear-img"),
    sampleChips: document.querySelectorAll(".sample-chip"),

    // Actions
    btnClassify: document.getElementById("btn-classify"),
    btnText: document.querySelector(".btn-text"),
    btnLoader: document.querySelector(".btn-loader"),

    // Result Containers
    emptyState: document.getElementById("empty-state"),
    loadingState: document.getElementById("loading-state"),
    scanningImgPreview: document.getElementById("scanning-img-preview"),
    singleResultContainer: document.getElementById("single-result-container"),
    compareContainer: document.getElementById("compare-container"),

    // Hero Elements
    heroCard: document.getElementById("hero-card"),
    heroStatusBadge: document.getElementById("hero-status-badge"),
    heroStatusText: document.getElementById("hero-status-text"),
    heroModelTag: document.getElementById("hero-model-tag"),
    heroDiseaseTitle: document.getElementById("hero-disease-title"),
    heroPathogen: document.getElementById("hero-pathogen"),
    heroConfVal: document.getElementById("hero-conf-val"),
    heroLatency: document.getElementById("hero-latency"),
    heroSummaryText: document.getElementById("hero-summary-text"),
    gaugeCircle: document.getElementById("gauge-circle"),

    // Probability List
    probList: document.getElementById("prob-list"),

    // Remediation Tabs & Content
    remediationTabs: document.querySelectorAll(".rem-tab"),
    tabPanes: document.querySelectorAll(".tab-pane"),
    remFavorableCond: document.getElementById("rem-favorable-cond"),
    remSymptomsList: document.getElementById("rem-symptoms-list"),
    remCulturalList: document.getElementById("rem-cultural-list"),
    remOrganicList: document.getElementById("rem-organic-list"),
    remChemicalList: document.getElementById("rem-chemical-list"),
    remPreventiveList: document.getElementById("rem-preventive-list"),

    // Compare Elements
    consensusAgreementBadge: document.getElementById("consensus-agreement-badge"),
    consensusDiseaseTitle: document.getElementById("consensus-disease-title"),
    consensusPathogen: document.getElementById("consensus-pathogen"),
    consensusSummary: document.getElementById("consensus-summary"),
    compYoloPred: document.getElementById("comp-yolo-pred"),
    compYoloBar: document.getElementById("comp-yolo-bar"),
    compYoloConf: document.getElementById("comp-yolo-conf"),
    compYoloTime: document.getElementById("comp-yolo-time"),
    compNovelPred: document.getElementById("comp-novel-pred"),
    compNovelBar: document.getElementById("comp-novel-bar"),
    compNovelConf: document.getElementById("comp-novel-conf"),
    compNovelTime: document.getElementById("comp-novel-time"),
    compMvitPred: document.getElementById("comp-mvit-pred"),
    compMvitBar: document.getElementById("comp-mvit-bar"),
    compMvitConf: document.getElementById("comp-mvit-conf"),
    compMvitTime: document.getElementById("comp-mvit-time"),
    compareRemediationCard: document.getElementById("compare-remediation-card"),

    // Modal
    catalogModal: document.getElementById("catalog-modal"),
    catalogContent: document.getElementById("catalog-content"),
    btnBrowseCatalog: document.getElementById("btn-browse-catalog"),
    btnCloseModal: document.getElementById("btn-close-modal"),
};

// Model specs descriptions
const MODEL_SPECS = {
    yolo: {
        arch: "YOLO11 Nano Classification (YOLO11n-cls)",
        framework: "Ultralytics YOLO / ONNX Runtime CPU",
        rec: "Real-time edge & field deployment with highest overall accuracy (99.8%)."
    },
    novel_cnn: {
        arch: "Novel Multi-Scale Inception-A/B + Depthwise Separable CNN",
        framework: "PyTorch / ONNX Runtime CPU (597K parameters)",
        rec: "Ultra-compact footprint for low-power IoT farm sensors & embedded hardware."
    },
    mobilevit: {
        arch: "apple/mobilevit-xx-small (Vision Transformer + Inverted Residuals)",
        framework: "Hugging Face Transformers / PyTorch CPU (1.3M parameters)",
        rec: "Global self-attention for fine-grained leaf texture & context reasoning."
    },
    compare_all: {
        arch: "Multi-Model Triangulation (YOLO + Novel CNN + MobileViT)",
        framework: "All 3 Engines in Parallel (Consensus Mode)",
        rec: "Cross-architectural verification and majority agreement analysis for critical decisions."
    }
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    setupModelSelector();
    setupDropzone();
    setupSampleChips();
    setupTabs();
    setupModal();
    setupClassifyAction();
    fetchBackendCatalog();
});

// Setup Model Switching
function setupModelSelector() {
    elements.modelPills.forEach(pill => {
        pill.addEventListener("click", () => {
            elements.modelPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            state.selectedModel = pill.dataset.model;

            const spec = MODEL_SPECS[state.selectedModel];
            if (spec) {
                elements.metaArchName.textContent = spec.arch;
                elements.metaFramework.textContent = spec.framework;
                elements.metaRecommendation.textContent = spec.rec;
            }
        });
    });
}

// Setup Drag & Drop + File Upload
function setupDropzone() {
    const { dropzone, fileInput, btnBrowse, btnClearImg } = elements;

    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => {
        if (!state.selectedFile) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("drag-over");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    btnClearImg.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });
}

// Handle file selection
function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file (JPG, PNG, or WEBP).");
        return;
    }

    state.selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImg.src = e.target.result;
        elements.scanningImgPreview.src = e.target.result;

        const img = new Image();
        img.onload = () => {
            elements.previewInfo.textContent = `${img.naturalWidth} × ${img.naturalHeight} • ${(file.size / 1024).toFixed(1)} KB`;
        };
        img.src = e.target.result;

        elements.dropzoneContent.classList.add("hidden");
        elements.previewWrapper.classList.remove("hidden");
        elements.btnClassify.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Clear selected image
function clearSelectedFile() {
    state.selectedFile = null;
    elements.fileInput.value = "";
    elements.previewImg.src = "";
    elements.previewWrapper.classList.add("hidden");
    elements.dropzoneContent.classList.remove("hidden");
    elements.btnClassify.disabled = true;

    // Reset results view
    elements.singleResultContainer.classList.add("hidden");
    elements.compareContainer.classList.add("hidden");
    elements.emptyState.classList.remove("hidden");
}

// Quick Sample Chips
function setupSampleChips() {
    elements.sampleChips.forEach(chip => {
        chip.addEventListener("click", async () => {
            const url = chip.dataset.sample;
            try {
                const res = await fetch(url);
                const blob = await res.blob();
                const filename = url.split("/").pop();
                const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
                handleSelectedFile(file);

                // Auto-trigger classification for instant feedback
                setTimeout(() => {
                    classifyImage();
                }, 150);
            } catch (err) {
                console.error("Failed to load sample image:", err);
            }
        });
    });
}

// Tab Switching inside Remediation
function setupTabs() {
    elements.remediationTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            elements.remediationTabs.forEach(t => t.classList.remove("active"));
            elements.tabPanes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(`pane-${tab.dataset.tab}`);
            if (targetPane) targetPane.classList.add("active");
        });
    });
}

// Disease Catalog Modal
function setupModal() {
    elements.btnBrowseCatalog.addEventListener("click", () => {
        elements.catalogModal.classList.remove("hidden");
    });

    elements.btnCloseModal.addEventListener("click", () => {
        elements.catalogModal.classList.add("hidden");
    });

    elements.catalogModal.addEventListener("click", (e) => {
        if (e.target === elements.catalogModal) {
            elements.catalogModal.classList.add("hidden");
        }
    });
}

// Setup Classification Trigger
function setupClassifyAction() {
    elements.btnClassify.addEventListener("click", () => {
        classifyImage();
    });
}

// Main Classification Handler
async function classifyImage() {
    if (!state.selectedFile || state.isProcessing) return;

    state.isProcessing = true;
    setLoadingState(true);

    const formData = new FormData();
    formData.append("file", state.selectedFile);

    try {
        if (state.selectedModel === "compare_all") {
            // Run all 3 models in parallel
            const res = await fetch("/api/predict_all", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Server error running multi-model inference.");
            }

            const data = await res.json();
            renderCompareResults(data);
        } else {
            // Single model inference
            formData.append("model_id", state.selectedModel);
            const res = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Server error running single model inference.");
            }

            const data = await res.json();
            renderSingleResult(data);
        }
    } catch (err) {
        alert(`Classification Error: ${err.message}`);
        console.error(err);
        elements.emptyState.classList.remove("hidden");
        elements.singleResultContainer.classList.add("hidden");
        elements.compareContainer.classList.add("hidden");
    } finally {
        state.isProcessing = false;
        setLoadingState(false);
    }
}

// Toggle Loading Spinner
function setLoadingState(isLoading) {
    if (isLoading) {
        elements.emptyState.classList.add("hidden");
        elements.singleResultContainer.classList.add("hidden");
        elements.compareContainer.classList.add("hidden");
        elements.loadingState.classList.remove("hidden");
        elements.btnText.classList.add("hidden");
        elements.btnLoader.classList.remove("hidden");
        elements.btnClassify.disabled = true;
    } else {
        elements.loadingState.classList.add("hidden");
        elements.btnText.classList.remove("hidden");
        elements.btnLoader.classList.add("hidden");
        elements.btnClassify.disabled = false;
    }
}

// Render Single Model Output
function renderSingleResult(data) {
    const info = data.disease_info;

    // 1. Hero Card
    elements.heroModelTag.textContent = data.model_name;
    elements.heroDiseaseTitle.textContent = data.predicted_class;
    elements.heroPathogen.textContent = `${info.scientific_name} • ${info.pathogen_type}`;
    elements.heroConfVal.textContent = `${data.confidence_percent}%`;
    elements.heroLatency.textContent = `${data.latency_ms} ms`;
    elements.heroSummaryText.textContent = info.summary;

    // Severity status badge styling
    const badge = elements.heroStatusBadge;
    badge.className = "diagnosis-status-badge";
    if (data.is_healthy) {
        badge.classList.add("status-optimal");
        elements.heroStatusText.textContent = "Healthy Foliage";
    } else {
        const sevClass = info.severity.toLowerCase();
        badge.classList.add(`status-${sevClass}`);
        elements.heroStatusText.textContent = `${info.severity.toUpperCase()} ALERT`;
    }

    // Gauge circle animation (circumference = 2 * PI * 42 ≈ 264)
    const offset = 264 - (264 * (data.confidence_score));
    elements.gaugeCircle.style.strokeDashoffset = offset;
    elements.gaugeCircle.style.stroke = data.is_healthy ? "var(--accent-mint)" : (info.severity_color || "var(--accent-emerald)");

    // 2. Render Probability Distribution Bars
    elements.probList.innerHTML = "";
    data.probabilities.forEach((item, idx) => {
        const isTop = (idx === 0);
        const div = document.createElement("div");
        div.className = `prob-item ${isTop ? "is-top" : ""}`;
        div.innerHTML = `
            <div class="prob-header">
                <span class="prob-name">${item.class_name}</span>
                <span class="prob-val">${item.percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: 0%;" data-target="${item.percentage}"></div>
            </div>
        `;
        elements.probList.appendChild(div);
    });

    // Trigger bar fill animation
    setTimeout(() => {
        document.querySelectorAll(".prob-bar-fill").forEach(bar => {
            bar.style.width = `${bar.dataset.target}%`;
        });
    }, 50);

    // 3. Render Remediation Advisory
    populateRemediationData(info);

    // Switch view
    elements.singleResultContainer.classList.remove("hidden");
    elements.compareContainer.classList.add("hidden");
}

// Render Compare All Output
function renderCompareResults(data) {
    const comp = data.models_comparison;
    const consensusInfo = data.consensus_disease_info;

    // 1. Consensus Hero
    elements.consensusAgreementBadge.textContent = data.consensus_agreement;
    elements.consensusDiseaseTitle.textContent = data.consensus_class;
    elements.consensusPathogen.textContent = `${consensusInfo.scientific_name} • ${consensusInfo.pathogen_type}`;
    elements.consensusSummary.textContent = consensusInfo.summary;

    // 2. YOLO Card
    const yolo = comp.yolo;
    elements.compYoloPred.textContent = yolo.predicted_class;
    elements.compYoloConf.textContent = `${yolo.confidence_percent}%`;
    elements.compYoloTime.textContent = `${yolo.latency_ms}ms`;
    elements.compYoloBar.style.width = `${yolo.confidence_percent}%`;

    // 3. Novel CNN Card
    const novel = comp.novel_cnn;
    elements.compNovelPred.textContent = novel.predicted_class;
    elements.compNovelConf.textContent = `${novel.confidence_percent}%`;
    elements.compNovelTime.textContent = `${novel.latency_ms}ms`;
    elements.compNovelBar.style.width = `${novel.confidence_percent}%`;

    // 4. MobileViT Card
    const mvit = comp.mobilevit;
    elements.compMvitPred.textContent = mvit.predicted_class;
    elements.compMvitConf.textContent = `${mvit.confidence_percent}%`;
    elements.compMvitTime.textContent = `${mvit.latency_ms}ms`;
    elements.compMvitBar.style.width = `${mvit.confidence_percent}%`;

    // Clone & populate remediation for consensus
    const remNode = document.getElementById("remediation-card").cloneNode(true);
    remNode.id = "cloned-consensus-remediation";
    elements.compareRemediationCard.innerHTML = "";
    elements.compareRemediationCard.appendChild(remNode);

    // Re-bind tabs for cloned element
    const clonedTabs = remNode.querySelectorAll(".rem-tab");
    const clonedPanes = remNode.querySelectorAll(".tab-pane");
    clonedTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            clonedTabs.forEach(t => t.classList.remove("active"));
            clonedPanes.forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            const target = remNode.querySelector(`#pane-${tab.dataset.tab}`);
            if (target) target.classList.add("active");
        });
    });

    populateRemediationData(consensusInfo, remNode);

    // Switch view
    elements.compareContainer.classList.remove("hidden");
    elements.singleResultContainer.classList.add("hidden");
}

// Populate Checklist Items in Remediation Card
function populateRemediationData(info, root = document) {
    const favorable = root.querySelector("#rem-favorable-cond");
    if (favorable) favorable.textContent = `Optimal Microclimate: ${info.favorable_conditions}`;

    const symptomsList = root.querySelector("#rem-symptoms-list");
    if (symptomsList) {
        symptomsList.innerHTML = info.symptoms.map(s => `<li>${s}</li>`).join("");
    }

    const culturalList = root.querySelector("#rem-cultural-list");
    if (culturalList) {
        culturalList.innerHTML = info.cultural_remediation.map(c => `<li>${c}</li>`).join("");
    }

    const organicList = root.querySelector("#rem-organic-list");
    if (organicList) {
        organicList.innerHTML = info.organic_treatments.map(o => `<li>${o}</li>`).join("");
    }

    const chemicalList = root.querySelector("#rem-chemical-list");
    if (chemicalList) {
        chemicalList.innerHTML = info.chemical_treatments.map(ch => `<li>${ch}</li>`).join("");
    }

    const preventiveList = root.querySelector("#rem-preventive-list");
    if (preventiveList) {
        preventiveList.innerHTML = info.preventive_measures.map(p => `<li>${p}</li>`).join("");
    }
}

// Fetch Encyclopedia from Backend
async function fetchBackendCatalog() {
    try {
        const res = await fetch("/api/diseases");
        const data = await res.json();
        state.diseaseCatalog = data.diseases;

        elements.catalogContent.innerHTML = "";
        Object.values(data.diseases).forEach(d => {
            const card = document.createElement("div");
            card.className = "catalog-item-card";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4>${d.canonical_name}</h4>
                    <span class="info-tag" style="color:${d.severity_color}; border-color:${d.severity_color};">${d.severity}</span>
                </div>
                <div class="cat-sci">${d.scientific_name} • ${d.pathogen_type}</div>
                <p><strong>Overview:</strong> ${d.summary}</p>
                <p style="margin-top:8px;"><strong>Field Remediation:</strong> ${d.cultural_remediation[0]} ${d.chemical_treatments[0]}</p>
            `;
            elements.catalogContent.appendChild(card);
        });
    } catch (err) {
        console.warn("Could not pre-load catalog:", err);
    }
}
