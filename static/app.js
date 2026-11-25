document.addEventListener("DOMContentLoaded", () => {
  const tabs = Array.from(document.querySelectorAll(".section-tab"));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

  const activateTab = (panelId, { focus = false, updateHash = true } = {}) => {
    const selectedTab = tabs.find((tab) => tab.getAttribute("aria-controls") === panelId);
    if (!selectedTab) return;

    tabs.forEach((tab) => {
      const selected = tab === selectedTab;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== panelId;
    });

    if (updateHash) {
      window.history.replaceState(null, "", `#${panelId}`);
    }
    if (focus) selectedTab.focus();
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab.getAttribute("aria-controls")));
    tab.addEventListener("keydown", (event) => {
      let nextIndex = index;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") nextIndex = 0;
      else if (event.key === "End") nextIndex = tabs.length - 1;
      else return;

      event.preventDefault();
      activateTab(tabs[nextIndex].getAttribute("aria-controls"), { focus: true });
    });
  });

  document.querySelectorAll("[data-tab-target]").forEach((link) => {
    if (link.matches(".section-tab")) return;
    link.addEventListener("click", (event) => {
      const panelId = link.dataset.tabTarget;
      if (!panels.some((panel) => panel.id === panelId)) return;
      event.preventDefault();
      activateTab(panelId);
      document.querySelector(".section-tabs").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  const initialPanel = window.location.hash.slice(1);
  activateTab(
    panels.some((panel) => panel.id === initialPanel) ? initialPanel : "panel-predictor",
    { updateHash: false },
  );

  const examples = window.cellScopeExamples || {};
  const form = document.getElementById("prediction-form");
  const submitButton = document.getElementById("submit-button");
  const clearButton = document.getElementById("clear-button");
  const searchInput = document.getElementById("dataset-search");
  const diagnosisFilter = document.getElementById("diagnosis-filter");
  const tableBody = document.getElementById("dataset-rows");
  const pageIndicator = document.getElementById("page-indicator");
  const datasetCount = document.getElementById("dataset-count");
  const previousPage = document.getElementById("previous-page");
  const nextPage = document.getElementById("next-page");
  const dataset = window.cellScopeDataset || [];
  const explorerFeatures = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "smoothness_mean",
    "compactness_mean",
  ];
  const pageSize = 12;
  let currentPage = 0;
  let filteredRows = dataset;

  const renderDataset = () => {
    if (!tableBody) return;
    const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
    currentPage = Math.min(currentPage, pageCount - 1);
    tableBody.replaceChildren();

    filteredRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize).forEach((row) => {
      const tr = document.createElement("tr");
      const diagnosis = document.createElement("td");
      diagnosis.textContent = row.diagnosis;
      diagnosis.className = row.diagnosis === "Benign" ? "class-benign" : "class-malignant";
      tr.appendChild(diagnosis);

      explorerFeatures.forEach((feature) => {
        const cell = document.createElement("td");
        cell.textContent = Number(row[feature]).toFixed(4);
        tr.appendChild(cell);
      });
      tableBody.appendChild(tr);
    });

    const firstItem = filteredRows.length ? currentPage * pageSize + 1 : 0;
    const lastItem = Math.min((currentPage + 1) * pageSize, filteredRows.length);
    pageIndicator.textContent = `Page ${currentPage + 1} of ${pageCount}`;
    datasetCount.textContent = `${firstItem}–${lastItem} of ${filteredRows.length} records`;
    previousPage.disabled = currentPage === 0;
    nextPage.disabled = currentPage >= pageCount - 1;
  };

  const filterDataset = () => {
    const query = searchInput.value.trim().toLowerCase();
    const diagnosis = diagnosisFilter.value;
    filteredRows = dataset.filter((row) => {
      const diagnosisMatches = diagnosis === "all" || row.diagnosis === diagnosis;
      const queryMatches = !query || [row.diagnosis, ...explorerFeatures.map((feature) => row[feature])]
        .join(" ")
        .toLowerCase()
        .includes(query);
      return diagnosisMatches && queryMatches;
    });
    currentPage = 0;
    renderDataset();
  };

  searchInput?.addEventListener("input", filterDataset);
  diagnosisFilter?.addEventListener("change", filterDataset);
  previousPage?.addEventListener("click", () => {
    currentPage = Math.max(0, currentPage - 1);
    renderDataset();
  });
  nextPage?.addEventListener("click", () => {
    currentPage += 1;
    renderDataset();
  });
  renderDataset();

  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      const sample = examples[button.dataset.example];
      if (!sample) return;

      Object.entries(sample).forEach(([name, value]) => {
        const input = document.getElementById(name);
        if (input) input.value = Number(value).toFixed(name.includes("smoothness") || name.includes("compactness") ? 4 : 2);
      });

      document.getElementById("panel-predictor").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  clearButton?.addEventListener("click", () => {
    form?.reset();
    form?.querySelector("input")?.focus();
  });

  form?.addEventListener("submit", () => {
    if (!form.checkValidity()) return;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> Analyzing measurements…';
  });
});
