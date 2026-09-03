"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const PALETTE = [
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
  "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99",
];
const STORAGE_KEY = "sir-classifier-state";

const state = {
  config: null,
  center: null,          // {lat, lon}
  classes: [],           // [{name, color}]
  selected: 0,
  points: [],            // [{lat, lon, cls}]
  trainingYear: 2025,    // year the points are labelled/trained on
  targetYear: 2025,      // year being classified/displayed
  compareA: 2017,        // Compare card: first year
  compareB: 2025,        // Compare card: second year
  trainedYear: null,
  classifier: "rf",
  opacity: 0.7,
  dirty: false,
  classifiedOnce: false,
};

let map, boxLayer, overlayLayer, boxBounds;
const markers = L.layerGroup();

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
init();

async function init() {
  setupMap();
  await loadConfig();
  restore();
  wireControls();
  renderClasses();
  renderPoints();
  if (state.center) goToArea(state.center.lat, state.center.lon, false);
  syncTrainUI();
  syncCompareUI();
  syncYearUI();
}

function setupMap() {
  map = L.map("map", { minZoom: 2, maxZoom: 19 }).setView([20, 0], 2);
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution:
        "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    }
  ).addTo(map);
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 19, opacity: 0.9 }
  ).addTo(map);

  map.createPane("classPane");
  map.getPane("classPane").style.zIndex = 350; // below markers, above imagery
  markers.addTo(map);

  map.on("click", (e) => onMapClick(e.latlng));

  // Right-click opens a small confirm menu at the cursor (rather than moving the
  // box instantly, which is easy to trigger by accident and would orphan any
  // points placed in the old area). Left-click still adds points; Leaflet's
  // contextmenu already suppresses the browser menu.
  map.on("contextmenu", (e) => showAreaMenu(e.latlng));

  // Live coordinate readout in the map's bottom-left corner.
  const coords = document.getElementById("coords");
  map.on("mousemove", (e) => {
    const ll = e.latlng.wrap();
    coords.textContent = `${ll.lat.toFixed(4)}, ${ll.lng.toFixed(4)}`;
    coords.hidden = false;
  });
  map.on("mouseout", () => { coords.hidden = true; });
}

async function loadConfig() {
  try {
    const r = await fetch("/api/config");
    state.config = await r.json();
  } catch (e) {
    return status("Could not load configuration from the server.", true);
  }
  const c = state.config;
  state.targetYear = c.default_year;
  state.trainingYear = c.default_year;
  state.classifier = c.default_classifier || "rf";
  document.getElementById("box-km").textContent = c.box_size_km;
  const yr = document.getElementById("year");
  yr.min = c.min_year;
  yr.max = c.max_year;
  yr.value = c.default_year;
  const trainYr = document.getElementById("train-year");
  for (let y = c.min_year; y <= c.max_year; y++) {
    const opt = document.createElement("option");
    opt.value = String(y);
    opt.textContent = String(y);
    trainYr.appendChild(opt);
  }
  trainYr.value = String(c.default_year);
  // Compare card: two year dropdowns spanning the full range (earliest → latest).
  state.compareA = c.min_year;
  state.compareB = c.max_year;
  for (const [id, val] of [["compare-a", c.min_year], ["compare-b", c.max_year]]) {
    const sel = document.getElementById(id);
    for (let y = c.min_year; y <= c.max_year; y++) {
      const opt = document.createElement("option");
      opt.value = String(y);
      opt.textContent = String(y);
      sel.appendChild(opt);
    }
    sel.value = String(val);
  }
  const presetSel = document.getElementById("preset");
  c.presets.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = `${p.lat},${p.lon}`;
    opt.textContent = `${p.name} — ${p.blurb}`;
    presetSel.appendChild(opt);
  });
  document.getElementById("classifier").value = state.classifier;
}

// ---------------------------------------------------------------------------
// Controls wiring
// ---------------------------------------------------------------------------
function wireControls() {
  document.getElementById("preset").addEventListener("change", (e) => {
    if (!e.target.value) return;
    const [lat, lon] = e.target.value.split(",").map(Number);
    document.getElementById("lat").value = lat;
    document.getElementById("lon").value = lon;
    goToArea(lat, lon, true);
  });
  document.getElementById("go").addEventListener("click", () => {
    const lat = parseFloat(document.getElementById("lat").value);
    const lon = parseFloat(document.getElementById("lon").value);
    if (Number.isNaN(lat) || Number.isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return status("Enter a valid latitude (−90..90) and longitude (−180..180).", true);
    }
    goToArea(lat, lon, true);
  });

  document.getElementById("add-class").addEventListener("click", addClass);
  document.getElementById("new-class").addEventListener("keydown", (e) => {
    if (e.key === "Enter") addClass();
  });
  document.getElementById("undo").addEventListener("click", () => {
    if (state.points.length) { state.points.pop(); afterPointsChange(); }
  });
  document.getElementById("clear-points").addEventListener("click", () => {
    if (state.points.length && confirm("Remove all training points?")) {
      state.points = []; afterPointsChange();
    }
  });

  document.getElementById("classifier").addEventListener("change", (e) => {
    state.classifier = e.target.value; markDirty(); persist();
  });
  document.getElementById("train-year").addEventListener("change", (e) => {
    state.trainingYear = +e.target.value;
    state.targetYear = state.trainingYear; // classify year matches the train year by default
    markDirty(); syncYearUI(); persist();
  });
  document.getElementById("train").addEventListener("click", () => classify(true));
  document.getElementById("classify").addEventListener("click", () => {
    if (!state.classifiedOnce) return status("Train a model first (step 4).", true);
    classify(false);
  });

  const yr = document.getElementById("year");
  yr.addEventListener("input", (e) => { state.targetYear = +e.target.value; syncYearUI(); });
  yr.addEventListener("change", () => persist());

  document.getElementById("opacity").addEventListener("input", (e) => {
    state.opacity = e.target.value / 100;
    if (overlayLayer) overlayLayer.setOpacity(state.opacity);
    persist();
  });

  document.getElementById("compare-a").addEventListener("change", (e) => {
    state.compareA = +e.target.value; persist();
  });
  document.getElementById("compare-b").addEventListener("change", (e) => {
    state.compareB = +e.target.value; persist();
  });
  document.getElementById("compare").addEventListener("click", compare);

  document.getElementById("export").addEventListener("click", exportProject);
  document.getElementById("import-btn").addEventListener("click", () =>
    document.getElementById("import").click()
  );
  document.getElementById("import").addEventListener("change", importProject);

  document.addEventListener("keydown", (e) => {
    if (/^[1-9]$/.test(e.key) && document.activeElement.tagName !== "INPUT") {
      const i = +e.key - 1;
      if (i < state.classes.length) { state.selected = i; renderClasses(); }
    }
  });
}

// ---------------------------------------------------------------------------
// Study area
// ---------------------------------------------------------------------------
// `newArea` distinguishes a deliberate move to a fresh area (preset / Go /
// right-click) from merely restoring a saved project (init / import).
function goToArea(lat, lon, newArea) {
  state.center = { lat, lon };
  boxBounds = L.latLngBounds(kmBox(lat, lon, state.config.box_size_km));
  if (boxLayer) map.removeLayer(boxLayer);
  boxLayer = L.rectangle(boxBounds, {
    color: "#ffd400", weight: 2, dashArray: "6", fill: false, interactive: false,
  }).addTo(map);
  map.fitBounds(boxBounds, { padding: [20, 20] });
  if (newArea) {
    // Moving resets the project — points belong to the old location and would be
    // meaningless here. Named classes (and their colours) are kept for reuse.
    state.points = [];
    clearOverlay();
    state.trainedYear = null;
    state.classifiedOnce = false;
    state.dirty = false;
    // Fallback training + matching classify year until the Esri imagery date
    // arrives and seeds them.
    state.trainingYear = state.config.default_year;
    state.targetYear = state.config.default_year;
    document.getElementById("train-accuracy").hidden = true;
    renderClasses(); // per-class counts back to 0
    renderPoints();  // clear the markers
    syncTrainUI();   // resets the Train button + year dropdown
    syncYearUI();
  }
  fetchImageryDate(lat, lon, newArea);
  persist();
}

function kmBox(lat, lon, km) {
  const half = km / 2;
  const dLat = half / 110.574;
  const dLon = half / (111.32 * Math.cos((lat * Math.PI) / 180));
  return [[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]];
}

// Confirmation popup for a right-click, so the box only moves on a deliberate
// second click. Warns that moving clears the existing points.
function showAreaMenu(latlng) {
  const ll = latlng.wrap(); // keep lon in -180..180 for the backend/inputs
  const lat = +ll.lat.toFixed(5), lon = +ll.lng.toFixed(5);
  const moving = state.center != null;
  const n = state.points.length;
  const warn = moving && n
    ? `<p class="map-menu-warn">WARNING: Moving the box deletes all (${n}) training point${n === 1 ? "" : "s"}.</p>`
    : "";

  const div = document.createElement("div");
  div.className = "map-menu";
  div.innerHTML =
    `<div class="map-menu-coord">${lat.toFixed(4)}, ${lon.toFixed(4)}</div>${warn}` +
    `<button class="btn small" type="button">${moving ? "Move" : "Place"} study box here</button>`;

  const popup = L.popup({ className: "map-menu-popup", closeButton: false, offset: [0, 0] })
    .setLatLng(latlng)
    .setContent(div)
    .openOn(map);

  div.querySelector("button").addEventListener("click", () => {
    document.getElementById("lat").value = lat;
    document.getElementById("lon").value = lon;
    goToArea(lat, lon, true);
    map.closePopup(popup);
  });
}

// The Esri basemap is an undated multi-date mosaic. Ask Esri's token-free identify
// op for the acquisition date of the imagery under the box centre and label it a
// mosaic, so students don't assume it matches the AEF year they're classifying.
// Display-only; any failure degrades to an honest note.
const IMAGERY_FALLBACK =
  "Basemap is a multi-date Esri mosaic — not year-matched to the data.";
let imageryReqId = 0;

async function fetchImageryDate(lat, lon, seed) {
  const el = document.getElementById("imagery-date");
  if (!el) return;
  const reqId = ++imageryReqId; // guard against a stale response for an old area
  el.textContent = "Checking Esri imagery date…";
  try {
    const size = map.getSize();
    const params = new URLSearchParams({
      f: "json",
      geometry: `${lon},${lat}`,
      geometryType: "esriGeometryPoint",
      sr: "4326",
      layers: "all:0", // the World Imagery footprint layer
      tolerance: "1",
      returnGeometry: "false",
      mapExtent: [
        boxBounds.getWest(), boxBounds.getSouth(),
        boxBounds.getEast(), boxBounds.getNorth(),
      ].join(","),
      imageDisplay: `${size.x},${size.y},96`,
    });
    const r = await fetch(
      "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/identify?" +
        params
    );
    const data = await r.json();
    if (reqId !== imageryReqId) return; // a newer area superseded this request
    const year = imageryYear(data);
    el.textContent = year
      ? `Satellite basemap imagery here is from: ${year}.`
      : IMAGERY_FALLBACK;
    // Suggest labelling on the year you actually see, and seed it on a new area.
    const hint = document.getElementById("train-year-hint");
    if (year) {
      const clamped = Math.min(state.config.max_year, Math.max(state.config.min_year, +year));
      hint.textContent = `Tip: match the satellite imagery (here: ${year}) you're labelling.`;
      if (seed) {
        state.trainingYear = clamped;
        state.targetYear = clamped; // classify year matches the train year by default
        syncTrainUI();
        syncYearUI();
        markDirty(); // a re-train is needed if a model already existed
        persist();
      }
    } else {
      hint.textContent = "";
    }
  } catch (e) {
    if (reqId === imageryReqId) el.textContent = IMAGERY_FALLBACK;
  }
}

function imageryYear(data) {
  const results = (data && data.results) || [];
  for (const res of results) {
    const d = res.attributes && res.attributes.SRC_DATE2;
    if (d) {
      const year = String(d).split("/").pop(); // "8/29/2025" -> "2025"
      if (/^\d{4}$/.test(year)) return year;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------
function addClass() {
  const input = document.getElementById("new-class");
  const name = input.value.trim();
  if (!name) return;
  if (state.classes.some((c) => c.name === name)) return status("That class already exists.", true);
  if (state.classes.length >= state.config.max_classes) {
    return status(`At most ${state.config.max_classes} classes.`, true);
  }
  const color = document.getElementById("new-color").value;
  state.classes.push({ name, color });
  state.selected = state.classes.length - 1;
  input.value = "";
  document.getElementById("new-color").value = PALETTE[state.classes.length % PALETTE.length];
  markDirty();
  renderClasses();
  persist();
}

function removeClass(i) {
  const name = state.classes[i].name;
  state.classes.splice(i, 1);
  state.points = state.points.filter((p) => p.cls !== name);
  if (state.selected >= state.classes.length) state.selected = Math.max(0, state.classes.length - 1);
  markDirty();
  renderClasses();
  renderPoints();
  persist();
}

function renderClasses() {
  const ul = document.getElementById("class-list");
  ul.innerHTML = "";
  const counts = {};
  state.points.forEach((p) => (counts[p.cls] = (counts[p.cls] || 0) + 1));
  state.classes.forEach((c, i) => {
    const li = document.createElement("li");
    li.className = "class-row" + (i === state.selected ? " active" : "");

    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.className = "swatch-input";
    swatch.value = c.color;
    swatch.title = "class colour";
    swatch.addEventListener("click", (e) => e.stopPropagation());
    swatch.addEventListener("input", (e) => { e.stopPropagation(); recolorClass(i, e.target.value); });

    const name = document.createElement("input");
    name.type = "text";
    name.className = "class-name-input";
    name.value = c.name; // set as a property, not innerHTML — injection-safe
    name.maxLength = 24;
    name.title = "rename class";
    name.addEventListener("click", (e) => e.stopPropagation());
    name.addEventListener("change", (e) => { e.stopPropagation(); renameClass(i, e.target.value); });
    name.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") e.target.blur(); // commit via the change handler
    });

    const count = document.createElement("span");
    count.className = "class-count";
    count.textContent = `${counts[c.name] || 0} pts`;

    const del = document.createElement("button");
    del.className = "class-del";
    del.title = "remove class";
    del.innerHTML = "&times;";

    li.append(swatch, name, count, del);
    li.addEventListener("click", (e) => {
      if (e.target.classList.contains("class-del")) { removeClass(i); return; }
      state.selected = i; renderClasses();
    });
    ul.appendChild(li);
  });
}

function renameClass(i, rawValue) {
  const oldName = state.classes[i].name;
  const name = rawValue.trim();
  if (!name || name === oldName) return renderClasses(); // revert the input
  if (state.classes.some((c, j) => j !== i && c.name === name)) {
    status("That class already exists.", true);
    return renderClasses(); // revert the input to the old name
  }
  state.classes[i].name = name;
  state.points.forEach((p) => { if (p.cls === oldName) p.cls = name; });
  markDirty();
  renderClasses();
  renderPoints();
  persist();
}

function recolorClass(i, color) {
  state.classes[i].color = color;
  renderPoints(); // recolour the markers
  // Live-update the legend swatch if results are showing (i-th li matches class i).
  const bar = document.getElementById("legend").children[i]?.querySelector(".bar");
  if (bar) bar.style.background = color;
  persist();
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------
function onMapClick(latlng) {
  if (!state.center) return status("Pick a study area first (step 1).", true);
  if (!state.classes.length) return status("Add a class first (step 2).", true);
  if (boxBounds && !boxBounds.contains(latlng)) {
    return status("Click inside the yellow box.", true);
  }
  state.points.push({ lat: latlng.lat, lon: latlng.lng, cls: state.classes[state.selected].name });
  afterPointsChange();
}

function afterPointsChange() {
  markDirty();
  renderClasses();
  renderPoints();
  persist();
}

function renderPoints() {
  markers.clearLayers();
  const colorOf = Object.fromEntries(state.classes.map((c) => [c.name, c.color]));
  state.points.forEach((p, i) => {
    const m = L.circleMarker([p.lat, p.lon], {
      radius: 6, weight: 2, color: "#fff", opacity: 1,
      fillColor: colorOf[p.cls] || "#888", fillOpacity: 1, className: "point-marker",
    });
    m.on("click", (e) => { L.DomEvent.stopPropagation(e); state.points.splice(i, 1); afterPointsChange(); });
    m.bindTooltip(p.cls, { direction: "top" });
    markers.addLayer(m);
  });
}

// ---------------------------------------------------------------------------
// Classify
// ---------------------------------------------------------------------------
async function classify(train) {
  if (!state.center) return status("Pick a study area first.", true);
  if (state.classes.length < 2) return status("Define at least two classes.", true);
  if (state.points.length < 2) return status("Add some training points first.", true);

  // "Train model" fits on the chosen training year; moving the target slider
  // re-applies the already-trained model (its fixed trainedYear).
  const trainingYear = train ? state.trainingYear : state.trainedYear ?? state.trainingYear;
  const payload = {
    lat: state.center.lat,
    lon: state.center.lon,
    training_year: trainingYear,
    target_year: state.targetYear,
    classifier: state.classifier,
    classes: state.classes,
    points: state.points.map((p) => ({ class: p.cls, lat: p.lat, lon: p.lon })),
  };

  setBusy(true);
  try {
    const r = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) return status(await errorMessage(r), true);
    const data = await r.json();
    if (train) { state.trainedYear = trainingYear; state.classifiedOnce = true; }
    setOverlay(data.image, data.bounds);
    renderResults(data);
    setActiveAction(train ? "train" : "classify");
    state.dirty = false;
    updateTrainBtn();
    syncYearUI();
    persist();
    hideStatus();
  } catch (e) {
    status("Network error contacting the server.", true);
  } finally {
    setBusy(false);
  }
}

function setOverlay(dataUrl, bounds) {
  clearOverlay();
  overlayLayer = L.imageOverlay(dataUrl, bounds, { opacity: state.opacity, pane: "classPane" });
  overlayLayer.addTo(map);
  if (boxLayer) boxLayer.setBounds(bounds); // align guide box to the true classified area
}

function clearOverlay() {
  if (overlayLayer) { map.removeLayer(overlayLayer); overlayLayer = null; }
  document.getElementById("results").hidden = true;
  // Classify and Compare share one overlay, so clear the other view's legend too.
  document.getElementById("transitions").innerHTML = "";
  document.getElementById("compare-note").hidden = true;
  setActiveAction(null); // nothing on the map now
}

function renderResults(data) {
  const el = document.getElementById("results");
  el.hidden = false;
  const total = Object.values(data.class_pixel_counts).reduce((a, b) => a + b, 0) || 1;
  const acc = data.accuracy == null ? "—" : (data.accuracy * 100).toFixed(0) + "%";
  const skipped = data.n_points_skipped
    ? ` · <span title="points outside the box or on masked pixels">${data.n_points_skipped} skipped</span>`
    : "";
  // Accuracy is a property of the trained model, so it lives in the Train card.
  const accEl = document.getElementById("train-accuracy");
  accEl.innerHTML = `Cross-validated accuracy: <b>${acc}</b>`;
  accEl.hidden = false;
  document.getElementById("stats").innerHTML = `
    <div class="metric"><span>Training points used</span><b>${data.n_points_used}${skipped}</b></div>
    <div class="metric"><span>Showing</span><b>${data.target_year} (trained on ${data.training_year})</b></div>`;

  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  state.classes.forEach((c) => {
    const count = data.class_pixel_counts[c.name] || 0;
    const pct = ((count / total) * 100).toFixed(1);
    const li = document.createElement("li");
    li.innerHTML = `<span class="bar" style="background:${c.color};width:${Math.max(4, pct)}px"></span>
      <span>${escapeHtml(c.name)}</span><span class="class-count">${pct}%</span>`;
    legend.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Compare (change detection between two years)
// ---------------------------------------------------------------------------
async function compare() {
  if (!state.center) return status("Pick a study area first.", true);
  if (state.classes.length < 2) return status("Define at least two classes.", true);
  if (state.points.length < 2) return status("Add some training points first.", true);
  if (state.compareA === state.compareB) return status("Pick two different years to compare.", true);

  const payload = {
    lat: state.center.lat,
    lon: state.center.lon,
    training_year: state.trainingYear,
    year_a: state.compareA,
    year_b: state.compareB,
    classifier: state.classifier,
    classes: state.classes,
    points: state.points.map((p) => ({ class: p.cls, lat: p.lat, lon: p.lon })),
  };

  setBusy(true);
  try {
    const r = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) return status(await errorMessage(r), true);
    const data = await r.json();
    setOverlay(data.image, data.bounds); // clears the single-year legend
    renderTransitions(data);
    setActiveAction("compare");
    persist();
    hideStatus();
  } catch (e) {
    status("Network error contacting the server.", true);
  } finally {
    setBusy(false);
  }
}

function renderTransitions(data) {
  const trans = data.transitions || [];
  // Percentages are of the total comparable area (changed + unchanged), so they
  // read as "X% of the area went from → to" and sum to the total changed fraction.
  const total = data.compared_pixels || 1;
  const changed = trans.reduce((a, t) => a + t.pixels, 0);
  const changedPct = ((changed / total) * 100).toFixed(1);

  const note = document.getElementById("compare-note");
  const acc = data.accuracy == null ? "—" : (data.accuracy * 100).toFixed(0) + "%";
  note.textContent =
    `Trained on ${data.training_year} · accuracy ${acc} · ` +
    `${data.year_a} → ${data.year_b} · ${changedPct}% of area changed`;
  note.hidden = false;

  const ul = document.getElementById("transitions");
  ul.innerHTML = "";
  if (!trans.length) {
    const li = document.createElement("li");
    li.textContent = `No class changes between ${data.year_a} and ${data.year_b}.`;
    ul.appendChild(li);
    return;
  }
  // Scale bars to the largest transition so small (but real) changes stay visible.
  const MAX_BAR_PX = 100;
  const maxPixels = Math.max(...trans.map((t) => t.pixels));
  trans.forEach((t) => {
    const pct = ((t.pixels / total) * 100).toFixed(1);
    const li = document.createElement("li");
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.background = t.color;
    bar.style.width = Math.max(6, Math.round((t.pixels / maxPixels) * MAX_BAR_PX)) + "px";
    const label = document.createElement("span");
    label.textContent = `${t.from} → ${t.to}`; // textContent — injection-safe
    const count = document.createElement("span");
    count.className = "class-count";
    count.textContent = pct + "%";
    li.append(bar, label, count);
    ul.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function syncYearUI() {
  document.getElementById("year-label").textContent = state.targetYear;
  document.getElementById("year").value = state.targetYear;
  const note = document.getElementById("trained-note");
  note.textContent = state.trainedYear != null ? `trained on ${state.trainedYear}` : "";
  const hint = document.getElementById("classify-hint");
  hint.textContent = state.classifiedOnce
    ? "Pick a year and click Classify to map it."
    : "Train a model (step 4) first.";
}

// Keeps the Train card's controls (year dropdown + button) in sync with state.
function syncTrainUI() {
  document.getElementById("train-year").value = String(state.trainingYear);
  updateTrainBtn();
}

function markDirty() {
  if (state.classifiedOnce) { state.dirty = true; updateTrainBtn(); }
}

function updateTrainBtn() {
  const btn = document.getElementById("train");
  btn.textContent = state.classifiedOnce && state.dirty ? "Re-train model" : "Train model";
  btn.classList.toggle("dirty", state.classifiedOnce && state.dirty);
  btn.classList.toggle("trained", state.classifiedOnce); // persistent check once trained
}

function setBusy(busy) {
  document.getElementById("train").disabled = busy;
  document.getElementById("classify").disabled = busy;
  document.getElementById("year").disabled = busy;
  document.getElementById("compare").disabled = busy;
  if (busy) status("Reading embeddings and classifying… the first look at a new area can take ~30 s.");
}

// Highlights the action whose result is currently on the map (Train / Classify /
// Compare), so users can see what they're looking at. Pass null to clear.
function setActiveAction(name) {
  for (const id of ["train", "classify", "compare"]) {
    document.getElementById(id).classList.toggle("active", id === name);
  }
}

// Keeps the Compare card's year dropdowns in sync with state.
function syncCompareUI() {
  document.getElementById("compare-a").value = String(state.compareA);
  document.getElementById("compare-b").value = String(state.compareB);
}

function status(msg, isError) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
  el.hidden = false;
  if (isError) setTimeout(() => { if (el.textContent === msg) hideStatus(); }, 4000);
}
function hideStatus() { document.getElementById("status").hidden = true; }

async function errorMessage(r) {
  if (r.status === 429) return "Too many requests — please wait a moment and try again.";
  try {
    const j = await r.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d) => d.msg).join("; ");
  } catch (e) {}
  return `Request failed (${r.status}).`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---------------------------------------------------------------------------
// Persistence + import/export
// ---------------------------------------------------------------------------
function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot()));
  } catch (e) {}
}

function snapshot() {
  return {
    center: state.center, classes: state.classes, points: state.points,
    trainingYear: state.trainingYear, targetYear: state.targetYear,
    compareA: state.compareA, compareB: state.compareB,
    trainedYear: state.trainedYear,
    classifier: state.classifier, opacity: state.opacity,
  };
}

function restore() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch (e) {}
  if (saved && saved.classes && saved.classes.length) {
    applyProject(saved);
  } else {
    state.classes = [
      { name: "water", color: "#1f78b4" },
      { name: "vegetation", color: "#33a02c" },
      { name: "urban", color: "#e31a1c" },
      { name: "bare", color: "#b15928" },
    ];
  }
  document.getElementById("classifier").value = state.classifier;
  document.getElementById("opacity").value = Math.round(state.opacity * 100);
}

function applyProject(p) {
  state.center = p.center || null;
  state.classes = p.classes || [];
  state.points = p.points || [];
  // `shownYear` fallback keeps projects saved before the train/target split loading.
  state.targetYear = p.targetYear ?? p.shownYear ?? state.config.default_year;
  state.trainingYear = p.trainingYear ?? state.config.default_year;
  state.compareA = p.compareA ?? state.config.min_year;
  state.compareB = p.compareB ?? state.config.max_year;
  state.trainedYear = p.trainedYear ?? null;
  state.classifier = p.classifier || "rf";
  state.opacity = p.opacity ?? 0.7;
  state.selected = 0;
  state.classifiedOnce = false;
  state.dirty = false;
}

function exportProject() {
  const blob = new Blob([JSON.stringify(snapshot(), null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "land-cover-project.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

function importProject(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      applyProject(JSON.parse(reader.result));
      clearOverlay();
      document.getElementById("train-accuracy").hidden = true;
      document.getElementById("classifier").value = state.classifier;
      renderClasses();
      renderPoints();
      if (state.center) goToArea(state.center.lat, state.center.lon, false);
      syncTrainUI();
      syncCompareUI();
      syncYearUI();
      persist();
      status("Project loaded.");
    } catch (err) {
      status("That file could not be read as a project.", true);
    }
  };
  reader.readAsText(file);
  e.target.value = "";
}
