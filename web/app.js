"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const PALETTE = [
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
  "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99",
];
const STORAGE_KEY = "sir-classifier-state";
const UI_KEY = "sir-classifier-ui"; // which steps are open (per browser, not exported)

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
  featureSource: "aef",  // id from config.feature_sources ("aef" | "landsat")
  imageryStyle: "",      // "" = Esri basemap, else a Landsat style id (step 3)
  imageryYear: 2025,     // year of the Landsat background imagery
  opacity: 0.7,
  dirty: false,
  classifiedOnce: false,
};

let map, boxLayer, overlayLayer, imageryLayer, boxBounds;
let currentAction = null; // "train" | "classify" | "compare" — what the map shows
// Collapsible steps: `open` = expanded step numbers; `reached` = furthest step the
// guided flow has opened (auto-advance only happens the first time through).
const ui = { open: new Set([1]), reached: 1, didClassify: false, didCompare: false };
const markers = L.layerGroup();

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
init();

async function init() {
  setupMap();
  await loadConfig();
  restore();
  setupSteps();
  wireControls();
  renderClasses();
  renderPoints();
  if (state.center) goToArea(state.center.lat, state.center.lon, false);
  syncFeatureUI();
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

  map.createPane("imageryPane");
  map.getPane("imageryPane").style.zIndex = 250; // Landsat: above the basemap tiles
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
  state.featureSource = c.default_feature_source || "aef";
  // Compare card: default to the full AEF range (earliest → latest).
  state.compareA = c.min_year;
  state.compareB = c.max_year;
  state.imageryYear = c.default_year;

  const featSel = document.getElementById("feature-source");
  c.feature_sources.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f.id;
    opt.textContent = `${f.label} (${f.min_year}–${f.max_year}, ${f.res_m} m)`;
    featSel.appendChild(opt);
  });
  const styleSel = document.getElementById("imagery-style");
  (c.imagery_styles || []).forEach((st) => {
    const opt = document.createElement("option");
    opt.value = st.id;
    opt.textContent = `Landsat — ${st.label}`;
    styleSel.appendChild(opt);
  });
  const lsat = imagerySource();
  if (lsat) fillYearSelect(document.getElementById("imagery-year"), lsat.min_year, lsat.max_year);
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
  document.getElementById("feature-source").addEventListener("change", (e) => {
    state.featureSource = e.target.value;
    markDirty();
    syncFeatureUI(); // re-ranges every year control for the new source
    syncTrainUI();
    syncCompareUI();
    syncYearUI();
    persist();
  });
  document.getElementById("train-year").addEventListener("change", (e) => {
    state.trainingYear = +e.target.value;
    state.targetYear = state.trainingYear; // classify year matches the train year by default
    markDirty(); syncYearUI(); syncImageryUI(); persist();
  });
  document.getElementById("evaluate").addEventListener("click", evaluateFeatures);

  document.getElementById("imagery-style").addEventListener("change", (e) => {
    const turningOn = !state.imageryStyle && e.target.value;
    state.imageryStyle = e.target.value;
    // Start on the year the map is showing (Compare's first year, or the classified year).
    if (turningOn) state.imageryYear = clampToSource(mapYear(), imagerySource());
    showImagery();
    persist();
  });
  document.getElementById("imagery-year").addEventListener("change", (e) => {
    state.imageryYear = +e.target.value; showImagery(); persist();
  });
  document.getElementById("imagery-prev").addEventListener("click", () => stepImageryYear(-1));
  document.getElementById("imagery-next").addEventListener("click", () => stepImageryYear(1));
  document.querySelectorAll("#imagery-jumps [data-jump]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const src = imagerySource();
      const year = { a: state.compareA, b: state.compareB }[btn.dataset.jump];
      state.imageryYear = clampToSource(year, src);
      showImagery();
      persist();
    })
  );
  document.getElementById("train").addEventListener("click", () => classify(true));
  document.getElementById("classify").addEventListener("click", () => {
    if (!state.classifiedOnce) return status("Train a model first (step 4).", true);
    classify(false);
  });

  const yr = document.getElementById("year");
  yr.addEventListener("input", (e) => { state.targetYear = +e.target.value; syncYearUI(); });
  yr.addEventListener("change", () => persist());

  // Two sliders (steps 5 and 7) drive the same overlay opacity.
  document.querySelectorAll(".opacity-input").forEach((el) =>
    el.addEventListener("input", (e) => {
      state.opacity = e.target.value / 100;
      if (overlayLayer) overlayLayer.setOpacity(state.opacity);
      document.querySelectorAll(".opacity-input").forEach((o) => { o.value = e.target.value; });
      persist();
    })
  );

  document.getElementById("compare-a").addEventListener("change", (e) => {
    state.compareA = +e.target.value; syncImageryUI(); persist();
  });
  document.getElementById("compare-b").addEventListener("change", (e) => {
    state.compareB = +e.target.value; syncImageryUI(); persist();
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
      if (i < state.classes.length) selectClass(i);
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
    document.getElementById("feature-compare").hidden = true;
    ui.didClassify = ui.didCompare = false;
    renderClasses(); // per-class counts back to 0
    renderPoints();  // clear the markers
    syncTrainUI();   // resets the Train button + year dropdown
    syncYearUI();
  }
  syncFeatureUI(); // preset-only sources are enabled just on the example places
  syncTrainUI();
  syncCompareUI();
  syncYearUI();
  showImagery();   // reload (or drop) the Landsat background for the new area
  fetchImageryDate(lat, lon, newArea);
  persist();
  if (newArea) advanceFrom(1);
}

function currentAreaIsPreset() {
  if (!state.center || !state.config) return false;
  return (state.config.presets || []).some(
    (p) => Math.abs(p.lat - state.center.lat) < 1e-3 && Math.abs(p.lon - state.center.lon) < 1e-3
  );
}

// ---------------------------------------------------------------------------
// Feature sources (AEF embeddings, Landsat bands, …) and their year ranges
// ---------------------------------------------------------------------------
function sourceById(id) {
  return (state.config.feature_sources || []).find((f) => f.id === id) || null;
}
function currentSource() {
  return sourceById(state.featureSource) || state.config.feature_sources[0];
}
// The source whose raw imagery step 3 can show (Landsat).
function imagerySource() {
  return (state.config.feature_sources || []).find((f) => f.imagery) || null;
}
function sourceAllowedHere(src) {
  return !!src && (!src.preset_only || currentAreaIsPreset());
}
function clampToSource(year, src) {
  return Math.min(src.max_year, Math.max(src.min_year, year));
}

function fillYearSelect(sel, min, max) {
  const keep = sel.value;
  sel.innerHTML = "";
  for (let y = min; y <= max; y++) {
    const opt = document.createElement("option");
    opt.value = String(y);
    opt.textContent = String(y);
    sel.appendChild(opt);
  }
  if (keep) sel.value = keep;
}

// Slow sources (Landsat) are pre-warmed for the example places only, so they are
// enabled just there. Also re-ranges every year control to the chosen source.
function syncFeatureUI() {
  const sel = document.getElementById("feature-source");
  const note = document.getElementById("feature-note");
  for (const f of state.config.feature_sources) {
    sel.querySelector(`option[value="${f.id}"]`).disabled = !sourceAllowedHere(f);
  }
  if (!sourceAllowedHere(currentSource())) state.featureSource = state.config.default_feature_source;
  sel.value = state.featureSource;

  const src = currentSource();
  const blocked = state.config.feature_sources.filter((f) => !sourceAllowedHere(f));
  note.textContent = blocked.length
    ? `${blocked.map((f) => f.label).join(", ")}: example places only, for now.`
    : src.id === "aef"
      ? "AlphaEarth embeddings summarise a whole year of many satellites, 2017 onward."
      : `${src.label}: the raw ${src.res_m} m reflectance, back to ${src.min_year}.`;

  // A model belongs to one source + year; drop it if that year no longer exists.
  if (state.trainedYear != null && (state.trainedYear < src.min_year || state.trainedYear > src.max_year)) {
    state.trainedYear = null;
    state.classifiedOnce = false;
    state.dirty = false;
    updateTrainBtn();
  }
  state.trainingYear = clampToSource(state.trainingYear, src);
  state.targetYear = clampToSource(state.targetYear, src);
  state.compareA = clampToSource(state.compareA, src);
  state.compareB = clampToSource(state.compareB, src);
  fillYearSelect(document.getElementById("train-year"), src.min_year, src.max_year);
  fillYearSelect(document.getElementById("compare-a"), src.min_year, src.max_year);
  fillYearSelect(document.getElementById("compare-b"), src.min_year, src.max_year);
  const yr = document.getElementById("year");
  yr.min = src.min_year;
  yr.max = src.max_year;
}

// ---------------------------------------------------------------------------
// Satellite imagery: Landsat composites over the Esri basemap (step 7)
// ---------------------------------------------------------------------------
let imageryReq = 0;
const imageryMemo = new Map(); // "lat,lon,year,style" -> response (instant re-flips)
const IMAGERY_MEMO_MAX = 80;

// The year the classification overlay is showing (for Compare: its first year).
function mapYear() {
  if (currentAction === "compare") return state.compareA;
  if (currentAction) return state.targetYear;
  return state.trainingYear;
}

function stepImageryYear(delta) {
  const src = imagerySource();
  if (!src) return;
  const next = clampToSource(state.imageryYear + delta, src);
  if (next === state.imageryYear) return;
  state.imageryYear = next;
  showImagery();
  persist();
}

function clearImagery() {
  if (imageryLayer) { map.removeLayer(imageryLayer); imageryLayer = null; }
}

function syncImageryUI(noteText, warn) {
  const src = imagerySource();
  const styleSel = document.getElementById("imagery-style");
  const allowed = sourceAllowedHere(src) && !!state.center;
  for (const opt of styleSel.options) if (opt.value) opt.disabled = !allowed;
  if (!allowed) state.imageryStyle = "";
  styleSel.value = state.imageryStyle;
  document.getElementById("imagery-year-row").hidden = !state.imageryStyle;
  const jumps = document.getElementById("imagery-jumps");
  jumps.hidden = !state.imageryStyle;
  const labels = { a: `Year A · ${state.compareA}`, b: `Year B · ${state.compareB}` };
  const years = { a: state.compareA, b: state.compareB };
  jumps.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.textContent = labels[btn.dataset.jump];
    btn.classList.toggle("current", years[btn.dataset.jump] === state.imageryYear);
  });
  document.getElementById("imagery-year").value = String(state.imageryYear);
  if (src) {
    document.getElementById("imagery-prev").disabled = state.imageryYear <= src.min_year;
    document.getElementById("imagery-next").disabled = state.imageryYear >= src.max_year;
  }
  const note = document.getElementById("imagery-note");
  if (noteText == null) {
    noteText = !src ? ""
      : !state.center ? ""
      : !allowed ? "Landsat imagery is available for the example places only, for now."
      : !state.imageryStyle ? `Switch to Landsat to see this place any year from ${src.min_year} to ${src.max_year}.`
      : "";
  }
  note.textContent = noteText;
  note.classList.toggle("warn", !!warn);
}

async function showImagery() {
  const src = imagerySource();
  syncImageryUI(); // also turns the style off when not allowed here
  if (!state.imageryStyle || !state.center || !src) { clearImagery(); return; }

  const { lat, lon } = state.center;
  const year = state.imageryYear;
  const style = state.imageryStyle;
  const key = `${lat},${lon},${year},${style}`;
  const reqId = ++imageryReq;
  let data = imageryMemo.get(key);
  if (!data) {
    syncImageryUI(`Loading Landsat ${year}… a year nobody has looked at yet can take ~20 s.`);
    try {
      const qs = new URLSearchParams({ lat, lon, year, style, source: src.id });
      const r = await fetch("/api/imagery?" + qs);
      if (reqId !== imageryReq) return; // the user has moved on to another year/area
      if (!r.ok) {
        clearImagery();
        return syncImageryUI(`No Landsat picture for ${year}: ${await errorMessage(r)}`, true);
      }
      data = await r.json();
    } catch (e) {
      if (reqId === imageryReq) syncImageryUI("Network error loading Landsat imagery.", true);
      return;
    }
    if (reqId !== imageryReq) return;
    imageryMemo.set(key, data);
    if (imageryMemo.size > IMAGERY_MEMO_MAX) imageryMemo.delete(imageryMemo.keys().next().value);
  }
  clearImagery();
  imageryLayer = L.imageOverlay(data.image, data.bounds, {
    pane: "imageryPane", className: "pixelated", interactive: false,
  }).addTo(map);
  if (boxLayer) boxLayer.setBounds(data.bounds); // the server's snapped box

  const gaps = data.coverage < 0.995
    ? ` · ${((1 - data.coverage) * 100).toFixed(0)}% gaps (always cloudy or no data)`
    : "";
  syncImageryUI(`Landsat ${year} · ${data.res_m} m pixels${gaps}`);
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
      const clamped = clampToSource(+year, currentSource());
      hint.textContent = `Tip: match the satellite imagery (here: ${year}) you're labelling.`;
      if (seed) {
        state.trainingYear = clamped;
        state.targetYear = clamped; // classify year matches the train year by default
        syncTrainUI();
        syncImageryUI();
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
    name.readOnly = true; // reads as plain text; single click selects the row
    name.title = "double-click to rename";
    // Double-click switches to edit mode (single click bubbles up to select).
    name.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      name.readOnly = false;
      name.focus();
      name.select();
    });
    name.addEventListener("keydown", (e) => {
      if (name.readOnly) return; // not editing — let shortcuts (number keys) through
      e.stopPropagation();
      if (e.key === "Enter") name.blur();
      if (e.key === "Escape") { name.value = c.name; name.blur(); }
    });
    name.addEventListener("blur", () => {
      if (name.readOnly) return;
      name.readOnly = true;
      renameClass(i, name.value); // validates, reverts dupes/empties, re-renders
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
      selectClass(i);
    });
    ul.appendChild(li);
  });
}

// Selecting is the frequent action, so do it in place (no full re-render) — a
// rebuild would destroy the row between clicks and break double-click-to-rename.
function selectClass(i) {
  state.selected = i;
  document.querySelectorAll("#class-list .class-row")
    .forEach((li, j) => li.classList.toggle("active", j === i));
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
    feature_source: state.featureSource,
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
    if (!train) ui.didClassify = true;
    persist();
    hideStatus();
    advanceFrom(train ? 4 : 5);
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
    <div class="metric"><span>Features</span><b>${escapeHtml(sourceById(data.feature_source)?.label || data.feature_source)}</b></div>
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
    feature_source: state.featureSource,
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
    ui.didCompare = true;
    persist();
    hideStatus();
    advanceFrom(6);
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
// Compare feature types (accuracy of each source on the same points)
// ---------------------------------------------------------------------------
async function evaluateFeatures() {
  if (!state.center) return status("Pick a study area first.", true);
  if (state.classes.length < 2) return status("Define at least two classes.", true);
  if (state.points.length < 2) return status("Add some training points first.", true);

  const payload = {
    lat: state.center.lat,
    lon: state.center.lon,
    training_year: state.trainingYear,
    classifier: state.classifier,
    classes: state.classes,
    points: state.points.map((p) => ({ class: p.cls, lat: p.lat, lon: p.lon })),
  };
  setBusy(true);
  status("Scoring your points with each feature type…");
  try {
    const r = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) return status(await errorMessage(r), true);
    renderFeatureCompare(await r.json());
    hideStatus();
  } catch (e) {
    status("Network error contacting the server.", true);
  } finally {
    setBusy(false);
  }
}

function renderFeatureCompare(data) {
  const box = document.getElementById("feature-compare");
  box.innerHTML = "";
  box.hidden = false;
  const pct = (a) => (a == null ? "—" : (a * 100).toFixed(0) + "%");
  const results = data.results;

  // One row per class, plus "All points"; one column per feature source. The
  // best score in each row is highlighted (ties all highlighted).
  const rows = [{ name: "All points", get: (r) => r.accuracy, overall: true }].concat(
    state.classes
      .filter((c) => results.some((r) => r.per_class[c.name] != null))
      .map((c) => ({ name: c.name, get: (r) => r.per_class[c.name] }))
  );
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  head.appendChild(document.createElement("th")).textContent = `Accuracy, ${data.training_year}`;
  results.forEach((r) => {
    const th = document.createElement("th");
    th.textContent = r.label.replace(/ (embeddings|bands)$/, "");
    th.title = `${r.label} (${r.res_m} m pixels)`;
    head.appendChild(th);
  });
  const body = table.createTBody();
  rows.forEach((row) => {
    const tr = body.insertRow();
    if (row.overall) tr.className = "overall";
    tr.insertCell().textContent = row.name;
    const vals = results.map(row.get);
    const best = Math.max(...vals.filter((v) => v != null));
    vals.forEach((v) => {
      const td = tr.insertCell();
      td.textContent = pct(v);
      if (results.length > 1 && v != null && v === best) td.className = "best";
    });
  });
  box.appendChild(table);

  const note = document.createElement("p");
  note.className = "hint";
  const dropped = data.n_points_total - data.n_points_shared;
  note.textContent =
    `Each point is hidden from the model in turn and predicted (cross-validation), ` +
    `using the same ${data.n_points_shared} points for every feature type` +
    (dropped ? ` (${dropped} skipped: outside the box or on a gap in one of them).` : ".");
  box.appendChild(note);
  if (results.some((r) => r.accuracy == null)) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "Add at least 2 points to every class to get an accuracy.";
    box.appendChild(p);
  }
  (data.skipped || []).forEach((s) => {
    const p = document.createElement("p");
    p.className = "hint warn";
    p.textContent = `${s.label}: not compared — ${s.reason}`;
    box.appendChild(p);
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
  document.getElementById("feature-compare").hidden = true; // scores are now stale
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
  document.getElementById("evaluate").disabled = busy;
  if (busy) {
    status(state.featureSource === "landsat"
      ? "Reading Landsat composites and classifying… a year nobody has used yet can take ~20 s."
      : "Reading embeddings and classifying… the first look at a new area can take ~30 s.");
  }
}

// Highlights the action whose result is currently on the map (Train / Classify /
// Compare), so users can see what they're looking at. Pass null to clear.
function setActiveAction(name) {
  currentAction = name;
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
  syncStepDone();
}

// ---------------------------------------------------------------------------
// Collapsible steps
// ---------------------------------------------------------------------------
const NEXT_BUTTON_STEPS = [2, 3, 5, 6]; // steps without a natural "done" action

function stepCards() {
  return [...document.querySelectorAll(".step-card")];
}

// Wraps each step's content in a collapsible body, makes its heading the toggle,
// and adds "Next step" buttons. Runs once at boot.
function setupSteps() {
  try {
    const saved = JSON.parse(localStorage.getItem(UI_KEY));
    if (saved && Array.isArray(saved.open)) {
      ui.open = new Set(saved.open);
      ui.reached = saved.reached || 1;
    }
  } catch (e) {}

  stepCards().forEach((card) => {
    const n = +card.dataset.step;
    const h2 = card.querySelector("h2");
    const num = h2.querySelector(".step");
    num.innerHTML = `<span class="num">${num.textContent}</span>`;
    const chev = document.createElement("span");
    chev.className = "chev";
    chev.textContent = "▼";
    chev.setAttribute("aria-hidden", "true");
    h2.appendChild(chev);
    h2.tabIndex = 0;
    h2.setAttribute("role", "button");

    const body = document.createElement("div");
    body.className = "step-body";
    [...card.children].filter((el) => el !== h2).forEach((el) => body.appendChild(el));
    card.appendChild(body);

    if (NEXT_BUTTON_STEPS.includes(n)) {
      const next = document.createElement("button");
      next.type = "button";
      next.className = "btn small ghost step-next";
      next.textContent = "Next step ▸";
      next.addEventListener("click", () => goToStep(n + 1, n));
      body.appendChild(next);
    }

    const toggle = () => setStepOpen(n, !ui.open.has(n));
    h2.addEventListener("click", toggle);
    h2.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
  });
  renderSteps();
}

function setStepOpen(n, open) {
  if (open) ui.open.add(n); else ui.open.delete(n);
  renderSteps();
}

// Collapse `from`, open `n`, and bring it into view.
function goToStep(n, from) {
  if (!stepCards().some((c) => +c.dataset.step === n)) return;
  if (from != null) ui.open.delete(from);
  ui.open.add(n);
  ui.reached = Math.max(ui.reached, n);
  renderSteps();
  const card = document.querySelector(`.step-card[data-step="${n}"]`);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Called when step `n` is completed: moves on only the first time through, so
// re-running a step later (e.g. classifying another year) doesn't jump around.
function advanceFrom(n) {
  if (ui.reached <= n) goToStep(n + 1, n);
}

function renderSteps() {
  stepCards().forEach((card) => {
    const open = ui.open.has(+card.dataset.step);
    card.classList.toggle("collapsed", !open);
    card.querySelector("h2").setAttribute("aria-expanded", String(open));
  });
  try {
    localStorage.setItem(UI_KEY, JSON.stringify({ open: [...ui.open], reached: ui.reached }));
  } catch (e) {}
}

// A green check on each step whose job is done.
function syncStepDone() {
  const byClass = {};
  state.points.forEach((p) => (byClass[p.cls] = (byClass[p.cls] || 0) + 1));
  const done = {
    1: !!state.center,
    2: state.classes.length >= 2,
    3: state.points.length >= 2 && Object.keys(byClass).length >= 2,
    4: state.classifiedOnce,
    5: ui.didClassify,
    6: ui.didCompare,
    7: !!state.imageryStyle,
  };
  stepCards().forEach((card) => card.classList.toggle("done", !!done[+card.dataset.step]));
}

function snapshot() {
  return {
    center: state.center, classes: state.classes, points: state.points,
    trainingYear: state.trainingYear, targetYear: state.targetYear,
    compareA: state.compareA, compareB: state.compareB,
    trainedYear: state.trainedYear,
    classifier: state.classifier, featureSource: state.featureSource,
    imageryStyle: state.imageryStyle, imageryYear: state.imageryYear,
    opacity: state.opacity,
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
  document.querySelectorAll(".opacity-input").forEach((o) => { o.value = Math.round(state.opacity * 100); });
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
  state.featureSource = sourceById(p.featureSource) ? p.featureSource : state.config.default_feature_source;
  state.imageryStyle = p.imageryStyle || "";
  state.imageryYear = p.imageryYear ?? state.config.default_year;
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
      document.getElementById("feature-compare").hidden = true;
      document.getElementById("classifier").value = state.classifier;
      renderClasses();
      renderPoints();
      if (state.center) goToArea(state.center.lat, state.center.lon, false);
      else showImagery();
      syncFeatureUI();
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
