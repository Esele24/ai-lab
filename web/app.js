/* AI Lab — shared frontend helpers. No framework, no build step. */

const NAV = [
  ["/", "Dashboard"], ["/doc", "01 Documents"], ["/data", "02 Data"],
  ["/model", "03 Model"], ["/prompt", "05 Prompts"], ["/industry", "06 Industry"],
  ["/voice", "07 Voice"],
];

/* --- theme: explicit choice wins, otherwise follow the OS --- */
function initTheme() {
  const stored = localStorage.getItem("ai-lab-theme");
  if (stored) document.documentElement.dataset.theme = stored;
  const button = document.getElementById("theme-toggle");
  if (!button) return;
  const paint = () => {
    const dark = document.documentElement.dataset.theme === "dark"
      || (!document.documentElement.dataset.theme
          && matchMedia("(prefers-color-scheme: dark)").matches);
    button.textContent = dark ? "☀︎ Light" : "☾ Dark";
    button.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  };
  button.onclick = () => {
    const dark = document.documentElement.dataset.theme === "dark"
      || (!document.documentElement.dataset.theme
          && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("ai-lab-theme", next);
    paint();
  };
  paint();
}

function buildNav() {
  const bar = document.querySelector(".topnav");
  if (!bar) return;
  const here = location.pathname;
  bar.innerHTML = NAV.map(([href, label]) =>
    `<a href="${href}"${href === here ? ' aria-current="page"' : ""}>${label}</a>`
  ).join("") + '<button id="theme-toggle" class="ghost small" style="margin-left:8px"></button>';
  initTheme();
}

/* --- API --- */
async function api(path, payload) {
  const options = payload === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`Server returned ${response.status} with a non-JSON body.`); }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

/* --- small DOM helpers --- */
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g,
  (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));

function show(target, html, kind) {
  const element = typeof target === "string" ? $(target) : target;
  element.innerHTML = kind ? `<div class="notice ${kind}">${html}</div>` : html;
}

function busy(button, on, label) {
  if (!button) return;
  button.disabled = on;
  if (on) {
    button.dataset.label = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span> ${label || "Working…"}`;
  } else if (button.dataset.label) {
    button.innerHTML = button.dataset.label;
  }
}

/* Read a File into base64. Uploads go as JSON because Python 3.13 dropped the
   `cgi` module that parsed multipart/form-data. */
function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function dropzone(element, onFiles) {
  const input = element.querySelector("input[type=file]");
  element.onclick = () => input.click();
  input.onchange = () => { if (input.files.length) onFiles([...input.files]); };
  ["dragenter", "dragover"].forEach((type) => element.addEventListener(type, (event) => {
    event.preventDefault(); element.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((type) => element.addEventListener(type, (event) => {
    event.preventDefault(); element.classList.remove("over");
  }));
  element.addEventListener("drop", (event) => {
    const files = [...(event.dataTransfer?.files || [])];
    if (files.length) onFiles(files);
  });
}

/* Minimal markdown: bold, italics, inline code, [1] citations, paragraphs.
   Escapes first, so model output can never inject HTML. */
function md(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(?<!\w)\*([^*\n]+)\*(?!\w)/g, "<i>$1</i>")
    .replace(/\[(\d+)\]/g, '<b style="color:var(--accent)">[$1]</b>')
    .split(/\n{2,}/).map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`).join("");
}

function barChart(rows, labelKey, valueKey) {
  const values = rows.map((row) => Number(row[valueKey]) || 0);
  const max = Math.max(...values.map(Math.abs), 1);
  return `<div class="bars">${rows.map((row) => {
    const value = Number(row[valueKey]) || 0;
    return `<div class="bar-row">
      <span class="muted" title="${escapeHtml(row[labelKey])}">${escapeHtml(String(row[labelKey]).slice(0, 22))}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(Math.abs(value) / max) * 100}%"></span></span>
      <b>${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</b>
    </div>`;
  }).join("")}</div>`;
}

function lineChart(rows, labelKey, valueKey) {
  const values = rows.map((row) => Number(row[valueKey]) || 0);
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  return `<div class="spark">${rows.map((row) => {
    const value = Number(row[valueKey]) || 0;
    const height = 6 + ((value - min) / span) * 94;
    return `<div style="height:${height}%" data-label="${escapeHtml(row[labelKey])}: ${value.toLocaleString()}"></div>`;
  }).join("")}</div>
  <div class="row tiny muted" style="justify-content:space-between;margin-top:6px">
    <span>${escapeHtml(rows[0]?.[labelKey] ?? "")}</span>
    <span>${escapeHtml(rows.at(-1)?.[labelKey] ?? "")}</span>
  </div>`;
}

function table(rows) {
  if (!rows?.length) return '<p class="muted small">No rows.</p>';
  const columns = Object.keys(rows[0]);
  return `<div class="scroll-x"><table><thead><tr>${
    columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")
  }</tr></thead><tbody>${
    rows.map((row) => `<tr>${columns.map((column) => {
      const value = row[column];
      return `<td>${escapeHtml(typeof value === "number"
        ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : value)}</td>`;
    }).join("")}</tr>`).join("")
  }</tbody></table></div>`;
}

document.addEventListener("DOMContentLoaded", buildNav);
