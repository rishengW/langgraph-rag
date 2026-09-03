/**
 * Frontend logic for the only Subcribers chat app.
 *
 * Talks to the FastAPI backend in src/frontend/chat/api.py:
 *   POST /chat                        -> { thread_id, source_urls, source_mode }
 *   POST /chat/{tid}/message          -> { answer, error? }
 *   GET  /chat/{tid}/history          -> { turns, source_urls, source_mode }
 *   DELETE /chat/{tid}                -> { status }
 *
 * The thread_id is persisted in localStorage so a page refresh keeps the
 * conversation alive (state lives in the server's MemorySaver).
 */

const STORAGE_KEY = "onlysub.chat.thread_id";
const API = window.location.origin;

const AMAP_JS_API_VERSION = "2.0";
const AMAP_COORDINATE_SYSTEM = "gcj02";
const AMAP_PROVIDER = "amap";
const AMAP_FALLBACK_ORIGIN = "https://uri.amap.com";
const MAX_ARTIFACTS_PER_TURN = 4;
const MAX_MARKERS_PER_ARTIFACT = 12;
const MAX_POLYLINE_POINTS = 500;
const DEFAULT_MARKER_ZOOM = 13;
const MAX_DOWNLOAD_FILENAME_CHARS = 200;
const MAX_DOWNLOAD_SIZE_BYTES = 100_000_000;
const FILE_DOWNLOAD_TYPES = Object.freeze({
    ".docx": Object.freeze({
        mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        label: "Word document",
    }),
    ".txt": Object.freeze({ mimeType: "text/plain", label: "Text file" }),
    ".md": Object.freeze({ mimeType: "text/markdown", label: "Markdown file" }),
    ".xlsx": Object.freeze({
        mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        label: "Excel workbook",
    }),
    ".pptx": Object.freeze({
        mimeType: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        label: "PowerPoint presentation",
    }),
});

const startScreen = document.getElementById("startScreen");
const chatScreen = document.getElementById("chatScreen");
const sessionControls = document.getElementById("sessionControls");

const seedField = document.getElementById("seedQuestion");
const startBtn = document.getElementById("startBtn");
const startError = document.getElementById("startError");

const transcript = document.getElementById("transcript");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const sessionInfo = document.getElementById("sessionInfo");
const errorBanner = document.getElementById("errorBanner");
const newChatBtn = document.getElementById("newChatBtn");
const fileInput = document.getElementById("fileInput");
const attachBtn = document.getElementById("attachBtn");
const attachments = document.getElementById("attachments");

let threadId = null;
let pending = false;
let chatConfigCache = null;
let chatConfigPromise = null;
let amapLoadPromise = null;
const liveMapInstances = new Set();

// ---- helpers ------------------------------------------------------------

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

function safeHref(url) {
    const value = String(url || "").trim().replace(/&amp;/gi, "&");
    if (/^(https?:|mailto:)/i.test(value)) {
        return escapeHtml(value);
    }
    return "";
}

// Validates a same-origin download path. Returns the value unescaped: it is
// assigned via the DOM `.href` property, which does not take HTML entities, so
// escaping here would corrupt any "&" in the path. Safety comes from rejecting
// every scheme and protocol-relative form below, leaving only "/..." paths.
function safeDownloadHref(url) {
    const value = String(url || "").trim();
    if (!value) return "";
    if (value.startsWith("//")) return "";
    if (!value.startsWith("/")) return "";
    if (/^\w+:/.test(value)) return "";
    const colonIndex = value.indexOf(":");
    const slashIndex = value.indexOf("/");
    if (colonIndex !== -1 && (slashIndex === -1 || colonIndex < slashIndex)) {
        return "";
    }
    return value;
}

function renderInlineMarkdown(text) {
    const codeTokens = [];
    let value = String(text).replace(/`([^`]+)`/g, (_match, code) => {
        const token = `\u0000CODE${codeTokens.length}\u0000`;
        codeTokens.push(`<code>${escapeHtml(code)}</code>`);
        return token;
    });

    // Protect math spans from the escape/emphasis passes below so LaTeX such
    // as x_1, a * b, and \sum_{i=1} survives intact for KaTeX. The restored
    // value is HTML-escaped text, not raw model content, so delimiters remain
    // available to KaTeX without turning math payloads into executable markup.
    const mathTokens = [];
    value = value.replace(/(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\s)[^$\n]+?(?<!\s)\$)/g, (m) => {
        const token = `\u0000MATH${mathTokens.length}\u0000`;
        mathTokens.push(escapeHtml(m));
        return token;
    });

    // Protect Markdown links before escaping the surrounding text. Parsing
    // links after escape turns query separators into literal "&amp;" text in
    // the DOM href, so downstream services receive parameters like "amp;to".
    const linkTokens = [];
    value = value.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, url) => {
        const href = safeHref(url);
        if (!href) return label;
        const token = `\u0000LINK${linkTokens.length}\u0000`;
        linkTokens.push(
            `<a href="${href}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`,
        );
        return token;
    });

    value = escapeHtml(value);
    value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    value = value.replace(/_([^_]+)_/g, "<em>$1</em>");

    for (const [index, html] of codeTokens.entries()) {
        value = value.replaceAll(`\u0000CODE${index}\u0000`, () => html);
    }
    for (const [index, mathText] of mathTokens.entries()) {
        value = value.replaceAll(`\u0000MATH${index}\u0000`, () => mathText);
    }
    for (const [index, html] of linkTokens.entries()) {
        value = value.replaceAll(`\u0000LINK${index}\u0000`, () => html);
    }
    return value;
}

// Typeset any LaTeX math inside an already-rendered element. Called only at
// final render (not per streaming token) to avoid main-thread blocking and
// incomplete-LaTeX flicker during streaming. No-op if KaTeX is unavailable.
function renderMath(target) {
    if (!target || typeof window.renderMathInElement !== "function") return;
    try {
        window.renderMathInElement(target, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "\\[", right: "\\]", display: true },
                { left: "\\(", right: "\\)", display: false },
                { left: "$", right: "$", display: false },
            ],
            throwOnError: false,
        });
    } catch (_) {
        // Leave math as raw text if KaTeX fails on a given expression.
    }
}

function renderTableRow(line) {
    // Split on pipes, dropping the leading/trailing empty cells that result
    // from the conventional surrounding pipes.
    let cells = line.trim().split("|");
    if (cells.length && cells[0].trim() === "") cells = cells.slice(1);
    if (cells.length && cells[cells.length - 1].trim() === "") cells = cells.slice(0, -1);
    return cells.map((cell) => cell.trim());
}

function isTableDivider(line) {
    // A divider row looks like: | --- | :--: | ---: |
    const cells = renderTableRow(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{1,}:?$/.test(cell));
}

function renderMarkdownBlocks(text) {
    const lines = String(text).replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let paragraph = [];
    let listType = null;

    function closeParagraph() {
        if (!paragraph.length) return;
        html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
        paragraph = [];
    }

    function closeList() {
        if (!listType) return;
        html.push(`</${listType}>`);
        listType = null;
    }

    for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) {
            closeParagraph();
            closeList();
            continue;
        }

        // Markdown table: a header row followed by a divider row, then body
        // rows. Detected on the header line by peeking at the next line.
        if (
            trimmed.includes("|") &&
            i + 1 < lines.length &&
            isTableDivider(lines[i + 1])
        ) {
            closeParagraph();
            closeList();
            const headers = renderTableRow(trimmed);
            const bodyRows = [];
            let j = i + 2;
            for (; j < lines.length; j += 1) {
                const bodyLine = lines[j].trim();
                if (!bodyLine || !bodyLine.includes("|")) break;
                bodyRows.push(renderTableRow(lines[j]));
            }
            const head = headers
                .map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`)
                .join("");
            const body = bodyRows
                .map(
                    (row) =>
                        `<tr>${row
                            .map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`)
                            .join("")}</tr>`,
                )
                .join("");
            html.push(
                `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`,
            );
            i = j - 1;
            continue;
        }

        const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
        if (heading) {
            closeParagraph();
            closeList();
            const level = heading[1].length;
            html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
            continue;
        }

        const unordered = trimmed.match(/^[-*]\s+(.+)$/);
        const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
        if (unordered || ordered) {
            closeParagraph();
            const nextType = unordered ? "ul" : "ol";
            if (listType !== nextType) {
                closeList();
                html.push(`<${nextType}>`);
                listType = nextType;
            }
            html.push(`<li>${renderInlineMarkdown((unordered || ordered)[1])}</li>`);
            continue;
        }

        const quote = trimmed.match(/^>\s?(.+)$/);
        if (quote) {
            closeParagraph();
            closeList();
            html.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
            continue;
        }

        closeList();
        paragraph.push(trimmed);
    }

    closeParagraph();
    closeList();
    return html.join("");
}

function renderMarkdownPreview(text) {
    const parts = String(text).replace(/\r\n/g, "\n").split(/```/);
    return parts.map((part, index) => {
        if (index % 2 === 0) {
            return renderMarkdownBlocks(part);
        }

        const lines = part.split("\n");
        if (/^[A-Za-z0-9_+.-]+$/.test(lines[0].trim())) {
            lines.shift();
        }
        return `<pre><code>${escapeHtml(lines.join("\n").trim())}</code></pre>`;
    }).join("");
}

function renderFinalAssistantBubble(bubble, content) {
    bubble.classList.add("markdown-preview");
    bubble.innerHTML = renderMarkdownPreview(content);
    renderMath(bubble);
}

function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove("hidden");
}

function clearError() {
    errorBanner.textContent = "";
    errorBanner.classList.add("hidden");
}

function appendTurn(role, content, opts = {}) {
    const div = document.createElement("div");
    div.className = `turn ${role}` + (opts.thinking ? " thinking" : "");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "assistant" && !opts.thinking && !opts.plain) {
        renderFinalAssistantBubble(bubble, content);
    } else {
        bubble.textContent = String(content);
    }
    if (role === "assistant" && opts.thinking) {
        bubble.setAttribute("role", "status");
        bubble.setAttribute("aria-live", "polite");
    }
    div.appendChild(bubble);
    if (role === "assistant" && !opts.thinking) {
        renderArtifactStack(div, opts.artifacts);
    }
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
    return div;
}

function destroyMapInstances() {
    for (const map of liveMapInstances) {
        try {
            if (map && typeof map.destroy === "function") {
                map.destroy();
            }
        } catch (_) {
            // Best-effort teardown; the transcript is being removed anyway.
        }
    }
    liveMapInstances.clear();
}

function clearTranscript() {
    destroyMapInstances();
    transcript.innerHTML = "";
}

function renderSessionInfo(sourceUrls, sourceMode) {
    const labels = {
        explicit: "Explicit URLs",
        web_search: "Web search",
        defaults: "Configured defaults",
    };
    const label = labels[sourceMode] || sourceMode;
    const count = (sourceUrls || []).length;
    sessionInfo.textContent = `Sources: ${label} · ${count} URL${count === 1 ? "" : "s"}`;
}

function showStart() {
    startScreen.classList.remove("hidden");
    chatScreen.classList.add("hidden");
    sessionControls.classList.add("hidden");
}

function showChat() {
    startScreen.classList.add("hidden");
    chatScreen.classList.remove("hidden");
    sessionControls.classList.remove("hidden");
    messageInput.focus();
}

// ---- AMap artifacts ------------------------------------------------------

function normalizeChatConfig(data) {
    const amap = data && typeof data === "object" && data.amap && typeof data.amap === "object"
        ? data.amap
        : {};
    return {
        amap: {
            enabled: amap.enabled === true,
            js_api_key: typeof amap.js_api_key === "string" ? amap.js_api_key.trim() : "",
            service_host: typeof amap.service_host === "string" ? amap.service_host.trim() : "",
            api_version: String(amap.api_version || AMAP_JS_API_VERSION),
            coordinate_system: String(amap.coordinate_system || ""),
        },
    };
}

async function getChatConfig() {
    if (chatConfigCache) return chatConfigCache;
    if (!chatConfigPromise) {
        chatConfigPromise = apiGet("/chat/config")
            .then((data) => {
                chatConfigCache = normalizeChatConfig(data);
                return chatConfigCache;
            })
            .catch((err) => {
                chatConfigPromise = null;
                throw err;
            });
    }
    return chatConfigPromise;
}

function absoluteSameOriginServiceHost(serviceHost) {
    const value = String(serviceHost || "").trim();
    if (!value) return "";
    try {
        const url = new URL(value, API);
        if (url.origin !== API) return "";
        return url.href;
    } catch (_) {
        return "";
    }
}

function applyAMapSecurityConfig(serviceHost) {
    const configured = String(serviceHost || "").trim();
    if (!configured) return;

    const absoluteServiceHost = absoluteSameOriginServiceHost(configured);
    if (!absoluteServiceHost) {
        throw new Error("Invalid AMap service host");
    }

    window._AMapSecurityConfig = Object.assign({}, window._AMapSecurityConfig || {}, {
        serviceHost: absoluteServiceHost,
    });
}

function injectAMapScript(jsApiKey, apiVersion) {
    return new Promise((resolve, reject) => {
        const previous = document.getElementById("amap-js-api");
        if (previous) previous.remove();

        const src = new URL("https://webapi.amap.com/maps");
        src.searchParams.set("v", apiVersion);
        src.searchParams.set("key", jsApiKey);

        const script = document.createElement("script");
        script.id = "amap-js-api";
        script.async = true;
        script.defer = true;
        script.src = src.toString();
        script.onload = () => {
            if (window.AMap && typeof window.AMap.Map === "function") {
                resolve(window.AMap);
            } else {
                reject(new Error("AMap failed to initialize"));
            }
        };
        script.onerror = () => {
            script.remove();
            reject(new Error("AMap failed to load"));
        };
        document.head.appendChild(script);
    });
}

function loadAMap() {
    if (window.AMap && typeof window.AMap.Map === "function") {
        return Promise.resolve(window.AMap);
    }
    if (amapLoadPromise) return amapLoadPromise;

    amapLoadPromise = getChatConfig()
        .then((config) => {
            const amap = config.amap || {};
            if (amap.enabled !== true) {
                throw new Error("AMap is disabled");
            }
            if (!amap.js_api_key) {
                throw new Error("AMap JS API key is not configured");
            }
            if (amap.api_version !== AMAP_JS_API_VERSION) {
                throw new Error("Unsupported AMap JS API version");
            }
            if (amap.coordinate_system !== AMAP_COORDINATE_SYSTEM) {
                throw new Error("Unsupported AMap coordinate system");
            }

            applyAMapSecurityConfig(amap.service_host);
            return injectAMapScript(amap.js_api_key, amap.api_version);
        })
        .catch((err) => {
            amapLoadPromise = null;
            throw err;
        });

    return amapLoadPromise;
}

function isPlainObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizePosition(value) {
    if (!isPlainObject(value)) return null;
    const lng = Number(value.lng);
    const lat = Number(value.lat);
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) return null;
    if (lng < -180 || lng > 180 || lat < -90 || lat > 90) return null;
    return {
        lng: Math.round(lng * 1_000_000) / 1_000_000,
        lat: Math.round(lat * 1_000_000) / 1_000_000,
    };
}

function boundedLabel(value, fallback) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    return (text || fallback).slice(0, 80);
}

function normalizeZoom(value) {
    const zoom = Number(value);
    if (!Number.isFinite(zoom)) return DEFAULT_MARKER_ZOOM;
    return Math.max(3, Math.min(20, Math.round(zoom)));
}

function normalizeNonNegativeNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0) return null;
    return Math.min(number, 1_000_000_000);
}

function safeAmapFallbackUrl(value) {
    try {
        const url = new URL(String(value || "").trim());
        if (
            url.origin === AMAP_FALLBACK_ORIGIN &&
            !url.username &&
            !url.password
        ) {
            return url.href;
        }
    } catch (_) {
        // Fall through to the safe AMap landing page.
    }
    return `${AMAP_FALLBACK_ORIGIN}/`;
}

function normalizeMarker(raw, index) {
    if (!isPlainObject(raw)) return null;
    const position = normalizePosition(isPlainObject(raw.position) ? raw.position : raw);
    if (!position) return null;
    return {
        position,
        label: boundedLabel(raw.label || raw.title || raw.name, `Marker ${index + 1}`),
    };
}

function normalizeMarkerArtifact(raw, fallbackUrl) {
    const markers = [];
    const rawMarkers = Array.isArray(raw.markers) ? raw.markers : [];
    for (const marker of rawMarkers) {
        if (markers.length >= MAX_MARKERS_PER_ARTIFACT) break;
        const normalized = normalizeMarker(marker, markers.length);
        if (normalized) markers.push(normalized);
    }

    const center = normalizePosition(raw.center) || (markers[0] ? markers[0].position : null);
    if (!center) return null;
    if (!markers.length) {
        markers.push({ position: center, label: "Marker" });
    }

    return {
        type: "amap",
        version: 1,
        kind: "marker",
        coordinateSystem: AMAP_COORDINATE_SYSTEM,
        fallbackUrl,
        center,
        markers,
        label: boundedLabel(raw.label || raw.title || markers[0].label, "Map marker"),
        zoom: normalizeZoom(raw.zoom),
    };
}

function normalizeEndpointPosition(value) {
    if (!isPlainObject(value)) return null;
    return normalizePosition(isPlainObject(value.position) ? value.position : value);
}

function normalizeRouteArtifact(raw, fallbackUrl) {
    const polyline = [];
    const rawPolyline = Array.isArray(raw.polyline) ? raw.polyline : [];
    for (const point of rawPolyline) {
        if (polyline.length >= MAX_POLYLINE_POINTS) break;
        const normalized = normalizePosition(point);
        if (normalized) polyline.push(normalized);
    }

    const rawMarkers = Array.isArray(raw.markers) ? raw.markers : [];
    const boundedMarkers = rawMarkers.slice(0, MAX_MARKERS_PER_ARTIFACT);
    const originMarker = boundedMarkers.find((marker) => marker && marker.role === "origin")
        || rawMarkers[0];
    const destinationMarker = boundedMarkers.find(
        (marker) => marker && marker.role === "destination",
    ) || rawMarkers[rawMarkers.length - 1];
    const rawPositions = Array.isArray(raw.positions) ? raw.positions : [];

    const origin = normalizeEndpointPosition(raw.origin)
        || normalizeEndpointPosition(originMarker)
        || normalizePosition(rawPositions[0])
        || polyline[0]
        || null;
    const destination = normalizeEndpointPosition(raw.destination)
        || normalizeEndpointPosition(destinationMarker)
        || normalizePosition(rawPositions[rawPositions.length - 1])
        || polyline[polyline.length - 1]
        || null;
    if (!origin || !destination) return null;

    const originLabel = boundedLabel(
        raw.originLabel || (isPlainObject(originMarker) && (
            originMarker.label || originMarker.title || originMarker.name
        )),
        "Origin",
    );
    const destinationLabel = boundedLabel(
        raw.destinationLabel || (isPlainObject(destinationMarker) && (
            destinationMarker.label || destinationMarker.title || destinationMarker.name
        )),
        "Destination",
    );

    return {
        type: "amap",
        version: 1,
        kind: "route",
        coordinateSystem: AMAP_COORDINATE_SYSTEM,
        fallbackUrl,
        origin,
        destination,
        polyline: polyline.length >= 2 ? polyline : [],
        distanceMeters: normalizeNonNegativeNumber(raw.distanceMeters),
        durationSeconds: normalizeNonNegativeNumber(raw.durationSeconds),
        label: boundedLabel(raw.label || raw.title, `${originLabel} to ${destinationLabel}`),
        originLabel,
        destinationLabel,
    };
}

function normalizeAMapArtifact(raw) {
    if (!isPlainObject(raw)) return null;
    if (raw.type !== "amap" || raw.version !== 1) return null;
    if (raw.provider !== AMAP_PROVIDER) return null;
    if (raw.coordinateSystem !== AMAP_COORDINATE_SYSTEM) return null;

    const fallbackUrl = safeAmapFallbackUrl(raw.fallbackUrl);
    if (raw.kind === "marker") {
        return normalizeMarkerArtifact(raw, fallbackUrl);
    }
    if (raw.kind === "route") {
        return normalizeRouteArtifact(raw, fallbackUrl);
    }
    return null;
}

function normalizeFileArtifact(raw) {
    if (!isPlainObject(raw)) return null;
    if (raw.type !== "file" || raw.version !== 1) return null;
    if (raw.kind !== "download") return null;
    if (raw.provider !== "chat_upload") return null;

    if (typeof raw.threadId !== "string" || !/^[A-Za-z0-9._-]{1,64}$/.test(raw.threadId)) {
        return null;
    }

    if (typeof raw.filename !== "string") return null;
    const filename = raw.filename.trim();
    if (filename !== raw.filename || !filename || filename.length > MAX_DOWNLOAD_FILENAME_CHARS) {
        return null;
    }
    if (filename === "." || filename === ".." || /[\\/\u0000]/.test(filename)) return null;

    const suffixIndex = filename.lastIndexOf(".");
    const suffix = suffixIndex >= 0 ? filename.slice(suffixIndex).toLowerCase() : "";
    const fileType = FILE_DOWNLOAD_TYPES[suffix];
    if (!fileType || raw.mimeType !== fileType.mimeType) return null;

    const sizeBytes = raw.sizeBytes;
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes < 0 || sizeBytes > MAX_DOWNLOAD_SIZE_BYTES) {
        return null;
    }

    const url = safeDownloadHref(raw.url);
    const expectedUrl = `/chat/${encodeURIComponent(raw.threadId)}/files/${encodeURIComponent(filename)}`;
    if (!url || url !== expectedUrl) return null;

    return {
        type: "file",
        version: 1,
        kind: "download",
        provider: "chat_upload",
        threadId: raw.threadId,
        filename,
        mimeType: fileType.mimeType,
        label: fileType.label,
        sizeBytes,
        url,
    };
}

function positionKey(position) {
    return [position.lng, position.lat];
}

function artifactDedupeKey(artifact) {
    if (artifact.type === "file") {
        return JSON.stringify(["file", 1, "download", artifact.url, artifact.sizeBytes]);
    }
    if (artifact.kind === "marker") {
        return JSON.stringify([
            "amap",
            1,
            "marker",
            artifact.fallbackUrl,
            positionKey(artifact.center),
            artifact.zoom,
            artifact.markers.map((marker) => [positionKey(marker.position), marker.label]),
        ]);
    }
    return JSON.stringify([
        "amap",
        1,
        "route",
        artifact.fallbackUrl,
        positionKey(artifact.origin),
        positionKey(artifact.destination),
        artifact.polyline.map(positionKey),
        artifact.distanceMeters,
        artifact.durationSeconds,
    ]);
}

function normalizeArtifacts(input) {
    const rawArtifacts = Array.isArray(input) ? input : (input ? [input] : []);
    const artifacts = new Map();
    for (const rawArtifact of rawArtifacts) {
        if (artifacts.size >= MAX_ARTIFACTS_PER_TURN) break;
        const artifact = normalizeAMapArtifact(rawArtifact) || normalizeFileArtifact(rawArtifact);
        if (!artifact) continue;
        const key = artifactDedupeKey(artifact);
        if (!artifacts.has(key)) {
            artifacts.set(key, artifact);
        }
    }
    return Array.from(artifacts.values());
}

function artifactsFromPayload(payload) {
    if (!isPlainObject(payload)) return [];
    const rawArtifacts = [];
    if (Array.isArray(payload.artifacts)) {
        rawArtifacts.push(...payload.artifacts);
    } else if (isPlainObject(payload.artifacts)) {
        rawArtifacts.push(payload.artifacts);
    }
    if (Array.isArray(payload.artifact)) {
        rawArtifacts.push(...payload.artifact);
    } else if (isPlainObject(payload.artifact)) {
        rawArtifacts.push(payload.artifact);
    }
    return rawArtifacts;
}

function collectArtifactsFromPayload(payload, artifactMap) {
    for (const artifact of normalizeArtifacts(artifactsFromPayload(payload))) {
        if (artifactMap.size >= MAX_ARTIFACTS_PER_TURN) return;
        const key = artifactDedupeKey(artifact);
        if (!artifactMap.has(key)) {
            artifactMap.set(key, artifact);
        }
    }
}

function toAMapPosition(position) {
    return [position.lng, position.lat];
}

function formatPosition(position) {
    return `${position.lng.toFixed(6)}, ${position.lat.toFixed(6)}`;
}

function formatDistance(meters) {
    if (meters === null) return "";
    if (meters >= 1000) {
        const digits = meters >= 10_000 ? 0 : 1;
        return `${(meters / 1000).toFixed(digits)} km`;
    }
    return `${Math.round(meters)} m`;
}

function formatDuration(seconds) {
    if (seconds === null) return "";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return remainingMinutes ? `${hours} hr ${remainingMinutes} min` : `${hours} hr`;
}

function setMapStatus(mapEl, message) {
    mapEl.textContent = "";
    const status = document.createElement("span");
    status.className = "amap-card__status";
    status.textContent = message;
    mapEl.appendChild(status);
}

function appendInfoChip(container, text) {
    const chip = document.createElement("span");
    chip.className = "amap-card__chip";
    chip.textContent = text;
    container.appendChild(chip);
}

function addMarkerDetails(card, artifact) {
    const labels = document.createElement("div");
    labels.className = "amap-card__labels";
    appendInfoChip(labels, `Center: ${formatPosition(artifact.center)}`);
    for (const marker of artifact.markers) {
        appendInfoChip(labels, `${marker.label}: ${formatPosition(marker.position)}`);
    }
    card.appendChild(labels);
}

function addRouteDetails(card, artifact) {
    const labels = document.createElement("div");
    labels.className = "amap-card__labels";
    appendInfoChip(labels, `${artifact.originLabel}: ${formatPosition(artifact.origin)}`);
    appendInfoChip(
        labels,
        `${artifact.destinationLabel}: ${formatPosition(artifact.destination)}`,
    );

    const distance = formatDistance(artifact.distanceMeters);
    const duration = formatDuration(artifact.durationSeconds);
    if (distance) appendInfoChip(labels, `Distance: ${distance}`);
    if (duration) appendInfoChip(labels, `Duration: ${duration}`);
    card.appendChild(labels);
}

function buildAMapArtifactCard(artifact) {
    const card = document.createElement("article");
    card.className = "amap-card";
    card.dataset.kind = artifact.kind;

    const header = document.createElement("div");
    header.className = "amap-card__header";

    const title = document.createElement("div");
    title.className = "amap-card__title";
    title.textContent = artifact.label;
    header.appendChild(title);

    const fallback = document.createElement("a");
    fallback.className = "amap-card__fallback";
    fallback.href = artifact.fallbackUrl;
    fallback.target = "_blank";
    fallback.rel = "noopener noreferrer";
    fallback.textContent = "Open in AMap";
    header.appendChild(fallback);

    card.appendChild(header);

    const mapEl = document.createElement("div");
    mapEl.className = "amap-card__map";
    setMapStatus(mapEl, "Map preview loading...");
    card.appendChild(mapEl);

    if (artifact.kind === "route") {
        addRouteDetails(card, artifact);
    } else {
        addMarkerDetails(card, artifact);
    }

    return { card, mapEl };
}

function fitMapView(map, overlays) {
    if (!overlays.length || typeof map.setFitView !== "function") return;
    try {
        map.setFitView(overlays, false, [40, 40, 40, 40], 17);
    } catch (_) {
        // Leave the map at its initial center/zoom if fit view is unavailable.
    }
}

function renderMarkerAMap(AMap, map, artifact) {
    const overlays = [];
    for (const marker of artifact.markers) {
        const overlay = new AMap.Marker({
            map,
            position: toAMapPosition(marker.position),
            title: marker.label,
        });
        overlays.push(overlay);
    }
    if (artifact.markers.length > 1) {
        fitMapView(map, overlays);
    } else {
        map.setCenter(toAMapPosition(artifact.center));
        map.setZoom(artifact.zoom);
    }
}

function renderRouteAMap(AMap, map, artifact) {
    const originMarker = new AMap.Marker({
        map,
        position: toAMapPosition(artifact.origin),
        title: artifact.originLabel,
    });
    const destinationMarker = new AMap.Marker({
        map,
        position: toAMapPosition(artifact.destination),
        title: artifact.destinationLabel,
    });
    const overlays = [originMarker, destinationMarker];
    if (artifact.polyline.length >= 2) {
        overlays.push(new AMap.Polyline({
            map,
            path: artifact.polyline.map(toAMapPosition),
            strokeColor: "#4f46e5",
            strokeOpacity: 0.85,
            strokeWeight: 6,
            lineJoin: "round",
            showDir: true,
        }));
    }
    fitMapView(map, overlays);
}

async function hydrateAMapCard(card, mapEl, artifact) {
    let map = null;
    try {
        const AMap = await loadAMap();
        if (!document.body.contains(card)) return;

        mapEl.textContent = "";
        const center = artifact.kind === "route" ? artifact.origin : artifact.center;
        map = new AMap.Map(mapEl, {
            center: toAMapPosition(center),
            zoom: artifact.kind === "marker" ? artifact.zoom : DEFAULT_MARKER_ZOOM,
            resizeEnable: true,
        });
        liveMapInstances.add(map);

        if (artifact.kind === "route") {
            renderRouteAMap(AMap, map, artifact);
        } else {
            renderMarkerAMap(AMap, map, artifact);
        }

        window.setTimeout(() => {
            try {
                if (document.body.contains(card) && map && typeof map.resize === "function") {
                    map.resize();
                }
            } catch (_) {
                // Non-fatal resize issue.
            }
        }, 0);
    } catch (_) {
        if (map) {
            liveMapInstances.delete(map);
            try {
                if (typeof map.destroy === "function") map.destroy();
            } catch (__) {
                // Ignore teardown failures after a render error.
            }
        }
        if (document.body.contains(card)) {
            setMapStatus(mapEl, "Map preview unavailable. Use the AMap link.");
        }
    }
}

function formatFileSize(sizeBytes) {
    if (sizeBytes < 1024) return `${sizeBytes} B`;
    if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(1)} KB`;
    return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function buildFileArtifactCard(artifact) {
    const card = document.createElement("article");
    card.className = "file-card";
    card.dataset.kind = artifact.kind;

    // The card is a flex row: this body column stacks the name over the meta
    // line, leaving the download link to sit at the far end.
    const body = document.createElement("div");
    body.className = "file-card__body";

    const name = document.createElement("div");
    name.className = "file-card__name";
    name.textContent = artifact.filename;
    name.title = artifact.filename;
    body.appendChild(name);

    const meta = document.createElement("div");
    meta.className = "file-card__meta";
    meta.textContent = `${artifact.label} - ${formatFileSize(artifact.sizeBytes)}`;
    body.appendChild(meta);

    card.appendChild(body);

    const link = document.createElement("a");
    link.className = "file-card__download";
    // Assigning to .href (not innerHTML) so the validated path is used as-is.
    link.href = artifact.url;
    link.download = artifact.filename;
    link.rel = "noopener noreferrer";
    link.textContent = "Download";
    card.appendChild(link);

    return card;
}

function renderArtifactStack(turn, artifactsInput) {
    const artifacts = normalizeArtifacts(artifactsInput);
    if (!turn || !artifacts.length) return;

    const stack = document.createElement("div");
    stack.className = "artifact-stack";
    const cards = [];
    for (const artifact of artifacts) {
        // File artifacts are static download cards; only AMap cards need a
        // map instance hydrated after they are attached to the document.
        if (artifact.type === "file") {
            stack.appendChild(buildFileArtifactCard(artifact));
            continue;
        }
        const { card, mapEl } = buildAMapArtifactCard(artifact);
        stack.appendChild(card);
        cards.push({ card, mapEl, artifact });
    }

    turn.classList.add("has-artifacts");
    turn.appendChild(stack);
    for (const { card, mapEl, artifact } of cards) {
        hydrateAMapCard(card, mapEl, artifact);
    }
}

function findSseSeparator(buffer) {
    const lf = buffer.indexOf("\n\n");
    const crlf = buffer.indexOf("\r\n\r\n");
    if (lf === -1 && crlf === -1) return null;
    if (lf === -1) return { index: crlf, length: 4 };
    if (crlf === -1) return { index: lf, length: 2 };
    return lf < crlf ? { index: lf, length: 2 } : { index: crlf, length: 4 };
}

// ---- API calls ----------------------------------------------------------

function errorMessage(data, res) {
    const detail = data.detail || data.error;
    if (typeof detail === "string") return detail || `HTTP ${res.status}`;
    if (detail != null) return JSON.stringify(detail);
    return `HTTP ${res.status}`;
}

async function apiPost(path, body) {
    const res = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorMessage(data, res));
    return data;
}

async function apiGet(path) {
    const res = await fetch(`${API}${path}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorMessage(data, res));
    return data;
}

async function apiDelete(path) {
    const res = await fetch(`${API}${path}`, { method: "DELETE" });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(errorMessage(data, res));
    }
}

// ---- start a new session ------------------------------------------------

startBtn.addEventListener("click", async () => {
    const seed = seedField.value.trim();

    startError.textContent = "";
    startBtn.disabled = true;
    startBtn.textContent = "Starting chat...";

    try {
        const data = await apiPost("/chat", {
            seed_question: seed || null,
        });
        threadId = data.thread_id;
        localStorage.setItem(STORAGE_KEY, threadId);
        clearTranscript();
        clearError();
        renderSessionInfo(data.source_urls, data.source_mode);
        showChat();

        // If the user typed a topic, send it as the first message automatically.
        if (seed) {
            await sendMessage(seed);
        }
    } catch (err) {
        startError.textContent = `Could not start chat: ${err.message}`;
    } finally {
        startBtn.disabled = false;
        startBtn.textContent = "Start chat";
    }
});

// ---- restore session on reload -----------------------------------------

async function restoreSession() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    try {
        const history = await apiGet(`/chat/${stored}/history`);
        threadId = stored;
        clearTranscript();
        renderSessionInfo(history.source_urls, history.source_mode);
        for (const turn of history.turns || []) {
            appendTurn(turn.role, turn.content, { artifacts: turn.artifacts });
        }
        showChat();
    } catch (err) {
        // Session no longer exists on the server; fall back to the start screen.
        localStorage.removeItem(STORAGE_KEY);
    }
}

// ---- send a message ----------------------------------------------------

const TOOL_STATUS_MIN_MS = 1200;

const TOOL_ACTION_LABELS = Object.freeze({
    retrieve_source_documents: "Searching source documents",
    live_web_search: "Searching the web",
    search_wikipedia: "Searching Wikipedia",
    get_weather: "Checking the weather",
    get_stock_quote: "Checking the stock market",
    convert_currency: "Converting currency",
    get_directions: "Finding directions",
    find_on_map: "Finding the location",
    solve_math: "Solving the calculation",
    compute_statistics: "Calculating statistics",
    linear_algebra: "Solving the linear algebra",
    number_theory: "Solving the number theory problem",
    calculate_datetime: "Calculating date and time",
    summarize_url: "Summarizing the web page",
    save_memory: "Saving to memory",
    recall_memory: "Recalling memory",
    forget_memory: "Updating memory",
});

function toolActionLabel(toolName) {
    const name = String(toolName || "");
    if (TOOL_ACTION_LABELS[name]) return TOOL_ACTION_LABELS[name];
    if (name.startsWith("read_") || name.startsWith("inspect_")) {
        return "Parsing the file(s)";
    }
    if (name.startsWith("edit_")) return "Editing the file(s)";
    if (name.startsWith("create_")) return "Creating file(s)";
    const readableName = name.replaceAll("_", " ").trim();
    return readableName ? `Using ${readableName}` : "Using a tool";
}

const stopBtn = document.getElementById("stopBtn");
let activeTurnController = null;

async function sendMessage(text) {
    if (!threadId || pending) return;
    const trimmed = text.trim();
    if (!trimmed) return;

    const controller = new AbortController();
    activeTurnController = controller;
    pending = true;
    sendBtn.disabled = true;
    stopBtn.disabled = false;
    stopBtn.classList.remove("hidden");
    appendTurn("user", trimmed);
    messageInput.value = "";
    autoresize();

    const thinking = appendTurn("assistant", "Thinking...", { thinking: true });

    try {
        const result = await streamMessage(trimmed, thinking, controller.signal);
        if (!result.stopped) {
            try {
                const history = await apiGet(`/chat/${threadId}/history`);
                renderSessionInfo(history.source_urls, history.source_mode);
            } catch (_) {
                // The answer is already complete; stale source metadata is non-fatal.
            }
        }
    } catch (err) {
        thinking.remove();
        showError(err.message);
        appendTurn("assistant", "(failed to fetch reply)");
    } finally {
        if (activeTurnController === controller) {
            activeTurnController = null;
        }
        pending = false;
        sendBtn.disabled = false;
        stopBtn.disabled = true;
        stopBtn.classList.add("hidden");
        messageInput.focus();
    }
}

// Consume the SSE token stream from POST /chat/{tid}/message/stream.
// Renders streamed text as plain text while accumulating the final answer;
// Markdown, KaTeX, and map artifacts are rendered once the stream completes.
async function streamMessage(message, thinking, signal) {
    let bubble = null;
    let answer = "";
    let streamError = "";
    let toolStatusVisibleUntil = 0;
    let answerRenderTimer = null;
    const responseArtifacts = new Map();
    const activeTools = new Map();
    const statusBubble = thinking.querySelector(".bubble");

    const updateToolStatus = () => {
        if (bubble || !thinking.isConnected || !statusBubble) return;
        const activeNames = Array.from(activeTools.values());
        if (activeNames.length) {
            statusBubble.textContent = toolActionLabel(activeNames[activeNames.length - 1]);
        } else if (Date.now() >= toolStatusVisibleUntil) {
            statusBubble.textContent = "Thinking...";
        }
        transcript.scrollTop = transcript.scrollHeight;
    };

    const ensureBubble = () => {
        if (bubble) return bubble;
        thinking.remove();
        clearError();
        const turn = appendTurn("assistant", "", { plain: true });
        bubble = turn.querySelector(".bubble");
        return bubble;
    };

    const renderStreamedAnswer = () => {
        if (!answer) return;
        const remaining = toolStatusVisibleUntil - Date.now();
        if (!bubble && remaining > 0) {
            clearTimeout(answerRenderTimer);
            answerRenderTimer = setTimeout(renderStreamedAnswer, remaining);
            return;
        }
        answerRenderTimer = null;
        const el = ensureBubble();
        el.textContent = answer;
        transcript.scrollTop = transcript.scrollHeight;
    };

    const handleEvent = (eventType, dataStr) => {
        let payload = {};
        try {
            payload = dataStr ? JSON.parse(dataStr) : {};
        } catch (_) {
            return;
        }
        if (eventType === "tool_start" && typeof payload.tool === "string") {
            const toolCallId = String(payload.tool_call_id || payload.tool);
            activeTools.set(toolCallId, payload.tool);
            toolStatusVisibleUntil = Math.max(
                toolStatusVisibleUntil,
                Date.now() + TOOL_STATUS_MIN_MS,
            );
            updateToolStatus();
        } else if (eventType === "tool_end") {
            const toolCallId = String(payload.tool_call_id || payload.tool || "");
            activeTools.delete(toolCallId);
            updateToolStatus();
        } else if (eventType === "token" && typeof payload.token === "string") {
            answer += payload.token;
            renderStreamedAnswer();
        } else if (eventType === "artifact") {
            collectArtifactsFromPayload(payload, responseArtifacts);
        } else if (eventType === "error") {
            streamError = payload.message || "stream error";
        } else if (eventType === "done") {
            // Prefer accumulated tokens; otherwise use the final answer.
            if (!answer && typeof payload.answer === "string") {
                answer = payload.answer;
            }
            collectArtifactsFromPayload(payload, responseArtifacts);
        }
    };

    try {
        const res = await fetch(`${API}/chat/${threadId}/message/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
            signal,
        });
        if (!res.ok || !res.body) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.detail || data.error || `HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // SSE frames are separated by a blank line.
            let sep;
            while ((sep = findSseSeparator(buffer)) !== null) {
                const frame = buffer.slice(0, sep.index);
                buffer = buffer.slice(sep.index + sep.length);
                let eventType = "message";
                let dataStr = "";
                for (const rawLine of frame.split(/\r?\n/)) {
                    const line = rawLine.trimEnd();
                    if (line.startsWith("event:")) eventType = line.slice(6).trim();
                    else if (line.startsWith("data:")) dataStr += line.slice(5).trimStart();
                }
                handleEvent(eventType, dataStr);
            }
        }
    } catch (err) {
        clearTimeout(answerRenderTimer);
        answerRenderTimer = null;
        if (err.name !== "AbortError") throw err;

        thinking.remove();
        const artifacts = Array.from(responseArtifacts.values());
        clearError();
        if (!bubble) {
            appendTurn("assistant", "(stopped)", { artifacts });
        } else {
            renderFinalAssistantBubble(bubble, answer || "(stopped)");
            const turn = bubble.closest(".turn");
            renderArtifactStack(turn, artifacts);
            transcript.scrollTop = transcript.scrollHeight;
        }

        // Server-side stream cleanup continues independently. The next turn
        // is serialized behind that rollback, so the composer can be restored
        // immediately without reusing the cancelled checkpoint.
        return { stopped: true };
    }

    const remainingToolStatusMs = toolStatusVisibleUntil - Date.now();
    if (!bubble && remainingToolStatusMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, remainingToolStatusMs));
    }
    clearTimeout(answerRenderTimer);
    answerRenderTimer = null;
    thinking.remove();
    const artifacts = Array.from(responseArtifacts.values());
    if (streamError && !answer) {
        showError(streamError);
        appendTurn("assistant", "(no reply produced)", { artifacts });
        return { stopped: false };
    }
    if (!bubble) {
        // No tokens streamed; render the final answer in one shot.
        clearError();
        appendTurn("assistant", answer || "(no reply produced)", { artifacts });
    } else {
        renderFinalAssistantBubble(bubble, answer || "(no reply produced)");
        const turn = bubble.closest(".turn");
        renderArtifactStack(turn, artifacts);
        transcript.scrollTop = transcript.scrollHeight;
    }
    return { stopped: false };
}

stopBtn.addEventListener("click", () => {
    stopBtn.disabled = true;
    activeTurnController?.abort();
});
sendBtn.addEventListener("click", () => sendMessage(messageInput.value));
messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(messageInput.value);
    }
});

// ---- file uploads ------------------------------------------------------

const FILE_CHIP_TYPES = {
    csv: { letter: "C", cls: "csv", label: "CSV" },
    tsv: { letter: "C", cls: "csv", label: "TSV" },
    xlsx: { letter: "X", cls: "xlsx", label: "XLSX" },
    xls: { letter: "X", cls: "xlsx", label: "XLS" },
    docx: { letter: "W", cls: "docx", label: "DOCX" },
    doc: { letter: "W", cls: "docx", label: "DOC" },
    pptx: { letter: "P", cls: "pptx", label: "PPTX" },
    pdf: { letter: "P", cls: "pdf", label: "PDF" },
    txt: { letter: "T", cls: "txt", label: "TXT" },
    md: { letter: "M", cls: "txt", label: "MD" },
    log: { letter: "L", cls: "txt", label: "LOG" },
};

function fileChipType(filename) {
    const dot = filename.lastIndexOf(".");
    const ext = dot >= 0 ? filename.slice(dot + 1).toLowerCase() : "";
    return (
        FILE_CHIP_TYPES[ext] || {
            letter: (ext[0] || "F").toUpperCase(),
            cls: "txt",
            label: (ext || "file").toUpperCase(),
        }
    );
}

function formatFileSize(bytes) {
    if (bytes >= 1024 * 1024) {
        const mb = bytes / (1024 * 1024);
        return `${mb >= 10 ? Math.round(mb) : mb.toFixed(1)}MB`;
    }
    return `${Math.max(1, Math.round(bytes / 1024))}KB`;
}

function buildFileChip(file) {
    const type = fileChipType(file.filename);
    const chip = document.createElement("span");
    chip.className = "file-chip";
    const icon = document.createElement("span");
    icon.className = `file-chip__icon ${type.cls}`;
    icon.textContent = type.letter;
    const body = document.createElement("span");
    body.className = "file-chip__body";
    const name = document.createElement("span");
    name.className = "file-chip__name";
    name.textContent = file.filename;
    name.title = file.filename;
    const meta = document.createElement("span");
    meta.className = "file-chip__meta";
    meta.textContent = `${type.label} · ${formatFileSize(file.size_bytes)}`;
    body.append(name, meta);
    chip.append(icon, body);
    return chip;
}

function renderAttachmentChips(files, errors) {
    attachments.innerHTML = "";
    for (const f of files || []) {
        attachments.appendChild(buildFileChip(f));
    }
    for (const err of errors || []) {
        const chip = document.createElement("span");
        chip.className = "chip error";
        chip.textContent = err;
        attachments.appendChild(chip);
    }
    attachments.classList.toggle("hidden", attachments.childElementCount === 0);
}

async function uploadFiles(fileList) {
    if (!threadId || !fileList || !fileList.length) return;

    const form = new FormData();
    for (const file of fileList) {
        form.append("files", file, file.name);
    }

    attachBtn.disabled = true;
    try {
        const res = await fetch(`${API}/chat/${threadId}/upload`, {
            method: "POST",
            body: form,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(data.detail || data.error || `HTTP ${res.status}`);
        }
        renderAttachmentChips(data.files, data.errors);
        clearError();

        const names = (data.files || []).map((f) => f.filename);
        if (names.length) {
            // Let the user know the files are ready and how to use them.
            const list = names.join(", ");
            appendTurn(
                "assistant",
                `Uploaded: ${list}. Ask me to read or summarize ` +
                    `${names.length === 1 ? "it" : "them"} by name.`,
            );
        }
        if ((data.errors || []).length && !names.length) {
            showError(`Upload failed: ${data.errors.join("; ")}`);
        }
    } catch (err) {
        showError(`Upload failed: ${err.message}`);
    } finally {
        attachBtn.disabled = false;
        fileInput.value = "";
    }
}

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => uploadFiles(fileInput.files));

function autoresize() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 240) + "px";
}
messageInput.addEventListener("input", autoresize);

// ---- new chat / reset --------------------------------------------------

newChatBtn.addEventListener("click", async () => {
    if (threadId) {
        try {
            await apiDelete(`/chat/${threadId}`);
        } catch (_) {
            // best-effort; session may already be gone
        }
    }
    localStorage.removeItem(STORAGE_KEY);
    threadId = null;
    clearTranscript();
    clearError();
    if (attachments) {
        attachments.innerHTML = "";
        attachments.classList.add("hidden");
    }
    seedField.value = "";
    showStart();
});

window.addEventListener("pagehide", destroyMapInstances);
document.addEventListener("DOMContentLoaded", restoreSession);
