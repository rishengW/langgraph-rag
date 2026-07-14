/**
 * Frontend logic for the only Subcribers chat app.
 *
 * Talks to the FastAPI backend in src/chat/api.py:
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

// ---- helpers ------------------------------------------------------------

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

function safeHref(url) {
    const value = String(url || "").trim();
    if (/^(https?:|mailto:)/i.test(value)) {
        return escapeHtml(value);
    }
    return "";
}

function renderInlineMarkdown(text) {
    const codeTokens = [];
    let value = String(text).replace(/`([^`]+)`/g, (_match, code) => {
        const token = `\u0000CODE${codeTokens.length}\u0000`;
        codeTokens.push(`<code>${escapeHtml(code)}</code>`);
        return token;
    });

    // Protect math spans from the escape/emphasis passes below so LaTeX such
    // as x_1, a * b, and \sum_{i=1} survives intact for KaTeX. Same
    // token-stash strategy used for inline code; restored raw (un-escaped)
    // after Markdown so auto-render sees the original delimiters.
    const mathTokens = [];
    value = value.replace(/(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\s)[^$\n]+?(?<!\s)\$)/g, (m) => {
        const token = `\u0000MATH${mathTokens.length}\u0000`;
        mathTokens.push(m);
        return token;
    });

    value = escapeHtml(value);
    value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    value = value.replace(/_([^_]+)_/g, "<em>$1</em>");
    value = value.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, url) => {
        const href = safeHref(url);
        if (!href) return label;
        return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    });

    for (const [index, html] of codeTokens.entries()) {
        value = value.replaceAll(`\u0000CODE${index}\u0000`, html);
    }
    for (const [index, m] of mathTokens.entries()) {
        value = value.replaceAll(`\u0000MATH${index}\u0000`, () => m);
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
    if (role === "assistant" && !opts.thinking) {
        bubble.classList.add("markdown-preview");
        bubble.innerHTML = renderMarkdownPreview(content);
        renderMath(bubble);
    } else {
        bubble.innerHTML = escapeHtml(content);
    }
    div.appendChild(bubble);
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
    return div;
}

function clearTranscript() {
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

// ---- API calls ----------------------------------------------------------

async function apiPost(path, body) {
    const res = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
    return data;
}

async function apiGet(path) {
    const res = await fetch(`${API}${path}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
    return data;
}

async function apiDelete(path) {
    const res = await fetch(`${API}${path}`, { method: "DELETE" });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
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
            appendTurn(turn.role, turn.content);
        }
        showChat();
    } catch (err) {
        // Session no longer exists on the server; fall back to the start screen.
        localStorage.removeItem(STORAGE_KEY);
    }
}

// ---- send a message ----------------------------------------------------

async function sendMessage(text) {
    if (!threadId || pending) return;
    const trimmed = text.trim();
    if (!trimmed) return;

    pending = true;
    sendBtn.disabled = true;
    appendTurn("user", trimmed);
    messageInput.value = "";
    autoresize();

    const thinking = appendTurn("assistant", "Thinking...", { thinking: true });

    try {
        await streamMessage(trimmed, thinking);
        try {
            const history = await apiGet(`/chat/${threadId}/history`);
            renderSessionInfo(history.source_urls, history.source_mode);
        } catch (_) {
            // The answer is already complete; stale source metadata is non-fatal.
        }
    } catch (err) {
        thinking.remove();
        showError(err.message);
        appendTurn("assistant", "(failed to fetch reply)");
    } finally {
        pending = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// Consume the SSE token stream from POST /chat/{tid}/message/stream.
// Renders assistant tokens incrementally; falls back to the final ``done``
// answer if no token deltas were received (e.g. token streaming disabled).
async function streamMessage(message, thinking) {
    const res = await fetch(`${API}/chat/${threadId}/message/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
    });
    if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || data.error || `HTTP ${res.status}`);
    }

    let bubble = null;
    let answer = "";
    let streamError = "";

    const ensureBubble = () => {
        if (bubble) return bubble;
        thinking.remove();
        clearError();
        const turn = appendTurn("assistant", "");
        bubble = turn.querySelector(".bubble");
        return bubble;
    };

    const handleEvent = (eventType, dataStr) => {
        let payload = {};
        try {
            payload = dataStr ? JSON.parse(dataStr) : {};
        } catch (_) {
            return;
        }
        if (eventType === "token" && typeof payload.token === "string") {
            answer += payload.token;
            const el = ensureBubble();
            el.innerHTML = renderMarkdownPreview(answer);
            transcript.scrollTop = transcript.scrollHeight;
        } else if (eventType === "error") {
            streamError = payload.message || "stream error";
        } else if (eventType === "done") {
            // Prefer accumulated tokens; otherwise use the final answer.
            if (!answer && typeof payload.answer === "string") {
                answer = payload.answer;
            }
        }
    };

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE frames are separated by a blank line.
        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            let eventType = "message";
            let dataStr = "";
            for (const line of frame.split("\n")) {
                if (line.startsWith("event:")) eventType = line.slice(6).trim();
                else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
            }
            handleEvent(eventType, dataStr);
        }
    }

    thinking.remove();
    if (streamError && !answer) {
        showError(streamError);
        appendTurn("assistant", "(no reply produced)");
        return;
    }
    if (!bubble) {
        // No tokens streamed; render the final answer in one shot.
        clearError();
        appendTurn("assistant", answer || "(no reply produced)");
    } else {
        bubble.innerHTML = renderMarkdownPreview(answer);
        renderMath(bubble);
    }
}

sendBtn.addEventListener("click", () => sendMessage(messageInput.value));
messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(messageInput.value);
    }
});

// ---- file uploads ------------------------------------------------------

function renderAttachmentChips(files, errors) {
    attachments.innerHTML = "";
    for (const f of files || []) {
        const chip = document.createElement("span");
        chip.className = "chip";
        const kb = Math.max(1, Math.round(f.size_bytes / 1024));
        chip.textContent = `${f.filename} (${kb} KB)`;
        attachments.appendChild(chip);
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
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
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

document.addEventListener("DOMContentLoaded", restoreSession);
