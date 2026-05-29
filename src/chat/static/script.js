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
const urlsField = document.getElementById("urls");
const webSearchCheckbox = document.getElementById("webSearch");
const startBtn = document.getElementById("startBtn");
const startError = document.getElementById("startError");

const transcript = document.getElementById("transcript");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const sessionInfo = document.getElementById("sessionInfo");
const errorBanner = document.getElementById("errorBanner");
const newChatBtn = document.getElementById("newChatBtn");

let threadId = null;
let pending = false;

// ---- helpers ------------------------------------------------------------

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
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
    bubble.innerHTML = escapeHtml(content);
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
    const urls = urlsField.value.trim();
    const webSearch = webSearchCheckbox.checked;

    if (!urls && !seed) {
        startError.textContent = "Provide a topic or some source URLs.";
        return;
    }
    startError.textContent = "";
    startBtn.disabled = true;
    startBtn.textContent = "Indexing sources...";

    try {
        const data = await apiPost("/chat", {
            urls: urls || null,
            web_search: webSearch,
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
        const data = await apiPost(`/chat/${threadId}/message`, { message: trimmed });
        thinking.remove();
        if (data.error) {
            showError(data.error);
            appendTurn("assistant", "(no reply produced)");
        } else {
            clearError();
            appendTurn("assistant", data.answer);
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

sendBtn.addEventListener("click", () => sendMessage(messageInput.value));
messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(messageInput.value);
    }
});

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
    seedField.value = "";
    urlsField.value = "";
    showStart();
});

document.addEventListener("DOMContentLoaded", restoreSession);
