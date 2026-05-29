/**
 * Frontend logic for the only Subcribers web interface
 */

const form = document.getElementById('queryForm');
const questionField = document.getElementById('question');
const urlsField = document.getElementById('urls');
const webSearchCheckbox = document.getElementById('webSearch');
const rebuildCheckbox = document.getElementById('rebuild');
const debugCheckbox = document.getElementById('debug');
const submitButton = form.querySelector('.btn-submit');
const submitText = document.getElementById('submitText');

const resultsSection = document.getElementById('resultsSection');
const statusBadge = document.getElementById('statusBadge');
const sourceModeContainer = document.getElementById('sourceMode');
const answerContainer = document.getElementById('answerContainer');
const errorContainer = document.getElementById('errorContainer');

// API endpoint
const API_BASE_URL = window.location.origin;
const QUERY_ENDPOINT = `${API_BASE_URL}/query`;

/**
 * Format error message for display
 */
function formatError(error) {
    if (typeof error === 'string') {
        return error;
    }
    if (error.detail) {
        return error.detail;
    }
    return 'An unexpected error occurred';
}

/**
 * Clear results section
 */
function clearResults() {
    resultsSection.classList.remove('show');
    answerContainer.innerHTML = '';
    errorContainer.innerHTML = '';
    statusBadge.innerHTML = '';
    if (sourceModeContainer) {
        sourceModeContainer.innerHTML = '';
    }
}

/**
 * Display source mode hint
 */
function renderSourceMode(mode, note, sourceUrls = []) {
    if (!sourceModeContainer) {
        return;
    }

    if (!mode) {
        sourceModeContainer.textContent = '';
        return;
    }

    const modeLabels = {
        explicit: 'Explicit URLs',
        web_search: 'Web search',
        defaults: 'Configured defaults',
    };

    const noteLabels = {
        web_search_failed: 'web search failed, used defaults',
        web_search_no_results: 'no search results, used defaults',
        web_search_disabled: 'web search disabled, used defaults',
    };

    const label = modeLabels[mode] || mode;
    const count = Array.isArray(sourceUrls) ? sourceUrls.length : 0;
    let text = `Source mode: ${label}`;
    if (count > 0) {
        text += ` (${count} URL${count === 1 ? '' : 's'})`;
    }
    if (note && noteLabels[note]) {
        text += ` - ${noteLabels[note]}`;
    }
    sourceModeContainer.textContent = text;
}

/**
 * Show loading state
 */
function showLoading() {
    clearResults();
    submitButton.disabled = true;
    submitText.innerHTML = '<span class="loading-indicator"><span class="spinner"></span></span>Processing...';
    resultsSection.classList.add('show');
    statusBadge.innerHTML = '<span class="status-badge badge-loading">Loading...</span>';
}

/**
 * Display success result
 */
function showSuccess(answer, sourceUrls = [], sourceMode = null, sourceNote = null) {
    statusBadge.innerHTML = '<span class="status-badge badge-success">✓ Success</span>';
    renderSourceMode(sourceMode, sourceNote, sourceUrls);
    answerContainer.innerHTML = `<div class="answer-box">${escapeHtml(answer)}</div>`;
    if (sourceUrls.length) {
        const links = sourceUrls
            .map(url => `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a></li>`)
            .join('');
        answerContainer.innerHTML += `<div class="help-text"><strong>Sources used:</strong><ul>${links}</ul></div>`;
    }
    submitButton.disabled = false;
    submitText.textContent = 'Ask Question';
}

/**
 * Display error result
 */
function showError(error, sourceMode = null, sourceNote = null, sourceUrls = []) {
    statusBadge.innerHTML = '<span class="status-badge badge-error">✗ Error</span>';
    renderSourceMode(sourceMode, sourceNote, sourceUrls);
    errorContainer.innerHTML = `<div class="error-box">${escapeHtml(error)}</div>`;
    submitButton.disabled = false;
    submitText.textContent = 'Ask Question';
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Handle form submission
 */
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const question = questionField.value.trim();
    const urls = urlsField.value.trim();
    const webSearch = webSearchCheckbox ? webSearchCheckbox.checked : true;
    const rebuild = rebuildCheckbox.checked;
    const debug = debugCheckbox ? debugCheckbox.checked : false;

    // Validate
    if (!question) {
        showError('Please enter a question');
        return;
    }

    showLoading();

    try {
        const response = await fetch(QUERY_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                urls: urls || null,
                web_search: webSearch,
                rebuild: rebuild,
                debug: debug,
            }),
        });

        const data = await response.json();

        const sourceUrls = data.source_urls || [];
        const sourceMode = data.source_mode || null;
        const sourceNote = data.source_note || null;

        if (!response.ok) {
            showError(formatError(data), sourceMode, sourceNote, sourceUrls);
            return;
        }

        if (data.error) {
            showError(data.error, sourceMode, sourceNote, sourceUrls);
            return;
        }

        if (!data.answer) {
            // If debug info is available, show it to help diagnosis
            if (data.messages && data.messages.length) {
                renderSourceMode(sourceMode, sourceNote, sourceUrls);
                answerContainer.innerHTML = `<div class="answer-box">No answer was generated. Debug messages below:</div>`;
                const msgs = data.messages.map(m => `<pre>${escapeHtml(m)}</pre>`).join('\n');
                answerContainer.innerHTML += `<div class="help-text">${msgs}</div>`;
                submitButton.disabled = false;
                submitText.textContent = 'Ask Question';
                return;
            }

            showError('No answer was generated. Please try again.', sourceMode, sourceNote, sourceUrls);
            return;
        }

        if (data.messages && data.messages.length) {
            // Show answer plus debug messages if present
            showSuccess(data.answer, sourceUrls, sourceMode, sourceNote);
            const msgs = data.messages.map(m => `<pre>${escapeHtml(m)}</pre>`).join('\n');
            answerContainer.innerHTML += `<div class="help-text">${msgs}</div>`;
        } else {
            showSuccess(data.answer, sourceUrls, sourceMode, sourceNote);
        }
    } catch (error) {
        console.error('Request error:', error);
        showError(`Request failed: ${error.message}`);
    }
});

// Clear results when user starts typing a new question
questionField.addEventListener('input', () => {
    if (resultsSection.classList.contains('show')) {
        clearResults();
    }
});

// Check API health on page load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (!data.graph_ready) {
            console.warn('Graph not yet ready, will be ready shortly');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Run health check when page loads
document.addEventListener('DOMContentLoaded', checkHealth);
