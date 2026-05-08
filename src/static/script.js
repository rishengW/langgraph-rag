/**
 * Frontend logic for RAG LangGraph web interface
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
function showSuccess(answer, sourceUrls = []) {
    statusBadge.innerHTML = '<span class="status-badge badge-success">✓ Success</span>';
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
function showError(error) {
    statusBadge.innerHTML = '<span class="status-badge badge-error">✗ Error</span>';
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

        if (!response.ok) {
            showError(formatError(data));
            return;
        }

        if (data.error) {
            showError(data.error);
            return;
        }

        if (!data.answer) {
            // If debug info is available, show it to help diagnosis
            if (data.messages && data.messages.length) {
                answerContainer.innerHTML = `<div class="answer-box">No answer was generated. Debug messages below:</div>`;
                const msgs = data.messages.map(m => `<pre>${escapeHtml(m)}</pre>`).join('\n');
                answerContainer.innerHTML += `<div class="help-text">${msgs}</div>`;
                submitButton.disabled = false;
                submitText.textContent = 'Ask Question';
                return;
            }

            showError('No answer was generated. Please try again.');
            return;
        }

        if (data.messages && data.messages.length) {
            // Show answer plus debug messages if present
            showSuccess(data.answer, data.source_urls || []);
            const msgs = data.messages.map(m => `<pre>${escapeHtml(m)}</pre>`).join('\n');
            answerContainer.innerHTML += `<div class="help-text">${msgs}</div>`;
        } else {
            showSuccess(data.answer, data.source_urls || []);
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
