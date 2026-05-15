# SSL and Dependency Fixes

## Issues Fixed

### 1. Missing DuckDuckGo Search Module
**Error**: `No module named 'duckduckgo_search'`

**Solution**: 
- Installed `ddgs` package from requirements.txt
- The code in `web_search.py` tries `ddgs` first, then falls back to `duckduckgo_search`
- Both packages are now supported via the import fallback mechanism

**Verification**:
```bash
pip list | grep -E "ddgs|duckduckgo"
```

### 2. Dashscope SSL Connection Errors
**Error**: 
```
HTTPSConnectionPool(host='dashscope.aliyuncs.com', port=443): Max retries exceeded with url: ...
SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
```

**Solutions Implemented**:

1. **Automatic Retry Logic with Exponential Backoff**
   - Added `_invoke_with_retry()` function with configurable retry attempts (default: 3)
   - Implements exponential backoff with jitter to avoid thundering herd
   - Catches SSL, connection, and timeout errors

2. **Error Handling in All LLM Calls**
   - `grade_documents_factory()`: Falls back to keyword-only matching on API error
   - `agent_factory()`: Calls the retriever directly if tool selection fails
   - `rewrite_factory()`: Returns original question on failure
   - `generate_factory()`: Returns an extractive answer from retrieved context on failure

3. **SSL Configuration**
   - Added `_configure_ssl()` function for safer SSL handling
   - Support for disabling SSL verification (debugging):
     ```bash
     export DISABLE_SSL_VERIFY=true
     ```
   - Suppresses SSL warnings when verification is disabled

## How to Use

### Normal Operation (with SSL verification)
```bash
python -m src.main
```

### Debugging (disable SSL verification, only if needed)
```bash
export DISABLE_SSL_VERIFY=true
python -m src.main
```

## Technical Details

### Retry Behavior
- Retries up to 3 times (configurable in `_invoke_with_retry()`)
- Exponential backoff: 1s, 2s, 4s (plus random jitter)
- Logs each retry attempt for debugging
- Fails gracefully with fallback responses

### Fallback Strategies
- **Grade Documents**: Uses keyword matching when LLM fails
- **Agent/Rewrite/Generate**: Returns error message or original input
- Application continues to work even if LLM API fails

## Environment Variables

- `DISABLE_SSL_VERIFY`: Set to `true` to disable SSL verification (not recommended for production)
- `EMBEDDING_MODEL`: Defaults to `text-embedding-v4` for DashScope embeddings
- `EMBEDDING_DIMENSION`: Optional dense vector size for `text-embedding-v4`
- `EMBEDDING_BATCH_SIZE`: Batch size for synchronous DashScope embedding calls
- `DASHSCOPE_REQUEST_TIMEOUT`: Request timeout in seconds for DashScope calls
- `DASHSCOPE_MAX_RETRIES`: Number of app-level retries for transient connection errors
- `DASHSCOPE_HTTP_BASE_URL`: Optional DashScope endpoint override
- Standard Dashscope configuration via `DASHSCOPE_API_KEY` in `.env`

## Testing

To verify the fixes are working:

```python
# In Python REPL
from src.config import load_settings
from src.graph import build_graph

settings = load_settings()
graph = build_graph(settings)
print("Graph built successfully!")

# Test a query
result = graph.invoke({
    "messages": [{"role": "user", "content": "What is Python?"}]
})
print("Query result:", result)
```

## Logs to Monitor

Watch for these log messages to verify functionality:

- "SSL/Connection error on attempt X/3" - Indicates retry is happening
- "Retrying in X.X seconds" - Shows backoff delay
- "Grade documents error" - Indicates LLM failure with fallback to keywords
- "Agent error" / "Rewrite error" / "Generate error" - Indicates node-level failures

## Next Steps

If SSL errors persist:
1. Check internet connectivity
2. Verify DASHSCOPE_API_KEY is valid
3. Check if Dashscope service is accessible
4. Try with `DISABLE_SSL_VERIFY=true` for testing (debugging only)
5. Check logs for specific error messages
