import requests

DATABRICKS_HOST = WORKSPACE_URL
LLM_ENDPOINT_NAME = "llama32-endpoint"


def call_llm_endpoint(prompt: str, max_new_tokens: int = 320, temperature: float = 0.0) -> str:
    """Sends a chat-style prompt to the Databricks-hosted Llama 3.2 endpoint."""
    
    url = f"{DATABRICKS_HOST}/serving-endpoints/{LLM_ENDPOINT_NAME}/invocations"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert complaints analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    
    if response.ok:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"LLM call failed: {response.status_code} {response.text[:500]}")
