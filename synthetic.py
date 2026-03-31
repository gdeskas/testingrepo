# Databricks notebook source
# MAGIC %md
# MAGIC # Minimal synthetic file generation notebook
# MAGIC
# MAGIC This notebook sends a single prompt to the `databricks-claude-sonnet-4-5` endpoint and writes the returned files to DBFS.
# MAGIC
# MAGIC Assumptions:
# MAGIC - You are running on a Databricks ML Runtime with the Databricks SDK already available.
# MAGIC - The model returns a JSON payload containing files, each with a filename and base64 content.

# COMMAND ----------
import base64
import json
import os
from databricks.sdk import WorkspaceClient

# COMMAND ----------
ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
OUTPUT_DIR = "/dbfs/tmp/synthetic_asset_finance_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = WorkspaceClient()

# COMMAND ----------
prompt = """
Generate synthetic test documents for an asset finance application workflow.

Requirements:
- Return files directly, not explanations.
- All data must be fictional and internally consistent.
- Create these documents:
  1. A bank statement PDF
  2. A multi-page PDF document
  3. A PDF document with orientation / scan-angle issues
  4. A UK driving licence document
  5. A UK passport document
- Make them suitable for OCR / document extraction testing.

Return valid JSON only in this exact shape:
{
  "files": [
    {
      "filename": "example.pdf",
      "content_base64": "...",
      "mime_type": "application/pdf"
    }
  ]
}
""".strip()

# COMMAND ----------
response = client.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    temperature=0.4,
    max_tokens=8000,
)

# COMMAND ----------
def extract_text(resp):
    if hasattr(resp, "choices") and resp.choices:
        choice = resp.choices[0]
        if hasattr(choice, "message") and choice.message and hasattr(choice.message, "content"):
            return choice.message.content
        if hasattr(choice, "text"):
            return choice.text
    if isinstance(resp, dict) and resp.get("choices"):
        choice = resp["choices"][0]
        if isinstance(choice, dict):
            if choice.get("message", {}).get("content"):
                return choice["message"]["content"]
            if choice.get("text"):
                return choice["text"]
    return str(resp)

raw_text = extract_text(response).strip()
start = raw_text.find("{")
end = raw_text.rfind("}")
payload = json.loads(raw_text[start:end+1])

# COMMAND ----------
written_files = []

for f in payload["files"]:
    path = os.path.join(OUTPUT_DIR, f["filename"])
    with open(path, "wb") as out:
        out.write(base64.b64decode(f["content_base64"]))
    written_files.append({
        "filename": f["filename"],
        "mime_type": f.get("mime_type", "application/octet-stream"),
        "path": path,
    })

# COMMAND ----------
for f in written_files:
    print(f["path"])
