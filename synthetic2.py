# Databricks notebook source
# MAGIC %md
# MAGIC # Minimal synthetic document pipeline: JSON -> render
# MAGIC
# MAGIC This notebook keeps the LLM output small and cheap:
# MAGIC 1. Ask `databricks-claude-sonnet-4-6` for structured synthetic JSON only
# MAGIC 2. Render that JSON into local files in a subfolder called `synthetic_test_data`
# MAGIC
# MAGIC Design goals:
# MAGIC - no `%pip install`
# MAGIC - only use Python libraries typically available in Databricks ML Runtime
# MAGIC - one prompt to generate all scenarios
# MAGIC - deterministic local rendering for repeatable tests

# COMMAND ----------
import json
import os
import textwrap
from databricks.sdk import WorkspaceClient

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# COMMAND ----------
ENDPOINT_NAME = "databricks-claude-sonnet-4-6"
OUTPUT_DIR = "synthetic_test_data"  # current working directory subfolder
JSON_PATH = os.path.join(OUTPUT_DIR, "synthetic_documents.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = WorkspaceClient()

# COMMAND ----------
prompt = """
Generate synthetic test data for an asset finance document workflow.

Rules:
- Return JSON only.
- All data must be fictional.
- No real people, real addresses, real passport numbers, real licence numbers, or real account numbers.
- Keep each document internally consistent.
- Prefer UK formatting.

Return exactly this JSON shape:
{
  "documents": [
    {
      "document_id": "doc_001",
      "document_type": "bank_statement | multipage_pdf | skewed_scanned_pdf | uk_driving_licence | uk_passport",
      "file_name": "doc_001.pdf",
      "metadata": {
        "pages": 1,
        "orientation_issue": false,
        "notes": ["optional note"]
      },
      "data": {}
    }
  ]
}

Document requirements:

1. bank_statement
- applicant_name
- address
- bank_name
- sort_code
- masked_account_number
- statement_period
- opening_balance
- closing_balance
- transactions: array of 20 to 35 items
- each transaction: date, description, amount, balance
- include salary, direct debit, card payments, transfer, cash withdrawal, and one edge case

2. multipage_pdf
- 3 to 5 pages
- data must contain pages_content: array
- each page item should have title and body_lines
- plausible asset finance pack content: customer summary, business details, affordability, asset details, invoice summary

3. skewed_scanned_pdf
- 2 to 4 pages
- same structure as multipage_pdf
- metadata.orientation_issue = true
- metadata.notes should mention scan/skew/rotation/contrast issue

4. uk_driving_licence
- surname
- given_names
- date_of_birth
- issue_date
- expiry_date
- issuing_authority
- licence_number
- address

5. uk_passport
- passport_type
- issuing_country
- passport_number
- surname
- given_names
- nationality
- date_of_birth
- sex
- place_of_birth
- issue_date
- expiry_date
- mrz_line_1
- mrz_line_2

Create exactly 5 documents total, one per type.
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
    temperature=0.3,
    max_tokens=12000,
)

# COMMAND ----------
def extract_text(resp):
    if hasattr(resp, "choices") and resp.choices:
        choice = resp.choices[0]
        if hasattr(choice, "message") and choice.message and hasattr(choice.message, "content"):
            return choice.message.content
        if hasattr(choice, "text") and choice.text:
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
payload = json.loads(raw_text[start:end + 1])

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(JSON_PATH)

# COMMAND ----------
def new_page(figsize=(8.27, 11.69)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_header(ax, title, subtitle=None):
    ax.text(0.05, 0.965, title, fontsize=16, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.05, 0.94, subtitle, fontsize=9, va="top")
    ax.plot([0.05, 0.95], [0.92, 0.92], linewidth=1)


def draw_wrapped_lines(ax, lines, start_y=0.89, line_height=0.022, width=90, fontsize=9, rotation=0):
    y = start_y
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=width) or [""]
        for part in wrapped:
            if y < 0.05:
                return
            ax.text(0.05, y, part, fontsize=fontsize, va="top", rotation=rotation)
            y -= line_height


def render_simple_text_pdf(path, title, sections, subtitle=None, rotation=0):
    with PdfPages(path) as pdf:
        fig, ax = new_page()
        draw_header(ax, title, subtitle)
        lines = []
        for section_title, section_lines in sections:
            lines.append(section_title)
            lines.extend(section_lines)
            lines.append("")
        draw_wrapped_lines(ax, lines, rotation=rotation)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def render_bank_statement(doc, path):
    data = doc["data"]
    txns = data.get("transactions", [])

    with PdfPages(path) as pdf:
        fig, ax = new_page()
        draw_header(ax, "Synthetic Bank Statement", doc["document_id"])

        summary_lines = [
            f"Account holder: {data.get('applicant_name', '')}",
            f"Address: {data.get('address', '')}",
            f"Bank: {data.get('bank_name', '')}",
            f"Sort code: {data.get('sort_code', '')}",
            f"Account number: {data.get('masked_account_number', '')}",
            f"Statement period: {data.get('statement_period', '')}",
            f"Opening balance: {data.get('opening_balance', '')}",
            f"Closing balance: {data.get('closing_balance', '')}",
            "",
            "Transactions"
        ]
        draw_wrapped_lines(ax, summary_lines, start_y=0.89, line_height=0.022, width=95, fontsize=9)

        y = 0.67
        ax.text(0.05, y, "Date", fontsize=8, fontweight="bold", va="top")
        ax.text(0.20, y, "Description", fontsize=8, fontweight="bold", va="top")
        ax.text(0.76, y, "Amount", fontsize=8, fontweight="bold", va="top")
        ax.text(0.90, y, "Balance", fontsize=8, fontweight="bold", va="top")
        y -= 0.018
        ax.plot([0.05, 0.95], [y, y], linewidth=0.8)
        y -= 0.012

        for txn in txns:
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = new_page()
                draw_header(ax, "Synthetic Bank Statement", f"{doc['document_id']} (cont.)")
                y = 0.89
            ax.text(0.05, y, str(txn.get("date", "")), fontsize=7.5, va="top")
            ax.text(0.20, y, str(txn.get("description", ""))[:48], fontsize=7.5, va="top")
            ax.text(0.76, y, str(txn.get("amount", "")), fontsize=7.5, va="top")
            ax.text(0.90, y, str(txn.get("balance", "")), fontsize=7.5, va="top")
            y -= 0.018

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def render_multipage(doc, path, skew=False):
    pages = doc["data"].get("pages_content", [])
    notes = ", ".join(doc.get("metadata", {}).get("notes", []))
    rotation = 7 if skew else 0

    with PdfPages(path) as pdf:
        for idx, page in enumerate(pages, start=1):
            fig, ax = new_page()
            subtitle = f"{doc['document_id']} | Page {idx}"
            if notes:
                subtitle += f" | {notes}"
            draw_header(ax, page.get("title", f"Page {idx}"), subtitle)

            if skew:
                ax.add_patch(Rectangle((0.03, 0.03), 0.94, 0.90, fill=False, linewidth=0.6, angle=0))

            body_lines = page.get("body_lines", [])
            draw_wrapped_lines(ax, body_lines, start_y=0.89, line_height=0.022, width=95, fontsize=9, rotation=rotation)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def render_driving_licence(doc, path):
    data = doc["data"]
    sections = [
        ("UK Driving Licence", [
            f"Surname: {data.get('surname', '')}",
            f"Given names: {data.get('given_names', '')}",
            f"Date of birth: {data.get('date_of_birth', '')}",
            f"Issue date: {data.get('issue_date', '')}",
            f"Expiry date: {data.get('expiry_date', '')}",
            f"Issuing authority: {data.get('issuing_authority', '')}",
            f"Licence number: {data.get('licence_number', '')}",
            f"Address: {data.get('address', '')}",
        ])
    ]
    render_simple_text_pdf(path, "Synthetic UK Driving Licence", sections, subtitle=doc["document_id"])


def render_passport(doc, path):
    data = doc["data"]
    sections = [
        ("UK Passport", [
            f"Type: {data.get('passport_type', '')}",
            f"Issuing country: {data.get('issuing_country', '')}",
            f"Passport number: {data.get('passport_number', '')}",
            f"Surname: {data.get('surname', '')}",
            f"Given names: {data.get('given_names', '')}",
            f"Nationality: {data.get('nationality', '')}",
            f"Date of birth: {data.get('date_of_birth', '')}",
            f"Sex: {data.get('sex', '')}",
            f"Place of birth: {data.get('place_of_birth', '')}",
            f"Issue date: {data.get('issue_date', '')}",
            f"Expiry date: {data.get('expiry_date', '')}",
            "",
            "MRZ:",
            data.get('mrz_line_1', ''),
            data.get('mrz_line_2', ''),
        ])
    ]
    render_simple_text_pdf(path, "Synthetic UK Passport", sections, subtitle=doc["document_id"])

# COMMAND ----------
rendered_files = []

for doc in payload["documents"]:
    path = os.path.join(OUTPUT_DIR, doc["file_name"])
    doc_type = doc["document_type"]

    if doc_type == "bank_statement":
        render_bank_statement(doc, path)
    elif doc_type == "multipage_pdf":
        render_multipage(doc, path, skew=False)
    elif doc_type == "skewed_scanned_pdf":
        render_multipage(doc, path, skew=True)
    elif doc_type == "uk_driving_licence":
        render_driving_licence(doc, path)
    elif doc_type == "uk_passport":
        render_passport(doc, path)
    else:
        continue

    rendered_files.append({
        "document_id": doc["document_id"],
        "document_type": doc_type,
        "path": path,
    })

# COMMAND ----------
for item in rendered_files:
    print(item["path"])

# COMMAND ----------
index_df = spark.createDataFrame(rendered_files)
display(index_df)

# COMMAND ----------
# Optional: save index to a Delta table
TABLE_NAME = "main.default.synthetic_asset_finance_rendered_files"
index_df.write.mode("overwrite").format("delta").saveAsTable(TABLE_NAME)
print(TABLE_NAME)
