# Databricks notebook source
# MAGIC %md
# MAGIC # Hardened synthetic document pipeline: JSON -> render
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Calls `databricks-claude-sonnet-4-6`
# MAGIC 2. Requests structured synthetic JSON only
# MAGIC 3. Validates the schema
# MAGIC 4. Renders local PDF files into `synthetic_test_data`
# MAGIC
# MAGIC Improvements included:
# MAGIC - safer response parsing
# MAGIC - schema validation
# MAGIC - clearer error messages
# MAGIC - continuation-page headers for bank statements
# MAGIC - no `%pip install`

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
OUTPUT_DIR = "synthetic_test_data"
JSON_PATH = os.path.join(OUTPUT_DIR, "synthetic_documents.json")
MAX_TOKENS = 12000
TEMPERATURE = 0.3

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
def extract_text(resp):
    """
    Extract text from a Databricks serving endpoint response across common shapes.
    """
    if hasattr(resp, "choices") and resp.choices:
        choice = resp.choices[0]

        if hasattr(choice, "message") and choice.message:
            message = choice.message
            if hasattr(message, "content"):
                content = message.content

                # Plain string
                if isinstance(content, str):
                    return content

                # Block-style content list
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif hasattr(item, "text"):
                            parts.append(item.text)
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)

        if hasattr(choice, "text") and choice.text:
            return choice.text

    if isinstance(resp, dict) and resp.get("choices"):
        choice = resp["choices"][0]

        if isinstance(choice, dict):
            message = choice.get("message", {})
            content = message.get("content")

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                return "\n".join(parts)

            if choice.get("text"):
                return choice["text"]

    return str(resp)


def parse_json_response(raw_text):
    """
    Parse model output into JSON with useful errors.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Model response was empty.")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            preview = raw_text[:1000]
            raise ValueError(
                "Could not locate a JSON object in the model response. "
                f"Response preview:\n{preview}"
            )

        candidate = raw_text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            preview = candidate[:1000]
            raise ValueError(
                "Model response contained text that looked like JSON, but parsing still failed. "
                f"JSON error: {e}. Response preview:\n{preview}"
            )


def require_field(obj, field_name, context):
    if field_name not in obj:
        raise ValueError(f"Missing required field '{field_name}' in {context}.")


def require_type(value, expected_type, context):
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_name = ", ".join(t.__name__ for t in expected_type)
        else:
            expected_name = expected_type.__name__
        raise ValueError(
            f"Invalid type for {context}. Expected {expected_name}, got {type(value).__name__}."
        )


def validate_bank_statement(doc):
    data = doc["data"]
    required = [
        "applicant_name",
        "address",
        "bank_name",
        "sort_code",
        "masked_account_number",
        "statement_period",
        "opening_balance",
        "closing_balance",
        "transactions",
    ]
    for field in required:
        require_field(data, field, f"bank_statement data for {doc['document_id']}")

    require_type(data["transactions"], list, f"transactions in {doc['document_id']}")

    for i, txn in enumerate(data["transactions"]):
        require_type(txn, dict, f"transaction {i} in {doc['document_id']}")
        for field in ["date", "description", "amount", "balance"]:
            require_field(txn, field, f"transaction {i} in {doc['document_id']}")


def validate_multipage(doc):
    data = doc["data"]
    require_field(data, "pages_content", f"{doc['document_type']} data for {doc['document_id']}")
    require_type(data["pages_content"], list, f"pages_content in {doc['document_id']}")

    for i, page in enumerate(data["pages_content"]):
        require_type(page, dict, f"page {i} in {doc['document_id']}")
        require_field(page, "title", f"page {i} in {doc['document_id']}")
        require_field(page, "body_lines", f"page {i} in {doc['document_id']}")
        require_type(page["body_lines"], list, f"body_lines in page {i} of {doc['document_id']}")


def validate_driving_licence(doc):
    data = doc["data"]
    required = [
        "surname",
        "given_names",
        "date_of_birth",
        "issue_date",
        "expiry_date",
        "issuing_authority",
        "licence_number",
        "address",
    ]
    for field in required:
        require_field(data, field, f"uk_driving_licence data for {doc['document_id']}")


def validate_passport(doc):
    data = doc["data"]
    required = [
        "passport_type",
        "issuing_country",
        "passport_number",
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "sex",
        "place_of_birth",
        "issue_date",
        "expiry_date",
        "mrz_line_1",
        "mrz_line_2",
    ]
    for field in required:
        require_field(data, field, f"uk_passport data for {doc['document_id']}")


def validate_payload(payload):
    require_type(payload, dict, "top-level payload")
    require_field(payload, "documents", "top-level payload")
    require_type(payload["documents"], list, "top-level 'documents'")

    expected_types = {
        "bank_statement",
        "multipage_pdf",
        "skewed_scanned_pdf",
        "uk_driving_licence",
        "uk_passport",
    }

    seen_types = set()

    for i, doc in enumerate(payload["documents"]):
        require_type(doc, dict, f"document {i}")

        for field in ["document_id", "document_type", "file_name", "metadata", "data"]:
            require_field(doc, field, f"document {i}")

        require_type(doc["metadata"], dict, f"metadata in {doc['document_id']}")
        require_type(doc["data"], dict, f"data in {doc['document_id']}")

        doc_type = doc["document_type"]
        if doc_type not in expected_types:
            raise ValueError(
                f"Unsupported document_type '{doc_type}' in {doc['document_id']}. "
                f"Expected one of: {sorted(expected_types)}"
            )

        seen_types.add(doc_type)

        if doc_type == "bank_statement":
            validate_bank_statement(doc)
        elif doc_type in ("multipage_pdf", "skewed_scanned_pdf"):
            validate_multipage(doc)
        elif doc_type == "uk_driving_licence":
            validate_driving_licence(doc)
        elif doc_type == "uk_passport":
            validate_passport(doc)

    missing_types = expected_types - seen_types
    if missing_types:
        raise ValueError(
            f"Payload did not contain all required document types. Missing: {sorted(missing_types)}"
        )

# COMMAND ----------
try:
    response = client.serving_endpoints.query(
        name=ENDPOINT_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to call serving endpoint '{ENDPOINT_NAME}'. "
        f"Check endpoint name, permissions, and cluster connectivity. Original error: {e}"
    )

raw_text = extract_text(response)

try:
    payload = parse_json_response(raw_text)
    validate_payload(payload)
except Exception as e:
    preview = raw_text[:2000] if raw_text else "<empty response>"
    raise RuntimeError(
        f"Model output could not be parsed/validated. Error: {e}\n\n"
        f"Response preview:\n{preview}"
    )

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Saved JSON payload to: {JSON_PATH}")

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
        pdf.savefig(fig)
        plt.close(fig)


def draw_bank_transaction_header(ax, y):
    ax.text(0.05, y, "Date", fontsize=8, fontweight="bold", va="top")
    ax.text(0.20, y, "Description", fontsize=8, fontweight="bold", va="top")
    ax.text(0.76, y, "Amount", fontsize=8, fontweight="bold", va="top")
    ax.text(0.90, y, "Balance", fontsize=8, fontweight="bold", va="top")
    y -= 0.018
    ax.plot([0.05, 0.95], [y, y], linewidth=0.8)
    return y - 0.012


def render_bank_statement(doc, path):
    data = doc["data"]
    txns = data.get("transactions", [])

    with PdfPages(path) as pdf:
        page_num = 1

        fig, ax = new_page()
        draw_header(ax, "Synthetic Bank Statement", f"{doc['document_id']} | Page {page_num}")

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
            "Transactions",
        ]
        draw_wrapped_lines(ax, summary_lines, start_y=0.89, line_height=0.022, width=95, fontsize=9)

        y = 0.67
        y = draw_bank_transaction_header(ax, y)

        for txn in txns:
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)

                page_num += 1
                fig, ax = new_page()
                draw_header(ax, "Synthetic Bank Statement", f"{doc['document_id']} | Page {page_num}")
                y = 0.89
                y = draw_bank_transaction_header(ax, y)

            ax.text(0.05, y, str(txn.get("date", "")), fontsize=7.5, va="top")
            ax.text(0.20, y, str(txn.get("description", ""))[:48], fontsize=7.5, va="top")
            ax.text(0.76, y, str(txn.get("amount", "")), fontsize=7.5, va="top")
            ax.text(0.90, y, str(txn.get("balance", "")), fontsize=7.5, va="top")
            y -= 0.018

        pdf.savefig(fig)
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
                ax.add_patch(Rectangle((0.03, 0.03), 0.94, 0.90, fill=False, linewidth=0.6))

            body_lines = page.get("body_lines", [])
            draw_wrapped_lines(
                ax,
                body_lines,
                start_y=0.89,
                line_height=0.022,
                width=95,
                fontsize=9,
                rotation=rotation,
            )

            pdf.savefig(fig)
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
            data.get("mrz_line_1", ""),
            data.get("mrz_line_2", ""),
        ])
    ]
    render_simple_text_pdf(path, "Synthetic UK Passport", sections, subtitle=doc["document_id"])

# COMMAND ----------
rendered_files = []
render_errors = []

for doc in payload["documents"]:
    try:
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
            raise ValueError(f"Unsupported document type: {doc_type}")

        rendered_files.append({
            "document_id": doc["document_id"],
            "document_type": doc_type,
            "path": path,
        })

    except Exception as e:
        render_errors.append({
            "document_id": doc.get("document_id", "<unknown>"),
            "document_type": doc.get("document_type", "<unknown>"),
            "error": str(e),
        })

if render_errors:
    print("Some documents failed to render:")
    for err in render_errors:
        print(json.dumps(err, indent=2))

if not rendered_files:
    raise RuntimeError("No files were rendered successfully.")

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
print(f"Saved rendered file index to: {TABLE_NAME}")
