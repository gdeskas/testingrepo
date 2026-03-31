# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Test Document Generator — LLM-Driven
# MAGIC **Asset Finance Application Processing — Test Data Pipeline**
# MAGIC
# MAGIC Claude Sonnet 4.5 generates complete Python rendering code for each document.
# MAGIC The notebook executes it directly to produce PDFs, then optionally degrades them
# MAGIC with scan artifacts.
# MAGIC
# MAGIC ```
# MAGIC Prompt (doc type + params) → Claude → Python/ReportLab code → exec() → PDF → scan artifacts
# MAGIC ```
# MAGIC
# MAGIC No JSON schema. No parsing. Maximum variety with minimum plumbing.

# COMMAND ----------

# MAGIC %pip install reportlab pdf2image Pillow numpy --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import os
import time
import shutil
import numpy as np
from openai import OpenAI
from PIL import Image, ImageFilter, ImageDraw

try:
    DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils() \
        .notebook().getContext().apiUrl().getOrElse(None)
    DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils() \
        .notebook().getContext().apiToken().getOrElse(None)
except NameError:
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

MODEL_ENDPOINT = "databricks-claude-sonnet-4-5"

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

LOCAL_TMP = "/tmp/synthetic_docs"
DBFS_OUTPUT = "/dbfs/tmp/synthetic_test_documents"
os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(DBFS_OUTPUT, exist_ok=True)

print(f"Workspace: {DATABRICKS_HOST}")
print(f"Endpoint:  {MODEL_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Core: Ask Claude to Write the Rendering Code, Then Run It

# COMMAND ----------

SYSTEM_PROMPT = """\
You are a Python code generator. You write complete, self-contained Python scripts \
that use ReportLab to generate realistic UK financial document PDFs.

RULES:
1. Output ONLY valid Python code. No markdown fences. No explanation. No preamble.
2. The script must be fully self-contained — all imports at the top.
3. The script must write a PDF to the file path stored in the variable `OUTPUT_PATH` \
   (this variable will be injected before execution).
4. Use only: reportlab, random, datetime, decimal, os — all pre-installed.
5. All data must be ENTIRELY FICTIONAL — no real people, companies, or account numbers.
6. Use fictional bank names — never real UK bank names (not Barclays, HSBC, Lloyds, etc).
7. The document must look like a genuine UK financial document: correct formatting, \
   realistic merchant names, proper UK sort codes (XX-XX-XX), 8-digit account numbers, \
   and plausible transaction patterns.
8. Use Helvetica (built-in). Don't register external fonts.
9. Every time you generate a document, vary the customer name, address, amounts, \
   transaction mix, and styling. Use random for variation.
10. For bank statements: the running balance must be arithmetically correct. \
    Compute it row-by-row from the opening balance.
"""


def generate_document(user_prompt: str, output_filename: str,
                      temperature: float = 1.0, max_retries: int = 3) -> str:
    """Ask Claude to write rendering code, then exec it to produce a PDF."""

    output_path = os.path.join(LOCAL_TMP, output_filename)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{attempt}/{max_retries}] Calling Claude...")
            t0 = time.time()

            response = client.chat.completions.create(
                model=MODEL_ENDPOINT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=16384,
            )

            code = response.choices[0].message.content.strip()
            elapsed = time.time() - t0
            print(f"  Code generated in {elapsed:.1f}s ({len(code)} chars)")

            # Strip markdown fences if model wraps them
            if code.startswith("```"):
                code = code.split("\n", 1)[1]
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            code = code.strip()

            # Inject the output path and execute
            exec_globals = {"OUTPUT_PATH": output_path}
            exec(code, exec_globals)

            if os.path.exists(output_path):
                size_kb = os.path.getsize(output_path) / 1024
                print(f"  ✓ {output_filename} ({size_kb:.1f} KB)")
                return output_path
            else:
                print(f"  ✗ Code ran but no file produced. Retrying...")

        except Exception as e:
            print(f"  ✗ Error (attempt {attempt}): {e}")

    raise RuntimeError(f"Failed to generate {output_filename} after {max_retries} attempts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Document Prompts

# COMMAND ----------

# ── 3a. Bank Statements ───────────────────────────────────────────────────

BANK_STATEMENT_PROMPTS = [
    # Style A — Barclays-like
    """Write Python code to generate a multi-page UK bank statement PDF.

Layout — Barclays-inspired:
- Cyan/dark blue colour scheme. Bank name in white on a coloured header bar.
- Fictional bank name (e.g. "Bluebird Bank", or invent your own).
- Customer name and address top-left. Account details in a shaded box top-right.
- Columns: Date | Description | Money out (£) | Money in (£) | Balance (£)
- Transaction type codes embedded in descriptions: "DD " for direct debit, "SO " for standing order, "BGC " for salary, "FPI " for faster payment in, "CARD " for debit card, "CL " for contactless, "CPT " for cash point.
- Date format: "DD MMM YY" (e.g. "03 Jan 25").
- Opening row: "Balance brought forward". Closing row: "Balance carried forward".
- Summary box at bottom with opening balance, total money in, total money out, closing balance.
- FCA/PRA registration footer.

Content — generate for January 2025:
- Personal current account for a fictional person in Birmingham.
- Salary credit of around £3,000-3,500 on the 28th.
- 5-7 direct debits (utilities, council tax, subscriptions, insurance).
- 1-2 standing orders (mortgage/rent, savings).
- 15-25 card purchases from UK merchants (Tesco, Sainsburys, Lidl, Costa, Amazon, Shell, Boots, etc).
- 2-4 contactless payments (small amounts, coffee shops and convenience stores).
- 1-3 ATM withdrawals (round amounts).
- 1 faster payment received.
- At least 35 transactions total so it spans 2+ pages.
- Include proper page continuation headers on page 2+.
- Running balance must be computed correctly row by row.

Write the PDF to OUTPUT_PATH.""",

    # Style B — HSBC-like
    """Write Python code to generate a multi-page UK bank statement PDF.

Layout — HSBC-inspired:
- Red/white colour scheme. Thin red line at top. Bank name bold and left-aligned.
- Fictional bank name (e.g. "Redwood Bank", or invent your own).
- "Statement of Account" title right-aligned. Statement period below.
- Customer details left, account details right (no box, just aligned text with labels).
- Include sort code, account number, IBAN, BIC, statement number.
- Columns: Date | Type | Description | Paid out (£) | Paid in (£) | Balance (£)
- Separate Type column with codes: DD, SO, BGC, FPI, FPO, DEB, ATM, CL, CHG.
- Date format: "DD MMM".
- "OPENING BALANCE" and "CLOSING BALANCE" labels.
- "Account Summary" box at bottom. FCA footer.

Content — generate for February 2025:
- Business current account for a fictional haulage company in Bristol.
- 5-8 client invoice payments (BACS credits) ranging £800-£12,000.
- HMRC VAT payment and HMRC PAYE payment.
- Vehicle lease, fleet insurance, fuel card charges.
- 3-5 staff payroll payments on the 28th.
- Trade supplier payments.
- Some card purchases (fuel, office supplies).
- At least 30 transactions. Running balance must be correct.

Write the PDF to OUTPUT_PATH.""",

    # Style C — Lloyds-like
    """Write Python code to generate a multi-page UK bank statement PDF.

Layout — Lloyds-inspired:
- Dark green colour scheme with a gold accent line under the header.
- Fictional bank name (e.g. "Greenfield Bank", or invent your own).
- "Your Statement" title. Statement number shown (e.g. "Statement 127").
- Customer details left, account details in a green-shaded box right.
- Columns: Date | Payment type | Details | Paid out (£) | Paid in (£) | Balance (£)
- Full payment type names in their own column: "DIRECT DEBIT", "STANDING ORDER", "BANK GIRO CREDIT", "FASTER PAYMENT", "DEBIT CARD", "CONTACTLESS", "CASH WITHDRAWAL".
- Date format: "DD MMM".
- "BALANCE BROUGHT FORWARD" and "BALANCE CARRIED FORWARD" labels.
- "Summary" box at bottom. FCA footer with gold accent line.

Content — generate for March 2025:
- Personal current account for a fictional person in Leeds.
- Lower income — salary around £1,900-2,200.
- Opening balance under £500.
- Balance should go negative at least twice during the month (arranged overdraft).
- Include an "ARRANGED OD INTEREST" charge near month end.
- Standard mix of direct debits, standing orders, card purchases.
- Include a Universal Credit (DWP UC) payment as additional income mid-month.
- At least 28 transactions. Running balance must be correct.

Write the PDF to OUTPUT_PATH.""",
]

# ── 3b. Supporting Documents ─────────────────────────────────────────────

SUPPORTING_DOCS_PROMPTS = [
    """Write Python code to generate a 4-page PDF containing an employer reference letter \
followed by 3 monthly payslips.

Page 1 — Employer reference letter:
- Fictional company letterhead at top (company name, address, phone, email).
- Dated February 2025. Addressed "To Whom It May Concern".
- Confirms employment of a fictional person: their name, DOB, start date, job title, \
  department, employment type (permanent full-time), gross annual salary (~£38,000-45,000), \
  pay frequency (monthly), pay method (BACS), and that they are not under notice.
- Signed by a fictional HR manager. Employee reference number included.
- Professional formatting, justified text.

Pages 2-4 — Three consecutive monthly payslips (Oct, Nov, Dec 2024):
- Company name header on each. "Payslip — [Month Year]" subtitle.
- Employee details table: name, employee number, NI number (AB 12 34 56 C format), \
  tax code (1257L), department, pay date, pay method, pay period.
- Payments section: Basic Salary. Optionally overtime on one month.
- Deductions section: Income Tax (PAYE ~20%), National Insurance (~12%), \
  Pension (5% employee), Student Loan Plan 2 (9% above £2,274/month) on 2 of 3 months.
- Gross pay, total deductions, net pay — clearly shown. Net pay in large bold font.
- Year-to-date summary: gross, tax, NI.
- Use a blue colour scheme for headers and dividers.

All data fictional. Write the PDF to OUTPUT_PATH.""",

    """Write Python code to generate a 4-page PDF containing an employer reference letter \
followed by 3 monthly payslips.

Page 1 — Employer reference letter:
- Fictional logistics/warehouse company letterhead.
- Confirms a warehouse supervisor, employed since 2020. Salary around £28,000-32,000.
- Part-time employee (30 hours/week). No student loan.
- Professional formatting.

Pages 2-4 — Payslips for Nov 2024, Dec 2024, Jan 2025:
- Include variable overtime on each month (£150-£400 varying).
- Standard UK deductions (tax, NI, pension).
- No student loan for this employee.
- Red/maroon colour scheme for headers.

All data fictional. Write the PDF to OUTPUT_PATH.""",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate All Clean PDFs

# COMMAND ----------

print("=" * 70)
print("PHASE 1: Generating clean PDFs via Claude")
print("=" * 70)

clean_pdfs = []

for i, prompt in enumerate(BANK_STATEMENT_PROMPTS):
    style_label = ["barclays", "hsbc", "lloyds"][i]
    fname = f"bank_statement_style_{style_label}.pdf"
    print(f"\n[{i+1}/{len(BANK_STATEMENT_PROMPTS)}] Bank statement — {style_label} layout")
    path = generate_document(prompt, fname)
    clean_pdfs.append(path)

for i, prompt in enumerate(SUPPORTING_DOCS_PROMPTS):
    fname = f"supporting_docs_{i+1}.pdf"
    print(f"\n[{len(BANK_STATEMENT_PROMPTS)+i+1}/"
          f"{len(BANK_STATEMENT_PROMPTS)+len(SUPPORTING_DOCS_PROMPTS)}] "
          f"Supporting docs set {i+1}")
    path = generate_document(prompt, fname)
    clean_pdfs.append(path)

print(f"\n✅ {len(clean_pdfs)} clean PDFs generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scan Artifact Engine

# COMMAND ----------

def apply_scan_artifacts(img, skew_deg=0, noise_level=0, blur_radius=0,
                         brightness_var=0, edge_shadow=False):
    """Apply scan artifacts to a PIL Image."""
    arr = np.array(img, dtype=np.float32)

    if noise_level > 0:
        arr = np.clip(arr + np.random.normal(0, noise_level, arr.shape), 0, 255)

    if brightness_var > 0:
        h = arr.shape[0]
        grad = np.linspace(1.0, 1.0 - brightness_var, h).reshape(-1, 1)
        if len(arr.shape) == 3:
            grad = np.stack([grad] * arr.shape[2], axis=-1)
        arr = np.clip(arr * grad, 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))

    if edge_shadow:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(int(w * 0.02)):
            a = int(80 * (1 - i / (w * 0.02)))
            draw.rectangle([i, i, w-1-i, h-1-i], outline=(a, a, a))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if abs(skew_deg) > 0.01:
        img = img.rotate(skew_deg, resample=Image.BICUBIC, expand=True,
                         fillcolor=(245, 245, 245))
    return img


SCAN_PROFILES = {
    "slightly_skewed": {
        "desc": "2° skew + light noise + edge shadow",
        "config": {"skew_deg": 2.3, "noise_level": 8, "blur_radius": 0.4,
                   "brightness_var": 0.08, "edge_shadow": True},
    },
    "heavily_rotated": {
        "desc": "6° rotation + heavy noise + blur",
        "config": {"skew_deg": -5.7, "noise_level": 18, "blur_radius": 0.9,
                   "brightness_var": 0.15, "edge_shadow": True},
    },
    "noisy_faded": {
        "desc": "Faded + heavy noise — old/worn document",
        "config": {"skew_deg": 0.6, "noise_level": 28, "blur_radius": 0.3,
                   "brightness_var": 0.22, "edge_shadow": False},
    },
    "upside_down_page": {
        "desc": "Page 2 fed upside down into scanner",
        "config": {"skew_deg": 0.5, "noise_level": 6, "blur_radius": 0.2,
                   "brightness_var": 0.05, "edge_shadow": True},
        "page_overrides": {1: {"skew_deg": 180}},
    },
    "mixed_orientation": {
        "desc": "Each page scanned at a different angle",
        "config": {"skew_deg": 1.2, "noise_level": 10, "blur_radius": 0.3,
                   "brightness_var": 0.06, "edge_shadow": True},
        "page_overrides": {0: {"skew_deg": -1.5}, 1: {"skew_deg": 3.8},
                           2: {"skew_deg": 90}},
    },
}


def create_scan_artifact(source_pdf: str, output_path: str, profile_name: str) -> str:
    """Rasterise a PDF and apply scan artifacts."""
    profile = SCAN_PROFILES[profile_name]
    base_cfg = profile["config"]
    overrides = profile.get("page_overrides", {})

    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(source_pdf, dpi=180)
    except Exception:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(source_pdf)
        writer = PdfWriter()
        for i, page in enumerate(reader.pages):
            rot = int({**base_cfg, **overrides.get(i, {})}.get("skew_deg", 0))
            if abs(rot) >= 45:
                page.rotate(rot)
            writer.add_page(page)
        with open(output_path, "wb") as f:
            writer.write(f)
        return output_path

    processed = []
    for i, page_img in enumerate(pages):
        cfg = {**base_cfg, **overrides.get(i, {})}
        result = apply_scan_artifacts(page_img, **cfg)
        if result.mode == "RGBA":
            result = result.convert("RGB")
        processed.append(result)

    if processed:
        processed[0].save(output_path, "PDF", resolution=150,
                          save_all=True, append_images=processed[1:])
    return output_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Scan Artifacts

# COMMAND ----------

print("=" * 70)
print("PHASE 2: Generating scan-artifact variants")
print("=" * 70)

# Apply each profile to the first bank statement (the richest multi-page doc)
source = clean_pdfs[0]

scan_pdfs = []
for profile_name, profile in SCAN_PROFILES.items():
    base = os.path.splitext(os.path.basename(source))[0]
    out_name = f"SCAN_{profile_name}__{base}.pdf"
    out_path = os.path.join(LOCAL_TMP, out_name)

    print(f"\n  {profile_name}: {profile['desc']}")
    create_scan_artifact(source, out_path, profile_name)

    if os.path.exists(out_path):
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✓ {out_name} ({size_kb:.0f} KB)")
        scan_pdfs.append(out_path)

print(f"\n✅ {len(scan_pdfs)} scan-artifact PDFs generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Copy to DBFS & Summary

# COMMAND ----------

print("=" * 70)
print("Copying to DBFS")
print("=" * 70)

all_files = clean_pdfs + scan_pdfs
for fpath in all_files:
    if os.path.exists(fpath):
        shutil.copy2(fpath, os.path.join(DBFS_OUTPUT, os.path.basename(fpath)))

print(f"\n{'─'*70}")
print(f"{'FILE':55s} {'SIZE':>10s}")
print(f"{'─'*70}")

total_size = 0
for fpath in all_files:
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        total_size += size
        tag = "📄" if "SCAN" not in os.path.basename(fpath) else "🔧"
        print(f"  {tag} {os.path.basename(fpath):52s} {size/1024:8.1f} KB")

print(f"{'─'*70}")
print(f"  {'TOTAL':52s} {total_size/1024:8.1f} KB")
print(f"\n✅ {len(all_files)} files saved to {DBFS_OUTPUT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Generating More Documents
# MAGIC
# MAGIC To create additional documents, just write a new prompt and call `generate_document()`:
# MAGIC
# MAGIC ```python
# MAGIC prompt = """Write Python code to generate a UK utility bill PDF.
# MAGIC - Fictional energy company. Quarterly gas bill.
# MAGIC - Customer name and supply address. MPAN/MPRN numbers.
# MAGIC - Meter readings, usage breakdown, charges, VAT at 5%.
# MAGIC - Payment due date and methods.
# MAGIC Write the PDF to OUTPUT_PATH."""
# MAGIC
# MAGIC generate_document(prompt, "utility_bill_gas.pdf")
# MAGIC ```
# MAGIC
# MAGIC This works for any document type — council tax bills, insurance certificates,
# MAGIC P60s, tenancy agreements, vehicle registration docs. Just describe what it
# MAGIC should look like and what data it should contain.
# MAGIC
# MAGIC ### Scaling
# MAGIC ```python
# MAGIC from concurrent.futures import ThreadPoolExecutor, as_completed
# MAGIC
# MAGIC prompts = [(prompt_text, f"statement_{i}.pdf") for i in range(50)]
# MAGIC
# MAGIC with ThreadPoolExecutor(max_workers=4) as pool:
# MAGIC     futures = {pool.submit(generate_document, p, f): f for p, f in prompts}
# MAGIC     for future in as_completed(futures):
# MAGIC         print(f"Done: {futures[future]}")
# MAGIC ```
