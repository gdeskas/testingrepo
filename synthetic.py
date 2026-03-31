# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Test Document Generator — LLM-Driven (No Pip Install)
# MAGIC **Asset Finance Application Processing — Test Data Pipeline**
# MAGIC
# MAGIC Uses **only libraries pre-installed in Databricks ML Runtime**: `matplotlib`, `Pillow`,
# MAGIC `numpy`, `openai`. No `%pip install` needed.
# MAGIC
# MAGIC ```
# MAGIC Prompt → Claude Sonnet 4.5 → Python code (matplotlib) → exec() → PDF → scan artifacts
# MAGIC ```

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

# ── Databricks auth ──
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
# MAGIC ## 2. Core Generator

# COMMAND ----------

SYSTEM_PROMPT = """\
You are a Python code generator. You write complete, self-contained Python scripts \
that generate realistic UK financial document PDFs.

CRITICAL LIBRARY CONSTRAINTS — you may ONLY use these pre-installed libraries:
- matplotlib (for PDF rendering via matplotlib.backends.backend_pdf.PdfPages)
- matplotlib.pyplot, matplotlib.patches, matplotlib.table
- numpy
- Pillow (PIL)
- random, datetime, decimal, os, math — standard library

You MUST NOT use: reportlab, fpdf, fpdf2, weasyprint, pdfkit, cairo, or any other \
PDF library. Use ONLY matplotlib to create PDF pages.

MATPLOTLIB PDF TECHNIQUE:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(OUTPUT_PATH) as pdf:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 in inches
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw text
    ax.text(x, y, "text", fontsize=10, fontfamily='sans-serif',
            transform=ax.transAxes, verticalalignment='top')

    # Draw rectangles
    rect = plt.Rectangle((x, y), width, height, facecolor='#003D6B',
                          edgecolor='none', transform=ax.transAxes)
    ax.add_patch(rect)

    # Draw lines
    ax.plot([x1, x2], [y1, y2], color='#003D6B', linewidth=0.5,
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # For page 2, create a new figure...
```

IMPORTANT RENDERING NOTES:
- Use ax.transAxes for all coordinates (0,0 = bottom-left, 1,1 = top-right).
- Use verticalalignment='top' for text since y=1.0 is the top of the page.
- For tables: position text in columns using precise x-coordinates.
- Use monospace font ('monospace') for numbers/amounts to ensure alignment.
- For multi-page: create a new fig per page, savefig each, close each.
- Set figure facecolor to 'white'.
- Use tight_layout() or bbox_inches='tight' for clean margins.

CONTENT RULES:
1. Output ONLY valid Python code. No markdown fences. No explanation.
2. All data must be ENTIRELY FICTIONAL — no real people, companies, or accounts.
3. Use fictional bank names — never Barclays, HSBC, Lloyds, NatWest, etc.
4. Sort codes: XX-XX-XX. Account numbers: 8 digits.
5. The running balance must be arithmetically correct, computed row by row.
6. The script must write a PDF to the variable OUTPUT_PATH (injected at runtime).
7. Use random for variation in amounts and names.
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

            # Strip markdown fences
            if code.startswith("```"):
                first_newline = code.index("\n")
                code = code[first_newline + 1:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            # Execute with OUTPUT_PATH injected
            exec_globals = {"OUTPUT_PATH": output_path}
            exec(code, exec_globals)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                size_kb = os.path.getsize(output_path) / 1024
                print(f"  ✓ {output_filename} ({size_kb:.1f} KB)")
                return output_path
            else:
                print(f"  ✗ No file produced or file too small. Retrying...")

        except Exception as e:
            print(f"  ✗ Error (attempt {attempt}): {type(e).__name__}: {e}")

    raise RuntimeError(f"Failed to generate {output_filename} after {max_retries} attempts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Document Prompts

# COMMAND ----------

# ── Bank Statements ──────────────────────────────────────────────────────

BANK_STATEMENT_PROMPTS = [

    # Style A — Barclays-like
    """Write Python code using matplotlib to generate a multi-page UK bank statement PDF.

Layout — Barclays-inspired:
- Cyan/dark blue colour scheme (#003D6B header bar across full page width at top, white text).
- Fictional bank name in the header bar (e.g. "Bluebird Bank"). Page number top-right.
- Below header: "Your statement" title, statement period "01 January 2025 to 31 January 2025".
- Customer name and address on the left. Account details on the right in a light blue (#E8F6FD) shaded box.
- Show sort code (XX-XX-XX), account number (8 digits), IBAN.
- Transaction table columns: Date | Description | Money out (£) | Money in (£) | Balance (£)
- Type codes embedded in description: "DD " prefix for direct debit, "SO " for standing order, "BGC " for salary, "CARD " for debit card, "CL " for contactless, "CPT " for cash withdrawal, "FPI " for faster payment in.
- Date format: "DD MMM YY".
- Alternating light grey (#F5F5F5) row shading.
- Opening row: "Balance brought forward". Closing row: "Balance carried forward" with totals.
- Summary box at bottom in light blue with opening balance, total in, total out, closing balance.
- FCA registration footer at bottom of each page.
- If transactions exceed one page, create a second page with a continuation header.

Content for January 2025:
- Personal current account, fictional person in Birmingham.
- Salary ~£3,200 on the 28th via BGC.
- 6 direct debits: council tax, gas, electric, water, mobile phone, Netflix.
- 1 standing order: mortgage ~£1,050.
- 18+ card purchases from: Tesco, Sainsburys, Lidl, Costa, Amazon.co.uk, Shell, Boots, Pret, Greggs, Deliveroo, Trainline, M&S.
- 3 contactless: small amounts at coffee shops.
- 2 ATM withdrawals (£50, £100).
- 1 faster payment in (£75 from a friend).
- At least 35 transactions total to force 2+ pages.
- Running balance computed correctly from opening balance.

Use monospace font for all amounts. Use 'sans-serif' for text. Write PDF to OUTPUT_PATH.""",

    # Style B — HSBC-like
    """Write Python code using matplotlib to generate a multi-page UK bank statement PDF.

Layout — HSBC-inspired:
- Red (#DB0011) colour scheme. Thin red line at very top of page.
- Bank name large and bold, top-left (fictional, e.g. "Redwood Bank").
- "Statement of Account" right-aligned. Period below it.
- Customer name/address left side. Account details right side (no box, just aligned text).
- Show: Account type, Sort code, Account number, IBAN, BIC, Statement number.
- Transaction table columns: Date | Type | Description | Paid out (£) | Paid in (£) | Balance (£)
- Separate Type column with short codes: DD, SO, BGC, FPI, FPO, DEB, ATM, CL, CHG.
- Date format: "DD MMM".
- "OPENING BALANCE" and "CLOSING BALANCE" rows.
- Alternating row shading. Red header bar for column headers.
- "Account Summary" box with light red (#FFF0F0) background. FCA footer.

Content for February 2025:
- Business current account for a fictional haulage company in Bristol.
- 5 client invoice payments (BACS credits) between £1,500-£9,000.
- HMRC VAT payment (~£3,200). HMRC PAYE (~£1,800).
- Vehicle lease payment (~£850). Fleet insurance (~£420).
- 2 fuel card payments (~£600 each). Tyre supplier payment.
- 4 staff payroll payments on 28th (~£2,000-£2,800 each).
- Card purchases: office supplies, fuel.
- At least 30 transactions. Running balance correct.

Write PDF to OUTPUT_PATH.""",

    # Style C — Lloyds-like
    """Write Python code using matplotlib to generate a multi-page UK bank statement PDF.

Layout — Lloyds-inspired:
- Dark green (#006A4E) colour scheme with a gold (#C5A000) accent line under the header bar.
- Fictional bank name in white on green header bar (e.g. "Greenfield Bank").
- Tagline under name: "By your side". Statement number shown (e.g. "Statement 127").
- "Your Statement" title. Customer details left, account details in green-shaded (#E6F2ED) box right.
- Transaction columns: Date | Payment type | Details | Paid out (£) | Paid in (£) | Balance (£)
- FULL payment type names in their own column: "DIRECT DEBIT", "STANDING ORDER", "BANK GIRO CREDIT", "FASTER PAYMENT", "DEBIT CARD", "CONTACTLESS", "CASH WITHDRAWAL".
- Date format: "DD MMM".
- "BALANCE BROUGHT FORWARD" and "BALANCE CARRIED FORWARD" rows.
- "Summary" box at bottom. FCA footer with gold accent line above.

Content for March 2025:
- Personal current account, fictional person in Leeds.
- Lower income: salary ~£2,050 on the 28th.
- Opening balance: £380.
- Balance goes negative twice during the month (arranged overdraft).
- "ARRANGED OD INTEREST" charge of ~£6.50 near month end.
- Universal Credit payment (DWP UC) of ~£340 mid-month.
- Standard mix of direct debits (5), standing order (rent ~£725), card purchases (12+).
- At least 28 transactions. Running balance correct (can go negative).

Write PDF to OUTPUT_PATH.""",
]

# ── Supporting Documents ─────────────────────────────────────────────────

SUPPORTING_DOCS_PROMPTS = [

    """Write Python code using matplotlib to generate a 4-page PDF: 1 employer letter + 3 payslips.

Page 1 — Employer reference letter:
- Fictional company name and address at top (blue #003D6B text).
- Phone number and email. Date: 3 February 2025.
- "To Whom It May Concern" then "Re: Employment Confirmation — [Name]".
- Body confirms: employee name, DOB, start date, job title (Senior Project Coordinator),
  department (Operations), permanent full-time, gross salary ~£41,000, paid monthly by BACS on 28th.
- States not under disciplinary or notice.
- Signed by fictional HR Manager. Employee reference number.
- Justified text, professional spacing.

Pages 2-4 — Payslips for October, November, December 2024:
- Company name header (same as letter). "Payslip — [Month Year]" subtitle.
- Employee details grid: name, employee number, NI number (AB 12 34 56 C), tax code (1257L), department, pay date, pay period.
- Two-column layout: PAYMENTS on left, DEDUCTIONS on right.
- Payments: Basic Salary only (~£3,485/month).
- Deductions: Income Tax (~£487), National Insurance (~£293), Pension 5% (~£174).
  Include Student Loan Plan 2 (~£109) on Nov and Dec only.
- Show: Gross pay, Total deductions, NET PAY (large/bold).
- Year-to-date section: gross, tax, NI.
- Blue (#003D6B) divider lines between sections.

Write PDF to OUTPUT_PATH.""",

    """Write Python code using matplotlib to generate a 4-page PDF: 1 employer letter + 3 payslips.

Page 1 — Employer reference letter:
- Fictional warehouse/logistics company in Bristol. Maroon (#800020) colour scheme.
- Confirms a Warehouse Supervisor employed since 2020.
- Part-time (30 hours/week). Salary ~£28,500. No student loan.

Pages 2-4 — Payslips for Nov 2024, Dec 2024, Jan 2025:
- Variable overtime each month (£180, £350, £220).
- Standard deductions: tax, NI, pension 5%. No student loan.
- Different colour scheme from first set (use maroon/dark red for headers).
- Year-to-date totals.

Write PDF to OUTPUT_PATH.""",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Clean PDFs

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
    idx = len(BANK_STATEMENT_PROMPTS) + i + 1
    total = len(BANK_STATEMENT_PROMPTS) + len(SUPPORTING_DOCS_PROMPTS)
    print(f"\n[{idx}/{total}] Supporting docs set {i+1}")
    path = generate_document(prompt, fname)
    clean_pdfs.append(path)

print(f"\n✅ {len(clean_pdfs)} clean PDFs generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scan Artifact Engine (Pillow + NumPy only)

# COMMAND ----------

def apply_scan_artifacts(img, skew_deg=0, noise_level=0, blur_radius=0,
                         brightness_var=0, edge_shadow=False):
    """Apply scan artifacts using only Pillow and NumPy."""
    arr = np.array(img, dtype=np.float32)

    # Gaussian noise
    if noise_level > 0:
        arr = np.clip(arr + np.random.normal(0, noise_level, arr.shape), 0, 255)

    # Uneven brightness (simulates scanner lid shadow)
    if brightness_var > 0:
        h = arr.shape[0]
        grad = np.linspace(1.0, 1.0 - brightness_var, h).reshape(-1, 1)
        if len(arr.shape) == 3:
            grad = np.stack([grad] * arr.shape[2], axis=-1)
        arr = np.clip(arr * grad, 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))

    # Edge shadow (dark border from scanner)
    if edge_shadow:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(int(w * 0.02)):
            a = int(80 * (1 - i / (w * 0.02)))
            draw.rectangle([i, i, w-1-i, h-1-i], outline=(a, a, a))

    # Slight blur (defocus)
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Rotation/skew
    if abs(skew_deg) > 0.01:
        img = img.rotate(skew_deg, resample=Image.BICUBIC, expand=True,
                         fillcolor=(245, 245, 245))

    return img


SCAN_PROFILES = {
    "slightly_skewed": {
        "desc": "2° skew, light noise, edge shadow — common scan issue",
        "config": {"skew_deg": 2.3, "noise_level": 8, "blur_radius": 0.4,
                   "brightness_var": 0.08, "edge_shadow": True},
    },
    "heavily_rotated": {
        "desc": "6° rotation, heavy noise, blur — bad flatbed scan",
        "config": {"skew_deg": -5.7, "noise_level": 18, "blur_radius": 0.9,
                   "brightness_var": 0.15, "edge_shadow": True},
    },
    "noisy_faded": {
        "desc": "Faded + heavy noise — old/worn document",
        "config": {"skew_deg": 0.6, "noise_level": 28, "blur_radius": 0.3,
                   "brightness_var": 0.22, "edge_shadow": False},
    },
    "upside_down_page": {
        "desc": "Page 2 fed upside down",
        "config": {"skew_deg": 0.5, "noise_level": 6, "blur_radius": 0.2,
                   "brightness_var": 0.05, "edge_shadow": True},
        "page_overrides": {1: {"skew_deg": 180}},
    },
    "mixed_orientation": {
        "desc": "Each page at a different angle",
        "config": {"skew_deg": 1.2, "noise_level": 10, "blur_radius": 0.3,
                   "brightness_var": 0.06, "edge_shadow": True},
        "page_overrides": {0: {"skew_deg": -1.5}, 1: {"skew_deg": 3.8},
                           2: {"skew_deg": 90}},
    },
}


def pdf_to_images(pdf_path, dpi=180):
    """Convert PDF pages to PIL Images using matplotlib (no pdf2image/poppler)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # matplotlib can read PDFs it produced — re-rasterise via Pillow
    # But matplotlib can't natively read arbitrary PDFs as images.
    # So we use a different approach: render via matplotlib's PdfPages in reverse
    # is not possible. Instead, we rasterise using PIL + a temporary PNG approach.

    # Best portable approach: use matplotlib's pdf backend to re-read isn't supported.
    # Use Pillow's built-in PDF reader (limited but works for matplotlib-generated PDFs).
    try:
        # Pillow can open PDF page 0
        img = Image.open(pdf_path)
        images = []
        page = 0
        while True:
            try:
                img.seek(page)
                # Convert to high-res RGB
                frame = img.copy().convert('RGB')
                # Scale up for quality
                w, h = frame.size
                scale = dpi / 72  # PDFs are 72dpi baseline
                new_w, new_h = int(w * scale), int(h * scale)
                frame = frame.resize((new_w, new_h), Image.LANCZOS)
                images.append(frame)
                page += 1
            except EOFError:
                break
        if images:
            return images
    except Exception:
        pass

    # Final fallback: return None and skip scan artifacts for this file
    return None


def create_scan_artifact(source_pdf: str, output_path: str, profile_name: str) -> str:
    """Rasterise a PDF and apply scan artifacts. Uses only Pillow + NumPy."""
    profile = SCAN_PROFILES[profile_name]
    base_cfg = profile["config"]
    overrides = profile.get("page_overrides", {})

    print(f"    Applying '{profile_name}': {profile['desc']}")

    images = pdf_to_images(source_pdf)

    if not images:
        print(f"    ⚠ Could not rasterise PDF. Skipping scan artifact.")
        return None

    processed = []
    for i, page_img in enumerate(images):
        cfg = {**base_cfg, **overrides.get(i, {})}
        result = apply_scan_artifacts(page_img, **cfg)
        if result.mode == "RGBA":
            result = result.convert("RGB")
        processed.append(result)

    if processed:
        processed[0].save(output_path, "PDF", resolution=dpi,
                          save_all=True, append_images=processed[1:])
        size_kb = os.path.getsize(output_path) / 1024
        print(f"    ✓ {len(processed)} page(s), {size_kb:.0f} KB → {os.path.basename(output_path)}")

    return output_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Scan Artifacts

# COMMAND ----------

print("=" * 70)
print("PHASE 2: Generating scan-artifact variants")
print("=" * 70)

source = clean_pdfs[0]  # Apply all profiles to the first bank statement
dpi = 180

scan_pdfs = []
for profile_name, profile in SCAN_PROFILES.items():
    base = os.path.splitext(os.path.basename(source))[0]
    out_name = f"SCAN_{profile_name}__{base}.pdf"
    out_path = os.path.join(LOCAL_TMP, out_name)

    print(f"\n  {profile_name}:")
    result = create_scan_artifact(source, out_path, profile_name)
    if result:
        scan_pdfs.append(result)

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
    if fpath and os.path.exists(fpath):
        shutil.copy2(fpath, os.path.join(DBFS_OUTPUT, os.path.basename(fpath)))

print(f"\n{'─'*70}")
print(f"  {'FILE':55s} {'SIZE':>10s}")
print(f"{'─'*70}")

total_size = 0
for fpath in all_files:
    if fpath and os.path.exists(fpath):
        size = os.path.getsize(fpath)
        total_size += size
        tag = "📄" if "SCAN" not in os.path.basename(fpath) else "🔧"
        print(f"  {tag} {os.path.basename(fpath):52s} {size/1024:8.1f} KB")

print(f"{'─'*70}")
print(f"  {'TOTAL':52s} {total_size/1024:8.1f} KB")
print(f"\n✅ {len(all_files)} files → {DBFS_OUTPUT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Adding New Document Types
# MAGIC
# MAGIC ```python
# MAGIC # Just write a prompt and call generate_document()
# MAGIC
# MAGIC prompt = """Write Python code using matplotlib to generate a UK council tax bill PDF.
# MAGIC - Fictional local authority name and address.
# MAGIC - Council tax band, annual charge, monthly instalments.
# MAGIC - Property address, account reference number.
# MAGIC - Payment schedule table for the year.
# MAGIC Write PDF to OUTPUT_PATH."""
# MAGIC
# MAGIC generate_document(prompt, "council_tax_bill.pdf")
# MAGIC ```
# MAGIC
# MAGIC Works for any document type: utility bills, P60s, insurance certs,
# MAGIC tenancy agreements, vehicle registration. Just describe what it looks like.
# MAGIC
# MAGIC ### Scaling
# MAGIC ```python
# MAGIC from concurrent.futures import ThreadPoolExecutor, as_completed
# MAGIC
# MAGIC prompts = [(p, f"stmt_{i}.pdf") for i, p in enumerate(BANK_STATEMENT_PROMPTS * 20)]
# MAGIC with ThreadPoolExecutor(max_workers=4) as pool:
# MAGIC     futures = {pool.submit(generate_document, p, f): f for p, f in prompts}
# MAGIC     for fut in as_completed(futures):
# MAGIC         print(f"Done: {futures[fut]}")
# MAGIC ```
