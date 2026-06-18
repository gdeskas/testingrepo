# Databricks notebook source
# MAGIC %md
# MAGIC # Ground-truth comparison — DEV scaffold (v3.1)
# MAGIC
# MAGIC Wires the notebook into the actual Gold-layer `composite_request_outbound` table produced by
# MAGIC `salesforce_composite_request_builder`. The full composite request —
# MAGIC **Account + Contact + Opportunity + OpportunityContactRole** — is loaded from Delta, parsed,
# MAGIC and diffed per object.
# MAGIC
# MAGIC **Mental model — the diff is field-complete; tracking is enrichment-only**
# MAGIC - SOQL returns every field we ask for, regardless of whether Salesforce has field-history
# MAGIC   tracking enabled on it. So **every field we sent is comparable**.
# MAGIC - `FIELDS_WITH_DIRECT_HISTORY` config is *metadata for downstream*, not a gate on this diff. It records
# MAGIC   which fields will additionally have rich change history available via
# MAGIC   `AccountHistory` / `OpportunityFieldHistory` / etc. — the change-type classifier (next
# MAGIC   notebook) uses that to enrich tracked rows with who/when/why.
# MAGIC - The `sf_history_available` column on the diff is therefore an *enrichment hint*, not a
# MAGIC   capability flag.
# MAGIC
# MAGIC **What's in v3.1**
# MAGIC - Section 2 reads from `{catalog}.{gld_schema}.composite_request_outbound`, keyed on `JOB_RUN_ID`.
# MAGIC - The `COMPOSITE_JSON` column is **Python-literal** serialised (`True`/`None`, not `true`/`null`)
# MAGIC   because of the regex post-processing in `salesforce_composite_request_builder`. Parsed with
# MAGIC   `ast.literal_eval`, not `json.loads`.
# MAGIC - All four object types are handled uniformly. The diff produces per-object rows keyed on
# MAGIC   `referenceId`, with a `changed: bool` column for easy filtering.
# MAGIC - A mock for `salesforce_response` is included so the linking layer can run end-to-end before
# MAGIC   real Salesforce write-back data is available.
# MAGIC
# MAGIC **What this notebook still doesn't do**
# MAGIC - Live Salesforce API calls (mock only — Alan's GET pattern goes here when ready).
# MAGIC - Change-type classification — separate notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets, imports, constants
# MAGIC
# MAGIC Same widget pattern as `salesforce_composite_request_builder` for consistency.

# COMMAND ----------

dbutils.widgets.text("catalog", "uc_comm_afl_dev", "catalog")
dbutils.widgets.text("gld_schema", "brkrflw_gld", "gld_schema")
dbutils.widgets.text("slr_schema", "brkrflw_slr", "slr_schema")
dbutils.widgets.text("brz_schema", "brkrflw_brz", "brz_schema")
dbutils.widgets.text("job_run_id", "718466553045729", "job_run_id")
dbutils.widgets.dropdown("data_source", "gold", ["gold", "mock"], "data_source")

# COMMAND ----------

import ast
import copy
import json
from typing import Any

import pandas as pd
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog")
GLD_SCHEMA = dbutils.widgets.get("gld_schema")
SLR_SCHEMA = dbutils.widgets.get("slr_schema")
BRZ_SCHEMA = dbutils.widgets.get("brz_schema")
JOB_RUN_ID = dbutils.widgets.get("job_run_id")
DATA_SOURCE = dbutils.widgets.get("data_source")   # "gold" or "mock"

SF_API_VERSION = "v57.0"                            # matches salesforce_composite_request_builder

print(f"catalog: {CATALOG}, gld_schema: {GLD_SCHEMA}, job_run_id: {JOB_RUN_ID}, data_source: {DATA_SOURCE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the composite request
# MAGIC
# MAGIC In `gold` mode, read `composite_request_outbound` for the given `JOB_RUN_ID`. In `mock` mode,
# MAGIC use a synthetic payload that matches the real structure — useful when no Gold rows exist yet
# MAGIC for the current job run.
# MAGIC
# MAGIC **Parsing note** — `salesforce_composite_request_builder` ends with:
# MAGIC ```python
# MAGIC regexp_replace(COMPOSITE_JSON, '"allOrNone":true',     '"allOrNone":True')
# MAGIC regexp_replace(COMPOSITE_JSON, '"Birthdate":null',     '"Birthdate":None')
# MAGIC ```
# MAGIC so the column is Python-literal, not JSON. Use `ast.literal_eval`.

# COMMAND ----------

MOCK_COMPOSITE_REQUEST = {
    "allOrNone": True,
    "compositeRequest": [
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/Account",
            "referenceId": "refAccount_065afd694939d8cf",
            "body": {
                "RecordTypeId": "012A00000012ABCDEF",
                "Name": "Acme Logistics Ltd",
                "Phone": "02012345678",
                "AccountSource": "Broker",
                "CB_Email__c": "info@acme-logistics.co.uk",
                "CB_Company_Registration_Number__c": "12345678",
                "CB_Customer_Type__c": "Limited Company",
                "CB_Company_Status__c": "Active",
                "BillingStreet": "123 High St",
                "BillingCity": "London",
                "BillingPostalCode": "EC1A 1BB",
                "BillingCountry": "United Kingdom",
                "BillingCountryCode": "GB",
                "ShippingStreet": "123 High St",
                "ShippingCity": "London",
                "ShippingPostalCode": "EC1A 1BB",
                "ShippingCountry": "United Kingdom",
                "ShippingCountryCode": "GB",
            },
        },
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/Contact",
            "referenceId": "refContact_a1b2c3d4",
            "body": {
                "AccountId": "@{refAccount_065afd694939d8cf.id}",
                "FirstName": "John",
                "LastName": "Smith",
                "Email": "john.smith@acme-logistics.co.uk",
                "Phone": "02012345679",
                "Title": "Director",
                "Birthdate": None,    # Python-literal None — see parsing note above
                "LeadSource": "Broker",
            },
        },
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/Opportunity",
            "referenceId": "refOpp",
            "body": {
                "AccountId": "@{refAccount_065afd694939d8cf.id}",
                "RecordTypeId": "012A00000034OPPXYZ",
                "Name": "Acme Logistics Ltd-Excavator-50000-GBP-2026/06/15",
                "StageName": "Submitted",
                "CloseDate": "2026-08-15",
                "Amount": 50000,
                "CurrencyIsoCode": "GBP",
                "CB_Main_Product__c": "Hire Purchase",
                "CB_Term_Months__c": 36,
                "CB_VAT_Deferral_Month__c": 3,
                "CB_Deposit__c": 5000,
                "CB_Partner_Account__c": "001A0000005XYZAB",
            },
        },
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/OpportunityContactRole",
            "referenceId": "refOCR_8f7e6d5c4b3a2918",
            "body": {
                "ContactId": "@{refContact_a1b2c3d4.id}",
                "OpportunityId": "@{refOpp.id}",
                "Role": "Director",
            },
        },
    ],
}

def load_composite_request() -> tuple[dict, str]:
    """Return (composite_request_dict, proposal_id) for the active job_run_id."""
    if DATA_SOURCE == "mock":
        print("Using MOCK composite request.")
        return MOCK_COMPOSITE_REQUEST, "mock_proposal_id"

    print(f"Loading from {CATALOG}.{GLD_SCHEMA}.composite_request_outbound where JOB_RUN_ID='{JOB_RUN_ID}'")
    row = (
        spark.table(f"{CATALOG}.{GLD_SCHEMA}.composite_request_outbound")
        .filter(F.col("JOB_RUN_ID") == JOB_RUN_ID)
        .select("PROPOSAL_ID", "COMPOSITE_JSON")
        .limit(1)
        .collect()
    )
    if not row:
        raise ValueError(
            f"No rows in composite_request_outbound for JOB_RUN_ID={JOB_RUN_ID}. "
            f"Set data_source=mock to develop without Gold data."
        )
    return ast.literal_eval(row[0]["COMPOSITE_JSON"]), row[0]["PROPOSAL_ID"]


COMPOSITE_REQUEST, PROPOSAL_ID = load_composite_request()
print(f"\nLoaded composite request for proposal_id={PROPOSAL_ID}")
print(f"  allOrNone: {COMPOSITE_REQUEST['allOrNone']}")
print(f"  items:     {len(COMPOSITE_REQUEST['compositeRequest'])}")
for item in COMPOSITE_REQUEST["compositeRequest"]:
    obj = item["url"].rsplit("/", 1)[-1]
    print(f"    - {obj:25s} referenceId={item['referenceId']}, {len(item['body'])} fields")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Field-history config — enrichment metadata
# MAGIC
# MAGIC **Important:** these sets do NOT limit what the diff can compare. SOQL returns every field
# MAGIC we select regardless of history-tracking config — so every field we sent will be diffed
# MAGIC against the current SF state in section 8. The sets below are *metadata for the downstream
# MAGIC change-type classifier*, telling it which fields will also have rich change history rows in
# MAGIC `AccountHistory` / `ContactHistory` / `OpportunityFieldHistory` (so it can enrich those rows
# MAGIC with who/when/why data).
# MAGIC
# MAGIC Possible values of the resulting `sf_history_available` column:
# MAGIC - **`direct`** — Salesforce has individual field history on this field
# MAGIC - **`compound`** — covered via parent compound field (e.g. `BillingStreet` via `BillingAddress`); history is at compound granularity
# MAGIC - **`none`** — no field history; classifier sees the diff only
# MAGIC - **`n/a`** — composite-placeholder rows where the field has no real value to track
# MAGIC
# MAGIC Sourced from the FieldDefinition CSVs Mithia shared. `OpportunityContactRole` is a junction
# MAGIC object and has no history config, so all OCR fields land as `none`.

# COMMAND ----------

FIELDS_WITH_DIRECT_HISTORY: dict[str, set[str]] = {
    "Account": {
        "Name", "RecordTypeId", "BillingAddress", "ShippingAddress", "Phone", "Website",
        "Industry", "Description", "OwnerId", "PersonBirthdate", "AccountSource",
        "CB_Company_Registration_Number__c", "CB_Customer_Type__c", "CB_Company_Status__c",
        "CB_Legal_Entity_Name__c",
        # See sf_tracking_gap_analysis.xlsx for the full Account history set.
    },
    "Contact": {
        "AccountId", "Name", "MailingAddress", "Phone", "MobilePhone", "Email", "Title",
        "LeadSource", "Birthdate", "Description", "OwnerId",
        "HasOptedOutOfEmail", "DoNotCall",
        # ... full set in gap analysis
    },
    "Opportunity": {
        "AccountId", "RecordTypeId", "Name", "StageName", "Amount", "CloseDate",
        "NextStep", "OwnerId",
        "CB_Partner_Account__c", "CB_Main_Product__c", "CB_Term_Months__c",
        "CB_VAT_Deferral_Month__c", "CB_Deposit__c",
        # ... full set in gap analysis
    },
    "OpportunityContactRole": set(),    # no field history on junction objects
}

COMPOUND_PARENTS = {
    "BillingStreet": "BillingAddress", "BillingCity": "BillingAddress",
    "BillingPostalCode": "BillingAddress", "BillingCountry": "BillingAddress",
    "BillingCountryCode": "BillingAddress", "BillingState": "BillingAddress",
    "ShippingStreet": "ShippingAddress", "ShippingCity": "ShippingAddress",
    "ShippingPostalCode": "ShippingAddress", "ShippingCountry": "ShippingAddress",
    "ShippingCountryCode": "ShippingAddress", "ShippingState": "ShippingAddress",
    "MailingStreet": "MailingAddress", "MailingCity": "MailingAddress",
    "MailingPostalCode": "MailingAddress", "MailingCountry": "MailingAddress",
    "MailingCountryCode": "MailingAddress", "MailingState": "MailingAddress",
}


def sf_history_available(object_type: str, field: str) -> str:
    """Return enrichment hint for downstream: 'direct' | 'compound' | 'none'."""
    direct = FIELDS_WITH_DIRECT_HISTORY.get(object_type, set())
    if field in direct:
        return "direct"
    parent = COMPOUND_PARENTS.get(field)
    if parent and parent in direct:
        return "compound"
    return "none"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extract per-record bodies
# MAGIC
# MAGIC The composite request has multiple Contacts and multiple OCRs in real data (one per person
# MAGIC party, one per person/role pair). Keyed by `referenceId` so we can pair against the
# MAGIC corresponding row in `salesforce_response`.

# COMMAND ----------

def extract_sent_records(composite_request: dict) -> dict[str, dict[str, Any]]:
    """Return {referenceId: {object_type, body}}."""
    out: dict[str, dict[str, Any]] = {}
    for item in composite_request["compositeRequest"]:
        url_parts = item["url"].rstrip("/").split("/")
        object_type = url_parts[url_parts.index("sobjects") + 1]
        out[item["referenceId"]] = {
            "object_type": object_type,
            "body": copy.deepcopy(item["body"]),
        }
    return out


sent_records = extract_sent_records(COMPOSITE_REQUEST)
sent_by_object: dict[str, list[str]] = {}
for ref_id, rec in sent_records.items():
    sent_by_object.setdefault(rec["object_type"], []).append(ref_id)

print("Sent records grouped by object type:")
for obj, refs in sent_by_object.items():
    print(f"  {obj}: {len(refs)} record(s) — {refs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load `salesforce_response` (real or mock)
# MAGIC
# MAGIC The write-acknowledgement table tells us the actual Salesforce IDs assigned to each
# MAGIC `referenceId`. We need those IDs to query current state in section 6.
# MAGIC
# MAGIC When no real responses exist yet, mock one row per sent record with a plausible SF Id.

# COMMAND ----------

def _fake_sf_id(object_type: str) -> str:
    prefix = {
        "Account": "001", "Opportunity": "006", "Contact": "003",
        "OpportunityContactRole": "00K",
    }.get(object_type, "000")
    return f"{prefix}A0000005{object_type[:4].upper():>04}MOCK"


def load_salesforce_response() -> dict[str, dict[str, Any]]:
    """Return {referenceId: {sf_id, success, http_status, errors}}."""
    if DATA_SOURCE == "mock":
        print("Using MOCK salesforce_response.")
        return {
            ref_id: {
                "sf_id": _fake_sf_id(rec["object_type"]),
                "success": True,
                "http_status": 201,
                "errors": [],
            }
            for ref_id, rec in sent_records.items()
        }

    rows = (
        spark.table(f"{CATALOG}.{GLD_SCHEMA}.salesforce_response")
        .filter(F.col("job_run_id") == JOB_RUN_ID)
        .filter(F.col("proposal_id") == PROPOSAL_ID)
        .select("referenceId", "id", "success", "httpStatusCode", "errors")
        .collect()
    )
    if not rows:
        raise ValueError(
            f"No rows in salesforce_response for proposal_id={PROPOSAL_ID}. "
            f"Either the payload hasn't been POSTed yet, or set data_source=mock."
        )
    return {
        r["referenceId"]: {
            "sf_id": r["id"],
            "success": r["success"],
            "http_status": r["httpStatusCode"],
            "errors": r["errors"] or [],
        }
        for r in rows
    }


sf_write_responses = load_salesforce_response()
failed = [r for r, v in sf_write_responses.items() if not v["success"]]
if failed:
    print(f"  WARNING — {len(failed)} write(s) failed: {failed}")
print(f"Salesforce IDs resolved for {len(sf_write_responses)} referenceId(s).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Mock the SOQL state queries
# MAGIC
# MAGIC For each (object_type, sf_id), produce a SOQL-shaped response with the **current state** of
# MAGIC the record. Apply realistic edits to simulate what the business does during their Salesforce
# MAGIC review. The Δ between (sent body) and (current state body) is the ground-truth signal.
# MAGIC
# MAGIC **Swap this whole section for Alan's GET pattern when access is wired in.** Downstream code
# MAGIC depends only on the response shape.

# COMMAND ----------

# Realistic edits per object type. Field names match what the pipeline sends.
EDITS_BY_OBJECT: dict[str, dict[str, Any]] = {
    "Account": {
        "BillingStreet": "123 High Street",                          # compound-tracked
        "ShippingStreet": "123 High Street",
        "Phone": "+44 20 1234 5678",                                 # tracked
        "CB_Email__c": "info@acmelogistics.co.uk",                   # untracked
    },
    "Contact": {
        "Email": "j.smith@acme-logistics.co.uk",                     # tracked
        "Phone": "+44 20 1234 5680",                                 # tracked
    },
    "Opportunity": {
        "Amount": 52000,                                             # tracked
        "StageName": "Originate",                                    # tracked — note: also the snapshot trigger
    },
    "OpportunityContactRole": {
        "Role": "Director and Guarantor",                            # untracked (OCR has no field history)
    },
}


def mock_soql_response(object_type: str, sf_id: str, sent_body: dict[str, Any]) -> dict[str, Any]:
    """Return a SOQL-shaped response containing one record with current SF state."""
    record = copy.deepcopy(sent_body)

    # Resolve composite-request placeholders to the actual Ids from the write response.
    for k, v in list(record.items()):
        if isinstance(v, str) and v.startswith("@{") and v.endswith(".id}"):
            placeholder_ref = v[2:-4]
            if placeholder_ref in sf_write_responses:
                record[k] = sf_write_responses[placeholder_ref]["sf_id"]

    # Apply business edits.
    for k, v in EDITS_BY_OBJECT.get(object_type, {}).items():
        record[k] = v

    # Wrap with SF metadata.
    return {
        "totalSize": 1,
        "done": True,
        "records": [{
            "attributes": {
                "type": object_type,
                "url": f"/services/data/{SF_API_VERSION}/sobjects/{object_type}/{sf_id}",
            },
            "Id": sf_id,
            "CreatedDate": "2026-06-10T09:14:22.000+0000",
            "CreatedById": "005A0000001USR01",
            "LastModifiedDate": "2026-06-11T15:42:08.000+0000",
            "LastModifiedById": "005A0000001BIZ02",
            **record,
        }],
    }


# Build current-state responses for every sent record.
current_state_responses: dict[str, dict[str, Any]] = {
    ref_id: mock_soql_response(rec["object_type"], sf_write_responses[ref_id]["sf_id"], rec["body"])
    for ref_id, rec in sent_records.items()
    if ref_id in sf_write_responses and sf_write_responses[ref_id]["success"]
}

print(f"Mocked current state for {len(current_state_responses)} record(s).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Unwrap and field-mapping seam
# MAGIC
# MAGIC `KNOWN_FIELD_MAPPINGS` is the seam where sent → received name aliases live. Currently empty
# MAGIC pending pipeline-team confirmation on `CB_Product__c` ↔ `CB_Main_Product__c` (the v2
# MAGIC notebook's open question — and as the screenshots confirm, the real pipeline writes
# MAGIC `CB_Main_Product__c`, so this v3 mock already uses the correct name).

# COMMAND ----------

SF_SYSTEM_FIELDS = {
    "Id", "attributes", "CreatedDate", "CreatedById", "LastModifiedDate",
    "LastModifiedById", "SystemModstamp", "IsDeleted", "OwnerId",
}

KNOWN_FIELD_MAPPINGS: dict[str, dict[str, str]] = {}


def unwrap_soql_response(response: dict, expected: int = 1) -> dict[str, Any]:
    records = response.get("records", [])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} record(s), got {len(records)}")
    return records[0]


def strip_system_fields(record: dict) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in SF_SYSTEM_FIELDS}


def map_sent_to_received_keys(sent_body: dict, object_type: str) -> dict[str, str]:
    overrides = KNOWN_FIELD_MAPPINGS.get(object_type, {})
    return {k: overrides.get(k, k) for k in sent_body}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. The diff — across all four object types
# MAGIC
# MAGIC One row per (referenceId, field). We diff **every field we sent** against the current SF state
# MAGIC retrieved via SOQL — none of this is gated on history-tracking config. The output columns:
# MAGIC
# MAGIC | column | meaning |
# MAGIC |---|---|
# MAGIC | `status` | one of `unchanged` / `changed` / `missing_in_response` / `added_by_sf` / `composite_placeholder` |
# MAGIC | `changed` | bool — `True` iff `status == "changed"`. Use this to filter for ground-truth signal. |
# MAGIC | `sf_history_available` | enrichment hint for downstream: `direct` / `compound` / `none` / `n/a` |
# MAGIC | `before` / `after` | values from sent payload vs current SF state |

# COMMAND ----------

def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and v.startswith("@{") and v.endswith("}")


def diff_record(
    sent_body: dict[str, Any],
    received_body: dict[str, Any],
    object_type: str,
    ref_id: str,
    sf_id: str,
) -> list[dict[str, Any]]:
    mapping = map_sent_to_received_keys(sent_body, object_type)
    rows: list[dict[str, Any]] = []
    received_keys_referenced = set(mapping.values())

    def _row(field, before, after, status, history_hint):
        return {
            "proposal_id": PROPOSAL_ID, "ref_id": ref_id, "object_type": object_type,
            "sf_id": sf_id, "field": field,
            "before": before, "after": after,
            "status": status,
            "changed": status == "changed",
            "sf_history_available": history_hint,
        }

    for sent_key, recv_key in mapping.items():
        sent_val = sent_body[sent_key]
        hint = sf_history_available(object_type, sent_key)

        if _is_placeholder(sent_val):
            rows.append(_row(sent_key, sent_val, received_body.get(recv_key),
                             "composite_placeholder", "n/a"))
            continue

        if recv_key not in received_body:
            rows.append(_row(sent_key, sent_val, None, "missing_in_response", hint))
            continue

        recv_val = received_body[recv_key]
        rows.append(_row(sent_key, sent_val, recv_val,
                         "unchanged" if sent_val == recv_val else "changed", hint))

    for recv_key in received_body:
        if recv_key in received_keys_referenced:
            continue
        rows.append(_row(recv_key, None, received_body[recv_key],
                         "added_by_sf", sf_history_available(object_type, recv_key)))

    return rows


all_rows: list[dict[str, Any]] = []
for ref_id, sent in sent_records.items():
    if ref_id not in current_state_responses:
        print(f"  skipping {ref_id} ({sent['object_type']}) — no current state response (write may have failed)")
        continue
    raw = unwrap_soql_response(current_state_responses[ref_id])
    received_body = strip_system_fields(raw)
    all_rows.extend(diff_record(
        sent_body=sent["body"],
        received_body=received_body,
        object_type=sent["object_type"],
        ref_id=ref_id,
        sf_id=raw["Id"],
    ))

diff_df = pd.DataFrame(all_rows)
diff_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary views

# COMMAND ----------

# Status × object — overall shape of the diff.
print("=== STATUS x OBJECT ===")
print(diff_df.groupby(["object_type", "status"], dropna=False).size().unstack(fill_value=0))

# COMMAND ----------

# For the changed rows, how many will have rich history available downstream?
# This is enrichment-hint info, not a filter on what's comparable.
print("=== CHANGED rows: downstream enrichment availability ===")
print(
    diff_df[diff_df["changed"]]
    .groupby(["object_type", "sf_history_available"], dropna=False)
    .size()
    .unstack(fill_value=0)
)

# COMMAND ----------

# Just the changed rows — filter on the boolean for cleanliness.
changes_only = (
    diff_df[diff_df["changed"]]
    [["object_type", "ref_id", "field", "before", "after", "sf_history_available"]]
    .reset_index(drop=True)
)
changes_only

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. What still needs swapping for production
# MAGIC
# MAGIC | Section | Mock today | Real once |
# MAGIC |---|---|---|
# MAGIC | 2 — composite request | `data_source=mock` or real Gold read | Always real |
# MAGIC | 5 — write response | mocked SF IDs | `salesforce_response` real rows |
# MAGIC | 6 — current state | `mock_soql_response` w/ planted edits | Alan's GET pattern: SOQL via REST API |
# MAGIC | 3 — `FIELDS_WITH_DIRECT_HISTORY` | abridged seed; full list in gap analysis | freeze when Mithia confirms tracking adds |
# MAGIC
# MAGIC **A note on what SOQL to issue in section 6 when we wire in real data.** The query should
# MAGIC select every field present in the sent body for that object — that's how we get apples-to-
# MAGIC apples comparison. Worth also pulling `Id`, `LastModifiedDate`, `LastModifiedById` so the
# MAGIC change-type classifier can use the timestamps and user IDs for who/when reasoning even on
# MAGIC fields that don't have direct history.
# MAGIC
# MAGIC **Adjacent work, not in this notebook**
# MAGIC - **Change-type classifier.** Consumes `diff_df` and uses `sf_history_available` to decide
# MAGIC   which rows to enrich with `AccountHistory` / `ContactHistory` / `OpportunityFieldHistory`.
# MAGIC   `direct` rows get who/when/why metadata; `compound` rows need compound-blob parsing;
# MAGIC   `none` rows fall back to `LastModifiedBy*` timestamps from the record itself.
# MAGIC - **OpportunityContactRole.** All OCR diffs land as `sf_history_available='none'` because
# MAGIC   junction objects can't have field history in Salesforce. If we need rich history on role
# MAGIC   assignments, fall back to Mithia's custom JSON snapshot pattern at stage transition.
# MAGIC - **Persist `diff_df` to Delta.** A `composite_request_groundtruth_diff` table in `gld` keyed on
# MAGIC   `(proposal_id, job_run_id, ref_id, field)` would mirror the existing Gold table conventions.
