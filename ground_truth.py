# Databricks notebook source
# MAGIC %md
# MAGIC # ground_truth
# MAGIC
# MAGIC Compares what was **sent** in the composite request against the **current SF state**
# MAGIC retrieved after the write, producing a field-level diff across all four object types.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Widgets, imports, constants

# COMMAND ----------

dbutils.widgets.text("catalog", "uc_comm_afl_dev", "catalog")
dbutils.widgets.text("gld_schema", "brkrflw_gld", "gld_schema")
dbutils.widgets.text("slr_schema", "brkrflw_slr", "slr_schema")
dbutils.widgets.text("brz_schema", "brkrflw_brz", "brz_schema")
dbutils.widgets.text("job_run_id", "718466553045729", "job_run_id")
dbutils.widgets.dropdown("data_source", "gold", ["gold", "mock"], "data_source")
dbutils.widgets.text("sf_server", "preprod.sandbox.my.salesforce.com", "sf_server")

# COMMAND ----------

import ast
import copy
import json
import requests
from typing import Any

import pandas as pd
from pyspark.sql import functions as F
from simple_salesforce import Salesforce

CATALOG     = dbutils.widgets.get("catalog")
GLD_SCHEMA  = dbutils.widgets.get("gld_schema")
SLR_SCHEMA  = dbutils.widgets.get("slr_schema")
BRZ_SCHEMA  = dbutils.widgets.get("brz_schema")
JOB_RUN_ID  = dbutils.widgets.get("job_run_id")
DATA_SOURCE = dbutils.widgets.get("data_source")   # "gold" or "mock"
SF_SERVER   = dbutils.widgets.get("sf_server")

SF_API_VERSION = "v57.0"   # matches salesforce_composite_request_builder

print(f"catalog: {CATALOG}, gld_schema: {GLD_SCHEMA}, job_run_id: {JOB_RUN_ID}, "
      f"data_source: {DATA_SOURCE}, sf_server: {SF_SERVER}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1b. Salesforce authentication
# MAGIC
# MAGIC Credentials are stored in Azure Key Vault–backed Databricks secrets.

# COMMAND ----------

key    = dbutils.secrets.get(scope="ss_comm_afl_brkrflw", key="DATABRICKS-SALESFORCE-CONSUMER-KEY")
secret = dbutils.secrets.get(scope="ss_comm_afl_brkrflw", key="DATABRICKS-SALESFORCE-CONSUMER-SECRET")

session = requests.Session()
# SF_SERVER widget is the full host, e.g. "preprod.sandbox.my.salesforce.com"
# simple_salesforce wants just the domain portion before ".salesforce.com"
domain = SF_SERVER.replace(".salesforce.com", "").replace(".my", ".my")

sf = Salesforce(
    consumer_key=key,
    consumer_secret=secret,
    domain=domain,
    session=session,
)
sf.session.timeout = 60
print(sf)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load the composite request
# MAGIC
# MAGIC In **gold** mode, read `composite_request_outbound` for the given `JOB_RUN_ID`.
# MAGIC In **mock** mode, use a synthetic payload that matches the real structure — useful
# MAGIC when no Gold rows exist yet for the current job run.
# MAGIC
# MAGIC **Parsing note** — `salesforce_composite_request_builder` ends with:
# MAGIC ```
# MAGIC regexp_replace(COMPOSITE_JSON, '"allOrNone":true',  '"allOrNone":True')
# MAGIC regexp_replace(COMPOSITE_JSON, '"Birthdate":null', '"Birthdate":None')
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
                "Birthdate": None,   # Python-literal None — see parsing note above
                "LeadSource": "Broker",
            },
        },
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/Opportunity",
            "referenceId": "refOpp",
            "body": {
                "AccountId": "@{refAccount_065afd694939d8cf.id}",
                "RecordTypeId": "012A000000340PPXYZ",
                "Name": "Acme Logistics Ltd-Excavator-50000-GBP-2026/06/15",
                "StageName": "Submitted",
                "CloseDate": "2026-08-15",
                "Amount": 50000,
                "CurrencyIsoCode": "GBP",
                "CB_Main_Product__c": "Hire Purchase",
                "CB_Term_Months__c": 36,
                "CB_VAT_Deferral_Month__c": 3,
                "CB_Deposit__c": 5000,
                "CB_Partner_Account__c": "001A0000005XYZ4B",
            },
        },
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/OpportunityContactRole",
            "referenceId": "refOCR_B7e6d5c4b3a2918",
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

    print(f"Loading from {CATALOG}.{GLD_SCHEMA}.composite_request_outbound "
          f"where JOB_RUN_ID='{JOB_RUN_ID}'")
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
            "Set data_source=mock to develop without Gold data."
        )
    return ast.literal_eval(row[0]["COMPOSITE_JSON"]), row[0]["PROPOSAL_ID"]


COMPOSITE_REQUEST, PROPOSAL_ID = load_composite_request()
print(f"\nLoaded composite request for proposal_id={PROPOSAL_ID}")
print(f"  allOrNone: {COMPOSITE_REQUEST['allOrNone']}")
print(f"  items:     {len(COMPOSITE_REQUEST['compositeRequest'])}")
for item in COMPOSITE_REQUEST["compositeRequest"]:
    obj = item["url"].rsplit("/", 1)[-1]
    print(f"    {obj:<30} referenceId={item['referenceId']}, "
          f"{len(item['body'])} fields")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Extract per-record bodies
# MAGIC
# MAGIC The composite request has multiple Contacts and multiple OCRs in real data
# MAGIC (one per person party, one per person/role pair). Keyed by `referenceId` so we
# MAGIC can pair against the corresponding row in `salesforce_response`.

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
# MAGIC ## 4. POST composite request to Salesforce and capture write responses
# MAGIC
# MAGIC Replaces the previous mock section. POSTs the composite payload to the real
# MAGIC Salesforce composite REST endpoint and returns `{referenceId: {sf_id, success,
# MAGIC http_status, errors}}`.

# COMMAND ----------

def call_salesforce_composite(composite_request: dict) -> dict:
    """POST the composite request to Salesforce and return the raw response dict."""
    try:
        result = sf.restful(
            "composite/",
            method="POST",
            data=json.dumps(composite_request),
        )
    except requests.exceptions.Timeout as e:
        raise Exception(f"Salesforce session timed out: {str(e)}")
    except (requests.exceptions.RequestException, KeyError) as e:
        raise Exception(f"Salesforce request failed: {str(e)}")

    # Fail fast if any sub-request returned an error status
    errors = []
    for r in result.get("compositeResponse", []):
        if r.get("httpStatusCode", 0) >= 400:
            body = r.get("body", [])
            if isinstance(body, dict):
                body = body.get("errors", [body])
            for e in body:
                if isinstance(e, dict):
                    errors.append(
                        f"{r.get('referenceId')} [{r.get('httpStatusCode')}]: "
                        f"{e.get('errorCode')}: {e.get('message')}"
                    )
                else:
                    errors.append(
                        f"{r.get('referenceId')} [{r.get('httpStatusCode')}]: {str(e)}"
                    )

    if errors:
        error_message = "\n".join(errors)
        raise Exception(f"Salesforce composite API failed:\n{error_message}")

    return result


def load_salesforce_response() -> dict[str, dict[str, Any]]:
    """POST composite request to real Salesforce and return
    {referenceId: {sf_id, success, http_status, errors}}."""
    print("POSTing composite request to Salesforce...")
    result = call_salesforce_composite(COMPOSITE_REQUEST)
    print(f"Received {len(result.get('compositeResponse', []))} sub-responses.")

    responses = {}
    for r in result.get("compositeResponse", []):
        body = r.get("body") or {}
        responses[r["referenceId"]] = {
            "sf_id":        body.get("id") if isinstance(body, dict) else None,
            "success":      body.get("success", False) if isinstance(body, dict) else False,
            "http_status":  r.get("httpStatusCode"),
            "errors":       body.get("errors", []) if isinstance(body, dict) else [],
        }
    return responses


sf_write_responses = load_salesforce_response()

failed = [r for r, v in sf_write_responses.items() if not v["success"]]
if failed:
    print(f"  WARNING — {len(failed)} write(s) failed: {failed}")
print(f"Salesforce IDs resolved for {len(sf_write_responses)} referenceId(s).")
for ref_id, v in sf_write_responses.items():
    print(f"  {ref_id} -> {v['sf_id']} (HTTP {v['http_status']}, success={v['success']})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Fetch current SF state via GET
# MAGIC
# MAGIC Replaces the SOQL mock. For every successfully written record, GETs the current
# MAGIC state from Salesforce using the SF ID returned by the composite write.
# MAGIC The response shape matches what the previous `mock_soql_response` produced so
# MAGIC all downstream diff logic is unchanged.

# COMMAND ----------

def get_sf_current_state(object_type: str, sf_id: str) -> dict[str, Any]:
    """GET current field values for a Salesforce record."""
    try:
        raw = sf.restful(
            f"sobjects/{object_type}/{sf_id}",
            method="GET",
        )
    except Exception as e:
        raise Exception(
            f"Failed to GET {object_type}/{sf_id} from Salesforce: {str(e)}"
        )
    return raw


# Build current_state_responses in the same shape as the old mock
current_state_responses: dict[str, dict[str, Any]] = {}
for ref_id, rec in sent_records.items():
    if ref_id not in sf_write_responses:
        print(f"  skipping {ref_id} — not in write responses")
        continue
    write_resp = sf_write_responses[ref_id]
    if not write_resp["success"]:
        print(f"  skipping {ref_id} — write was unsuccessful")
        continue

    sf_id = write_resp["sf_id"]
    raw   = get_sf_current_state(rec["object_type"], sf_id)

    current_state_responses[ref_id] = {
        "totalSize": 1,
        "done": True,
        "records": [{
            "attributes": {
                "type": rec["object_type"],
                "url": f"/services/data/{SF_API_VERSION}/sobjects/{rec['object_type']}/{sf_id}",
            },
            "Id": sf_id,
            **raw,
        }],
    }

print(f"Fetched current state for {len(current_state_responses)} record(s).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Unwrap and field-mapping seam
# MAGIC
# MAGIC `KNOWN_FIELD_MAPPINGS` is the seam where sent → received name aliases live.
# MAGIC Currently empty pending pipeline-team confirmation on
# MAGIC `CB_Product__c ↔ CB_Main_Product__c` (the v2 notebook's open question —
# MAGIC and as the screenshots confirm, the real pipeline writes `CB_Main_Product__c`,
# MAGIC so this v3 mock already uses the correct name).

# COMMAND ----------

SF_SYSTEM_FIELDS = {
    "Id", "attributes", "CreatedDate", "CreatedById",
    "LastModifiedDate", "LastModifiedById", "SystemModstamp",
    "IsDeleted", "OwnerId",
}

KNOWN_FIELD_MAPPINGS: dict[str, dict[str, str]] = {}


def unwrap_soql_response(response: dict, expected: int = 1) -> dict[str, Any]:
    records = response.get("records", [])
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} record(s), got {len(records)}"
        )
    return records[0]


def strip_system_fields(record: dict) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in SF_SYSTEM_FIELDS}


def map_sent_to_received_keys(sent_body: dict, object_type: str) -> dict[str, str]:
    overrides = KNOWN_FIELD_MAPPINGS.get(object_type, {})
    return {k: overrides.get(k, k) for k in sent_body}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. The diff — across all four object types
# MAGIC
# MAGIC One row per (referenceId, field). We diff **every field we sent** against the
# MAGIC current SF state retrieved via GET — none of this is gated on history-tracking
# MAGIC config. The output columns:
# MAGIC
# MAGIC | column | meaning |
# MAGIC |--------|---------|
# MAGIC | status | one of `unchanged / changed / missing_in_response / added_by_sf / composite_placeholder` |
# MAGIC | changed | bool — True iff status == "changed". Use this to filter for ground-truth signal. |
# MAGIC | before / after | values from sent payload vs current SF state |

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

    def _row(field, before, after, status):
        return {
            "proposal_id": PROPOSAL_ID,
            "ref_id": ref_id,
            "object_type": object_type,
            "sf_id": sf_id,
            "field": field,
            "before": before,
            "after": after,
            "status": status,
            "changed": status == "changed",
        }

    for sent_key, recv_key in mapping.items():
        sent_val = sent_body[sent_key]

        if _is_placeholder(sent_val):
            rows.append(_row(sent_key, sent_val,
                             received_body.get(recv_key), "composite_placeholder"))
            continue

        if recv_key not in received_body:
            rows.append(_row(sent_key, sent_val, None, "missing_in_response"))
            continue

        recv_val = received_body[recv_key]
        rows.append(_row(
            sent_key, sent_val, recv_val,
            "unchanged" if sent_val == recv_val else "changed",
        ))

    for recv_key in received_body:
        if recv_key in received_keys_referenced:
            continue
        rows.append(_row(recv_key, None, received_body[recv_key], "added_by_sf"))

    return rows


all_rows: list[dict[str, Any]] = []
for ref_id, sent in sent_records.items():
    if ref_id not in current_state_responses:
        print(f"  skipping {ref_id} ({sent['object_type']}) — no current state response "
              "(write may have failed)")
        continue

    raw           = unwrap_soql_response(current_state_responses[ref_id])
    received_body = strip_system_fields(raw)
    sf_id         = sf_write_responses[ref_id]["sf_id"]

    all_rows.extend(diff_record(
        sent_body=sent["body"],
        received_body=received_body,
        object_type=sent["object_type"],
        ref_id=ref_id,
        sf_id=sf_id,
    ))

diff_df = pd.DataFrame(all_rows)
diff_df

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Summary views

# COMMAND ----------

# Status x object — overall shape of the diff
print("=== STATUS x OBJECT ===")
print(
    diff_df.groupby(["object_type", "status"], dropna=False)
    .size()
    .unstack(fill_value=0)
)

# COMMAND ----------

# Just the changed rows — filter on the boolean for cleanliness
changes_only = (
    diff_df[diff_df["changed"]]
    [["object_type", "ref_id", "field", "before", "after"]]
    .reset_index(drop=True)
)
changes_only
