# Databricks notebook source
# MAGIC %md
# MAGIC # Ground-truth comparison — DEV scaffold (v2)
# MAGIC
# MAGIC Builds the online ground-truth diff end-to-end against mock data.
# MAGIC When real Salesforce access is wired in (Alan's GET pattern + Mithia's snapshot/field-history),
# MAGIC only the `mock_salesforce_response` function needs to be swapped for the real API call.
# MAGIC
# MAGIC **What changed in v2** (after the call with Mithia and the FieldDefinition gap analysis)
# MAGIC - Per-object **tracked-field config** added — each diff row is now annotated with whether
# MAGIC   Salesforce has change-history tracking on that field. The change-type classifier (next
# MAGIC   notebook) needs this to know which fields it can query `AccountHistory` /
# MAGIC   `OpportunityFieldHistory` against.
# MAGIC - Known **field-name mismatches** between what the pipeline sends and what Salesforce stores
# MAGIC   are surfaced explicitly. The biggest open one is `CB_Product__c` (sent) vs
# MAGIC   `CB_Main_Product__c` (tracked) — flagged as a TODO rather than silently mapped.
# MAGIC - Compound address fields (`BillingAddress`, `ShippingAddress`) noted as the
# MAGIC   tracking unit, even though we send and diff at component level.
# MAGIC - Summary breaks down changes by tracking status, not just status.
# MAGIC
# MAGIC **What this notebook still doesn't do** — change-type classification (correction vs
# MAGIC business-update vs format-normalisation) and any actual SF API calls; both belong elsewhere.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and constants

# COMMAND ----------

import copy
import json
from typing import Any

import pandas as pd

SF_API_VERSION = "v60.0"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The composite request we send to Salesforce
# MAGIC
# MAGIC Mirrors the structure of the JSON the pipeline POSTs to
# MAGIC `/services/data/v60.0/composite/`. Account is created first; Opportunity is second with a
# MAGIC `@{newAccount.id}` forward reference.
# MAGIC
# MAGIC The field list below was reconstructed from an OCR'd reference; cross-checked against
# MAGIC the FieldDefinition CSVs Mithia shared but a couple of names still need pipeline-side
# MAGIC confirmation (see the TODO under "field-name mismatches" further down).

# COMMAND ----------

ACCOUNT_BODY = {
    "RecordTypeId": "012A00000012ABCDEF",
    "AccountSource": "Broker",
    "Name": "Acme Logistics Ltd",
    "Phone": "02012345678",
    "clcommon__Legal_Entity_Type__c": "Private Limited Company",
    "CB_Email__c": "info@acme-logistics.co.uk",
    "CB_Company_Registration_Number__c": "12345678",
    "CB_Customer_Type__c": "Limited Company",
    "CB_Company_Status__c": "Active",
    "BillingStreet": "123 High St",
    "BillingCity": "London",
    "BillingPostalCode": "EC1A 1BB",
    "BillingCountry": "United Kingdom",
    "BillingCountryCode": "GB",
    "CB_RegisteredAddressStreet__c": "123 High St",
    "CB_RegisteredAddressCity__c": "London",
    "CB_RegisteredAddressPostalCode__c": "EC1A 1BB",
    "ShippingStreet": "123 High St",
    "ShippingCity": "London",
    "ShippingPostalCode": "EC1A 1BB",
    "ShippingCountry": "United Kingdom",
    "ShippingCountryCode": "GB",
}

OPPORTUNITY_BODY = {
    "AccountId": "@{newAccount.id}",
    "LeadSource": "Broker",
    "CB_Partner_Account__c": "001A0000005XYZAB",
    "CurrencyIsoCode": "GBP",
    "CloseDate": "2026-08-15",
    "Amount": 50000,
    # TODO: confirm with the pipeline team — Mithia's tracking list has CB_Main_Product__c,
    # not CB_Product__c. Either (a) the pipeline is writing to a deprecated/wrong field,
    # (b) these are the same field under different names, or (c) they're genuinely different
    # and one of them is unused. Until confirmed, see KNOWN_FIELD_MAPPINGS below for the
    # candidate alias.
    "CB_Product__c": "Hire Purchase",
    "CB_Regulation__c": "Unregulated",
    "CB_Term_Months__c": 36,
    "CB_VAT_Deferral_Month__c": 3,
    "CB_Deposit__c": 5000,
    "CB_Underwriting_Decision_required_by__c": "2026-07-01",
    "CB_Broker_Email_Received__c": "broker@finance.co.uk",
    # Still unclear whether these are Boolean__c fields, a picklist value on Type, or
    # something else entirely. They appeared at the end of the OCR'd source list with
    # no surrounding context — confirm with the SF engineer.
    "inbound": True,
    "outbound": False,
}

COMPOSITE_REQUEST = {
    "allOrNone": True,
    "compositeRequest": [
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/Account",
            "referenceId": "newAccount",
            "body": ACCOUNT_BODY,
        },
        {
            "method": "POST",
            "url": f"/services/data/{SF_API_VERSION}/sobjects/Opportunity",
            "referenceId": "newOpportunity",
            "body": OPPORTUNITY_BODY,
        },
    ],
}

print(json.dumps(COMPOSITE_REQUEST, indent=2, default=str)[:600], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Field-tracking config (NEW in v2)
# MAGIC
# MAGIC From the FieldDefinition CSVs Mithia shared. Each set is the Salesforce-side fields with
# MAGIC `enableHistoryTracking = true`. Used downstream to annotate the diff with `tracking_status`,
# MAGIC which determines what the change-type classifier can ask of `AccountHistory` /
# MAGIC `OpportunityFieldHistory`.
# MAGIC
# MAGIC **Compound addresses** — `BillingAddress` / `ShippingAddress` / `MailingAddress` are tracked
# MAGIC as single compound fields. Their components (Street, City, PostalCode, Country, CountryCode)
# MAGIC are covered, but history is recorded at compound level: when the business edits BillingCity,
# MAGIC `AccountHistory` will show one row for `BillingAddress` with a before/after blob.
# MAGIC
# MAGIC **Keep this list in sync** with what's actually enabled in Salesforce. The gap analysis
# MAGIC spreadsheet lists the gaps; if Mithia enables tracking on additional fields, add them here.

# COMMAND ----------

TRACKED_FIELDS = {
    "Account": {
        # Standard
        "Name", "RecordTypeId", "BillingAddress", "ShippingAddress", "Phone", "Website",
        "Industry", "Description", "OwnerId", "PersonBirthdate", "AccountSource",
        # Custom (CB_*, CL_*, etc.) — abridged for the fields we send; full list in gap analysis
        "CB_Company_Registration_Number__c", "CB_Customer_Type__c", "CB_Company_Status__c",
        "CB_Legal_Entity_Name__c",
        # ... see sf_tracking_gap_analysis.xlsx for the full set
    },
    "Opportunity": {
        "AccountId", "RecordTypeId", "Name", "StageName", "Amount", "CloseDate",
        "NextStep", "OwnerId",
        "CB_Partner_Account__c", "CB_Main_Product__c", "CB_Term_Months__c",
        "CB_VAT_Deferral_Month__c", "CB_Deposit__c",
        # ... see sf_tracking_gap_analysis.xlsx for the full set
    },
    "Contact": set(),  # tracked extensively in SF but the pipeline doesn't currently write Contacts
}

# Compound-address parents: when sending the component but querying tracking, we should treat
# the component as "tracked via compound" rather than untracked.
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


def tracking_status(object_type: str, field: str) -> str:
    """Return 'tracked' / 'compound' / 'untracked' for a given (object, field)."""
    tracked = TRACKED_FIELDS.get(object_type, set())
    if field in tracked:
        return "tracked"
    parent = COMPOUND_PARENTS.get(field)
    if parent and parent in tracked:
        return "compound"
    return "untracked"


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Helper — extract per-record bodies from the composite request

# COMMAND ----------

def extract_sent_records(composite_request: dict) -> dict[str, dict[str, Any]]:
    """Pull each compositeRequest item out into {referenceId: {object_type, body}}."""
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
for ref_id, rec in sent_records.items():
    print(f"  {ref_id} ({rec['object_type']}): {len(rec['body'])} fields")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Mock the Salesforce response
# MAGIC
# MAGIC SOQL response shape:
# MAGIC
# MAGIC ```json
# MAGIC {"totalSize": 1, "done": true, "records": [
# MAGIC   {"attributes": {...}, "Id": "...", "<field>": "<value>", ...}
# MAGIC ]}
# MAGIC ```
# MAGIC
# MAGIC The mock applies realistic edits and adds standard SF system fields.
# MAGIC
# MAGIC **Replace this function with a real SOQL call when DEV access is wired in.** The
# MAGIC downstream code only depends on the response shape.

# COMMAND ----------

# Edits that simulate what the business does during their Salesforce review.
# At least one edit per tracking status — handy for end-to-end testing.
ACCOUNT_EDITS = {
    "BillingStreet": "123 High Street",                     # compound-tracked: history at BillingAddress level
    "ShippingStreet": "123 High Street",                    # compound-tracked
    "CB_RegisteredAddressStreet__c": "123 High Street",     # UNTRACKED — diff visible, no history
    "Phone": "+44 20 1234 5678",                            # tracked
    "CB_Email__c": "info@acmelogistics.co.uk",              # UNTRACKED
}

OPPORTUNITY_EDITS = {
    "Amount": 52000,                                                # tracked
    "CB_Broker_Email_Received__c": "broker.smith@finance.co.uk",    # UNTRACKED
}


def _fake_sf_id(object_type: str) -> str:
    prefix = {"Account": "001", "Opportunity": "006", "Contact": "003"}.get(object_type, "000")
    return f"{prefix}A0000005MOCK{object_type[:2].upper()}1"


def mock_salesforce_response(
    object_type: str,
    sent_body: dict[str, Any],
    edits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a SOQL-shaped response containing one record that has been 'edited' in SF."""
    edits = edits or {}
    sf_id = _fake_sf_id(object_type)

    record = copy.deepcopy(sent_body)
    if record.get("AccountId", "").startswith("@{"):
        record["AccountId"] = _fake_sf_id("Account")

    for k, v in edits.items():
        record[k] = v

    record = {
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
    }

    return {"totalSize": 1, "done": True, "records": [record]}


mock_responses = {
    "newAccount": mock_salesforce_response("Account", ACCOUNT_BODY, ACCOUNT_EDITS),
    "newOpportunity": mock_salesforce_response("Opportunity", OPPORTUNITY_BODY, OPPORTUNITY_EDITS),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Unwrap responses → flat records

# COMMAND ----------

SF_SYSTEM_FIELDS = {
    "Id", "attributes", "CreatedDate", "CreatedById", "LastModifiedDate",
    "LastModifiedById", "SystemModstamp", "IsDeleted", "OwnerId",
}


def unwrap_soql_response(response: dict, expected_records: int = 1) -> dict[str, Any]:
    records = response.get("records", [])
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} record(s), got {len(records)}")
    return records[0]


def strip_system_fields(record: dict, keep_id: bool = True) -> dict[str, Any]:
    keep = SF_SYSTEM_FIELDS - {"Id"} if keep_id else SF_SYSTEM_FIELDS
    return {k: v for k, v in record.items() if k not in keep}


received_records: dict[str, dict[str, Any]] = {}
for ref_id, resp in mock_responses.items():
    raw = unwrap_soql_response(resp)
    received_records[ref_id] = {
        "object_type": raw["attributes"]["type"],
        "sf_id": raw["Id"],
        "body": strip_system_fields(raw, keep_id=False),
    }
    print(f"  {ref_id} ({received_records[ref_id]['object_type']}): "
          f"{len(received_records[ref_id]['body'])} fields, Id={received_records[ref_id]['sf_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Field-name mapping (sent ↔ received)
# MAGIC
# MAGIC Most fields are identity-mapped. Two known mismatches need pipeline-side confirmation:
# MAGIC
# MAGIC | Sent (pipeline)       | Salesforce-side candidate | Decision needed                                  |
# MAGIC |-----------------------|---------------------------|--------------------------------------------------|
# MAGIC | `CB_Product__c`       | `CB_Main_Product__c`      | Same field renamed, different field, or pipeline writing to a deprecated alias? |
# MAGIC | `LeadSource` on Opp   | `LeadSource` on Contact   | Pipeline writes to Opportunity; SF tracks it on Contact. Where should it live? |
# MAGIC
# MAGIC Until confirmed, the mapping is **identity** — `CB_Product__c` will appear in the diff as
# MAGIC `missing_in_response` (because SF doesn't have a field by that exact name). Surfacing the
# MAGIC issue rather than silently aliasing.

# COMMAND ----------

KNOWN_FIELD_MAPPINGS: dict[str, dict[str, str]] = {
    # Uncomment the candidate alias only after pipeline-team confirmation.
    # "Opportunity": {
    #     "CB_Product__c": "CB_Main_Product__c",
    # },
}


def map_sent_to_received_keys(sent_body: dict, object_type: str) -> dict[str, str]:
    """Return a {sent_field_name: received_field_name} mapping."""
    overrides = KNOWN_FIELD_MAPPINGS.get(object_type, {})
    return {k: overrides.get(k, k) for k in sent_body}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. The diff (now tracking-aware)
# MAGIC
# MAGIC Each row gets two annotations:
# MAGIC
# MAGIC | column | values | what it tells you |
# MAGIC |---|---|---|
# MAGIC | `status` | `unchanged` / `changed` / `missing_in_response` / `added_by_sf` / `composite_placeholder` | what the comparison found |
# MAGIC | `tracking_status` | `tracked` / `compound` / `untracked` | whether SF will give us field history for this field |
# MAGIC
# MAGIC The intersection matters most: `(status='changed', tracking_status='tracked')` is the rich
# MAGIC ground-truth signal. `(changed, untracked)` is still a usable diff but can't be enriched
# MAGIC with who-changed-it-when later, so the change-type classifier has less to go on.

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

    for sent_key, recv_key in mapping.items():
        sent_val = sent_body[sent_key]
        track = tracking_status(object_type, sent_key)

        if _is_placeholder(sent_val):
            rows.append({
                "ref_id": ref_id, "object_type": object_type, "sf_id": sf_id,
                "field": sent_key, "before": sent_val, "after": received_body.get(recv_key),
                "status": "composite_placeholder", "tracking_status": "n/a",
            })
            continue

        if recv_key not in received_body:
            rows.append({
                "ref_id": ref_id, "object_type": object_type, "sf_id": sf_id,
                "field": sent_key, "before": sent_val, "after": None,
                "status": "missing_in_response", "tracking_status": track,
            })
            continue

        recv_val = received_body[recv_key]
        rows.append({
            "ref_id": ref_id, "object_type": object_type, "sf_id": sf_id,
            "field": sent_key, "before": sent_val, "after": recv_val,
            "status": "unchanged" if sent_val == recv_val else "changed",
            "tracking_status": track,
        })

    for recv_key in received_body:
        if recv_key in received_keys_referenced:
            continue
        rows.append({
            "ref_id": ref_id, "object_type": object_type, "sf_id": sf_id,
            "field": recv_key, "before": None, "after": received_body[recv_key],
            "status": "added_by_sf", "tracking_status": tracking_status(object_type, recv_key),
        })

    return rows


all_rows: list[dict[str, Any]] = []
for ref_id, sent in sent_records.items():
    if ref_id not in received_records:
        print(f"  ! {ref_id}: no response — skipping")
        continue
    recv = received_records[ref_id]
    if sent["object_type"] != recv["object_type"]:
        print(f"  ! {ref_id}: object_type mismatch — skipping")
        continue
    all_rows.extend(
        diff_record(
            sent_body=sent["body"],
            received_body=recv["body"],
            object_type=sent["object_type"],
            ref_id=ref_id,
            sf_id=recv["sf_id"],
        )
    )

diff_df = pd.DataFrame(all_rows)
diff_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary — by object and by status × tracking

# COMMAND ----------

summary_by_status = (
    diff_df.groupby(["object_type", "status"], dropna=False)
    .size()
    .unstack(fill_value=0)
)
summary_by_status

# COMMAND ----------

# Status × tracking-status breakdown for the changed rows.
# This is the table you'd take to a stand-up: how many changes have rich history available
# vs how many are state-diff only.
changed_breakdown = (
    diff_df[diff_df["status"] == "changed"]
    .groupby(["object_type", "tracking_status"], dropna=False)
    .size()
    .unstack(fill_value=0)
)
changed_breakdown

# COMMAND ----------

# Just the changed rows, with tracking annotation.
changes_only = (
    diff_df[diff_df["status"] == "changed"]
    [["object_type", "field", "before", "after", "tracking_status"]]
    .reset_index(drop=True)
)
changes_only

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. What to swap when DEV access is live
# MAGIC
# MAGIC Three things change.
# MAGIC
# MAGIC **a) Section 5 — replace the mock with a real SOQL call.**
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC def fetch_salesforce_record(object_type: str, sf_id: str, fields: list[str]) -> dict:
# MAGIC     fields_csv = ",".join(fields)
# MAGIC     query = f"SELECT {fields_csv} FROM {object_type} WHERE Id = '{sf_id}'"
# MAGIC     url = f"{SF_BASE_URL}/services/data/{SF_API_VERSION}/query"
# MAGIC     resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, params={"q": query})
# MAGIC     resp.raise_for_status()
# MAGIC     return resp.json()
# MAGIC ```
# MAGIC Reuse whatever auth helper the existing pipeline uses for its POSTs.
# MAGIC
# MAGIC **b) Section 3 — confirm and freeze `TRACKED_FIELDS`.**
# MAGIC The lists here are seeded from the FieldDefinition CSVs. If Mithia enables tracking on
# MAGIC additional fields to close the gap (see `sf_tracking_gap_analysis.xlsx`), add them here.
# MAGIC The change-type classifier will read from this config when deciding which fields it can
# MAGIC enrich with `AccountHistory` / `OpportunityFieldHistory` data.
# MAGIC
# MAGIC **c) Section 7 — resolve the field-name mismatches.**
# MAGIC Get pipeline-team confirmation on:
# MAGIC - `CB_Product__c` ↔ `CB_Main_Product__c`
# MAGIC - `LeadSource` on Opportunity vs on Contact
# MAGIC
# MAGIC Then either populate `KNOWN_FIELD_MAPPINGS` (if it's an alias) or fix the pipeline (if it's
# MAGIC writing to the wrong field).
# MAGIC
# MAGIC **Adjacent work, not in this notebook:**
# MAGIC - **Change-type classifier.** Uses `tracking_status` from the diff above to decide which
# MAGIC   changed rows to enrich with field-history data. Tracked rows get `who/when/why`
# MAGIC   metadata from `AccountHistory`; untracked rows get the diff only.
# MAGIC - **Trigger.** Per the business call, the snapshot moment is the opportunity transition to
# MAGIC   `Originate`. That belongs in the pipeline orchestration, not here.
# MAGIC - **Compound-address history parsing.** When `AccountHistory` returns a `BillingAddress`
# MAGIC   change row, the old/new values are serialised compound objects. The classifier needs to
# MAGIC   parse those out into component-level deltas to align with this diff.
