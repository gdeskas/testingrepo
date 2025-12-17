# Databricks notebook source

# MAGIC %md
# MAGIC # RAG System with MLflow Experiment Tracking
# MAGIC 
# MAGIC This notebook implements a complete RAG (Retrieval-Augmented Generation) system for complaint summarization with:
# MAGIC - Multiple LLM endpoint support
# MAGIC - LLM-as-a-judge evaluation framework
# MAGIC - MLflow experiment tracking and comparison
# MAGIC - Comprehensive visualization and analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import json
import requests
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import mlflow
import mlflow.pyfunc
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
from math import pi

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# COMMAND ----------

# Configuration
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
EMBEDDING_ENDPOINT = "gte-endpoint"
WORKSPACE_URL = "https://adb-7941446833400015.15.azuredatabricks.net"

# Available LLM endpoints for RAG
AVAILABLE_LLMS = {
    "claude-sonnet-4.5": "databricks-claude-sonnet-4-5",
    "claude-opus-4.5": "databricks-claude-opus-4-5",
    "gpt-oss-120b": "databricks-gpt-oss-120b"
}

# Judge LLM for evaluation
JUDGE_LLM_ENDPOINT = "databricks-claude-opus-4-5"

# Vector Search Setup
VECTOR_SEARCH_ENDPOINT = "complaint-vector-endpoint"
VECTOR_SEARCH_INDEX = "cntrl-busops-dev.complaints-1kh-gld.complaints_chunk_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Loading and Preprocessing

# COMMAND ----------

# Load data from PySpark table
df_spark = spark.sql("SELECT * FROM `cntrl-busops-dev`.`complaints-1kh-gld`.`vw_complaint`")

# Convert to Pandas for processing
df = df_spark.toPandas()

print(f"Loaded {len(df)} complaints")

# COMMAND ----------

def extract_reference_number(text: str) -> str:
    """Extract reference_number from the all_columns text field."""
    if pd.isna(text) or not text:
        return "unknown"
    
    # Try to extract reference_number from the beginning of the text
    match = re.search(r'reference_number:\s*([^\|]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: try common complaint ID patterns
    id_match = re.search(r'(CDCR\w[|C-\w]{5,})', text)
    if id_match:
        return id_match.group(1)
    
    return "unknown"


# Extract complaint IDs from all_columns
df['reference_number'] = df['all_columns'].apply(extract_reference_number)

# Get all unique complaint IDs
all_ids = df['reference_number'].unique().tolist()
print(f"Found {len(all_ids)} unique complaints")

# Build documents (each row's all_columns is the full document)
docs = df['all_columns'].fillna("(no content)").tolist()
print(f"Built {len(docs)} documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Document Chunking

# COMMAND ----------

def _chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, breaking on sentence boundaries where possible."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            search_start = end - int(chunk_size * 0.2)
            search_region = text[search_start:end]
            last_period = search_region.rfind('. ')
            last_newline = search_region.rfind('\n')
            last_pipe = search_region.rfind(' | ')
            last_break = max(last_period, last_newline, last_pipe)
            
            if last_break != -1:
                end = search_start + last_break + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def smart_chunk_document(
    doc: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    preserve_sections: bool = True
) -> List[Tuple[str, Dict[str, str]]]:
    """Smart chunking with section preservation."""
    
    chunks = []
    reference_number = extract_reference_number(doc)
    
    if preserve_sections:
        sections = re.split(r'(^##\s+\$)', doc, flags=re.MULTILINE)
        current_section = ""
        section_name = "Main"
        
        for i, part in enumerate(sections):
            if part.strip().startswith("## "):
                section_name = part.strip().replace("## ", "")
                current_section = part + "\n"
            elif part.strip():
                current_section += part
                
                if len(current_section) >= chunk_size:
                    section_chunks = _chunk_text_with_overlap(current_section, chunk_size, chunk_overlap)
                    
                    for j, chunk_text in enumerate(section_chunks):
                        metadata = {
                            "reference_number": reference_number,
                            "section": section_name,
                            "chunk_index": j,
                            "total_chunks": len(section_chunks)
                        }
                        chunks.append((chunk_text, metadata))
                    current_section = ""
        
        if current_section.strip():
            metadata = {
                "reference_number": reference_number,
                "section": section_name,
                "chunk_index": 0,
                "total_chunks": 1
            }
            chunks.append((current_section, metadata))
    else:
        chunk_texts_list = _chunk_text_with_overlap(doc, chunk_size, chunk_overlap)
        for i, chunk_text in enumerate(chunk_texts_list):
            metadata = {
                "reference_number": reference_number,
                "section": "Full-doc",
                "chunk_index": i,
                "total_chunks": len(chunk_texts_list)
            }
            chunks.append((chunk_text, metadata))
    
    if not chunks and doc.strip():
        metadata = {
            "reference_number": reference_number,
            "section": "Main",
            "chunk_index": 0,
            "total_chunks": 1
        }
        chunks.append((doc, metadata))
    
    return chunks

# COMMAND ----------

# Build chunks from all documents
chunk_texts = []
chunk_metadata = []

for i, doc in enumerate(docs):
    chunks = smart_chunk_document(doc, chunk_size=512, chunk_overlap=128, preserve_sections=True)
    for chunk_text, metadata in chunks:
        chunk_texts.append(chunk_text)
        chunk_metadata.append(metadata)

print(f"Created {len(chunk_texts)} chunks from {len(docs)} complaints")
print(f"Average chunks per complaint: {len(chunk_texts) / len(docs):.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Embedding Generation

# COMMAND ----------

def _invocations_url(workspace_url: str, endpoint_name: str) -> str:
    return f"{workspace_url.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"


def _make_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _parse_embeddings(resp_json):
    """Handle common embedding response shapes."""
    if isinstance(resp_json, dict):
        if "embeddings" in resp_json and isinstance(resp_json["embeddings"], list):
            return resp_json["embeddings"]
        if "data" in resp_json:
            data = resp_json["data"]
            if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
                return [row["embedding"] for row in data]
            if isinstance(data, dict) and "embeddings" in data:
                return data["embeddings"]
    raise ValueError(f"Unrecognized embeddings response shape: {list(resp_json.keys()) if isinstance(resp_json, dict) else type(resp_json)}")


def embed_databricks(
    texts: List[str],
    workspace_url: str = WORKSPACE_URL,
    endpoint_name: str = EMBEDDING_ENDPOINT,
    token: str = DATABRICKS_TOKEN,
    batch_size: int = 64,
) -> np.ndarray:
    """Returns L2-normalized embeddings (float32) shaped [N, D]."""
    url = _invocations_url(workspace_url, endpoint_name)
    headers = _make_headers(token)
    
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {"input": batch}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.ok:
            vecs = _parse_embeddings(response.json())
            all_vecs.extend(vecs)
        else:
            raise RuntimeError(f"Embedding call failed: {response.status_code} {response.text[:500]}")
    
    arr = np.array(all_vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    arr = arr / norms
    return arr

# COMMAND ----------

print("Embedding all chunks...")
embeddings = embed_databricks(chunk_texts).astype("float32")
print(f"Embedded {len(chunk_texts)} chunks, dimension: {embeddings.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Vector Search and Retrieval

# COMMAND ----------

def retrieve_similar_chunks(
    reference_number: str,
    query_text: Optional[str] = None,
    k: int = 10,
    return_full_complaints: bool = False,
    exclude_duplicates: bool = True
) -> List[Tuple[str, float, str, Dict]]:
    """Retrieve similar chunks using Mosaic AI Vector Search REST API."""
    
    # Build query vector
    if query_text is not None:
        q_vec = embed_databricks([query_text])[0].tolist()
    else:
        complaint_chunk_indices = []
        for i, meta in enumerate(chunk_metadata):
            if meta['reference_number'] == reference_number:
                complaint_chunk_indices.append(i)
        
        if not complaint_chunk_indices:
            return []
        
        q_idx = complaint_chunk_indices[0]
        q_vec = embeddings[q_idx].tolist()
    
    # Call Vector Search REST API
    search_k = k * 5 if exclude_duplicates else k * 2
    
    url = f"{WORKSPACE_URL}/api/2.0/vector-search/indexes/{VECTOR_SEARCH_INDEX}/query"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query_vector": q_vec,
        "num_results": search_k,
        "columns": ["chunk_id", "reference_number", "section", "chunk_index", "chunk_text"]
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    
    if not response.ok:
        raise RuntimeError(f"Vector Search failed: {response.status_code} {response.text[:500]}")
    
    results_raw = response.json()
    
    seen_complaints = set([reference_number])
    results = []
    duplicate_exclusion_set = set()
    
    for row in results_raw.get("result", {}).get("data_array", []):
        chunk_id, chunk_cid, section, chunk_idx, text, score = row
        
        if exclude_duplicates and int(chunk_id) in duplicate_exclusion_set:
            continue
        
        if chunk_cid == reference_number:
            continue
        
        meta = {
            "reference_number": chunk_cid,
            "section": section,
            "chunk_index": chunk_idx
        }
        
        if return_full_complaints:
            if chunk_cid not in seen_complaints:
                seen_complaints.add(chunk_cid)
                try:
                    doc_idx = all_ids.index(chunk_cid)
                    full_doc = docs[doc_idx]
                except ValueError:
                    full_doc = text
                results.append((chunk_cid, float(score), full_doc, meta))
        else:
            results.append((chunk_cid, float(score), text, meta))
        
        if len(results) >= k:
            break
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. LLM Integration

# COMMAND ----------

def call_llm_endpoint(
    prompt: str, 
    endpoint_name: str,
    max_new_tokens: int = 320, 
    temperature: float = 0.0
) -> str:
    """Sends a chat-style prompt to a Databricks-hosted LLM endpoint."""
    
    url = f"{WORKSPACE_URL}/serving-endpoints/{endpoint_name}/invocations"
    
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


SUMMARY_PROMPT = """You are an expert complaints analyst writing a clear, factual and chronological summary of a customer complaint.

Use ONLY the provided context - do not add information that isn't present.

Your goal is to create a concise narrative (5-10 sentences) that describes the complaint journey in order of events.
Include:
- When and how the complaint was received
- What the customer raised or alleged
- How the bank investigated and communicated during the process
- Key findings, decisions, or redress outcomes
- Any SLA breaches, delays, or escalation to FOS if applicable
- When and how the complaint was closed

Write it in professional plain English, past tense, and in chronological order (oldest events first).
Avoid bullet points or headings - return one coherent paragraph.

---
TARGET COMPLAINT CONTEXT:
{target}

---
SIMILAR CASES (for style reference only, do not copy facts):
{neighbors}

Now write the chronological summary of the target complaint only.
"""


def rag_summarize(
    reference_number: str,
    llm_endpoint: str,
    k_neighbors: int = 4,
    exclude_duplicates: bool = True,
    temperature: float = 0.6,
    max_tokens: int = 5000
) -> str:
    """Generate a RAG-enhanced summary using specified LLM endpoint."""
    
    if reference_number not in all_ids:
        return f"Complaint {reference_number} not found."
    
    target_doc = docs[all_ids.index(reference_number)]
    
    similar = retrieve_similar_chunks(
        reference_number,
        k=k_neighbors,
        return_full_complaints=True,
        exclude_duplicates=exclude_duplicates
    )
    
    neigh_text = "\n\n---\n\n".join([
        f"# Similar {i} ({cid}) (score: {score:.3f})\n{doc[:6000]}"
        for i, (cid, score, doc, meta) in enumerate(similar)
    ])[:12000]
    
    prompt = SUMMARY_PROMPT.format(
        target=target_doc[:24000],
        neighbors=neigh_text
    )
    
    return call_llm_endpoint(prompt, endpoint_name=llm_endpoint, max_new_tokens=max_tokens, temperature=temperature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. LLM-as-a-Judge Evaluation

# COMMAND ----------

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of complaint summaries.

You will be given:
1. The original complaint context
2. A generated summary

Evaluate the summary on the following criteria (rate each 1-5, where 5 is best):

**ACCURACY**: Does the summary contain only factual information from the original complaint? Are there any hallucinations or invented details?
- 5: Completely accurate, no hallucinations
- 3: Mostly accurate with minor discrepancies
- 1: Contains significant inaccuracies or hallucinations

**COMPLETENESS**: Does the summary capture all key events, decisions, and outcomes from the complaint?
- 5: All critical information included
- 3: Some important details missing
- 1: Major gaps in coverage

**CHRONOLOGY**: Are events presented in the correct temporal order?
- 5: Perfect chronological flow
- 3: Mostly chronological with minor issues
- 1: Confusing or incorrect ordering

**CLARITY**: Is the summary well-written, concise, and easy to understand?
- 5: Exceptionally clear and professional
- 3: Adequate but could be clearer
- 1: Confusing or poorly written

**CONCISENESS**: Is the summary appropriately brief without unnecessary detail?
- 5: Perfect balance of detail and brevity
- 3: Somewhat verbose or too terse
- 1: Far too long or missing critical context

---
ORIGINAL COMPLAINT:
{original_context}

---
GENERATED SUMMARY:
{summary}

---
Provide your evaluation in the following JSON format:
{{
    "accuracy": <score 1-5>,
    "accuracy_reasoning": "<brief explanation>",
    "completeness": <score 1-5>,
    "completeness_reasoning": "<brief explanation>",
    "chronology": <score 1-5>,
    "chronology_reasoning": "<brief explanation>",
    "clarity": <score 1-5>,
    "clarity_reasoning": "<brief explanation>",
    "conciseness": <score 1-5>,
    "conciseness_reasoning": "<brief explanation>",
    "overall_score": <average of all scores>,
    "overall_assessment": "<2-3 sentence summary of strengths and weaknesses>"
}}

Return ONLY the JSON, no other text.
"""


def judge_summary(
    reference_number: str,
    summary: str,
    judge_endpoint: str = JUDGE_LLM_ENDPOINT
) -> Dict:
    """Use LLM-as-a-judge to evaluate a generated summary."""
    
    if reference_number not in all_ids:
        raise ValueError(f"Complaint {reference_number} not found")
    
    original_context = docs[all_ids.index(reference_number)]
    
    prompt = JUDGE_PROMPT.format(
        original_context=original_context[:20000],
        summary=summary
    )
    
    response = call_llm_endpoint(
        prompt, 
        endpoint_name=judge_endpoint,
        max_new_tokens=2000,
        temperature=0.1
    )
    
    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        eval_result = json.loads(response)
        return eval_result
    except json.JSONDecodeError as e:
        print(f"Failed to parse judge response: {e}")
        print(f"Response was: {response}")
        return {
            "accuracy": 3,
            "completeness": 3,
            "chronology": 3,
            "clarity": 3,
            "conciseness": 3,
            "overall_score": 3,
            "overall_assessment": "Failed to parse evaluation",
            "raw_response": response
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MLflow Experiment Framework

# COMMAND ----------

def run_rag_experiment(
    experiment_name: str = "/Users/your_email@domain.com/rag_llm_comparison",
    test_complaints: List[str] = None,
    llm_configs: Dict[str, Dict] = None,
    k_neighbors: int = 4
):
    """Run RAG experiments with different LLMs and track results in MLflow."""
    
    mlflow.set_experiment(experiment_name)
    
    if test_complaints is None:
        test_complaints = all_ids[:10] if len(all_ids) >= 10 else all_ids
    
    if llm_configs is None:
        llm_configs = {
            "claude-sonnet-4.5-temp0.3": {
                "endpoint": AVAILABLE_LLMS["claude-sonnet-4.5"],
                "temperature": 0.3,
                "max_tokens": 5000
            },
            "claude-sonnet-4.5-temp0.6": {
                "endpoint": AVAILABLE_LLMS["claude-sonnet-4.5"],
                "temperature": 0.6,
                "max_tokens": 5000
            },
            "claude-opus-4.5": {
                "endpoint": AVAILABLE_LLMS["claude-opus-4.5"],
                "temperature": 0.6,
                "max_tokens": 5000
            },
            "gpt-oss-120b": {
                "endpoint": AVAILABLE_LLMS["gpt-oss-120b"],
                "temperature": 0.6,
                "max_tokens": 5000
            }
        }
    
    results_summary = []
    
    for llm_name, config in llm_configs.items():
        print(f"\n{'='*80}")
        print(f"Running experiment: {llm_name}")
        print(f"{'='*80}\n")
        
        with mlflow.start_run(run_name=f"{llm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            mlflow.log_param("llm_name", llm_name)
            mlflow.log_param("llm_endpoint", config["endpoint"])
            mlflow.log_param("temperature", config["temperature"])
            mlflow.log_param("max_tokens", config["max_tokens"])
            mlflow.log_param("k_neighbors", k_neighbors)
            mlflow.log_param("num_test_complaints", len(test_complaints))
            mlflow.log_param("judge_endpoint", JUDGE_LLM_ENDPOINT)
            
            all_summaries = []
            all_evaluations = []
            
            for i, complaint_id in enumerate(test_complaints):
                print(f"Processing complaint {i+1}/{len(test_complaints)}: {complaint_id}")
                
                try:
                    summary = rag_summarize(
                        reference_number=complaint_id,
                        llm_endpoint=config["endpoint"],
                        k_neighbors=k_neighbors,
                        temperature=config["temperature"],
                        max_tokens=config["max_tokens"]
                    )
                    
                    evaluation = judge_summary(
                        reference_number=complaint_id,
                        summary=summary,
                        judge_endpoint=JUDGE_LLM_ENDPOINT
                    )
                    
                    all_summaries.append({
                        "complaint_id": complaint_id,
                        "summary": summary,
                        "summary_length": len(summary)
                    })
                    
                    all_evaluations.append({
                        "complaint_id": complaint_id,
                        **evaluation
                    })
                    
                    mlflow.log_metric(f"accuracy_{i}", evaluation["accuracy"])
                    mlflow.log_metric(f"completeness_{i}", evaluation["completeness"])
                    mlflow.log_metric(f"chronology_{i}", evaluation["chronology"])
                    mlflow.log_metric(f"clarity_{i}", evaluation["clarity"])
                    mlflow.log_metric(f"conciseness_{i}", evaluation["conciseness"])
                    mlflow.log_metric(f"overall_score_{i}", evaluation["overall_score"])
                    
                except Exception as e:
                    print(f"Error processing {complaint_id}: {e}")
                    continue
            
            if all_evaluations:
                avg_accuracy = np.mean([e["accuracy"] for e in all_evaluations])
                avg_completeness = np.mean([e["completeness"] for e in all_evaluations])
                avg_chronology = np.mean([e["chronology"] for e in all_evaluations])
                avg_clarity = np.mean([e["clarity"] for e in all_evaluations])
                avg_conciseness = np.mean([e["conciseness"] for e in all_evaluations])
                avg_overall = np.mean([e["overall_score"] for e in all_evaluations])
                avg_summary_length = np.mean([s["summary_length"] for s in all_summaries])
                
                mlflow.log_metric("avg_accuracy", avg_accuracy)
                mlflow.log_metric("avg_completeness", avg_completeness)
                mlflow.log_metric("avg_chronology", avg_chronology)
                mlflow.log_metric("avg_clarity", avg_clarity)
                mlflow.log_metric("avg_conciseness", avg_conciseness)
                mlflow.log_metric("avg_overall_score", avg_overall)
                mlflow.log_metric("avg_summary_length", avg_summary_length)
                mlflow.log_metric("std_accuracy", np.std([e["accuracy"] for e in all_evaluations]))
                mlflow.log_metric("std_overall_score", np.std([e["overall_score"] for e in all_evaluations]))
                
                summaries_df = pd.DataFrame(all_summaries)
                evaluations_df = pd.DataFrame(all_evaluations)
                
                summaries_df.to_csv("summaries.csv", index=False)
                evaluations_df.to_csv("evaluations.csv", index=False)
                
                mlflow.log_artifact("summaries.csv")
                mlflow.log_artifact("evaluations.csv")
                
                os.remove("summaries.csv")
                os.remove("evaluations.csv")
                
                results_summary.append({
                    "llm_name": llm_name,
                    "avg_accuracy": avg_accuracy,
                    "avg_completeness": avg_completeness,
                    "avg_chronology": avg_chronology,
                    "avg_clarity": avg_clarity,
                    "avg_conciseness": avg_conciseness,
                    "avg_overall_score": avg_overall,
                    "avg_summary_length": avg_summary_length
                })
                
                print(f"\n{llm_name} Results:")
                print(f"  Avg Accuracy: {avg_accuracy:.2f}")
                print(f"  Avg Completeness: {avg_completeness:.2f}")
                print(f"  Avg Chronology: {avg_chronology:.2f}")
                print(f"  Avg Clarity: {avg_clarity:.2f}")
                print(f"  Avg Conciseness: {avg_conciseness:.2f}")
                print(f"  Avg Overall Score: {avg_overall:.2f}")
                print(f"  Avg Summary Length: {avg_summary_length:.0f} chars")
    
    comparison_df = pd.DataFrame(results_summary)
    comparison_df = comparison_df.sort_values("avg_overall_score", ascending=False)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY - RANKED BY OVERALL SCORE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    return comparison_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Run Experiment

# COMMAND ----------

# Select test complaints
test_complaint_ids = [
    "CDCR-23502631", 
    "CDCR-21029850", 
    "CDCR-33381031", 
    "CDCR-35951561"
]

# Define LLM configurations to test
llm_configs_to_test = {
    "claude-sonnet-4.5-balanced": {
        "endpoint": AVAILABLE_LLMS["claude-sonnet-4.5"],
        "temperature": 0.6,
        "max_tokens": 5000
    },
    "claude-sonnet-4.5-conservative": {
        "endpoint": AVAILABLE_LLMS["claude-sonnet-4.5"],
        "temperature": 0.3,
        "max_tokens": 5000
    },
    "claude-opus-4.5": {
        "endpoint": AVAILABLE_LLMS["claude-opus-4.5"],
        "temperature": 0.6,
        "max_tokens": 5000
    },
    "gpt-oss-120b": {
        "endpoint": AVAILABLE_LLMS["gpt-oss-120b"],
        "temperature": 0.6,
        "max_tokens": 5000
    }
}

# Run the experiment
results = run_rag_experiment(
    experiment_name="/Users/your_email@domain.com/rag_llm_comparison",
    test_complaints=test_complaint_ids,
    llm_configs=llm_configs_to_test,
    k_neighbors=4
)

print("\nExperiment complete! Check MLflow UI for detailed results.")
print(f"Best performing model: {results.iloc[0]['llm_name']}")
print(f"Overall score: {results.iloc[0]['avg_overall_score']:.2f}/5.0")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Load and Analyze Experiment Results

# COMMAND ----------

# Initialize MLflow client
client = MlflowClient()

# Get the experiment
experiment_name = "/Users/your_email@domain.com/rag_llm_comparison"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Experiment '{experiment_name}' not found!")
else:
    print(f"Found experiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    
    # Get all runs from the experiment
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    print(f"Total runs: {len(runs)}")
    
    # Display basic info about runs
    display(runs[['run_id', 'start_time', 'status', 'params.llm_name', 
                  'metrics.avg_overall_score']].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Model Performance Comparison

# COMMAND ----------

# Extract key metrics for comparison
comparison_metrics = runs[[
    'params.llm_name',
    'params.temperature',
    'metrics.avg_accuracy',
    'metrics.avg_completeness',
    'metrics.avg_chronology',
    'metrics.avg_clarity',
    'metrics.avg_conciseness',
    'metrics.avg_overall_score',
    'metrics.avg_summary_length',
    'metrics.std_overall_score'
]].copy()

comparison_metrics = comparison_metrics.sort_values('metrics.avg_overall_score', ascending=False)

print("="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
display(comparison_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Visualization: Overall Scores by Model

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 6))

models = comparison_metrics['params.llm_name'].values
scores = comparison_metrics['metrics.avg_overall_score'].values
std_devs = comparison_metrics['metrics.std_overall_score'].values

bars = ax.barh(models, scores, xerr=std_devs, capsize=5, color='steelblue', alpha=0.8)

for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(score + 0.1, i, f'{score:.2f}', va='center', fontweight='bold')

ax.set_xlabel('Average Overall Score (1-5)', fontsize=12, fontweight='bold')
ax.set_ylabel('LLM Configuration', fontsize=12, fontweight='bold')
ax.set_title('RAG Summarization: Overall Model Performance', fontsize=14, fontweight='bold')
ax.set_xlim(0, 5.5)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Multi-Dimensional Performance - Radar Chart

# COMMAND ----------

categories = ['Accuracy', 'Completeness', 'Chronology', 'Clarity', 'Conciseness']
metric_cols = [
    'metrics.avg_accuracy',
    'metrics.avg_completeness', 
    'metrics.avg_chronology',
    'metrics.avg_clarity',
    'metrics.avg_conciseness'
]

top_models = comparison_metrics.head(4)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for idx, (_, row) in enumerate(top_models.iterrows()):
    values = [row[col] for col in metric_cols]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=row['params.llm_name'], color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, fontweight='bold')

ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], size=9)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

ax.set_title('Multi-Dimensional Performance Comparison', 
             size=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Detailed Metrics Distribution

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Score Distributions Across Models', fontsize=16, fontweight='bold')

metrics_to_plot = [
    ('metrics.avg_accuracy', 'Accuracy'),
    ('metrics.avg_completeness', 'Completeness'),
    ('metrics.avg_chronology', 'Chronology'),
    ('metrics.avg_clarity', 'Clarity'),
    ('metrics.avg_conciseness', 'Conciseness'),
    ('metrics.avg_overall_score', 'Overall Score')
]

for idx, (metric_col, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    plot_data = []
    labels = []
    for _, row in top_models.iterrows():
        plot_data.append(row[metric_col])
        labels.append(row['params.llm_name'])
    
    bars = ax.barh(range(len(labels)), plot_data, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 5)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, plot_data)):
        ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Temperature Impact Analysis

# COMMAND ----------

temp_analysis = runs[runs['params.llm_name'].str.contains('claude-sonnet', na=False)].copy()

if len(temp_analysis) > 0:
    temp_analysis = temp_analysis[[
        'params.llm_name',
        'params.temperature',
        'metrics.avg_overall_score',
        'metrics.avg_accuracy',
        'metrics.avg_clarity',
        'metrics.std_overall_score'
    ]].sort_values('params.temperature')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    temps = temp_analysis['params.temperature'].astype(float)
    scores = temp_analysis['metrics.avg_overall_score']
    std_devs = temp_analysis['metrics.std_overall_score']
    
    ax1.errorbar(temps, scores, yerr=std_devs, marker='o', markersize=8, 
                 capsize=5, linewidth=2, color='steelblue')
    ax1.set_xlabel('Temperature', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Overall Score', fontsize=12, fontweight='bold')
    ax1.set_title('Impact of Temperature on Performance', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 5.5)
    ax1.grid(alpha=0.3)
    
    accuracy = temp_analysis['metrics.avg_accuracy']
    clarity = temp_analysis['metrics.avg_clarity']
    
    scatter = ax2.scatter(accuracy, clarity, c=temps, s=200, cmap='coolwarm', 
                         edgecolors='black', linewidths=1.5, alpha=0.7)
    
    for temp, acc, clar in zip(temps, accuracy, clarity):
        ax2.annotate(f'T={temp:.1f}', (acc, clar), fontsize=9, 
                    ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Clarity Score', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Clarity Trade-off', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Temperature', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for temperature analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Summary Length Analysis

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

models = comparison_metrics['params.llm_name'].values
lengths = comparison_metrics['metrics.avg_summary_length'].values

bars = ax1.barh(models, lengths, color='coral', alpha=0.8)
for i, (bar, length) in enumerate(zip(bars, lengths)):
    ax1.text(length + 50, i, f'{length:.0f}', va='center', fontweight='bold')

ax1.set_xlabel('Average Summary Length (characters)', fontsize=12, fontweight='bold')
ax1.set_ylabel('LLM Configuration', fontsize=12, fontweight='bold')
ax1.set_title('Average Summary Length by Model', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

ax2.scatter(comparison_metrics['metrics.avg_summary_length'],
           comparison_metrics['metrics.avg_overall_score'],
           s=150, alpha=0.6, color='steelblue', edgecolors='black', linewidths=1.5)

for _, row in comparison_metrics.iterrows():
    ax2.annotate(row['params.llm_name'], 
                (row['metrics.avg_summary_length'], row['metrics.avg_overall_score']),
                fontsize=8, ha='left', va='bottom')

ax2.set_xlabel('Average Summary Length (characters)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Overall Score', fontsize=12, fontweight='bold')
ax2.set_title('Length vs Quality Trade-off', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Winner Declaration and Recommendations

# COMMAND ----------

best_model = comparison_metrics.iloc[0]

print("="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print()
print(f"WINNER: {best_model['params.llm_name']}")
print(f"Overall Score: {best_model['metrics.avg_overall_score']:.3f}/5.0")
print(f"Temperature: {best_model['params.temperature']}")
print()
print("Breakdown:")
print(f"  Accuracy:     {best_model['metrics.avg_accuracy']:.2f}/5.0")
print(f"  Completeness: {best_model['metrics.avg_completeness']:.2f}/5.0")
print(f"  Chronology:   {best_model['metrics.avg_chronology']:.2f}/5.0")
print(f"  Clarity:      {best_model['metrics.avg_clarity']:.2f}/5.0")
print(f"  Conciseness:  {best_model['metrics.avg_conciseness']:.2f}/5.0")
print()
print("="*80)
print()

print("RECOMMENDATIONS:")
print()

best_accuracy = comparison_metrics.nlargest(1, 'metrics.avg_accuracy').iloc[0]
print(f"For maximum accuracy: {best_accuracy['params.llm_name']}")
print(f"  (Accuracy score: {best_accuracy['metrics.avg_accuracy']:.2f})")
print()

best_clarity = comparison_metrics.nlargest(1, 'metrics.avg_clarity').iloc[0]
print(f"For best clarity: {best_clarity['params.llm_name']}")
print(f"  (Clarity score: {best_clarity['metrics.avg_clarity']:.2f})")
print()

best_concise = comparison_metrics.nlargest(1, 'metrics.avg_conciseness').iloc[0]
print(f"For conciseness: {best_concise['params.llm_name']}")
print(f"  (Conciseness score: {best_concise['metrics.avg_conciseness']:.2f})")
print()

print(f"For overall balance: {best_model['params.llm_name']}")
print()
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Export Results

# COMMAND ----------

report_df = comparison_metrics.copy()
report_df.columns = [col.replace('params.', '').replace('metrics.', '') for col in report_df.columns]

spark.createDataFrame(report_df).write.mode("overwrite").saveAsTable("rag_experiment_results")

print("Results saved to Delta table: rag_experiment_results")
print()
print("Query with:")
print("  SELECT * FROM rag_experiment_results ORDER BY avg_overall_score DESC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Sample Summaries Comparison

# COMMAND ----------

def display_sample_summaries(complaint_id: str):
    """Display summaries from all models for a given complaint."""
    
    print(f"\n{'='*100}")
    print(f"COMPLAINT ID: {complaint_id}")
    print(f"{'='*100}\n")
    
    for _, row in comparison_metrics.iterrows():
        run_id = runs[runs['params.llm_name'] == row['params.llm_name']].iloc[0]['run_id']
        
        try:
            summaries_path = client.download_artifacts(run_id, "summaries.csv")
            summaries_df = pd.read_csv(summaries_path)
            
            summary_row = summaries_df[summaries_df['complaint_id'] == complaint_id]
            
            if not summary_row.empty:
                summary = summary_row.iloc[0]['summary']
                model_name = row['params.llm_name']
                score = row['metrics.avg_overall_score']
                
                print(f"{model_name} (Score: {score:.2f})")
                print(f"{'-'*100}")
                print(summary)
                print(f"\n")
        except:
            continue

# Display sample summaries for first test complaint
test_complaint_id = "CDCR-23502631"
display_sample_summaries(test_complaint_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook provides a complete RAG system with MLflow experiment tracking:
# MAGIC 
# MAGIC 1. Data loading and preprocessing
# MAGIC 2. Document chunking with section preservation
# MAGIC 3. Embedding generation using Databricks endpoints
# MAGIC 4. Vector search for similarity retrieval
# MAGIC 5. LLM integration for summarization
# MAGIC 6. LLM-as-a-judge evaluation framework
# MAGIC 7. MLflow experiment tracking
# MAGIC 8. Comprehensive visualization and analysis
# MAGIC 9. Results export and recommendations
# MAGIC 
# MAGIC The framework enables systematic comparison of different LLMs, temperatures, and RAG parameters to identify the optimal configuration for complaint summarization.
