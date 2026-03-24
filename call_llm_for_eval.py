def call_llm_for_eval(text_content: str, prompt_name: str) -> dict:
    pcfg = prompt_configs[prompt_name]
    ccfg = PROMPT_CATALOGUE[prompt_name]
    output_fields = ccfg["output_fields"]

    try:
        text_truncated = text_content[:8000] if text_content else ""
        prompt_df = spark.createDataFrame([(text_truncated,)], ["doc_text"])

        resp_fmt_str = json.dumps(pcfg["response_format"]).replace("'", "\\'")
        prompt_escaped = pcfg["prompt_text"].replace("'", "\\'")

        result_df = prompt_df.selectExpr(f"""
            ai_query(
                '{pcfg["model_endpoint"]}',
                CONCAT('{prompt_escaped}', '\\n\\nDocument text:\\n', doc_text),
                responseFormat => '{resp_fmt_str}'
            ) AS llm_response
        """)

        raw = result_df.collect()[0]["llm_response"]

        if isinstance(raw, str):
            parsed = json.loads(raw)
        elif isinstance(raw, dict):
            parsed = raw
        else:
            parsed = json.loads(str(raw))

        return {field: parsed.get(field, None) for field in output_fields}

    except Exception as e:
        print(f"    ERROR: {e}")
        return {field: None for field in output_fields}
