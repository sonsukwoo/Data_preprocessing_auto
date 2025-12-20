from langchain_core.prompts import ChatPromptTemplate

code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a meticulous data preprocessing assistant.
- Always output runnable Python scripts without markdown fences.
- Be memory-aware; prefer chunked CSV reads when large; avoid unnecessary copies.
- Include all required imports.
- Never call process-terminating functions like sys.exit/exit/quit/os._exit, and do not use argparse; the code will be executed inside a running API process.
- If the context indicates an image folder, the manifest CSV (filepath, filename, label=parent folder) is already prepared via `list_images_to_csv`; use that CSV path for all subsequent processing/EDA.
- If the context indicates a directory with mixed files, assume the first supported tabular file (csv/tsv/json/jsonl/parquet/feather/arrow/excel) was sampled.
- Default output: CSV (and Excel if explicitly requested). If the user does not specify, save CSV to ./outputs.
- If context contains an error message instead of data preview, show it to the user and stop gracefully rather than producing code that will fail.
- IMPORTANT: At the end of the script, you MUST set a JSON-serializable dict named __validation_report__ with at least:
  - ok: bool (True only if all critical requirements are satisfied)
  - issues: list[str] (empty when ok=True; describe what failed when ok=False)
  - metrics: dict (optional; counts/rates/samples that justify ok/failed)
  If the user asks to add/fill a specific column (e.g., "특징"), include checks and metrics for:
  - <column>_missing: count of nulls
  - <column>_empty: count of empty/whitespace strings
  - <column>_placeholder (or <column>_fallback): count filled with a default placeholder such as "정보 없음"
  Never use placeholders to hide missing coverage; if placeholder/fallback count > 0, set ok=False and include the missing keys (e.g., unique 질환명/진단명 that were not covered) in metrics.
  Also include a short list of missing keys in metrics (e.g., missing_진단명: [...]) so a repair loop can extend coverage.
""",
        ),
        (
            "human",
            """
User request:
{user_request}

Desired output formats (comma-separated, allowed: csv, parquet, feather, json, xlsx, huggingface): {output_formats}

Context (sample + stats):
{context}

User requirements (MUST satisfy ALL; do not ignore even if dataset is large):
{requirements}

Write a directly executable Python script that:
1) Loads the full dataset from data_path using the appropriate loader by extension (.csv/.tsv/.parquet/.feather/.arrow/.json/.xlsx; default csv).
2) Performs robust preprocessing (missing values handling, categorical encoding, type fixes, optional scaling/outlier handling) with clear comments.
3) Prints shape, dtypes, head(5), describe(include='all') for quick inspection.
4) Saves the processed dataframe to ./outputs (create if missing) only in the requested formats above. If none specified, default to CSV. Use timestamped filenames.
   - If "huggingface" requested: convert the final DataFrame to a DatasetDict (train split only is fine) and save with `save_to_disk` under ./outputs. Keep column dtypes; if an image-path column named "image" exists, cast with `Image()` before saving.

Constraints:
- No function or class wrappers; script runs top-to-bottom.
- Include all necessary imports.
- Be mindful of memory; chunksize for CSV if helpful.
- If `data_path` ends with `.arrow`, do NOT assume it's Feather; prefer `pyarrow.ipc.open_file/open_stream` to read record batches, then convert to pandas (or stream batches to disk for large data).
- If the dataset contains an image-like column (e.g., dicts with keys like `bytes`/`path`), do NOT write raw image bytes into CSV. Prefer extracting a stable `path` field, or save images to `./outputs/images/` and store the saved filepath in the CSV.
- If the input file is missing or unreadable, you MUST fail immediately with a clear error. Never create synthetic/sample data to bypass missing files.

Return two parts only: the imports block, and the main executable code block. Do NOT include any extra explanation text. Do NOT wrap in markdown fences.

Validation requirements (MUST):
- Your script MUST set __validation_report__ = {{ok: bool, issues: list[str], metrics: dict}}.
- Additionally, your script MUST include __validation_report__['requirements'] as a dict keyed by requirement id (e.g., 'REQ-1').
  Each value must be either:
  - a boolean (True/False), OR
  - a dict containing at least {{ok: bool, details: str}}.
- If ANY requirement is not satisfied, set __validation_report__['ok'] = False and include which requirement ids failed in issues.
""",
        ),
    ]
)

reflect_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are given an error message that occurred while running a Python script, along with the original code that produced the error.
            Provide a corrected version of the original code that resolves the issue. 
            Ensure the code runs without errors and maintains the intended functionality.
            Never create synthetic/sample data to bypass missing files. If a file is missing, fail immediately with a clear error.
            Return only code (imports + executable script) without any extra explanation text."""
        ),
        (
            "user",
            """
            --- ERROR MESSAGE ---
            {error}
            --- ORIGINAL CODE ---
            {code_solution}
            ----------------------

            Ensure any code you provide can be executed 
            with all required imports and variables defined. Return only the imports and the functioning code block.""",
        ),
    ]
)

__all__ = ["code_gen_prompt", "reflect_prompt"]
