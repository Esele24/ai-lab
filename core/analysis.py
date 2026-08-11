"""02 — natural-language questions over a CSV.

THE DESIGN DECISION THAT MATTERS, and the reason this is not a toy:

The obvious build is "ask the model to write pandas code, then exec() it". That is
how most demos do it, and it is both a remote-code-execution hole and a liar --
the model will happily invent a number when its own code errors.

This does the opposite. The model's only job is to choose an OPERATION from a
fixed vocabulary, returned under a strict responseSchema. Pandas computes every
number. Then a second call writes prose around numbers that already exist.

Consequence: the model cannot state a total that pandas did not compute, and it
cannot run anything. The worst failure available to it is picking the wrong column,
which is visible in the output because the operation is shown alongside the answer.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core import gemini

MAX_ROWS = 200_000

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["aggregate", "group_by", "time_series", "top_n", "describe", "count_rows"],
        },
        "metric_column": {"type": "string", "description": "Numeric column to measure. Empty for count_rows/describe."},
        "aggregation": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count"]},
        "dimension_column": {"type": "string", "description": "Column to group by. Empty if not grouping."},
        "date_column": {"type": "string", "description": "Date column for time_series. Empty otherwise."},
        "sort_descending": {"type": "boolean"},
        "limit": {"type": "integer"},
        "chart": {"type": "string", "enum": ["bar", "line", "pie", "none"]},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "comparator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains"]},
                    "value": {"type": "string"},
                },
                "required": ["column", "comparator", "value"],
            },
        },
        "reasoning": {"type": "string", "description": "One sentence on why this operation answers the question."},
        "answerable": {"type": "boolean", "description": "False if the data cannot answer this question."},
        "unanswerable_reason": {"type": "string"},
    },
    "required": ["operation", "metric_column", "aggregation", "dimension_column",
                 "date_column", "sort_descending", "limit", "chart", "filters",
                 "reasoning", "answerable", "unanswerable_reason"],
}


def profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe the dataframe for the model -- schema and samples, never all the rows.

    Sending the whole CSV would cost a fortune and blow the context window on a
    50,000-row file. Three sample values per column is enough for the model to tell
    a currency column from a quantity column.
    """
    columns = []
    for name in frame.columns:
        series = frame[name]
        entry: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "samples": [str(v)[:60] for v in series.dropna().head(3).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            entry["min"] = float(series.min())
            entry["max"] = float(series.max())
            entry["mean"] = round(float(series.mean()), 4)
        columns.append(entry)
    return {"rows": int(len(frame)), "columns": columns}


def _coerce(series: pd.Series, value: str):
    """Turn a filter value (always a string from the schema) into the column's type."""
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except ValueError:
            return value
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value, errors="coerce")
    return value


def _apply_filters(frame: pd.DataFrame, filters: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    applied: list[str] = []
    for rule in filters or []:
        column = rule.get("column", "")
        if column not in frame.columns:
            continue  # A hallucinated column name is dropped, not guessed at.
        series = frame[column]
        value = _coerce(series, str(rule.get("value", "")))
        comparator = rule.get("comparator", "==")
        try:
            if comparator == "==":
                mask = series.astype(str).str.lower() == str(value).lower() if series.dtype == object else series == value
            elif comparator == "!=":
                mask = series.astype(str).str.lower() != str(value).lower() if series.dtype == object else series != value
            elif comparator == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, na=False)
            elif comparator == ">":
                mask = series > value
            elif comparator == ">=":
                mask = series >= value
            elif comparator == "<":
                mask = series < value
            else:
                mask = series <= value
        except TypeError:
            continue
        frame = frame[mask]
        applied.append(f"{column} {comparator} {rule.get('value')}")
    return frame, applied


def execute(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """Run the chosen operation in pandas. This function does all the arithmetic."""
    working, applied = _apply_filters(frame, spec.get("filters", []))
    operation = spec.get("operation", "describe")
    metric = spec.get("metric_column") or ""
    dimension = spec.get("dimension_column") or ""
    date_column = spec.get("date_column") or ""
    aggregation = spec.get("aggregation") or "sum"
    limit = max(1, min(int(spec.get("limit") or 10), 100))
    descending = bool(spec.get("sort_descending", True))

    result: dict[str, Any] = {
        "operation": operation, "filters_applied": applied,
        "rows_after_filters": int(len(working)), "chart": spec.get("chart", "none"),
    }

    if working.empty:
        result["kind"] = "empty"
        result["note"] = "No rows matched those filters."
        return result

    if operation == "count_rows":
        result.update(kind="scalar", label="Row count", value=int(len(working)))
        return result

    if operation == "describe":
        numeric = working.select_dtypes("number")
        if numeric.empty:
            result.update(kind="empty", note="No numeric columns to describe.")
            return result
        table = numeric.describe().T.reset_index().rename(columns={"index": "column"})
        result.update(kind="table", table=table.round(4).to_dict("records"),
                      columns=list(table.columns))
        return result

    if metric and metric not in working.columns:
        result.update(kind="error",
                      note=f"The model asked for column '{metric}', which is not in this file.")
        return result

    if operation == "aggregate":
        series = pd.to_numeric(working[metric], errors="coerce").dropna()
        if series.empty:
            result.update(kind="error", note=f"'{metric}' has no numeric values.")
            return result
        value = float(getattr(series, aggregation)()) if aggregation != "count" else int(series.count())
        result.update(kind="scalar", label=f"{aggregation} of {metric}",
                      value=round(value, 4) if isinstance(value, float) else value)
        return result

    if operation in ("group_by", "top_n"):
        if dimension not in working.columns:
            result.update(kind="error",
                          note=f"The model asked to group by '{dimension}', which is not in this file.")
            return result
        grouped = working.groupby(dimension, dropna=False)
        if aggregation == "count" or not metric:
            series = grouped.size().rename("value")
        else:
            working = working.assign(**{metric: pd.to_numeric(working[metric], errors="coerce")})
            series = getattr(working.groupby(dimension, dropna=False)[metric], aggregation)().rename("value")
        series = series.sort_values(ascending=not descending).head(limit)
        table = series.reset_index()
        table.columns = [dimension, "value"]
        result.update(kind="series", table=table.round(4).to_dict("records"),
                      x=dimension, y=f"{aggregation} of {metric}" if metric else "row count")
        return result

    if operation == "time_series":
        if date_column not in working.columns:
            result.update(kind="error",
                          note=f"The model asked for date column '{date_column}', which is not in this file.")
            return result
        parsed = pd.to_datetime(working[date_column], errors="coerce")
        working = working.assign(_period=parsed.dt.to_period("M").astype(str)).dropna(subset=["_period"])
        if aggregation == "count" or not metric:
            series = working.groupby("_period").size().rename("value")
        else:
            working = working.assign(**{metric: pd.to_numeric(working[metric], errors="coerce")})
            series = getattr(working.groupby("_period")[metric], aggregation)().rename("value")
        table = series.sort_index().reset_index()
        table.columns = ["period", "value"]
        result.update(kind="series", table=table.round(4).to_dict("records"),
                      x="period", y=f"{aggregation} of {metric}" if metric else "row count")
        return result

    result.update(kind="error", note=f"Unsupported operation: {operation}")
    return result


def narrate(question: str, spec: dict, result: dict) -> str:
    """Write prose around numbers pandas already computed.

    The computed result is passed in as fact. The model is explicitly forbidden from
    producing a number that is not in it -- which is enforceable here precisely
    because it never had access to the raw rows.
    """
    if result.get("kind") in ("error", "empty"):
        return result.get("note", "No result.")
    payload = {k: v for k, v in result.items() if k != "table"}
    if "table" in result:
        payload["table_preview"] = result["table"][:15]
    return gemini.generate(
        f"Question: {question}\n\nOperation chosen: {spec.get('operation')} "
        f"({spec.get('reasoning')})\n\nComputed result: {payload}",
        system=(
            "You explain a computed data result in 2-3 sentences for a business owner.\n"
            "Every figure you state must appear in the computed result given to you. "
            "You have NOT seen the raw data and must not imply otherwise. "
            "Do not invent trends, causes, or comparisons that are not in the result. "
            "No preamble, no bullet points."
        ),
        temperature=0.2,
    )


def ask(frame: pd.DataFrame, question: str) -> dict[str, Any]:
    if len(frame) > MAX_ROWS:
        raise ValueError(f"{len(frame):,} rows is over the {MAX_ROWS:,} row cap for this demo.")
    schema_text = profile(frame)
    spec = gemini.generate_json(
        f"Dataset profile:\n{schema_text}\n\nUser question: {question}",
        SPEC_SCHEMA,
        system=(
            "You translate a question about a dataset into ONE operation.\n"
            "Use ONLY column names exactly as they appear in the profile -- never invent one.\n"
            "If the dataset genuinely cannot answer the question, set answerable=false and "
            "explain why in unanswerable_reason.\n"
            "Choose chart='none' for single numbers, 'line' for time series, 'bar' for "
            "comparisons across categories, 'pie' only for parts of one whole with under 7 groups."
        ),
    )
    if not spec.get("answerable", True):
        return {"spec": spec, "result": {"kind": "unanswerable",
                                         "note": spec.get("unanswerable_reason", "")},
                "narrative": spec.get("unanswerable_reason", "")}
    result = execute(frame, spec)
    return {"spec": spec, "result": result, "narrative": narrate(question, spec, result)}
