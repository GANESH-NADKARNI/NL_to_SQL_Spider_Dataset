"""
Evaluation utilities for NL2SQL models.
Computes template accuracy and exact match accuracy.
"""
import re
import numpy as np


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    sql = sql.upper().strip().rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r"'[^']*'", "'<STR>'", sql)
    sql = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', sql)
    return sql


def extract_template(sql: str) -> str:
    """Extract SQL structural template."""
    sql_up = sql.upper()
    parts = []
    for kw in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY',
               'HAVING', 'LIMIT', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
        if kw in sql_up:
            parts.append(kw.replace(' ', '_'))
    tables = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.IGNORECASE)
    table_names = sorted(set(t for tup in tables for t in tup if t))
    return '|'.join(parts) + '||' + '|'.join(table_names)


def exact_match(pred: str, true: str) -> bool:
    return normalize_sql(pred) == normalize_sql(true)


def template_match(pred: str, true: str) -> bool:
    return extract_template(pred) == extract_template(true)


def compute_metrics(predictions: list, ground_truths: list) -> dict:
    """
    Compute all metrics for a set of predictions.
    Returns dict with exact_match, template_match, per-query results.
    """
    assert len(predictions) == len(ground_truths)

    results = []
    exact_matches = 0
    template_matches = 0

    for pred, true in zip(predictions, ground_truths):
        em = exact_match(pred, true)
        tm = template_match(pred, true)
        exact_matches += int(em)
        template_matches += int(tm)
        results.append({
            'predicted': pred,
            'expected': true,
            'exact_match': em,
            'template_match': tm,
        })

    n = len(predictions)
    return {
        'exact_match_accuracy': exact_matches / n if n > 0 else 0.0,
        'template_match_accuracy': template_matches / n if n > 0 else 0.0,
        'total': n,
        'exact_correct': exact_matches,
        'template_correct': template_matches,
        'per_query': results
    }


def evaluate_model(model, nl_list: list, sql_list: list) -> dict:
    """Evaluate a model on a dataset."""
    predictions = model.predict_batch(nl_list)
    return compute_metrics(predictions, sql_list)
