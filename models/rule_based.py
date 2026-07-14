"""
Rule-Based NLP Model for NL2SQL
Uses pattern matching and keyword extraction to map NL to SQL templates.
"""
import re
import pickle
import os
import numpy as np
from collections import defaultdict


class RuleBasedNLP:
    """
    Rule-Based NL2SQL Model.
    
    Strategy:
    1. Extract keywords from NL input
    2. Match to closest training NL using TF-IDF similarity
    3. Return the associated SQL template
    """

    def __init__(self):
        self.nl_to_sql = {}          # Direct NL→SQL mapping from training
        self.keyword_index = defaultdict(list)  # keyword → [(nl, sql)]
        self.templates = {}          # template_id → sql
        self.nl_list = []
        self.sql_list = []
        self.is_trained = False

    def _extract_keywords(self, text: str) -> set:
        text = text.lower()
        # Remove stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'of', 'in',
                      'for', 'to', 'by', 'with', 'and', 'or', 'that', 'which',
                      'from', 'on', 'at', 'as', 'all', 'any', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did'}
        tokens = re.findall(r'[a-zA-Z0-9]+', text)
        keywords = {t for t in tokens if t not in stop_words and len(t) > 2}
        return keywords

    def train(self, nl_list: list, sql_list: list):
        """Build lookup index from training data."""
        self.nl_list = nl_list
        self.sql_list = sql_list
        self.nl_to_sql = {}
        self.keyword_index = defaultdict(list)

        for i, (nl, sql) in enumerate(zip(nl_list, sql_list)):
            nl_clean = nl.lower().strip()
            self.nl_to_sql[nl_clean] = sql
            keywords = self._extract_keywords(nl_clean)
            for kw in keywords:
                self.keyword_index[kw].append(i)

        self.is_trained = True
        print(f"[RuleBasedNLP] Trained on {len(nl_list)} pairs")

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def predict(self, nl_query: str) -> str:
        """Predict SQL for given NL query."""
        if not self.is_trained:
            return "-- Model not trained"

        nl_clean = nl_query.lower().strip()

        # Exact match first
        if nl_clean in self.nl_to_sql:
            return self.nl_to_sql[nl_clean]

        # Keyword-based candidate retrieval
        query_kws = self._extract_keywords(nl_clean)
        candidate_counts = defaultdict(int)
        for kw in query_kws:
            for idx in self.keyword_index.get(kw, []):
                candidate_counts[idx] += 1

        if not candidate_counts:
            # Fallback: return most common SQL
            return self.sql_list[0] if self.sql_list else "SELECT * FROM unknown;"

        # Re-rank by Jaccard similarity
        top_candidates = sorted(candidate_counts.keys(),
                                key=lambda x: candidate_counts[x],
                                reverse=True)[:20]

        best_idx = top_candidates[0]
        best_score = 0.0

        for idx in top_candidates:
            train_kws = self._extract_keywords(self.nl_list[idx].lower())
            score = self._jaccard_similarity(query_kws, train_kws)
            if score > best_score:
                best_score = score
                best_idx = idx

        return self.sql_list[best_idx]

    def predict_batch(self, nl_queries: list) -> list:
        return [self.predict(q) for q in nl_queries]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'nl_list': self.nl_list,
                'sql_list': self.sql_list,
                'nl_to_sql': self.nl_to_sql,
                'keyword_index': dict(self.keyword_index),
                'is_trained': self.is_trained
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.nl_list = data['nl_list']
        self.sql_list = data['sql_list']
        self.nl_to_sql = data['nl_to_sql']
        self.keyword_index = defaultdict(list, data['keyword_index'])
        self.is_trained = data['is_trained']
        return self
