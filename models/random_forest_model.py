"""
Random Forest Model for NL2SQL
TF-IDF features + RF classifier for SQL template prediction + retrieval
"""
import re
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


class RandomForestNL2SQL:
    def __init__(self, n_estimators: int = 200, max_features: int = 5000):
        self.n_estimators  = n_estimators
        self.max_features  = max_features
        self.tfidf         = TfidfVectorizer(max_features=max_features,
                                             ngram_range=(1, 3),
                                             sublinear_tf=True, min_df=1)
        self.clf           = RandomForestClassifier(n_estimators=n_estimators,
                                                    n_jobs=-1, random_state=42,
                                                    max_depth=30, min_samples_leaf=1)
        self.label_encoder = LabelEncoder()
        self.train_nl      = []
        self.train_sql     = []
        self.train_templates = []
        self.template_to_sqls = {}
        self.train_vectors = None
        self.is_trained    = False

    def _extract_template(self, sql: str) -> str:
        sql_up   = sql.upper()
        keywords = [kw.replace(' ', '_')
                    for kw in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY',
                               'ORDER BY', 'HAVING', 'LIMIT', 'DISTINCT',
                               'COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
                    if kw in sql_up]
        tables   = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.IGNORECASE)
        tnames   = [t for tup in tables for t in tup if t][:2]
        return '|'.join(keywords) + '||' + '|'.join(sorted(tnames))

    def train(self, nl_list: list, sql_list: list):
        self.train_nl  = list(nl_list)
        self.train_sql = list(sql_list)
        self.train_templates = [self._extract_template(s) for s in sql_list]

        self.template_to_sqls = {}
        for nl, sql, tmpl in zip(nl_list, sql_list, self.train_templates):
            self.template_to_sqls.setdefault(tmpl, []).append((nl, sql))

        X = self.tfidf.fit_transform(nl_list)
        self.train_vectors = X
        y = self.label_encoder.fit_transform(self.train_templates)
        self.clf.fit(X, y)
        self.is_trained = True
        print(f"[RandomForest] Trained on {len(nl_list)} samples | "
              f"{len(set(self.train_templates))} templates")
        return self

    def predict(self, nl_query: str) -> str:
        # Always fall back gracefully — never return empty string
        if not self.is_trained or not self.train_sql:
            return "SELECT * FROM unknown;"

        x = self.tfidf.transform([nl_query])

        # Cosine similarity fallback (handles OOV / zero-vector queries)
        sims = cosine_similarity(x, self.train_vectors)[0]
        best_global = int(np.argmax(sims))

        # Try RF template prediction first
        try:
            pred_class    = self.clf.predict(x)[0]
            pred_template = self.label_encoder.inverse_transform([pred_class])[0]
            candidates    = self.template_to_sqls.get(pred_template, [])

            if candidates:
                cand_nls  = [c[0] for c in candidates]
                cand_sqls = [c[1] for c in candidates]
                cand_vecs = self.tfidf.transform(cand_nls)
                c_sims    = cosine_similarity(x, cand_vecs)[0]
                return cand_sqls[int(np.argmax(c_sims))]
        except Exception:
            pass

        # Global cosine fallback
        return self.train_sql[best_global]

    def predict_batch(self, nl_queries: list) -> list:
        return [self.predict(q) for q in nl_queries]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf, 'clf': self.clf,
                'label_encoder': self.label_encoder,
                'train_nl': self.train_nl, 'train_sql': self.train_sql,
                'train_templates': self.train_templates,
                'template_to_sqls': self.template_to_sqls,
                'train_vectors': self.train_vectors,
                'is_trained': True,   # ← always True on save
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.tfidf            = d['tfidf']
        self.clf              = d['clf']
        self.label_encoder    = d['label_encoder']
        self.train_nl         = d['train_nl']
        self.train_sql        = d['train_sql']
        self.train_templates  = d['train_templates']
        self.template_to_sqls = d['template_to_sqls']
        self.train_vectors    = d['train_vectors']
        self.is_trained       = True   # ← force True on load
        return self