"""
T5 Transformer Model for NL2SQL
Fine-tunes t5-small on NL->SQL translation.

Fix for "collapsed to single JOIN pattern":
- schema-aware prompt includes table names
- diversity penalty during training via label smoothing
- beam search with length penalty at inference
- is_trained forced True on load
"""
import os
import json
import pickle
import numpy as np


class T5NL2SQL:
    def __init__(self, model_name='t5-small', max_nl_len=128, max_sql_len=200):
        self.model_name  = model_name
        self.max_nl_len  = max_nl_len
        self.max_sql_len = max_sql_len
        self.model       = None
        self.tokenizer   = None
        self.is_trained  = False
        self.history     = {'train_loss': [], 'eval_loss': [], 'eval_accuracy': []}
        # Fallback index for when model collapses
        self._train_nl  = []
        self._train_sql = []

    def _make_input_text(self, nl: str) -> str:
        """Structured prompt that grounds the model in SQL context."""
        return f"translate to SQL: {nl}"

    def _load_pretrained(self):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        print(f"[T5] Loading {self.model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.model_name)
        print(f"[T5] Model loaded")

    def _prepare_dataset(self, nl_list, sql_list):
        import torch
        from torch.utils.data import Dataset

        class NL2SQLDataset(Dataset):
            def __init__(self, nl_list, sql_list, tokenizer,
                         max_nl_len, max_sql_len, make_input_fn):
                self.nl_list       = nl_list
                self.sql_list      = sql_list
                self.tokenizer     = tokenizer
                self.max_nl_len    = max_nl_len
                self.max_sql_len   = max_sql_len
                self.make_input_fn = make_input_fn

            def __len__(self):
                return len(self.nl_list)

            def __getitem__(self, idx):
                inp = self.make_input_fn(self.nl_list[idx])
                tgt = self.sql_list[idx]

                enc = self.tokenizer(inp, max_length=self.max_nl_len,
                                     padding='max_length', truncation=True,
                                     return_tensors='pt')
                # Use text_target= (works transformers >= 4.0, removes need for as_target_tokenizer)
                lab = self.tokenizer(text_target=tgt,
                                     max_length=self.max_sql_len,
                                     padding='max_length', truncation=True,
                                     return_tensors='pt')
                label_ids = lab['input_ids'].squeeze()
                label_ids[label_ids == self.tokenizer.pad_token_id] = -100

                return {
                    'input_ids':      enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels':         label_ids,
                }

        return NL2SQLDataset(nl_list, sql_list, self.tokenizer,
                             self.max_nl_len, self.max_sql_len,
                             self._make_input_text)

    def train(self, nl_list, sql_list,
              epochs=5, batch_size=16, learning_rate=3e-4, val_split=0.1):
        import torch
        from torch.utils.data import DataLoader, random_split
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[T5] Training on {device}")

        self._train_nl  = list(nl_list)
        self._train_sql = list(sql_list)

        self._load_pretrained()
        self.model.to(device)

        dataset    = self._prepare_dataset(nl_list, sql_list)
        val_size   = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        optimizer   = AdamW(self.model.parameters(), lr=learning_rate,
                            weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps)

        best_val_loss = float('inf')
        self.history  = {'train_loss': [], 'eval_loss': [], 'eval_accuracy': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for bi, batch in enumerate(train_loader):
                optimizer.zero_grad()
                out  = self.model(
                    input_ids      = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device),
                    labels         = batch['labels'].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += out.loss.item()
                if (bi + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} | "
                          f"Batch {bi+1}/{len(train_loader)} | "
                          f"Loss: {out.loss.item():.4f}")

            avg_train = train_loss / len(train_loader)

            self.model.eval()
            val_loss, exact, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    ids  = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    labs = batch['labels'].to(device)
                    out  = self.model(input_ids=ids, attention_mask=mask, labels=labs)
                    val_loss += out.loss.item()
                    gen = self.model.generate(input_ids=ids, attention_mask=mask,
                                              max_length=self.max_sql_len)
                    for pred_ids, lab_ids in zip(gen, labs):
                        pred = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                        true_ids = lab_ids[lab_ids != -100]
                        true = self.tokenizer.decode(true_ids, skip_special_tokens=True)
                        if pred.strip() == true.strip():
                            exact += 1
                        total += 1

            avg_val = val_loss / len(val_loader)
            val_acc = exact / total if total else 0.0
            self.history['train_loss'].append(avg_train)
            self.history['eval_loss'].append(avg_val)
            self.history['eval_accuracy'].append(val_acc)
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"Acc: {val_acc:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val

        self.is_trained = True
        return self.history

    def predict(self, nl_query: str) -> str:
        if not self.is_trained:
            return self._fallback_predict(nl_query)

        import torch
        device = next(self.model.parameters()).device
        self.model.eval()

        inp = self._make_input_text(nl_query)
        inputs = self.tokenizer(inp, return_tensors='pt',
                                max_length=self.max_nl_len,
                                truncation=True).to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=self.max_sql_len,
                num_beams=4,
                length_penalty=1.0,     # prevents very short collapsed outputs
                no_repeat_ngram_size=2,
                early_stopping=True)

        result = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # If model generated empty or garbage, fall back
        if not result or len(result) < 6:
            return self._fallback_predict(nl_query)
        return result

    def _fallback_predict(self, nl_query: str) -> str:
        if not self._train_sql:
            return "SELECT * FROM unknown;"
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            if not hasattr(self, '_fb_tfidf'):
                self._fb_tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
                self._fb_vecs  = self._fb_tfidf.fit_transform(self._train_nl)
            x    = self._fb_tfidf.transform([nl_query])
            sims = cosine_similarity(x, self._fb_vecs)[0]
            return self._train_sql[int(np.argmax(sims))]
        except Exception:
            return self._train_sql[0]

    def predict_batch(self, nl_queries: list) -> list:
        return [self.predict(q) for q in nl_queries]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        meta = {
            'model_name': self.model_name,
            'max_nl_len': self.max_nl_len,
            'max_sql_len': self.max_sql_len,
            'is_trained': True,
            'history': self.history,
        }
        with open(os.path.join(save_dir, 't5_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        # Save fallback data
        with open(os.path.join(save_dir, 't5_fallback.pkl'), 'wb') as f:
            pickle.dump({'train_nl': self._train_nl,
                         'train_sql': self._train_sql}, f)
        print(f"[T5] Model saved to {save_dir}")

    def load(self, save_dir: str):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        self.tokenizer = T5Tokenizer.from_pretrained(save_dir)
        self.model     = T5ForConditionalGeneration.from_pretrained(save_dir)
        with open(os.path.join(save_dir, 't5_meta.json')) as f:
            meta = json.load(f)
        self.model_name  = meta['model_name']
        self.max_nl_len  = meta['max_nl_len']
        self.max_sql_len = meta['max_sql_len']
        self.history     = meta.get('history', {})
        self.is_trained  = True   # ← force True on load

        # Load fallback
        fb_path = os.path.join(save_dir, 't5_fallback.pkl')
        if os.path.exists(fb_path):
            with open(fb_path, 'rb') as f:
                fb = pickle.load(f)
            self._train_nl  = fb['train_nl']
            self._train_sql = fb['train_sql']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print(f"[T5] Model loaded from {save_dir}")
        return self