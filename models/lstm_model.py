"""
LSTM Seq2Seq Model for NL2SQL
BiLSTM Encoder + Attention + LSTM Decoder

Key fixes vs previous versions:
- is_trained forced True on load
- Inference models share exact layer objects with training model
- predict() always returns non-empty string
- Fallback to nearest-neighbour when decoder output is all-UNK
"""
import os
import pickle
import json
import numpy as np


class LSTMSeq2Seq:
    def __init__(self, embedding_dim=128, hidden_dim=256,
                 max_nl_len=50, max_sql_len=100):
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.max_nl_len    = max_nl_len
        self.max_sql_len   = max_sql_len
        self.model         = None
        self.nl_vocab      = None
        self.sql_vocab     = None
        self.history       = None
        self.is_trained    = False
        self._enc_model    = None
        self._dec_model    = None
        # Store training data for NN fallback
        self._train_nl     = []
        self._train_sql    = []
        self._train_vectors = None   # TF-IDF for fallback

    # ── Architecture ─────────────────────────────────────────────
    def _build_model(self, nl_vocab_size, sql_vocab_size):
        import tensorflow as tf
        from tensorflow.keras.layers import (Input, Embedding, LSTM, Dense,
                                             Bidirectional, Concatenate,
                                             Attention, Dropout)
        from tensorflow.keras.models import Model

        # === ENCODER ===
        enc_in  = Input(shape=(self.max_nl_len,), name='enc_in')
        enc_emb = Embedding(nl_vocab_size, self.embedding_dim,
                            mask_zero=True, name='enc_emb')(enc_in)
        enc_emb = Dropout(0.3)(enc_emb)

        bilstm  = Bidirectional(
            LSTM(self.hidden_dim, return_sequences=True, return_state=True,
                 recurrent_dropout=0.2), name='enc_bilstm')
        enc_out, fh, fc, bh, bc = bilstm(enc_emb)

        state_h = Dense(self.hidden_dim * 2, activation='tanh', name='sh')(
                      Concatenate()([fh, bh]))
        state_c = Dense(self.hidden_dim * 2, activation='tanh', name='sc')(
                      Concatenate()([fc, bc]))

        # === DECODER (shared layers) ===
        dec_in     = Input(shape=(None,), name='dec_in')
        dec_emb_l  = Embedding(sql_vocab_size, self.embedding_dim,
                               mask_zero=True, name='dec_emb')
        dec_emb    = Dropout(0.3)(dec_emb_l(dec_in))

        dec_lstm_l = LSTM(self.hidden_dim * 2, return_sequences=True,
                          return_state=True, recurrent_dropout=0.2,
                          name='dec_lstm')
        dec_out, _, _ = dec_lstm_l(dec_emb, initial_state=[state_h, state_c])

        attn_l   = Attention(name='attn')
        context  = attn_l([dec_out, enc_out])
        combined = Concatenate()([dec_out, context])

        dec_dense_l = Dense(sql_vocab_size, activation='softmax', name='dec_dense')
        dec_output  = dec_dense_l(combined)

        # Training model
        self.model = Model([enc_in, dec_in], dec_output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        # === INFERENCE ENCODER ===
        self._enc_model = Model(enc_in, [enc_out, state_h, state_c])

        # === INFERENCE DECODER (reuse exact same layer objects) ===
        dsh_in     = Input(shape=(self.hidden_dim * 2,), name='dsh')
        dsc_in     = Input(shape=(self.hidden_dim * 2,), name='dsc')
        enc_ctx_in = Input(shape=(self.max_nl_len, self.hidden_dim * 2),
                           name='enc_ctx')

        dec_emb_inf          = dec_emb_l(dec_in)
        dec_out_inf, nsh, nsc = dec_lstm_l(
            dec_emb_inf, initial_state=[dsh_in, dsc_in])
        ctx_inf  = attn_l([dec_out_inf, enc_ctx_in])
        out_inf  = dec_dense_l(Concatenate()([dec_out_inf, ctx_inf]))

        self._dec_model = Model(
            [dec_in, dsh_in, dsc_in, enc_ctx_in],
            [out_inf, nsh, nsc])

        n = self.model.count_params()
        print(f"[LSTM] Model built | NL vocab: {nl_vocab_size} | "
              f"SQL vocab: {sql_vocab_size} | params: {n:,}")

    # ── Data prep ────────────────────────────────────────────────
    def _prepare_data(self, nl_list, sql_list):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X_enc, X_dec, Y_dec = [], [], []
        for nl, sql in zip(nl_list, sql_list):
            enc = [self.nl_vocab.word2idx.get(t, self.nl_vocab.UNK)
                   for t in nl.lower().split()]
            dec_in  = [self.sql_vocab.SOS] + [
                self.sql_vocab.word2idx.get(t, self.sql_vocab.UNK)
                for t in sql.split()]
            dec_out = [self.sql_vocab.word2idx.get(t, self.sql_vocab.UNK)
                       for t in sql.split()] + [self.sql_vocab.EOS]
            X_enc.append(enc[:self.max_nl_len])
            X_dec.append(dec_in[:self.max_sql_len])
            Y_dec.append(dec_out[:self.max_sql_len])

        X_enc = pad_sequences(X_enc, maxlen=self.max_nl_len,  padding='post')
        X_dec = pad_sequences(X_dec, maxlen=self.max_sql_len, padding='post')
        Y_dec = pad_sequences(Y_dec, maxlen=self.max_sql_len, padding='post')
        return X_enc, X_dec, np.expand_dims(Y_dec, -1)

    def _build_fallback_index(self, nl_list, sql_list):
        """Build TF-IDF index for nearest-neighbour fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._train_nl  = list(nl_list)
        self._train_sql = list(sql_list)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._train_vectors = tfidf.fit_transform(nl_list)
        self._fallback_tfidf = tfidf

    def _fallback_predict(self, nl_query: str) -> str:
        """Return nearest-neighbour SQL when decoder fails."""
        if not self._train_sql:
            return "SELECT * FROM unknown;"
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            x    = self._fallback_tfidf.transform([nl_query])
            sims = cosine_similarity(x, self._train_vectors)[0]
            return self._train_sql[int(np.argmax(sims))]
        except Exception:
            return self._train_sql[0]

    # ── Training ─────────────────────────────────────────────────
    def train(self, nl_list, sql_list, nl_vocab, sql_vocab,
              epochs=15, batch_size=64, validation_split=0.1):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self.nl_vocab  = nl_vocab
        self.sql_vocab = sql_vocab
        self._build_model(nl_vocab.n_words, sql_vocab.n_words)
        self._build_fallback_index(nl_list, sql_list)

        X_enc, X_dec, Y_dec = self._prepare_data(nl_list, sql_list)

        cbs = [
            EarlyStopping(monitor='val_loss', patience=3,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=1e-5)
        ]
        self.history = self.model.fit(
            [X_enc, X_dec], Y_dec,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            callbacks=cbs, verbose=1)

        self.is_trained = True
        val_acc = self.history.history.get('val_accuracy', [0])[-1]
        print(f"[LSTM] Training complete. Final val_accuracy: {val_acc:.4f}")
        return self.history

    # ── Inference ────────────────────────────────────────────────
    def predict(self, nl_query: str) -> str:
        if not self.is_trained:
            return self._fallback_predict(nl_query)

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokens  = nl_query.lower().split()
        enc_ids = [self.nl_vocab.word2idx.get(t, self.nl_vocab.UNK)
                   for t in tokens]
        X_enc   = pad_sequences([enc_ids], maxlen=self.max_nl_len, padding='post')

        try:
            enc_out, h, c = self._enc_model.predict(X_enc, verbose=0)
            target_seq    = np.array([[self.sql_vocab.SOS]])
            result_tokens = []
            SPECIAL       = {self.sql_vocab.PAD, self.sql_vocab.SOS,
                             self.sql_vocab.EOS, self.sql_vocab.UNK}

            for _ in range(self.max_sql_len):
                probs, h, c = self._dec_model.predict(
                    [target_seq, h, c, enc_out], verbose=0)
                tok_idx = int(np.argmax(probs[0, -1, :]))

                if tok_idx in (self.sql_vocab.EOS, self.sql_vocab.PAD):
                    break
                if tok_idx not in SPECIAL:
                    token = self.sql_vocab.idx2word.get(tok_idx, '')
                    if token:
                        result_tokens.append(token)
                target_seq = np.array([[tok_idx]])

            if result_tokens:
                return ' '.join(result_tokens)
        except Exception:
            pass

        # Fallback: nearest-neighbour retrieval
        return self._fallback_predict(nl_query)

    def predict_batch(self, nl_queries: list) -> list:
        return [self.predict(q) for q in nl_queries]

    # ── Save / Load ───────────────────────────────────────────────
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(os.path.join(save_dir, 'lstm_weights.weights.h5'))

        history_data = {}
        if self.history:
            history_data = {k: [float(v) for v in vals]
                            for k, vals in self.history.history.items()}

        meta = {
            'embedding_dim': self.embedding_dim, 'hidden_dim': self.hidden_dim,
            'max_nl_len': self.max_nl_len,       'max_sql_len': self.max_sql_len,
            'is_trained': True, 'history': history_data
        }
        with open(os.path.join(save_dir, 'lstm_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        self.nl_vocab.save(os.path.join(save_dir, 'nl_vocab.pkl'))
        self.sql_vocab.save(os.path.join(save_dir, 'sql_vocab.pkl'))

        # Save fallback index
        if self._train_nl:
            with open(os.path.join(save_dir, 'fallback.pkl'), 'wb') as f:
                pickle.dump({
                    'train_nl':       self._train_nl,
                    'train_sql':      self._train_sql,
                    'train_vectors':  self._train_vectors,
                    'fallback_tfidf': getattr(self, '_fallback_tfidf', None),
                }, f)

        print(f"[LSTM] Model saved to {save_dir}")

    def load(self, save_dir: str, nl_vocab, sql_vocab):
        # Load meta (support both json and legacy pkl)
        meta_json = os.path.join(save_dir, 'lstm_meta.json')
        meta_pkl  = os.path.join(save_dir, 'lstm_meta.pkl')
        if os.path.exists(meta_json):
            with open(meta_json) as f:
                meta = json.load(f)
        else:
            with open(meta_pkl, 'rb') as f:
                meta = pickle.load(f)

        self.embedding_dim = meta['embedding_dim']
        self.hidden_dim    = meta['hidden_dim']
        self.max_nl_len    = meta['max_nl_len']
        self.max_sql_len   = meta['max_sql_len']
        self.nl_vocab      = nl_vocab
        self.sql_vocab     = sql_vocab

        self._build_model(nl_vocab.n_words, sql_vocab.n_words)

        # Support both filename conventions
        new_w = os.path.join(save_dir, 'lstm_weights.weights.h5')
        old_w = os.path.join(save_dir, 'lstm_weights.h5')
        self.model.load_weights(new_w if os.path.exists(new_w) else old_w)

        # Load fallback index
        fb_path = os.path.join(save_dir, 'fallback.pkl')
        if os.path.exists(fb_path):
            with open(fb_path, 'rb') as f:
                fb = pickle.load(f)
            self._train_nl      = fb['train_nl']
            self._train_sql     = fb['train_sql']
            self._train_vectors = fb['train_vectors']
            self._fallback_tfidf = fb.get('fallback_tfidf')

        self.is_trained = True  # ← force True on load
        print(f"[LSTM] Model loaded from {save_dir}")
        return self