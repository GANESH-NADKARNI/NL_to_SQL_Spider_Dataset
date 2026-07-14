"""
NL2SQL Research - Training Script
==================================
Run this script ONCE before launching the Streamlit app.

Usage:
    python train.py [--models all] [--epochs-lstm 15] [--epochs-t5 5]

Models trained:
    1. Rule-Based NLP    (fast, ~30s)
    2. Random Forest     (medium, ~2-5min)
    3. LSTM Seq2Seq      (slow, ~10-30min depending on hardware)
    4. T5 Transformer    (slowest, ~30min-2hrs depending on hardware)

Outputs saved to: ./trained_models/
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocess import load_and_preprocess, build_vocabularies, tokenize_nl, tokenize_sql
from data.split_data import create_familiar_unseen_split, save_splits
from utils.evaluate import compute_metrics

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
DATA_SPLITS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'splits')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'NL2SQL_Query_Dataset.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Train NL2SQL models')
    parser.add_argument('--models', nargs='+', 
                        choices=['rule', 'rf', 'lstm', 't5', 'all'],
                        default=['all'],
                        help='Which models to train')
    parser.add_argument('--epochs-lstm', type=int, default=15, help='LSTM training epochs')
    parser.add_argument('--epochs-t5', type=int, default=5, help='T5 training epochs')
    parser.add_argument('--batch-lstm', type=int, default=64, help='LSTM batch size')
    parser.add_argument('--batch-t5', type=int, default=16, help='T5 batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (None=all). Use small number for quick test.')
    return parser.parse_args()


def prepare_data(max_samples=None):
    """Load and prepare all data splits."""
    print("\n" + "="*60)
    print("STEP 1: Data Preprocessing & Splitting")
    print("="*60)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        print("Please place NL2SQL_Query_Dataset.csv in the data/ directory")
        sys.exit(1)

    print(f"Loading dataset from {CSV_PATH}...")
    df = load_and_preprocess(CSV_PATH)
    print(f"Loaded {len(df)} samples | {df['Query'].nunique()} unique SQL queries")

    # Create splits
    train_df, familiar_test_df, unseen_test_df = create_familiar_unseen_split(df)
    save_splits(train_df, familiar_test_df, unseen_test_df, DATA_SPLITS_DIR)

    if max_samples:
        train_df = train_df.head(max_samples)
        print(f"Using {max_samples} samples for quick testing")

    # Build vocabularies from training data
    print("\nBuilding vocabularies...")
    nl_vocab, sql_vocab = build_vocabularies(train_df)
    print(f"NL vocab size: {nl_vocab.n_words} | SQL vocab size: {sql_vocab.n_words}")

    # Save vocabularies
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    nl_vocab.save(os.path.join(TRAINED_MODELS_DIR, 'nl_vocab.pkl'))
    sql_vocab.save(os.path.join(TRAINED_MODELS_DIR, 'sql_vocab.pkl'))

    return train_df, familiar_test_df, unseen_test_df, nl_vocab, sql_vocab


def evaluate_and_save_metrics(model, model_name, train_df, familiar_test_df, unseen_test_df):
    """Evaluate model on all splits and save metrics."""
    print(f"\n  Evaluating {model_name}...")

    def safe_eval(nl_list, sql_list, name):
        try:
            preds = model.predict_batch(list(nl_list))
            return compute_metrics(preds, list(sql_list))
        except Exception as e:
            print(f"  Warning: Evaluation failed on {name}: {e}")
            return {'exact_match_accuracy': 0, 'template_match_accuracy': 0, 'total': 0}

    # Evaluate on subsets to save time
    train_sample = train_df.sample(min(200, len(train_df)), random_state=42)
    fam_sample = familiar_test_df.sample(min(100, len(familiar_test_df)), random_state=42)
    unseen_sample = unseen_test_df.sample(min(100, len(unseen_test_df)), random_state=42)

    train_metrics = safe_eval(train_sample['Prompt'], train_sample['Query'], 'train')
    fam_metrics = safe_eval(fam_sample['Prompt'], fam_sample['Query'], 'familiar')
    unseen_metrics = safe_eval(unseen_sample['Prompt'], unseen_sample['Query'], 'unseen')

    metrics = {
        'model_name': model_name,
        'train_exact': train_metrics['exact_match_accuracy'],
        'train_template': train_metrics['template_match_accuracy'],
        'familiar_exact': fam_metrics['exact_match_accuracy'],
        'familiar_template': fam_metrics['template_match_accuracy'],
        'unseen_exact': unseen_metrics['exact_match_accuracy'],
        'unseen_template': unseen_metrics['template_match_accuracy'],
    }

    print(f"  Train: exact={metrics['train_exact']:.1%} | template={metrics['train_template']:.1%}")
    print(f"  Familiar: exact={metrics['familiar_exact']:.1%} | template={metrics['familiar_template']:.1%}")
    print(f"  Unseen: exact={metrics['unseen_exact']:.1%} | template={metrics['unseen_template']:.1%}")

    return metrics


def train_rule_based(train_df, familiar_test_df, unseen_test_df):
    """Train Rule-Based NLP model."""
    print("\n" + "="*60)
    print("Training: Rule-Based NLP")
    print("="*60)
    t0 = time.time()

    from models.rule_based import RuleBasedNLP
    model = RuleBasedNLP()
    model.train(list(train_df['Prompt']), list(train_df['Query']))

    save_path = os.path.join(TRAINED_MODELS_DIR, 'rule_based.pkl')
    model.save(save_path)
    print(f"Saved to {save_path}")

    metrics = evaluate_and_save_metrics(model, 'Rule-Based NLP', train_df, familiar_test_df, unseen_test_df)
    metrics['training_time_sec'] = time.time() - t0
    print(f"Done in {metrics['training_time_sec']:.1f}s")
    return metrics


def train_random_forest(train_df, familiar_test_df, unseen_test_df):
    """Train Random Forest model."""
    print("\n" + "="*60)
    print("Training: Random Forest")
    print("="*60)
    t0 = time.time()

    from models.random_forest_model import RandomForestNL2SQL
    model = RandomForestNL2SQL(n_estimators=200, max_features=5000)
    model.train(list(train_df['Prompt']), list(train_df['Query']))

    save_path = os.path.join(TRAINED_MODELS_DIR, 'random_forest.pkl')
    model.save(save_path)
    print(f"Saved to {save_path}")

    metrics = evaluate_and_save_metrics(model, 'Random Forest', train_df, familiar_test_df, unseen_test_df)
    metrics['training_time_sec'] = time.time() - t0
    print(f"Done in {metrics['training_time_sec']:.1f}s")
    return metrics


def train_lstm(train_df, familiar_test_df, unseen_test_df, nl_vocab, sql_vocab,
               epochs=15, batch_size=64):
    """Train LSTM Seq2Seq model."""
    print("\n" + "="*60)
    print("Training: LSTM Seq2Seq")
    print("="*60)
    t0 = time.time()

    from models.lstm_model import LSTMSeq2Seq
    model = LSTMSeq2Seq(embedding_dim=128, hidden_dim=256, max_nl_len=50, max_sql_len=100)

    history = model.train(
        list(train_df['Prompt']),
        list(train_df['Query']),
        nl_vocab=nl_vocab,
        sql_vocab=sql_vocab,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    save_dir = os.path.join(TRAINED_MODELS_DIR, 'lstm')
    model.save(save_dir)
    print(f"Saved to {save_dir}")

    # Save training history
    history_data = {
        'loss': [float(x) for x in history.history.get('loss', [])],
        'accuracy': [float(x) for x in history.history.get('accuracy', [])],
        'val_loss': [float(x) for x in history.history.get('val_loss', [])],
        'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
    }
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history_data, f)

    metrics = evaluate_and_save_metrics(model, 'LSTM', train_df, familiar_test_df, unseen_test_df)
    metrics['training_time_sec'] = time.time() - t0
    metrics['history'] = history_data
    print(f"Done in {metrics['training_time_sec']:.1f}s")
    return metrics


def train_t5(train_df, familiar_test_df, unseen_test_df, epochs=5, batch_size=16):
    """Train T5 Transformer model."""
    print("\n" + "="*60)
    print("Training: T5 Transformer")
    print("="*60)
    t0 = time.time()

    from models.t5_model import T5NL2SQL
    model = T5NL2SQL(model_name='t5-small', max_nl_len=128, max_sql_len=200)

    model.train(
        list(train_df['Prompt']),
        list(train_df['Query']),
        epochs=epochs,
        batch_size=batch_size,
        val_split=0.1
    )

    save_dir = os.path.join(TRAINED_MODELS_DIR, 't5')
    model.save(save_dir)
    print(f"Saved to {save_dir}")

    metrics = evaluate_and_save_metrics(model, 'T5 Transformer', train_df, familiar_test_df, unseen_test_df)
    metrics['training_time_sec'] = time.time() - t0
    metrics['history'] = model.history
    print(f"Done in {metrics['training_time_sec']:.1f}s")
    return metrics


def main():
    args = parse_args()

    train_models = set()
    if 'all' in args.models:
        train_models = {'rule', 'rf', 'lstm', 't5'}
    else:
        train_models = set(args.models)

    print("\n" + "="*60)
    print("NL2SQL Research - Training Pipeline")
    print("="*60)
    print(f"Models to train: {train_models}")

    # Prepare data
    train_df, familiar_test_df, unseen_test_df, nl_vocab, sql_vocab = prepare_data(
        max_samples=args.max_samples
    )

    all_metrics = {}

    if 'rule' in train_models:
        try:
            m = train_rule_based(train_df, familiar_test_df, unseen_test_df)
            all_metrics['Rule-Based NLP'] = m
        except Exception as e:
            print(f"ERROR training Rule-Based NLP: {e}")

    if 'rf' in train_models:
        try:
            m = train_random_forest(train_df, familiar_test_df, unseen_test_df)
            all_metrics['Random Forest'] = m
        except Exception as e:
            print(f"ERROR training Random Forest: {e}")

    if 'lstm' in train_models:
        try:
            m = train_lstm(train_df, familiar_test_df, unseen_test_df, nl_vocab, sql_vocab,
                          epochs=args.epochs_lstm, batch_size=args.batch_lstm)
            all_metrics['LSTM'] = m
        except Exception as e:
            print(f"ERROR training LSTM: {e}")
            import traceback; traceback.print_exc()

    if 't5' in train_models:
        try:
            m = train_t5(train_df, familiar_test_df, unseen_test_df,
                        epochs=args.epochs_t5, batch_size=args.batch_t5)
            all_metrics['T5 Transformer'] = m
        except Exception as e:
            print(f"ERROR training T5: {e}")
            import traceback; traceback.print_exc()

    # Save all metrics
    metrics_path = os.path.join(TRAINED_MODELS_DIR, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Metrics saved to: {metrics_path}")
    print("\nSummary:")
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Familiar (template/exact): {metrics.get('familiar_template', 0):.1%} / {metrics.get('familiar_exact', 0):.1%}")
        print(f"  Unseen   (template/exact): {metrics.get('unseen_template', 0):.1%} / {metrics.get('unseen_exact', 0):.1%}")

    print("\n✓ Now run: streamlit run app.py")


if __name__ == '__main__':
    main()
