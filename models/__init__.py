from models.rule_based import RuleBasedNLP
from models.random_forest_model import RandomForestNL2SQL
from models.lstm_model import LSTMSeq2Seq
from models.t5_model import T5NL2SQL

MODEL_REGISTRY = {
    'Rule-Based NLP': RuleBasedNLP,
    'Random Forest': RandomForestNL2SQL,
    'LSTM': LSTMSeq2Seq,
    'T5 Transformer': T5NL2SQL,
}

MODEL_COLORS = {
    'Rule-Based NLP': '#FF6B9D',
    'Random Forest': '#4ECDC4',
    'LSTM': '#45B7D1',
    'T5 Transformer': '#FFA07A',
}
