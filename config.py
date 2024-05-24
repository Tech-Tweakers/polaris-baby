HP = {
    "embed_dim": 8,            # Embedding dimension: Size of the embedding vectors.
    "hidden_dim": 64,          # Hidden dimension: Size of the hidden layers in the model.
    "num_layers": 2,            # Number of layers: The number of layers in the model (e.g., in LSTM or Transformer models).
    "num_heads": 8,            # Number of heads: The number of heads in the multi-head attention mechanism.
    "learning_rate": 0.005,     # Learning rate: The step size at each iteration while moving toward a minimum of a loss function.
    "epochs": 2,                # Epochs: The number of complete passes through the training dataset.
    "batch_size": 32,          # Batch size: The number of training examples utilized in one iteration.
    "context_window": 32,      # Context window: The size of the window of context used for models that require a fixed input size.
    "log_interval": 32,         # Log interval: The interval (in iterations) at which training progress (e.g., loss) is logged.
    "dropout": 0.25,            # Dropout: The probability of dropout for regularization in the model.
    "freq_penalty": 0.009        # Frequency penalties
}

HP_configs = [
    {'context_window': 8, 'batch_size': 8, 'embed_dim': 8, 'hidden_dim': 64, 'dropout': 0.4, 'num_layers': 2, 'num_heads': 8, 'learning_rate': 0.004, 'epochs': 3, 'log_interval': 8},
    {'context_window': 16, 'batch_size': 16, 'embed_dim': 8, 'hidden_dim': 64, 'dropout': 0.3, 'num_layers': 2, 'num_heads': 8, 'learning_rate': 0.003, 'epochs': 3, 'log_interval': 16},
    {'context_window': 32, 'batch_size': 32, 'embed_dim': 8, 'hidden_dim': 64, 'dropout': 0.2, 'num_layers': 2, 'num_heads': 8, 'learning_rate': 0.002, 'epochs': 3, 'log_interval': 32},
    {'context_window': 64, 'batch_size': 64, 'embed_dim': 8, 'hidden_dim': 64, 'dropout': 0.1, 'num_layers': 2, 'num_heads': 8, 'learning_rate': 0.001, 'epochs': 3, 'log_interval': 64},
    {'context_window': 128, 'batch_size': 128, 'embed_dim': 8, 'hidden_dim': 64, 'dropout': 0.0, 'num_layers': 2, 'num_heads': 8, 'learning_rate': 0.001, 'epochs': 3, 'log_interval': 128},
]

CC = {
    "vocab_size": None,
    "vocab": None,
    "val": None,
    "encoded_text": None,
    "weights_matrix": None,
}

class Colors:
    HEADER = '\033[95m'             # Purple
    OKBLUE = '\033[94m'             # Blue
    OKGREEN = '\033[92m'            # Green
    WARNING = '\033[93m'            # Yellow
    FAIL = '\033[91m'               # Red
    ENDC = '\033[0m'                # Reset color
    BOLD = '\033[1m'                # Bold text
    UNDERLINE = '\033[4m'           # Underline text
    CYAN = '\033[96m'               # Cyan
    WHITE = '\033[97m'              # White
    YELLOW_BACKGROUND = '\033[43m'  # Yellow background
    BLUE_BACKGROUND = '\033[44m'    # Blue background
