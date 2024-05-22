HP = {
    "embed_dim": 8,            # Embedding dimension: Size of the embedding vectors.
    "hidden_dim": 32,          # Hidden dimension: Size of the hidden layers in the model.
    "num_layers": 4,            # Number of layers: The number of layers in the model (e.g., in LSTM or Transformer models).
    "num_heads": 16,            # Number of heads: The number of heads in the multi-head attention mechanism.
    "learning_rate": 0.005,     # Learning rate: The step size at each iteration while moving toward a minimum of a loss function.
    "epochs": 4,                # Epochs: The number of complete passes through the training dataset.
    "batch_size": 8,          # Batch size: The number of training examples utilized in one iteration.
    "context_window": 32,      # Context window: The size of the window of context used for models that require a fixed input size.
    "log_interval": 1,         # Log interval: The interval (in iterations) at which training progress (e.g., loss) is logged.
    "dropout": 0.25,            # Dropout: The probability of dropout for regularization in the model.
    "freq_penalty": 0.01        # Frequency penalties
}

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
