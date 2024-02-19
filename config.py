import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

HP = {
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    "learning_rate": 0.0005,
    "epochs": 5,
    "batch_size": 32,
    "loss_threshold": 0.4,
    "context_window": 128,
    "log_interval": 32,
}

CC = {
    "vocab_size": None,
    "vocab": None,
    "val": None,
    "encoded_text": None,
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