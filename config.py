import os

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

HP = {
    "embed_dim": 64,
    "hidden_dim": 128,
    "learning_rate": 0.005,
    "epochs": 1000,
    "batch_size": 32,
    "loss_threshold": 0.4,
    "num_layers": 2,
    "context_window": 64,
    "vocab_size": None,  # To be updated after loading data
    "vocab": None,  # To be updated after loading data
    "val": None,  # To be updated after validation
    "log_interval": 10,
    "encoded_text": None,  # To be updated after loading data  
}
