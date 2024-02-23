import torch
import argparse
import torch.nn.functional as F
from config import Colors, HP
from model import SmallRNNModel

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        print(f"{Colors.FAIL}Error opening or reading input file: {e}{Colors.ENDC}")
        return ""

lines = read_file('input.txt')

vocab = sorted(set(lines))
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi.get(ch, 0) for ch in s]
def top_k_top_p_filtering(logits, top_k=40, top_p=0.5, filter_value=-float('Inf')):
    assert logits.dim() == 1 

    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_text(seed_text, model, max_length, temperature=0.7, top_k=40, top_p=0.3):
    model.eval()
    text_generated = [seed_text]
    input_eval = torch.tensor(encode(seed_text), dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            predictions = model(input_eval)[:,-1,:]
            predictions = predictions / temperature
            filtered_logits = top_k_top_p_filtering(predictions.squeeze(), top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            if itos[predicted_id] == '\n':  
                print("\n")
                break

            generated_character = itos[predicted_id]
            print(generated_character, end='', flush=True)

            predicted_id_tensor = torch.tensor([[predicted_id]], dtype=torch.long)
            input_eval = torch.cat([input_eval, predicted_id_tensor], dim=1)

            text_generated.append(generated_character)

    return ''.join(text_generated)

def main(args):
    vocab_size = 1800
    embed_dim = HP['embed_dim']
    hidden_dim = HP['hidden_dim']
    dropout = HP['dropout']
    num_layers = HP['num_layers']
    model = SmallRNNModel(vocab_size, embed_dim, hidden_dim, dropout, num_layers)

    try:
        model.load_state_dict(torch.load("small_rnn_model_final.pth"))
        model.eval()
    except Exception as e:
        print(f"{Colors.FAIL}Error loading model: {e}{Colors.ENDC}")
        return

    generate_text(args.seed_text, model, args.max_length, args.temperature, args.top_k, args.top_p)
    print(f"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument("--seed_text", type=str, default=" ", help="Initial text to start generating from.")
    parser.add_argument("--max_length", type=int, default=2000, help="Maximum length of the generated text.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k filtering threshold.")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p (nucleus) filtering threshold.")

    args = parser.parse_args()
    main(args)
