from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

# Function to read the input file and return a list of sentences
def load_sentences(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    return sentences

# Function to tokenize each sentence into words
def tokenize_sentences(sentences):
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    return tokenized_sentences

# Load and process the input text
input_file = 'input.txt'  # Adjust the file path if necessary
sentences = load_sentences(input_file)
tokenized_corpus = tokenize_sentences(sentences)

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=64, window=5, min_count=3, workers=4)

# Save the model
model.save("word2vec.model")

print("Model trained and saved successfully.")
