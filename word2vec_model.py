from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

class Word2VecTrainer:
    def __init__(self, input_file, vector_size=64, window=5, min_count=3, workers=4):
        self.input_file = input_file
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def load_sentences(self):
        with open(self.input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        sentences = sent_tokenize(text)  # Tokenize the text into sentences
        return sentences

    def tokenize_sentences(self, sentences):
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
        return tokenized_sentences

    def train_model(self):
        sentences = self.load_sentences()
        tokenized_corpus = self.tokenize_sentences(sentences)
        model = Word2Vec(sentences=tokenized_corpus, vector_size=self.vector_size, 
                         window=self.window, min_count=self.min_count, workers=self.workers)
        model.save("word2vec.model")
        print("Model trained and saved successfully.")
