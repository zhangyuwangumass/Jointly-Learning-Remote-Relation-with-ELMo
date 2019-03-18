from utils.char_vectorizer import CharVectorizer

vectorizer = CharVectorizer()

f = open('../data/vocab', 'r')

vocab = f.readlines()
vectorizer.vocab_vec_and_save(vocab, 50)
