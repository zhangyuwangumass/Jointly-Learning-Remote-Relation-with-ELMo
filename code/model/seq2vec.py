from utils.char_vectorizer import CharVectorizer

vectorizer = CharVectorizer()

f = open('../data/sentence', 'r')

sentences = f.readlines()
vectorizer.vec_and_save(sentences, 100)
