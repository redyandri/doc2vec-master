from gensim.models import Word2Vec
from gensim.models import FastText

model_path=r"model\fasttext100.bin"
fasttext_model = FastText.load(model_path)
print(fasttext_model.wv.most_similar("sql server"))
model_path=r"model\word2vec100.bin"
w2v_model = Word2Vec.load(model_path)
print(w2v_model.wv.most_similar("sql server"))