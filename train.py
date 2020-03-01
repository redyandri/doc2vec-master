from victorinox import victorinox
from gensim.models import FastText

#https://radimrehurek.com/gensim/models/fasttext.html
dataset_path=r"data/dataset_lower_clean_stem_sentence.csv"
model_path=r"model/fasttext300.bin"
fasttext_model= FastText(dataset_path, size=300, window=5, min_count=5, workers=4,sg=1)
fasttext_model.save(model_path)
fasttext_model.wv.most_similar("nadine")