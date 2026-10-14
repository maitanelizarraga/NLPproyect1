import fasttext
import fasttext.util
from datasets import load_dataset

# Descargar 157 word vectors preentrenados con Common Crawl y Wikipedia utilizando fastText (CBOW), caracteres con n-gramas de largura 5, window size de 5 y 10 negativos.
fasttext.util.download_model('es', if_exists='ignore')
ft=fasttext.load_model('cc.es.300.bin')

ds = load_dataset("somosnlp/lenguaje-claro-dataset")
print(ds['train'][0]) 