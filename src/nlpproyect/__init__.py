import fasttext
import fasttext.util
from datasets import load_dataset
import pandas as pd


set_seed = 123


def descarga_docs():
    # Descargar 157 word vectors preentrenados con Common Crawl y Wikipedia utilizando fastText (CBOW), caracteres con n-gramas de largura 5, window size de 5 y 10 negativos.
    fasttext.util.download_model('es', if_exists='ignore')
    ft=fasttext.load_model('cc.es.300.bin')
    ds = load_dataset("somosnlp/lenguaje-claro-dataset")
    print(ds['train'][0]) 

    print(len(ds['train']))

    return ds
        

def division(ds):
    TAMAÑO_ENTRENAMIENTO = 2800
    TAMAÑO_TEST = 647

    fin_train = TAMAÑO_ENTRENAMIENTO
    fin_test = TAMAÑO_ENTRENAMIENTO + TAMAÑO_TEST

    train = ds.select(range(0, fin_train))
    test = ds.select(range(fin_train, fin_test))
    validation = ds.select(range(fin_test, len(ds)))

    return train, test, validation
    


def main():
    #ds=descarga_docs()
    train, test, validation= division(ds)


if __name__ == "__main__":
    main()

