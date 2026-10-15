def carga_y_analisis():
    # Cargar el dataset
    df = pd.read_csv("lenguaje_claro_dataset.csv")

    # Inspeccionar los datos
    print("Dimensiones:", df.shape)
    print("\nColumnas:", df.columns.tolist())
    print("\nValores nulos:")
    print(df.isnull().sum())

    # Comprobar duplicados
    print("\nDuplicados:", df.duplicated().sum())

    # Eliminar filas sin texto original o simplificado
    df = df.dropna(subset=["question", "answer"]).copy()

    # Eliminar textos vacíos
    df = df[
        df["question"].astype(str).str.strip().ne("")
        & df["answer"].astype(str).str.strip().ne("")
    ].copy()

    # Conservar un identificador
    df = df.reset_index(drop=True)
    df["document_id"] = df.index

    # Longitud de los textos
    df["longitud_original"] = df["question"].str.split().str.len()
    df["longitud_simplificada"] = df["answer"].str.split().str.len()

    print("\nLongitud media del original:",
        df["longitud_original"].mean())

    print("Longitud media de la simplificación:",
        df["longitud_simplificada"].mean())




import spacy
'''
pip install pandas spacy
python -m spacy download es_core_news_md
'''
def limpieza_preprocesamiento(df):
    nlp = spacy.load("es_core_news_md")

    def procesar_texto(texto):
        doc = nlp(str(texto))

        tokens = [
            token.text
            for token in doc
            if not token.is_space
        ]

        oraciones = [
            oracion.text
            for oracion in doc.sents
        ]

        lemas = [
            token.lemma_
            for token in doc
            if not token.is_space
        ]

        entidades = [
            {
                "texto": entidad.text,
                "tipo": entidad.label_
            }
            for entidad in doc.ents
        ]

        return {
            "tokens": tokens,
            "oraciones": oraciones,
            "lemas": lemas,
            "entidades": entidades
        }

    # Aplicar al texto original
    df["procesado_original"] = df["question"].apply(procesar_texto)

    # Aplicar al texto simplificado
    df["procesado_simplificado"] = df["answer"].apply(procesar_texto)

    print(df[[
        "question",
        "procesado_original",
        "answer",
        "procesado_simplificado"
    ]].head())





from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
def bow_tfidf(df):

    # Usar los textos originales
    textos = df["question"].astype(str).tolist()

    # Bag of Words
    bow = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2
    )

    X_bow = bow.fit_transform(textos)

    # TF-IDF
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X_tfidf = tfidf.fit_transform(textos)

    print("Dimensiones BoW:", X_bow.shape)
    print("Dimensiones TF-IDF:", X_tfidf.shape)
    print("Vocabulario BoW:", len(bow.vocabulary_))
    print("Vocabulario TF-IDF:", len(tfidf.vocabulary_))




from gensim.models import Word2Vec
def wordembedding_nocontx(df):
    # Utilizar los tokens generados con spaCy
    corpus_tokens = [
        registro["tokens"]
        for registro in df["procesado_original"]
    ]

    # Entrenar Word2Vec
    modelo_w2v = Word2Vec(
        sentences=corpus_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=20,
        seed=42
    )

    # Guardar el modelo
    modelo_w2v.save("word2vec_juridico.model")

    # Consultar palabras similares
    if "sentencia" in modelo_w2v.wv:
        print(modelo_w2v.wv.most_similar("sentencia", topn=10))



def analisis(corpus_tokens, modelo_w2v):
    total_tokens = 0
    tokens_desconocidos = 0
    palabras_desconocidas = set()

    for tokens in corpus_tokens:
        for palabra in tokens:
            total_tokens += 1

            if palabra not in modelo_w2v.wv:
                tokens_desconocidos += 1
                palabras_desconocidas.add(palabra)

    porcentaje_oov = (
        100 * tokens_desconocidos / total_tokens
        if total_tokens else 0
    )

    print("Total de tokens:", total_tokens)
    print("Tokens no representados:", tokens_desconocidos)
    print("Palabras desconocidas distintas:", len(palabras_desconocidas))
    print(f"Porcentaje de tokens no representados: {porcentaje_oov:.2f}%")
    print("Ejemplos:", list(palabras_desconocidas)[:30])



import torch
from transformers import AutoTokenizer, AutoModel
'''
pip install transformers torch
'''
def wordembb_context(df):
    modelo_nombre = "PlanTL-GOB-ES/roberta-base-bne"

    tokenizador = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModel.from_pretrained(modelo_nombre)

    modelo.eval()

    def obtener_embeddings(texto):
        entradas = tokenizador(
            str(texto),
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            salida = modelo(**entradas)

        # Vectores contextuales de los tokens
        embeddings_tokens = salida.last_hidden_state

        # Máscara para ignorar el padding
        mascara = entradas["attention_mask"].unsqueeze(-1)

        # Promedio de los vectores de tokens válidos
        embedding_texto = (
            (embeddings_tokens * mascara).sum(dim=1)
            / mascara.sum(dim=1).clamp(min=1)
        )

        return embedding_texto.squeeze(0).cpu().numpy()

    vector = obtener_embeddings(df["question"].iloc[0])

    print("Dimensiones del embedding:", vector.shape)



#hay q guardar en csv