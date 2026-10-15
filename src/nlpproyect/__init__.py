"""
Entrega 2 - Procesamiento y análisis de datos
Proyecto: Simplificación de textos jurídicos mediante PLN.

Instalación (en el entorno virtual donde se ejecutará el script):
    pip install pandas datasets spacy scikit-learn gensim torch transformers joblib
    python -m spacy download es_core_news_md

Ejecución:
    python entrega2.py

Si existe lenguaje_claro_dataset.csv en la carpeta del script, se utiliza ese archivo.
En caso contrario, se descarga somosnlp/lenguaje-claro-dataset desde Hugging Face.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import spacy
import torch
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
SEED = 123
TAMANO_TRAIN = 2800
TAMANO_TEST = 647
CARPETA_SALIDA = Path("resultados_e2")
ARCHIVO_LOCAL = Path("lenguaje_claro_dataset.csv")
MODELO_SPACY = "es_core_news_md"
MODELO_CONTEXTUAL = "PlanTL-GOB-ES/roberta-base-bne"
MAX_TOKENS_ROBERTA = 512
BATCH_SIZE_ROBERTA = 8


def cargar_dataset():
    """Carga el CSV local o, si no existe, el dataset de Hugging Face."""
    if ARCHIVO_LOCAL.exists():
        df = pd.read_csv(ARCHIVO_LOCAL)
        print(f"Dataset cargado desde {ARCHIVO_LOCAL.resolve()}")
    else:
        print("No se encuentra el CSV local. Descargando dataset de Hugging Face...")
        dataset = load_dataset("somosnlp/lenguaje-claro-dataset")
        if "train" not in dataset:
            raise ValueError("El dataset descargado no contiene la partición 'train'.")
        df = dataset["train"].to_pandas()

    columnas_necesarias = {"question", "answer"}
    faltan = columnas_necesarias - set(df.columns)
    if faltan:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(faltan)}. "
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    print("\n--- Análisis inicial ---")
    print("Dimensiones iniciales:", df.shape)
    print("Columnas:", df.columns.tolist())
    print("\nValores nulos por columna:\n", df.isnull().sum())
    print("\nFilas duplicadas:", int(df.duplicated().sum()))

    # Eliminar filas que no tienen texto original o simplificado.
    df = df.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")].copy()

    # Se eliminan duplicados exactos de los pares de texto para no repetir ejemplos.
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    df.insert(0, "document_id", np.arange(len(df), dtype=int))

    df["longitud_original"] = df["question"].str.split().str.len()
    df["longitud_simplificada"] = df["answer"].str.split().str.len()

    print("\nDimensiones tras limpieza:", df.shape)
    print("Longitud media del texto original:",
          round(float(df["longitud_original"].mean()), 2))
    print("Longitud media del texto simplificado:",
          round(float(df["longitud_simplificada"].mean()), 2))

    return df


def dividir_dataset(df):
    """
    Divide los datos de forma reproducible.
    Se mantienen los tamaños propuestos en el código inicial si hay suficientes filas.
    Si el dataset tiene otro tamaño, se aplica una división aproximada 68,5/15,8/15,7.
    """
    n = len(df)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 documentos para crear train/test/validación.")

    if n >= TAMANO_TRAIN + TAMANO_TEST + 1:
        train = df.iloc[:TAMANO_TRAIN].copy()
        test = df.iloc[TAMANO_TRAIN:TAMANO_TRAIN + TAMANO_TEST].copy()
        validation = df.iloc[TAMANO_TRAIN + TAMANO_TEST:].copy()
        estrategia = "Tamaños del código inicial: 2800 train, 647 test y resto validación."
    else:
        # División aleatoria reproducible para corpus más pequeños.
        train, resto = train_test_split(
            df, test_size=0.315, random_state=SEED, shuffle=True
        )
        test, validation = train_test_split(
            resto, test_size=0.50, random_state=SEED, shuffle=True
        )
        train, test, validation = train.copy(), test.copy(), validation.copy()
        estrategia = "División aleatoria reproducible aproximada 68,5/15,8/15,7."

    # Exportar cada partición. No se reinician los document_id.
    train.to_csv(CARPETA_SALIDA / "train.csv", index=False)
    test.to_csv(CARPETA_SALIDA / "test.csv", index=False)
    validation.to_csv(CARPETA_SALIDA / "validation.csv", index=False)

    print("\n--- División del corpus ---")
    print(estrategia)
    print(f"Train: {len(train)} | Test: {len(test)} | Validación: {len(validation)}")
    return train, test, validation


def cargar_spacy():
    """Carga el modelo español y muestra un mensaje útil si no está instalado."""
    try:
        return spacy.load(MODELO_SPACY)
    except OSError as exc:
        raise RuntimeError(
            f"No está instalado el modelo {MODELO_SPACY}. Ejecuta:\n"
            f"python -m spacy download {MODELO_SPACY}"
        ) from exc


def procesar_textos(df, nlp):
    """
    Tokeniza, separa en oraciones, obtiene lemas y entidades.
    Se procesan tanto los textos originales (question) como los simplificados (answer).
    """
    df = df.copy()
    originales = df["question"].tolist()
    simplificados = df["answer"].tolist()

    def extraer(doc):
        return {
            "tokens": [t.text for t in doc if not t.is_space],
            "oraciones": [s.text for s in doc.sents],
            "lemas": [t.lemma_ for t in doc if not t.is_space],
            "entidades": [
                {"texto": ent.text, "tipo": ent.label_}
                for ent in doc.ents
            ],
        }

    print("\nProcesando textos con spaCy...")
    docs_originales = list(nlp.pipe(originales, batch_size=32))
    docs_simplificados = list(nlp.pipe(simplificados, batch_size=32))

    df["procesado_original"] = [extraer(doc) for doc in docs_originales]
    df["procesado_simplificado"] = [extraer(doc) for doc in docs_simplificados]
    return df


def ajustar_bow_tfidf(train, test, validation):
    """
    Ajusta los vocabularios SOLO con train y transforma test/validación con esos vocabularios.
    Evita fuga de información entre particiones.
    """
    textos_train = train["question"].astype(str).tolist()
    textos_test = test["question"].astype(str).tolist()
    textos_val = validation["question"].astype(str).tolist()

    # min_df=2 puede fallar en corpus muy pequeños; en ese caso se usa min_df=1.
    min_df = 2 if len(train) >= 10 else 1
    bow = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df)
    tfidf = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True
    )

    X_bow_train = bow.fit_transform(textos_train)
    X_bow_test = bow.transform(textos_test)
    X_bow_val = bow.transform(textos_val)

    X_tfidf_train = tfidf.fit_transform(textos_train)
    X_tfidf_test = tfidf.transform(textos_test)
    X_tfidf_val = tfidf.transform(textos_val)

    joblib.dump(bow, CARPETA_SALIDA / "vectorizador_bow.joblib")
    joblib.dump(tfidf, CARPETA_SALIDA / "vectorizador_tfidf.joblib")

    matrices = {
        "bow_train": X_bow_train, "bow_test": X_bow_test, "bow_validation": X_bow_val,
        "tfidf_train": X_tfidf_train, "tfidf_test": X_tfidf_test,
        "tfidf_validation": X_tfidf_val,
    }
    for nombre, matriz in matrices.items():
        # Formato comprimido para conservar matrices dispersas sin convertirlas a densas.
        from scipy.sparse import save_npz
        save_npz(CARPETA_SALIDA / f"{nombre}.npz", matriz)

    print("\n--- BoW y TF-IDF ---")
    print("Vocabulario BoW:", len(bow.vocabulary_))
    print("Vocabulario TF-IDF:", len(tfidf.vocabulary_))
    for nombre, matriz in matrices.items():
        print(f"{nombre}: {matriz.shape}")

    return bow, tfidf


def entrenar_word2vec(train):
    """Entrena Word2Vec no contextual con los tokens del texto original de train."""
    corpus_tokens = [
        registro["tokens"]
        for registro in train["procesado_original"]
    ]

    modelo = Word2Vec(
        sentences=corpus_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,       # Skip-gram
        epochs=20,
        seed=SEED,
    )
    modelo.save(str(CARPETA_SALIDA / "word2vec_juridico.model"))

    print("\n--- Word2Vec no contextual ---")
    print("Vocabulario aprendido:", len(modelo.wv))
    if "sentencia" in modelo.wv:
        print("Palabras similares a 'sentencia':")
        print(modelo.wv.most_similar("sentencia", topn=min(10, len(modelo.wv))))

    return modelo


def analizar_oov(df, modelo_w2v):
    """
    Calcula el porcentaje de tokens que no tienen vector en Word2Vec.
    Se cuentan tokens de los textos originales y se informa también de palabras únicas OOV.
    """
    total_tokens = 0
    tokens_oov = 0
    palabras_oov = set()

    for registro in df["procesado_original"]:
        for palabra in registro["tokens"]:
            # Word2Vec es sensible a mayúsculas/minúsculas; se evalúan ambos formatos.
            candidatos = (palabra, palabra.lower())
            total_tokens += 1
            if not any(p in modelo_w2v.wv for p in candidatos):
                tokens_oov += 1
                palabras_oov.add(palabra)

    porcentaje = 100 * tokens_oov / total_tokens if total_tokens else 0.0
    resultado = {
        "total_tokens": total_tokens,
        "tokens_no_representados": tokens_oov,
        "palabras_desconocidas_distintas": len(palabras_oov),
        "porcentaje_tokens_no_representados": round(porcentaje, 4),
        "ejemplos_palabras_oov": sorted(palabras_oov)[:100],
    }
    with open(CARPETA_SALIDA / "analisis_oov_word2vec.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print("\n--- Análisis de palabras no representadas por Word2Vec ---")
    print("Total de tokens:", total_tokens)
    print("Tokens no representados:", tokens_oov)
    print("Palabras desconocidas distintas:", len(palabras_oov))
    print(f"Porcentaje de tokens no representados: {porcentaje:.2f}%")
    print("Ejemplos:", sorted(palabras_oov)[:30])
    if porcentaje > 10:
        print(
            "Nota: el porcentaje OOV es elevado. Se podría comparar Word2Vec con "
            "FastText preentrenado, que aprovecha subpalabras, o entrenar con más corpus."
        )
    return resultado


def generar_embeddings_contextuales(df):
    """
    Genera un vector contextual por texto con RoBERTa en español.
    Se usa el promedio de los vectores de tokens no-padding. Por defecto se generan
    embeddings para question y answer, guardados en archivos .npz.
    """
    print("\nCargando modelo contextual:", MODELO_CONTEXTUAL)
    tokenizador = AutoTokenizer.from_pretrained(MODELO_CONTEXTUAL)
    modelo = AutoModel.from_pretrained(MODELO_CONTEXTUAL)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)
    modelo.eval()
    print("Dispositivo:", dispositivo)

    def obtener_embeddings(textos, nombre):
        todos = []
        for inicio in range(0, len(textos), BATCH_SIZE_ROBERTA):
            lote = textos[inicio:inicio + BATCH_SIZE_ROBERTA]
            entradas = tokenizador(
                lote,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS_ROBERTA,
            )
            entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
            with torch.no_grad():
                salida = modelo(**entradas).last_hidden_state

            mascara = entradas["attention_mask"].unsqueeze(-1).to(salida.dtype)
            vectores = (salida * mascara).sum(dim=1) / mascara.sum(dim=1).clamp(min=1)
            todos.append(vectores.cpu().numpy())

            procesados = min(inicio + len(lote), len(textos))
            if procesados % 100 == 0 or procesados == len(textos):
                print(f"{nombre}: {procesados}/{len(textos)} textos")

        if todos:
            matriz = np.concatenate(todos, axis=0).astype(np.float32)
        else:
            matriz = np.empty((0, int(modelo.config.hidden_size)), dtype=np.float32)
        np.savez_compressed(
            CARPETA_SALIDA / f"embeddings_roberta_{nombre}.npz",
            document_id=df["document_id"].to_numpy(),
            embeddings=matriz,
        )
        print(f"Embeddings {nombre}: {matriz.shape}")
        return matriz

    embeddings_originales = obtener_embeddings(df["question"].astype(str).tolist(), "originales")
    embeddings_simplificados = obtener_embeddings(df["answer"].astype(str).tolist(), "simplificados")
    return embeddings_originales, embeddings_simplificados


def exportar_resultados(df):
    """Guarda los textos procesados en CSV y JSONL para su posterior reutilización."""
    df.to_csv(CARPETA_SALIDA / "corpus_procesado.csv", index=False)

    ruta_jsonl = CARPETA_SALIDA / "corpus_procesado.jsonl"
    with open(ruta_jsonl, "w", encoding="utf-8") as f:
        for registro in df.to_dict(orient="records"):
            # Convierte valores NumPy y NaN a tipos compatibles con JSON.
            registro_limpio = {}
            for clave, valor in registro.items():
                if pd.isna(valor) if not isinstance(valor, (dict, list, tuple)) else False:
                    registro_limpio[clave] = None
                elif isinstance(valor, (np.integer,)):
                    registro_limpio[clave] = int(valor)
                elif isinstance(valor, (np.floating,)):
                    registro_limpio[clave] = float(valor)
                else:
                    registro_limpio[clave] = valor
            f.write(json.dumps(registro_limpio, ensure_ascii=False) + "\n")

    print("\nCorpus exportado a:")
    print(" -", (CARPETA_SALIDA / "corpus_procesado.csv").resolve())
    print(" -", ruta_jsonl.resolve())


def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    train, test, validation = dividir_dataset(df)

    nlp = cargar_spacy()
    df_procesado = procesar_textos(df, nlp)

    # Recuperar las particiones usando document_id para conservar el procesamiento.
    train_p = df_procesado[df_procesado["document_id"].isin(train["document_id"])].copy()
    test_p = df_procesado[df_procesado["document_id"].isin(test["document_id"])].copy()
    validation_p = df_procesado[
        df_procesado["document_id"].isin(validation["document_id"])
    ].copy()

    # Guardar también las particiones con tokenización, lemas y entidades.
    train_p.to_csv(CARPETA_SALIDA / "train_procesado.csv", index=False)
    test_p.to_csv(CARPETA_SALIDA / "test_procesado.csv", index=False)
    validation_p.to_csv(CARPETA_SALIDA / "validation_procesado.csv", index=False)

    ajustar_bow_tfidf(train_p, test_p, validation_p)

    modelo_w2v = entrenar_word2vec(train_p)
    analizar_oov(df_procesado, modelo_w2v)

    # Genera embeddings contextuales para originales y simplificaciones.
    generar_embeddings_contextuales(df_procesado)

    exportar_resultados(df_procesado)
    print("\nProceso completado. Todos los resultados están en:", CARPETA_SALIDA.resolve())


if __name__ == "__main__":
    main()
