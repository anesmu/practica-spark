import nltk
from nltk import WordNetLemmatizer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, IDF, HashingTF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

import Constants


def characterization_idf(df):
    nltk.download('omw-1.4')
    nltk.download("wordnet")

    lemmatizer_udf = udf(lemmatize_tokens, ArrayType(StringType()))

    for column in Constants.col:
        tokenizer = Tokenizer(inputCol=column, outputCol=column + "_tokens")
        df = tokenizer.transform(df)

        remover = StopWordsRemover(inputCol=column + "_tokens", outputCol=column + "_filtered_tokens")
        df = remover.transform(df)

        df = df.withColumn(column + "_lemmatized_tokens",
                           lemmatizer_udf(df[column + "_filtered_tokens"]))
        df = df.drop(column + "_tokens")
        df = df.drop(column + "_filtered_tokens")
        df = df.withColumnRenamed(column + "_lemmatized_tokens", column + "_tokens")

        hashing_tf = HashingTF(inputCol=column + "_tokens", outputCol=column + "_hash")
        df = hashing_tf.transform(df)

        idf = IDF(inputCol=column + "_hash", outputCol=column + "_vector")
        idf_model = idf.fit(df)
        df = idf_model.transform(df)
        df = df.drop(column + "_hash")
    return df


# All bellow are private methods that only are use on first method

def lemmatize_tokens(tokens):
    if tokens is None:
        return None
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
