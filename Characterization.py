import nltk
from nltk import WordNetLemmatizer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, IDF, HashingTF, MinHashLSH
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.functions import udf, collect_list, when, col, explode, expr, array
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import split, size

import Constants


def lemmatize_tokens(tokens):
    if tokens is None:
        return None
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


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

    grouped = df.groupBy("name_vector").agg(collect_list("id").alias("id"))

    udf_split_list_http = udf(split_list_http, ArrayType(StringType()))
    udf_split_list_no_http = udf(split_list_no_http, ArrayType(StringType()))

    grouped = grouped.withColumn("idGoogle", udf_split_list_http(col("id")))
    grouped = grouped.withColumn("idAmazon", udf_split_list_no_http(col("id")))

    df_filtered = grouped.select("idGoogle", "idAmazon").filter(
        (col("idGoogle").isNotNull()) & (col("idAmazon").isNotNull()) & (col("idGoogle") != array([])) & (
                col("idAmazon") != array([])))
    final = df_filtered.select("idAmazon", explode("idGoogle").alias("idGoogle"))
    finaler = final.select("idGoogle", explode("idAmazon").alias("idAmazon"))

    finaler.show(truncate=False)
    final.printSchema()
    return df


def split_list_http(l):
    contains_http = [elem for elem in l if "http" in elem]
    return contains_http


def split_list_no_http(l):
    rest = [elem for elem in l if "http" not in elem]
    return rest
