import util

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.sql.functions import col

def main(spark):
    path = 'data/'
    filename_google = 'GoogleProducts.csv'
    filename_amazon = 'Amazon.csv'

    absolute_file_path_google = util.get_absolute_file_path(path, filename_google)
    absolute_file_path_amazon = util.get_absolute_file_path(path, filename_amazon)

    df_google = spark.read.csv(absolute_file_path_google, header=True, inferSchema=True)
    df_google = df_google.withColumnRenamed("title", "name")
    df_amazon = spark.read.csv(absolute_file_path_amazon, header=True, inferSchema=True)
    df_product = df_google.union(df_amazon)

    df_product = df_product.select(
        [lower(col(col_name)).alias(col_name) if col_name in util.col else col(col_name) for col_name in
         df_product.columns])

    special_characters = "!@#%^&*()_+-={}|[]\\:\";'<>,.?/\""
    for col_name in util.col:
        if col_name != util.price:
            df_product = df_product.withColumn(col_name, translate(col(col_name), special_characters, ""))
        else:
            df_product = df_product.withColumn(util.price, regexp_replace(col(util.price), " gbp", ""))

    tokenizer = Tokenizer(inputCol="name", outputCol="name_tokens")
    df_product = tokenizer.transform(df_product)

    remover = StopWordsRemover(inputCol="name_tokens", outputCol="name_filtered_tokens")
    df_product = remover.transform(df_product)

    hashing_tf = HashingTF(inputCol="name_filtered_tokens", outputCol="name_raw_features")
    df_product = hashing_tf.transform(df_product)

    idf = IDF(inputCol="name_raw_features", outputCol="name_tfidf_features")
    idf_model = idf.fit(df_product)
    df_product = idf_model.transform(df_product)
    df_product.show(500, truncate=False)

    df_distance = df_product.alias("df1").crossJoin(df_product.alias("df2")) \
        .where(col("df1.id") != col("df2.id")) \
        .select(col("df1.id").alias("id1"), col("df2.id").alias("id2"),
                levenshtein(col("df1.name"), col("df2.name")).alias("name_distance"),
                levenshtein(col("df1.description"), col("df2.description")).alias("description_distance"))
    df_distance.show(500, truncate=False)


if __name__ == '__main__':
    spark_session = SparkSession.builder.appName("Offers ") \
        .master("local[*]").getOrCreate()

    spark_session.sparkContext.setLogLevel('warn')
    main(spark_session)
    spark_session.stop()
