from functools import reduce

from graphframes import GraphFrame
from pyspark.sql import DataFrame
from pyspark.sql.functions import explode, col, collect_list, array, udf
from pyspark.sql.types import StringType, ArrayType

import Constants


def entity_gen(df):
    df_candidates = get_pair_candidates(df)
    gen_graph_frame(df, df_candidates)
    return df_candidates


# All bellow are private methods that only are use on first method


def get_pair_candidates(df):
    grouped_list = []

    for column in Constants.col:
        a = df.groupBy(column + "_vector").agg(collect_list("id").alias("id"))
        grouped_list.append(a)

    grouped = reduce(DataFrame.union, grouped_list)

    udf_split_list_http = udf(split_list_http, ArrayType(StringType()))
    udf_split_list_no_http = udf(split_list_no_http, ArrayType(StringType()))

    grouped = grouped.withColumn("idGoogle", udf_split_list_http(col("id")))
    grouped = grouped.withColumn("idAmazon", udf_split_list_no_http(col("id")))

    df_filtered = grouped.select("idGoogle", "idAmazon").filter(
        (col("idGoogle").isNotNull()) & (col("idAmazon").isNotNull()) & (col("idGoogle") != array([])) & (
                col("idAmazon") != array([])))
    df_filtered_explode = df_filtered.select("idAmazon", explode("idGoogle").alias("idGoogle"))
    df_candidates = df_filtered_explode.select("idGoogle", explode("idAmazon").alias("idAmazon"))
    df_candidates = df_candidates.dropDuplicates()

    return df_candidates


def gen_graph_frame(df, df_candidates):
    edges = df_candidates.select("idAmazon", "idGoogle").withColumnRenamed("idAmazon", "src").withColumnRenamed(
        "idGoogle", "dst")
    g = GraphFrame(df, edges)
    g.vertices.show()
    g.edges.show()
    g.degrees.show()

    return


def split_list_http(element_list):
    contains_http = [elem for elem in element_list if "http" in elem]
    return contains_http


def split_list_no_http(element_list):
    rest = [elem for elem in element_list if "http" not in elem]
    return rest
