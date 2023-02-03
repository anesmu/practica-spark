import Candidate_pairs
import Characterization
import Standardization
import Util

from pyspark.sql import SparkSession


def main(spark):
    # Load df from  /data
    df = Util.load_df_from_data_folder(spark)
    df.cache()

    # Part 1: standardization
    df_standard = Standardization.standardization_df(df)
    df_standard.cache()

    # Part 2: characterization
    df_char = Characterization.characterization_idf(df_standard)
    df_char.cache()

    # Part 3: Candidate_pairs
    df_distance = Candidate_pairs.get_pair_cadidates(df_char)
    df_distance.show(200, truncate=False)
    print(df_distance.row())
    return


if __name__ == '__main__':
    spark_session = SparkSession.builder.appName("Offers ") \
        .master("local[*]") \
        .getOrCreate()

    spark_session.sparkContext.setLogLevel('warn')
    main(spark_session)
    spark_session.stop()
