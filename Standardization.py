from pyspark.sql.functions import col, lower, regexp_replace, translate

import Constants


def standardization_df(df):
    df = df_to_lower_case(df)
    df = remove_special_characters(df)
    df = remove_gbp_from_price(df)
    df = fill_null_fields(df)

    return df


# All bellow are private methods that only are use on first method

def df_to_lower_case(df):
    df = df.select(
        [lower(col(col_name)).alias(col_name) if col_name in Constants.col else col(col_name) for col_name in
         df.columns])

    return df


def remove_special_characters(df):
    special_characters = "!@#%^&*()_+-={}|[]\\:\";'<>,.?/\""
    for col_name in Constants.col:
        if col_name != Constants.price:
            df = df.withColumn(col_name, translate(col(col_name), special_characters, ""))

    return df


def remove_gbp_from_price(df):
    special_characters = "!@#%^&*()_+-={}|[]\\:\";'<>,.?/\"" + " gbp"
    df = df.withColumn(Constants.price, translate(col(Constants.price), special_characters, ""))

    return df


def fill_null_fields(df):
    df = df.fillna("null_value", subset=None)
    return df
