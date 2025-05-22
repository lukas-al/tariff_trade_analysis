"""
Utility functions, shared across multiple pipelines
"""

from pathlib import Path

import polars as pl


def vectorized_hs_translation(input_lf: pl.LazyFrame, mapping_dir: str = "data/raw/hs_reference") -> pl.LazyFrame:
    hs_versions = ["H1", "H2", "H3", "H4", "H5", "H6"]
    original_columns = input_lf.collect_schema().names()

    # Instrumentation: Unique codes before
    unique_codes_before = input_lf.select(pl.col("product_code").n_unique()).collect().item()
    print(f"Number of unique product codes before translation: {unique_codes_before}")

    mapping_lfs = []
    for hs_version in hs_versions:
        path = Path(mapping_dir) / f"{hs_version}_to_H0.CSV"

        current_mapping_df = pl.read_csv(
            source=path,
            has_header=True,
            columns=[0, 2],
            new_columns=["source_code", "target_code"],
            encoding="iso-8859-1",
            schema_overrides={"source_code": pl.Utf8, "target_code": pl.Utf8},
        )

        current_mapping_lf = current_mapping_df.with_columns(pl.lit(hs_version).cast(pl.Utf8).alias("hs_revision")).lazy()
        mapping_lfs.append(current_mapping_lf)

    mapping_all = pl.concat(mapping_lfs)

    df_h0 = input_lf.filter(pl.col("hs_revision") == "H0")
    df_non_h0 = input_lf.filter(pl.col("hs_revision") != "H0")

    # Instrumentation: Count H0 codes
    count_h0_codes = df_h0.select(pl.len()).collect().item()
    print(f"Number of H0 codes (not translated): {count_h0_codes}")

    df_non_h0_joined = df_non_h0.join(
        mapping_all,
        left_on=["hs_revision", "product_code"],
        right_on=["hs_revision", "source_code"],
        how="left",
    )

    # Instrumentation: Codes renamed (translated)
    count_translated_lf = df_non_h0_joined.filter(pl.col("target_code").is_not_null())
    number_translated = count_translated_lf.select(pl.len()).collect().item()
    print(f"Number of HS codes successfully translated (renamed): {number_translated}")

    # Instrumentation: Codes left alone
    count_left_alone_lf = df_non_h0_joined.filter(pl.col("target_code").is_null())
    number_left_alone = count_left_alone_lf.select(pl.len()).collect().item()
    print(f"Number of non-H0 HS codes left alone (no translation found): {number_left_alone}")

    df_non_h0_translated = df_non_h0_joined.with_columns(pl.coalesce(pl.col("target_code"), pl.col("product_code")).alias("product_code"))

    processed_df_non_h0 = df_non_h0_translated.select(original_columns)

    df_final = pl.concat(
        [df_h0.select(original_columns), processed_df_non_h0],
        how="vertical_relaxed",
    )

    # Instrumentation: Unique codes after
    unique_codes_after = df_final.select(pl.col("product_code").n_unique()).collect().item()
    print(f"Number of unique product codes after translation: {unique_codes_after}")

    df_final = df_final.drop("hs_revision")

    return df_final
