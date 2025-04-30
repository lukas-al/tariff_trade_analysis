import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import random
    import shutil
    import time

    import marimo as mo
    import polars as pl

    return mo, os, pl, random, shutil, time


@app.cell
def _(mo):
    mo.md(
        r"""
        # Sample Pipeline
        Create some sub-samples of the data to develop and experiment with
        """
    )
    return


@app.cell
def _(pl):
    unified_data = pl.scan_parquet(
        "data/final/unified_trade_tariff_partitioned/",
    )
    return (unified_data,)


@app.cell
def _(os, pl, shutil, time):
    print("--- CREATING DATASET SUB-SAMPLES ---")

    def create_and_save_filter(
        lf: pl.LazyFrame, min_trade_value: int, top_n_countries: int
    ) -> pl.LazyFrame:
        """
        Create a filtered version of the unified dataset based on minimum trade value
        and top N trading countries (by total import + export value).
        Save the filtered dataset to the final folder, partitioned by year.

        Args:
            lf: The input Polars LazyFrame containing trade data. Expected columns
                include 'reporter_country', 'partner_country', 'volume', and 'year'.
            min_trade_value: The minimum trade value (inclusive) to keep a record (in thousands USD).
            top_n_countries: The number of top trading countries to consider. Only trades
                             *between* these top countries will be kept.
        Returns:
            pl.LazyFrame: filter lazyframe pre sink
        """
        # --- Configuration ---
        importer_col = "partner_country"  # Column name for the importing country
        exporter_col = "reporter_country"  # Column name for the exporting country
        value_col = "volume"  # Column name for the trade value
        year_col = "year"  # Column name for the year, used for partitioning
        # Construct a dynamic output path based on parameters
        output_path = (
            f"data/final/unified_filtered_{min_trade_value}minval_top{top_n_countries}countries"
        )

        print("--- Starting Filter Process ---")
        print(f"Parameters: min_trade_value={min_trade_value}, top_n_countries={top_n_countries}")
        print(f"Output path: {output_path}")

        # --- Clean up existing output directory ---
        if os.path.isdir(output_path):
            print(f"Removing existing directory: {output_path}")
            shutil.rmtree(output_path)

        start_time = time.time()

        # --- Step 1: Calculate total volume per country and find Top N ---
        print(f"Identifying top {top_n_countries} countries by total trade volume...")

        # Select exports: country and value
        ldf_exports = lf.select(pl.col(exporter_col).alias("country"), pl.col(value_col)).filter(
            pl.col("country").is_not_null() & pl.col(value_col).is_not_null()
        )

        # Select imports: country and value
        ldf_imports = lf.select(pl.col(importer_col).alias("country"), pl.col(value_col)).filter(
            pl.col("country").is_not_null() & pl.col(value_col).is_not_null()
        )

        # Combine exports and imports vertically to get all flows per country
        ldf_all_flows = pl.concat([ldf_exports, ldf_imports], how="vertical")

        # Group by country and sum the trade values
        ldf_volumes = ldf_all_flows.group_by("country").agg(pl.sum(value_col).alias("total_volume"))

        # Determine the top N countries based on total volume
        # Use collect(streaming=True) for potentially large grouping operations
        top_countries_df = (
            ldf_volumes.sort("total_volume", descending=True)
            .head(top_n_countries)
            .collect(engine="streaming")
        )
        top_countries_list = top_countries_df["country"].to_list()  # Get list of country names

        print(f"Identified Top {len(top_countries_list)} countries.")
        if top_countries_list:
            print(
                f"Sample of top countries: {top_countries_list[:min(10, len(top_countries_list))]}"
            )
        else:
            print(
                "Warning: No top countries identified. Filtering might result in an empty dataset."
            )

        # --- Step 2: Filter the original data ---
        print("Applying filters to the dataset (lazy execution)...")

        ldf_filtered = lf.filter(
            # Filter 1: Keep rows where trade value meets the minimum threshold
            pl.col(value_col) >= min_trade_value
        ).filter(
            # Filter 2: Keep rows where BOTH importer and exporter are in the top N list
            pl.col(importer_col).is_in(top_countries_list)
            & pl.col(exporter_col).is_in(top_countries_list)
        )

        # --- Step 3: Write filtered data to partitioned Parquet files ---
        print(f"Writing filtered data to: {output_path}")
        ldf_filtered.sink_parquet(
            pl.PartitionByKey(
                base_path=output_path,
                by=pl.col("year"),
            ),
            mkdir=True,
        )
        print("Successfully wrote filtered data.")

        end_time = time.time()
        print("--- Finished Filtering ---")
        print(f"Total duration: {end_time - start_time:.2f} seconds")

        # The function implicitly returns None as specified by the type hint
        return ldf_filtered

    return (create_and_save_filter,)


@app.cell
def _(create_and_save_filter, unified_data):
    min_trade_value = 10000
    num_top_countries = 100
    filtered_lf = create_and_save_filter(unified_data, min_trade_value, num_top_countries)
    return filtered_lf, min_trade_value, num_top_countries


@app.cell
def _(mo):
    mo.md(r"""# Print some summary stats, and create smaller filters""")
    return


@app.cell
def _(filtered_lf, pl):
    num_rows = filtered_lf.select(pl.len()).collect(engine="streaming").item()
    num_cols = len(filtered_lf.collect_schema().names())

    null_count_lazy = filtered_lf.select(pl.col("effective_tariff").is_null().sum())
    null_count_effective_tariff = null_count_lazy.collect(engine="streaming").item()

    print("--- Summary Statistics for filtered data ---")
    print(f"Dimensions (rows, columns): ({num_rows}, {num_cols})")
    print(f"Number of null values in 'effective_tariff': {null_count_effective_tariff}")
    print(f"Null percentage = {null_count_effective_tariff / num_rows}")

    print("\n--- First 5 Rows (Collected) ---")
    print(filtered_lf.head(5).collect(engine="streaming"))

    print("\n--- DataFrame Schema (from LazyFrame) ---")
    print(filtered_lf.collect_schema())
    return (num_rows,)


@app.cell
def _(mo):
    mo.md(r"""# Write an even smaller sample""")
    return


@app.cell
def _(
    filtered_lf,
    min_trade_value,
    num_rows,
    num_top_countries,
    os,
    pl,
    random,
    shutil,
):
    sample_size = 10000000
    random.seed(42)
    sample_output_path = f"data/final/unified_filtered_{min_trade_value}val_{num_top_countries}c_sample_{int(sample_size/1000)}krows_filter"

    if os.path.isdir(sample_output_path):
        print("Removing existing dir")
        shutil.rmtree(sample_output_path)

    print(f"--- Saving Random Sample of {sample_size} Rows to {sample_output_path} ---")

    sampled_indices = random.sample(range(num_rows), sample_size)
    indices_series = pl.Series("idx_to_keep", sorted(sampled_indices), dtype=pl.UInt32)

    filtered_lf_sample = (
        filtered_lf.with_row_index(name="index")
        .filter(pl.col("index").is_in(indices_series))
        .drop("index")
    )

    print(f"Random sample head:\n{filtered_lf_sample.head().collect(engine='streaming')}")

    # To maintain parity between the larger data, write it the same way
    filtered_lf_sample.sink_parquet(
        pl.PartitionByKey(
            base_path=sample_output_path,
            by=pl.col("year"),
        ),
        mkdir=True,
    )

    print("--- FINISHED ---")
    return


if __name__ == "__main__":
    app.run()
