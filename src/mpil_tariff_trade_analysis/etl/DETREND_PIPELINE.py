import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse

    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _(mo):
    mo.md(
        """
    # Implement the detrending & further post-processing
    1. Load the unified data
    2. Detrend
    3. Remove outliers
    4. Save

    ## First define convenience functions
    """
    )
    return


@app.cell
def _(pl):
    def apply_detrending_to_lazyframe(
        target_lf: pl.LazyFrame,
        source_lf: pl.LazyFrame,
        target_metric_to_detrend_col: str,
        source_value_col: str = "value",
        source_quantity_col: str = "quantity",
        trend_line_output_col_name: str = "product_global_trend",
    ) -> pl.LazyFrame:
        """
        Detrends a specified metric in a target LazyFrame by subtracting a global trend.

        The global trend is calculated from `source_lf` as an average price
        (sum of `source_value_col` / sum of `source_quantity_col`) for each
        `product_code` and `year`. This calculated trend is then added to `target_lf`
        as a new column named by `trend_line_output_col_name`.

        A second new column is added to `target_lf`, named by appending "_detrended"
        to `target_metric_to_detrend_col`. This column contains the original metric's
        values minus the calculated global trend.

        If the sum of `source_quantity_col` for a given `product_code` and `year` is zero,
        the calculated trend for that group will be null.

        Args:
            target_lf: The Polars LazyFrame containing the data to be detrended.
                It must include `product_code`, `year`, and the column specified by
                `target_metric_to_detrend_col`.
            source_lf: The Polars LazyFrame used as the source for calculating the
                global trend. It must include `product_code`, `year`, and the columns
                specified by `source_value_col` and `source_quantity_col`.
            target_metric_to_detrend_col: The name of the column in `target_lf`
                whose values will be detrended.
            source_value_col: The name of the column in `source_lf` representing
                total values for calculating the trend (e.g., total sales value).
                Defaults to 'value'.
            source_quantity_col: The name of the column in `source_lf` representing
                total quantities for calculating the trend (e.g., total units sold).
                Defaults to 'quantity'.
            trend_line_output_col_name: The name for the new column that will be
                added to `target_lf` to store the calculated global trend values.
                Defaults to "product_global_trend".

        Returns:
            A new Polars LazyFrame based on `target_lf` with two additional columns:
            - A column named by `trend_line_output_col_name` (e.g., "product_global_trend"),
              containing the calculated global trend (average price) for the
              corresponding `product_code` and `year`.
            - A column named `f"{target_metric_to_detrend_col}_detrended"`,
              containing the result of `target_metric_to_detrend_col` minus
              the `trend_line_output_col_name`.
            The returned LazyFrame is sorted by the 'year' column.
        """
        _sum_value_alias = "_sum_value_for_trend_calc"
        _sum_quantity_alias = "_sum_quantity_for_trend_calc"

        # Calc global average price trend from source_lf
        global_avg_price_trend_lf = (
            source_lf.group_by(["product_code", "year"], maintain_order=False)
            .agg(
                pl.sum(source_value_col).alias(_sum_value_alias),
                pl.sum(source_quantity_col).alias(_sum_quantity_alias),
            )
            .with_columns(
                pl.when(pl.col(_sum_quantity_alias) != 0)
                .then(pl.col(_sum_value_alias) / pl.col(_sum_quantity_alias))
                .otherwise(None)  # Results in null if sum_quantity is 0
                .alias(trend_line_output_col_name)
            )
            .select(["product_code", "year", trend_line_output_col_name])
        )

        detrended_lf = target_lf.join(
            global_avg_price_trend_lf, on=["product_code", "year"], how="left"
        )

        detrended_lf = detrended_lf.with_columns(
            (
                pl.col(target_metric_to_detrend_col)
                - pl.col(trend_line_output_col_name)
            ).alias(target_metric_to_detrend_col + "_detrended")
        )

        # detrended_lf = detrended_lf.sort("year") # Leads to OOM on the full dataset

        return detrended_lf
    return (apply_detrending_to_lazyframe,)


@app.cell
def _():
    # parser = argparse.ArgumentParser(description="Marimo visualise unified")

    # # Add your parameters/arguments here
    # parser.add_argument("--fullfat", action="store_true", help="Using this flag will run on all the data")
    # args = parser.parse_args()

    # args.fullfat
    # return (args,)
    return


@app.cell
def _(pl):
    print("Loading Unified DF")

    # if args.fullfat:
    #     file_to_detrend_and_overwrite = "data/final/unified_trade_tariff_partitioned/"
    #     print("Running detrend across all the data")
    # else:
    #     file_to_detrend_and_overwrite = "data/final/unified_filtered_10000minval_top100countries"
    #     print("Running detrend across subset of the data")

    # HARDCODE FULLFAT
    file_to_detrend_and_overwrite = "data/final/unified_trade_tariff_partitioned/"

    unified_lf = pl.scan_parquet(file_to_detrend_and_overwrite)
    unified_lf = unified_lf.with_columns(
        (pl.col("value") / pl.col("quantity")).alias("unit_value")
    )
    print(
        "Loaded unified_lf with unit_value:\n",
        unified_lf.head().collect(engine="streaming"),
    )
    return file_to_detrend_and_overwrite, unified_lf


@app.cell
def _(pl, unified_lf):
    # Interpolate values, quantities, and effective tariffs.
    def interpolate_groupwise_lazy(
        lf: pl.LazyFrame,
        group_keys: list[str],
        sort_key: str,
        interpolate_col: str,
        fill_na_value: any = 0.0,
    ) -> pl.LazyFrame:
        """
        Interpolates a specified column within groups in a LazyFrame,
        after sorting each group by a specified sort key. Remaining NaNs
        after interpolation are filled with a specified value.

        Args:
            lf: The input Polars LazyFrame.
            group_keys: A list of column names to group by.
            sort_key: The column name to sort by within each group before interpolation.
            interpolate_col: The name of the column to interpolate.
            fill_na_value: The value to use to fill any NaNs remaining after interpolation.
                           Defaults to 0.0.

        Returns:
            A Polars LazyFrame with the specified column interpolated.
        """
        processed_lf = lf.sort(group_keys + [sort_key]).with_columns(
            pl.col(interpolate_col)
            .interpolate()
            .over(group_keys)
            .fill_null(fill_na_value)
            .alias(interpolate_col)  # Ensure the original column is updated
        )
        return processed_lf


    group_keys = ["product_code", "partner_country", "reporter_country"]

    # Apply interpolation to the average tariff column
    print("Interpolating and filling nulls for average tariff column")
    unified_lf_nanfilled = interpolate_groupwise_lazy(
        unified_lf,
        group_keys=group_keys,
        sort_key="year",
        interpolate_col="average_tariff",
        fill_na_value=0.0,
    )
    return


@app.cell
def _(apply_detrending_to_lazyframe, unified_lf):
    # Detrend the series. Drop some unnecessary series.
    print("Applying detrending to entire table...")
    unified_lf_detrend = apply_detrending_to_lazyframe(
        target_lf=unified_lf,
        source_lf=unified_lf,
        target_metric_to_detrend_col="value",
        trend_line_output_col_name="value_global_trend",
    )

    unified_lf_detrend = apply_detrending_to_lazyframe(
        target_lf=unified_lf_detrend,
        source_lf=unified_lf_detrend,
        target_metric_to_detrend_col="quantity",
        trend_line_output_col_name="quantity_global_trend",
    )

    unified_lf_detrend = apply_detrending_to_lazyframe(
        target_lf=unified_lf_detrend,
        source_lf=unified_lf_detrend,
        target_metric_to_detrend_col="unit_value",
        trend_line_output_col_name="price_global_trend",
    )
    return (unified_lf_detrend,)


@app.cell
def _():
    # unified_lf_trf_detrend = unified_lf_detrend.with_columns(
    # pl.when(pl.col("unit_value") > 0)
    #   .then(pl.col("unit_value").log())
    #   .otherwise(pl.lit(0, dtype=pl.Float64)) # Set to null if not positive
    #   .alias("unit_value_log"),
    # pl.when(pl.col("unit_value_detrended") > 0)
    #   .then(pl.col("unit_value_detrended").log())
    #   .otherwise(pl.lit(0, dtype=pl.Float64)) # Set to null if not positive
    #   .alias("unit_value_log_detrended"),
    # pl.when(pl.col("effective_tariff") > 0)
    #   .then((pl.col("effective_tariff") + 1).log())
    #   .otherwise(pl.lit(0, dtype=pl.Float64)) # Set to null if not positive
    #   .alias("effective_tariff_log"),

    #
    #

    # pl.col("unit_value").arcsinh().alias("unit_value_arcsinh"),
    # pl.col("unit_value_detrended")
    # .arcsinh()
    # .alias("unit_value_arcsinh_detrended"),
    # pl.col("effective_tariff").arcsinh().alias("effective_tariff_arcsinh"),
    # )
    return


@app.cell
def _(unified_lf_detrend):
    print("New unified_lf schema:", unified_lf_detrend.collect_schema())
    return


@app.cell
def _(file_to_detrend_and_overwrite, pl, unified_lf_detrend):
    # Sink output
    print("Sinking back to unified_lf")
    unified_lf_detrend.sink_parquet(
        pl.PartitionByKey(
            base_path=file_to_detrend_and_overwrite,
            by=pl.col("year"),
        ),
        mkdir=True,
    )
    return


if __name__ == "__main__":
    app.run()
