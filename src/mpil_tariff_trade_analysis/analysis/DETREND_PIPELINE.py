

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import shutil
    import os
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
        value_column_name: str,
        detrended_output_col_name: str = "detrended_value",
        trend_line_output_col_name: str = "trend_line_value"
    ) -> pl.LazyFrame:
        """
        Applies detrending to a Polars LazyFrame.

        This function calculates trend parameters (slope and intercept) for each 'product_code'
        group using the 'year' column from `source_for_trend_params_lf` and the specified
        `value_column_name`. It then joins these parameters to `target_lf` and
        calculates the trend line and the detrended values.

        It relies on the globally defined `calculate_trend_params_statsmodels` UDF and
        `udf_output_schema_sm` which hardcode 'product_code' as the grouping key and
        'year' as the time variable for OLS regression.

        Args:
            target_lf: The LazyFrame to which detrending will be applied.
                           Must contain 'product_code', 'year', and `value_column_name`.
            source_lf: The LazyFrame used to calculate trend parameters.
                                        Must contain 'product_code', 'year', and `value_column_name`.
                                        Can be the same as `target_lf`.
            value_column_name: The name of the column in `source_for_trend_params_lf` (for fitting)
                               and `target_lf` (for detrending) that contains the values to be detrended.
            detrended_output_col_name: The desired name for the new column containing detrended values.
            trend_line_output_col_name: The desired name for the new column containing trend line values.

        Returns:
            A new LazyFrame with the added trend line and detrended value columns.
        """
        source_lf = source_lf.with_columns(
                pl.col('year').cast(pl.Float64)
            ).sort('year')

        trend_params_lf = source_lf.group_by(
            "product_code", maintain_order=True
        ).agg(
            [
                # Manually calculate the OLS linear regression
                pl.cov(pl.col("year"), pl.col(value_column_name)).alias("cov_xy"),
                pl.col("year").var().alias("var_x"),
                pl.col("year").mean().alias("mean_x"),
                pl.col(value_column_name).mean().alias("mean_y"),
            ]
        )

        trend_params_lf = trend_params_lf.with_columns(
            (
                pl.col("cov_xy") / pl.col("var_x")
            ).alias('slope')
        )

        trend_params_lf = trend_params_lf.with_columns(
            (
                pl.col("mean_y") - (pl.col("slope") * pl.col("mean_x"))
            ).alias('intercept')
        ).select(['product_code', 'slope', 'intercept'])

        # Join trend parameters with the target LazyFrame.
        # The join is on 'product_code'.
        detrended_lf = target_lf.join(
            trend_params_lf,
            on='product_code', # Fixed by the UDF's output
            how='left',
            allow_parallel=True,
        ).sort('year')

        # Calculate the global trend at each year and the detrended value.
        # Cast 'year' to Float64 for consistency with OLS fitting in the UDF.
        detrended_lf = detrended_lf.with_columns(
            (
                pl.col("slope") * pl.col("year").cast(pl.Float64) + pl.col("intercept")
            ).alias(trend_line_output_col_name)
        ).with_columns(
            (
                pl.col(value_column_name) - pl.col(trend_line_output_col_name)
            ).alias(detrended_output_col_name)
        ).drop(
            'slope', 'intercept'
        ).sort('year')

        return detrended_lf

    return


@app.cell
def _(pl):
    def apply_detrending_to_lazyframe2(
        target_lf: pl.LazyFrame,
        source_lf: pl.LazyFrame,
        target_metric_to_detrend_col: str,
        source_value_col: str = 'value',
        source_quantity_col: str = 'quantity',
        trend_line_output_col_name: str = "product_global_trend"
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
        global_avg_price_trend_lf = source_lf.group_by(
            ["product_code", "year"], maintain_order=False
        ).agg(
            pl.sum(source_value_col).alias(_sum_value_alias),
            pl.sum(source_quantity_col).alias(_sum_quantity_alias)
        ).with_columns(
            pl.when(pl.col(_sum_quantity_alias) != 0)
            .then(pl.col(_sum_value_alias) / pl.col(_sum_quantity_alias))
            .otherwise(None)  # Results in null if sum_quantity is 0
            .alias(trend_line_output_col_name)
        ).select(["product_code", "year", trend_line_output_col_name])



        detrended_lf = target_lf.join(
            global_avg_price_trend_lf,
            on=['product_code', 'year'],
            how='left'
        )

        detrended_lf = detrended_lf.with_columns(
            (pl.col(target_metric_to_detrend_col) - pl.col(trend_line_output_col_name)).alias(
                target_metric_to_detrend_col+"_detrended"
            )
        )

        detrended_lf = detrended_lf.sort("year")

        return detrended_lf
    return (apply_detrending_to_lazyframe2,)


@app.cell
def _(pl):
    print("Loading Unified DF")
    # file_to_detrend_and_overwrite = "data/final/unified_filtered_10000minval_top100countries"
    # file_to_detrend_and_overwrite = "data/final/unified_filtered_10000val_100c_sample_1000krows_filter"
    file_to_detrend_and_overwrite = "data/final/unified_trade_tariff_partitioned/"
    unified_lf = pl.scan_parquet(file_to_detrend_and_overwrite)

    unified_lf = unified_lf.with_columns(
        (pl.col('value') / pl.col('quantity')).alias('unit_value')
    )
    print("Loaded unified_lf with unit_value:\n", unified_lf.head().collect(engine='streaming'))
    return file_to_detrend_and_overwrite, unified_lf


@app.cell
def _(unified_lf):
    unified_lf.collect()
    return


@app.cell
def _(apply_detrending_to_lazyframe2, unified_lf):
    # Detrend the series. Drop some unnecessary series.
    print("Applying detrending to entire table...")
    unified_lf_detrend = apply_detrending_to_lazyframe2(
        target_lf=unified_lf,
        source_lf=unified_lf,
        target_metric_to_detrend_col='unit_value',
        trend_line_output_col_name="product_global_trend",
    )
    return (unified_lf_detrend,)


@app.cell
def _():
    # unified_lf_detrend.head().collect(engine='streaming')
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
