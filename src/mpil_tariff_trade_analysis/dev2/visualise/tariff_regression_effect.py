

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import itertools
    import functools
    import operator
    import time
    import math
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pycountry
    import pickle
    from functools import partial
    from tqdm import tqdm
    from typing import List
    return List, functools, mo, operator, pl, px, tqdm


@app.cell
def _(pl):
    def remove_outliers_by_percentile(
        df: pl.DataFrame,
        column_names: list[str], # Modified: now accepts a list of column names
        lower_p: float = 0.01,
        upper_p: float = 0.99
    ) -> pl.DataFrame:
        """
        Removes outliers from specified columns in a Polars DataFrame based on percentiles.
        Outlier bounds for each column are determined from its distribution in the original input DataFrame.
        Filters are then applied sequentially.

        Args:
            df: The input Polars DataFrame.
            column_names: A list of names of columns from which to remove outliers.
                          These columns must be numeric.
            lower_p: The lower percentile (e.g., 0.01 for 1st percentile).
                           Must be between 0 and 1.
            upper_p: The upper percentile (e.g., 0.99 for 99th percentile).
                           Must be between 0 and 1, and greater than lower_p.

        Returns:
            A new Polars DataFrame with outliers removed from the specified columns.
            For each specified column, if its outlier bounds cannot be determined
            (e.g., the column in the original DataFrame is all nulls, empty, or has too few distinct values)
            or if its calculated lower_bound > upper_bound, filtering for that
            specific column is skipped, a warning is printed, and processing continues
            with the next column. The finally returned DataFrame will reflect all
            successful outlier removal operations.
            Errors from Polars (e.g., column not found during quantile calculation)
            for a specific column will cause that column's processing to be skipped with a warning.
        """

        # Validate percentile inputs for the function's contract
        if not (0 <= lower_p < 1 and 0 <= upper_p < 1 and lower_p < upper_p):
            raise ValueError(
                "Percentiles must be between 0 and 1, and lower_p must be less than upper_p. "
                f"Received: lower_p={lower_p}, upper_p={upper_p}"
            )

        df_cleaned = df  # Start with the original df; Polars filter creates new DataFrames
        initial_shape = df.shape
        print(f"Initial DataFrame shape: {initial_shape}")

        if not column_names:
            print("Warning: No column names provided for outlier removal. Returning original DataFrame.")
            return df

        for column_name in column_names:
            print(f"\n--- Processing outlier removal for column: '{column_name}' ---")
            # Shape of the DataFrame before attempting to filter for the current column
            shape_before_this_column_filter = df_cleaned.shape

            # Attempt to calculate quantile bounds using the original DataFrame's data for the current column
            try:
                # Crucially, quantiles are calculated from the original 'df' for each column
                # to ensure bounds are independent of previous filtering steps on other columns.
                col_data_for_quantile = df.get_column(column_name) # Ensures column exists before quantile
                lower_bound = col_data_for_quantile.quantile(lower_p, interpolation='linear')
                upper_bound = col_data_for_quantile.quantile(upper_p, interpolation='linear')
            except Exception as e: # Catches PolarsErrors like ColumnNotFoundError or non-numeric errors
                print(f"Error calculating quantiles for column '{column_name}': {e}")
                print(f"Ensure the column exists and is numeric. Skipping outlier removal for this column.")
                continue  # Skip to the next column_name

            # Handle cases where bounds might be None (e.g., column is empty or all nulls in original df)
            if lower_bound is None or upper_bound is None:
                print(
                    f"Warning: Could not determine sensible quantile bounds for '{column_name}' "
                    f"(column in original DataFrame might be empty, all nulls, or have too few distinct non-null values). "
                    f"Skipping outlier removal for this column."
                )
                continue

            # Handle cases where, due to unusual data, lower_bound might exceed upper_bound
            if lower_bound > upper_bound:
                print(
                    f"Warning: Calculated lower bound ({lower_bound:.4f}) is greater than upper bound ({upper_bound:.4f}) "
                    f"for column '{column_name}'. This can happen with unusual data distributions or percentile choices. "
                    f"Skipping outlier removal for this column."
                )
                continue

            print(f"Shape before filtering based on '{column_name}': {shape_before_this_column_filter}")
            print(f"Using percentiles for '{column_name}': lower={lower_p*100:.1f}th (value: {lower_bound:.4f}), upper={upper_p*100:.1f}th (value: {upper_bound:.4f})")

            height_before_filter_for_this_col = df_cleaned.height

            # Apply filter to the currently processed DataFrame
            df_cleaned = df_cleaned.filter(
                (pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound)
            )

            removed_count_this_column = height_before_filter_for_this_col - df_cleaned.height

            if removed_count_this_column > 0:
                print(f"Number of outliers removed based on '{column_name}': {removed_count_this_column}")
            else:
                print(f"No outliers removed based on '{column_name}'.")
            print(f"Shape after filtering based on '{column_name}': {df_cleaned.shape}")

        final_shape = df_cleaned.shape
        total_rows_removed = initial_shape[0] - final_shape[0]

        print(f"\n--- Outlier Removal Summary ---")
        print(f"Initial DataFrame shape: {initial_shape}")
        print(f"Final DataFrame shape after all successful filters: {final_shape}")
        print(f"Total rows removed: {total_rows_removed}")

        return df_cleaned
    return (remove_outliers_by_percentile,)


@app.cell
def _(pl):
    import statsmodels.api as sm
    import numpy as np

    def calculate_trend_params_statsmodels(group_df: pl.DataFrame, target_col_name: str) -> pl.DataFrame:
        """
        Calculates slope and intercept for a given group using statsmodels.OLS.
        Assumes data is clean and sufficient for regression.
        """
        x_series = group_df["year"].cast(pl.Float64)
        y_series = group_df[target_col_name].cast(pl.Float64)
        product_code = group_df['product_code'][0]

        # Prepare data for statsmodels (drop NaNs in y, ensure x aligns)
        valid_indices = y_series.is_not_nan()
        x_clean_pd = x_series.filter(valid_indices).to_pandas() # Independent variable (year)
        y_clean_pd = y_series.filter(valid_indices).to_pandas() # Dependent variable (metric_for_trend)

        slope = 0.0
        intercept = 0.0 # Default to 0.0 if conditions below aren't met or y_clean_pd is empty

        if not y_clean_pd.empty: # Ensure there's data to process
            # Calculate initial intercept as mean of y, if y_clean_pd is not empty.
            initial_intercept_candidate = np.nanmean(y_clean_pd.to_numpy())
            intercept = 0.0 if np.isnan(initial_intercept_candidate) else initial_intercept_candidate

        # Check if enough data points and distinct 'x' values for regression
        if len(x_clean_pd) >= 2 and x_clean_pd.nunique() >= 2:
            X_with_constant = sm.add_constant(x_clean_pd, prepend=True) # Add intercept column
            model = sm.OLS(y_clean_pd, X_with_constant, missing='drop') # OLS handles missing if any still exist
            results = model.fit()
            intercept = results.params.iloc[0]  # 'const' coefficient
            slope = results.params.iloc[1]      # 'year' coefficient

        elif not y_clean_pd.empty:
            slope = 0.0

        else:
            slope = 0.0
            intercept = 0.0


        return pl.DataFrame({"slope": [slope], "intercept": [intercept], "product_code": product_code})

    udf_output_schema_sm = {"slope": pl.Float64, "intercept": pl.Float64, "product_code": pl.String}
    return


@app.cell
def _(List, pl):
    # Index value and volume to make plotting more intuitive
    def simple_rebase(
        df: pl.DataFrame,
        baseline_date: str,
        value_cols: List[str],
        suffix: str = "_rebased"
    ) -> pl.DataFrame:
        """
        Simpler version: Rebases specified value columns in a Polars DataFrame to 100
        at the overall start date.

        Args:
            df: The input Polars DataFrame.
            baseline_date: str of the indexation date
            value_cols: A list of column names to be rebased.
            suffix: Suffix to append to the names of the new rebased columns.

        Returns:
            A Polars DataFrame with the added rebased columns.
        """
        date_col = 'year'
        # 2. Get the row containing the baseline values for this date
        # Using head(1) in case multiple rows exist for the baseline_date.
        baseline_row_df = df.filter(pl.col(date_col) == baseline_date).collect().head(1)

        # 3. Prepare expressions for new rebased columns
        new_column_expressions = []
        for col_name in value_cols:
            # This will error if col_name doesn't exist in baseline_row_df.
            base_value = baseline_row_df.select(pl.col(col_name)).item()

            # No explicit check for base_value being None or 0.
            # Polars handles division by None as null, division by 0 as inf.
            expression = ((pl.col(col_name) / base_value) * 100).alias(f"{col_name}{suffix}")
            new_column_expressions.append(expression)

        # If value_cols was empty, new_column_expressions will be empty,
        # and with_columns([]) will return the original df.
        if not new_column_expressions:
            return df

        return df.with_columns(new_column_expressions)
    return (simple_rebase,)


@app.cell
def _(functools, operator, pl, tqdm):
    def analyze_tariff_changes(
        unified_lf: pl.LazyFrame,
        start_year: str,
        end_year: str,
        year_gap: int,
        year_unit_value_end_gap: int,
        years_before_tariff_change_unit_value: int = 0,
        reporter_countries: list[str] = None,
        partner_countries: list[str] = None,
        product_codes: list[str] = None,
        tariff_col_name: str = 'effective_tariff',
        price_col_name: str = 'unit_value'
    ) -> pl.DataFrame:
        """
        Filters a Polars LazyFrame, identifies tariff changes over a configured period,
        extracts changes in unit value, value, and quantity, and returns these instances.

        Args:
            unified_lf: The input Polars LazyFrame containing trade data.
                        Expected columns: 'reporter_country', 'partner_country',
                                          'product_code', 'year', 'effective_tariff',
                                          'unit_value', 'value', 'quantity'.
            start_year: Starting year for analysis (e.g., "2000").
            end_year: End year for analysis (e.g., "2023").
            year_gap: Gap in years to identify and calculate the tariff change
                      (e.g., if 1, compares year Y with Y+1).
            year_unit_value_end_gap: Year gap from the start of the tariff change period (y1)
                                     to calculate the change in unit value up to
                                     (e.g., if 2, unit value change is y1 vs y1+2).
             years_before_tariff_change_unit_value: Year gap before the start of the tariff change
                                                    (first year) from which we measure unit value
                                                    change. (e.g. if 1, makes y1 = y1-1)
            reporter_countries: Optional list of reporter country codes to filter by.
            partner_countries: Optional list of partner country codes to filter by.
            product_codes: Optional list of product codes (or prefixes) to filter by.

        Returns:
            A Polars DataFrame containing instances of tariff changes and their
            relevant information.
        """

        # --- 1. FILTER DATA ---
        filtered_lf = unified_lf

        if reporter_countries:
            filtered_lf = filtered_lf.filter(
                pl.col("reporter_country").is_in(reporter_countries)
            )

        if partner_countries:
            filtered_lf = filtered_lf.filter(
                pl.col("partner_country").is_in(partner_countries)
            )

        if product_codes:
            conditions = [
                pl.col("product_code").str.slice(0, len(p)) == p
                for p in product_codes
            ]
            # Combine conditions with an OR
            combined_condition = functools.reduce(operator.or_, conditions)
            filtered_lf = filtered_lf.filter(combined_condition)

        # --- 2. PROCESS TARIFF CHANGES BY YEAR PAIR ---
        year_pairs = [
            (str(y1_int), str(y1_int + year_gap))
            for y1_int in range(int(start_year), int(end_year) + 1 - year_gap)
        ]

        group_cols = ["reporter_country", "partner_country", "product_code"]
        extracted_dfs_list = []

        for y1, y2 in tqdm(year_pairs, desc="Processing year pairs"):
            # The years between which we want to measure the price change
            year_before_imposition = str(int(y1) - years_before_tariff_change_unit_value)
            year_unit_value_end = str(int(y1) + int(year_unit_value_end_gap))

            # Ensure relevant_years are distinct and sorted, useful for filtering
            relevant_years = sorted(
                list(set([year_before_imposition, y1, y2, year_unit_value_end]))
            )

            changed_groups_lf = (
                filtered_lf.filter(pl.col("year").is_in(relevant_years))
                .group_by(group_cols, maintain_order=True)
                .agg(
                    # Tariff difference calculated between y1 and y2
                    (
                        pl.col(tariff_col_name)
                        .filter(pl.col("year") == y2)
                        .mean()
                        - pl.col(tariff_col_name)
                        .filter(pl.col("year") == y1)
                        .mean()
                    ).alias("tariff_difference"),
                    # Tariff percentage change calculated between y1 and y2
                    (
                        (
                            pl.col(tariff_col_name)
                            .filter(pl.col("year") == y2)
                            .mean()
                            - pl.col(tariff_col_name)
                            .filter(pl.col("year") == y1)
                            .mean()
                        )
                        / pl.col(tariff_col_name)
                        .filter(pl.col("year") == y1)
                        .mean()
                        * 100
                    ).alias("tariff_perc_change"),
                    # Unit value percentage change calculated between year_before_imposition and year_unit_value_end
                    ((
                        (
                            pl.col(price_col_name)
                            .filter(pl.col("year") == year_unit_value_end)
                            .mean()
                            - pl.col(price_col_name)
                            .filter(pl.col("year") == year_before_imposition)
                            .mean()
                        )
                        / pl.col(price_col_name)
                        .filter(pl.col("year") == year_before_imposition)
                        .mean()
                    ) * 100).alias("unit_value_perc_change"),
                    # Unit value change
                    (
                         pl.col(price_col_name)
                        .filter(pl.col("year") == year_unit_value_end)
                        .mean()
                        - pl.col(price_col_name)
                        .filter(pl.col("year") == year_before_imposition)
                        .mean()
                    ).alias('unit_value_difference'),
                    # Value difference calculated between year_before_imposition and year_unit_value_end
                    (
                        pl.col("value")
                        .filter(pl.col("year") == year_unit_value_end)
                        .sum()
                        - pl.col("value")
                        .filter(pl.col("year") == year_before_imposition)
                        .sum()
                    ).alias("value_difference"),
                    # Quantity difference calculated between year_before_imposition and year_unit_value_end
                    (
                        pl.col("quantity")
                        .filter(pl.col("year") == year_unit_value_end)
                        .sum()
                        - pl.col("quantity")
                        .filter(pl.col("year") == year_before_imposition)
                        .sum()
                    ).alias("quantity_difference"),
                )
                .filter(
                    (pl.col("tariff_difference").is_not_null())
                    & (pl.col("tariff_difference") != 0.0)
                    & (pl.col("unit_value_perc_change").is_not_null())
                )
                .with_columns(
                    # pl.lit(y1).alias("year_period_start"),
                    pl.lit(y1+"-"+y2).alias("tariff_change_range"),
                    pl.lit(year_before_imposition+"-"+year_unit_value_end).alias(
                        "unit_value_change_range"
                    ),
                )
            )

            extracted_dfs_list.append(changed_groups_lf.collect())

        combined_df = pl.concat(extracted_dfs_list)
        return combined_df
    return (analyze_tariff_changes,)


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

        # ## THIS IS TAKING TOO LONG ---->> 

        # detrend_calc_func = partial(
        #     calculate_trend_params_statsmodels, # User's UDF from CELL 1
        #     target_col_name=value_column_name
        # )

        # # Calculate trend parameters. Grouping is by 'product_code' and schema uses 'product_code'
        # trend_params_lf = source_lf.group_by(
        #     "product_code", maintain_order=True
        # ).map_groups(
        #     detrend_calc_func, schema=udf_output_schema_sm # User's schema from CELL 1
        # )

        # ### <<---- REPLACE WITH POLARS OPTIMISED OLS

        ### OPTIMISED VERSION ---->>

        source_lf = source_lf.with_columns(
                pl.col('year').cast(pl.Float64)
            ).sort('year')
        #     .filter(
        #     (pl.col(value_column_name).is_not_null() & pl.col('year').is_not_null())
        # )

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
        ### <<------

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
        ).sort('year')

        return detrended_lf

    return (apply_detrending_to_lazyframe,)


@app.cell
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_filtered_10000val_100c_sample_1000krows_filter")
    # unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")
    unified_lf = unified_lf.with_columns(
        (pl.col('value') / pl.col('quantity')).alias('unit_value')
    )
    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _(unified_lf):
    # Identify elements where there's change in tariff level between t and t+x
    # This means we need to create time series of each product and country pair. Or we could simply iterate over the whole table? Maybe that's actually fine.

    # Unique list of all countries
    country_list = set(unified_lf.select('reporter_country').unique().collect()['reporter_country'].to_list()).intersection(
        unified_lf.select('partner_country').unique().collect()['partner_country'].to_list()
    )

    # List of all years
    year_list = unified_lf.select('year').unique().collect()['year'].to_list()

    # List of all products
    product_list = unified_lf.select('product_code').unique().collect()['product_code'].to_list()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # 1. Effect of tariffs on bilateral trade prices
        How can we measure the effect of tariffs on bilateral trade values / prices?

        ## Steps
        1. Simple scatter plot of unitprice change vs tariff change at different intervals
        2. Detrend the unit value series using the global product trend methodology
        3. Repeat scatter plots - what's the effect of detrending?
        4. Estimate the passthrough for all instances of tariff imposition
        5. Plot the distribution of these passthrough effects in aggregate
        6. Repeat all the above, but sampling interesting cases such as US-China, time periods, etc.
        7. Perform a panel regression, controlling for fixed effects (other than trend)

        ## 1. Simple scatter plot of different subsets of the data
        """
    )
    return


@app.cell
def _():
    # 1. Filter the subset
    # 2. Identify all cases of tariff increase or decrease
    # 3. Extract the change in tariffs to the change in price
    # 4. Store in dataframe
    # 5. Plot on a scatter. Use the metadata to group products, countries, etc.
    # 6. Apply this methodology to the 2018 US-China tariffs on specific goods.
    # 7. Estimate the coefficient. Estimate statistical significance.
    return


@app.cell
def _(analyze_tariff_changes, unified_lf):
    year_gap = 1
    year_unit_value_end_gap = 2

    tariff_changes_df = analyze_tariff_changes(
        unified_lf,
        start_year="1996",
        end_year="2023",
        year_gap=year_gap,
        year_unit_value_end_gap=year_unit_value_end_gap,
        years_before_tariff_change_unit_value=0,
        reporter_countries=['156'], # China exporter, 
        partner_countries=['840'],
    )

    tariff_changes_df.head()
    return tariff_changes_df, year_gap, year_unit_value_end_gap


@app.cell
def _(tariff_changes_df):
    tariff_changes_df.describe()
    return


@app.cell
def _(remove_outliers_by_percentile, tariff_changes_df):
    # --- 3. REMOVE OUTLIERS --- 
    # Should probably run this earlier.
    normalised_combined_df = remove_outliers_by_percentile(
        tariff_changes_df,
        column_names=[
            'unit_value_perc_change_extended', 'tariff_difference', "tariff_perc_change", "unit_value_difference"
    ],
        upper_p=0.99,
        lower_p=0.01,
    )
    return (normalised_combined_df,)


@app.cell
def _(normalised_combined_df, px, year_gap, year_unit_value_end_gap):
    # --- 4. CREATE INITIAL TOTAL SCATTER ---
    scatter_fig = px.scatter(
        normalised_combined_df,
        x="tariff_difference",
        # x="tariff_perc_change",
        # y="unit_value_perc_change",
        y="unit_value_difference",
        title=f"""
            {year_gap}-year abs change in tariff vs {year_unit_value_end_gap}-year abs change in unit value US (importer) & China (exporter)
        """,
        hover_data=['tariff_change_range', "unit_value_change_range", 'product_code'],
        color='tariff_change_range',
        # trendline='ols'
    )

    scatter_fig.show()
    return


@app.cell
def _():
    # Expand year sample </
    # Detrend: remove the average price change across product lines and leave the residuals. 

    # Dumping - for those products where unit value went up, what changes affect other countries' imports. Quantity, value
    # Type of products analysis - are those poorly-correlated values just those where demand is super sensitive and so china absorbs costs? Are they the opposite where demand is super
    # % change of unit values

    # Detrend

    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Detrend

        To remove the trend from each product line, we might want to do something such:

        1. For each product line *k*
        2. Find the line of best fit for unit value growth across all cases
        3. Treat this as the underlying common price growth for the product line.
        4. Remove this from disaggregated unit_price for each *i-j*.
        """
    )
    return


@app.cell
def _(apply_detrending_to_lazyframe, functools, operator, pl, unified_lf):
    product_codes=['320417']
    reporter_countries=['156'] # China exporter, 
    partner_countries=['840'] # US importer

    test_lf = unified_lf
    if reporter_countries:
        test_lf = test_lf.filter(
            pl.col("reporter_country").is_in(reporter_countries)
        )

    if partner_countries:
        test_lf = test_lf.filter(
            pl.col("partner_country").is_in(partner_countries)
        )

    if product_codes:
        conditions = [
            pl.col("product_code").str.slice(0, len(p)) == p
            for p in product_codes
        ]
        # Combine conditions with an OR
        combined_condition = functools.reduce(operator.or_, conditions)
        test_lf = test_lf.filter(combined_condition)
    # 

    detrend_lf_filtered = apply_detrending_to_lazyframe(
        target_lf=test_lf,
        source_lf=unified_lf,
        value_column_name="unit_value",
        detrended_output_col_name='unit_value_detrend',
        trend_line_output_col_name='product_global_trend_vals'
    )

    return (
        detrend_lf_filtered,
        partner_countries,
        product_codes,
        reporter_countries,
    )


@app.cell
def _(detrend_lf_filtered):
    detrend_lf_filtered.head(25).collect(engine='streaming') # < ----- TEST
    return


@app.cell
def _(
    detrend_lf_filtered,
    partner_countries,
    pl,
    product_codes,
    px,
    reporter_countries,
    simple_rebase,
):
    # --- 6. VISUALISE EFFECTS OF DETRENDING AND REBASE ---
    # Investigate effects of detrending

    # Filter to a specific product code to make visualisation easier.
    # We've obviously previously filtered down to just the US / China

    # Rebase
    detrend_df_rebased_filtered = simple_rebase(
        detrend_lf_filtered, '1995', ['value', 'quantity']
    ).collect(engine='streaming')

    # If we previously selected multiple countries, etc. We need to group the data down
    detrend_df_rebased_filtered = detrend_df_rebased_filtered.group_by(
        'year'
    ).agg(
        pl.sum('quantity_rebased'),
        pl.sum('value_rebased'),
        pl.mean('effective_tariff'),
        pl.mean('unit_value_detrend'),
        pl.mean('unit_value')
    ).sort('year')
    # .with_columns(
    #     (pl.col('value_rebased') / pl.col('quantity_rebased')).alias('unit_value_rebased'),
    #     (pl.col('effective_tariff').pct_change(n=1)*100).alias('effective_tariff_pct_change')
    # )

    # Chart of aggregate quantities an values
    px.line(
        detrend_df_rebased_filtered,
        x='year',
        y=['quantity_rebased', 'value_rebased'],
        title=f"Aggregate export quantities and values for product: {product_codes[0]} between {reporter_countries} to {partner_countries} "
    ).update_traces(textposition='top center').show()

    # Chart of unit value trend and the OLS
    fig_unitvaluetrend = px.scatter(
        detrend_df_rebased_filtered,
        x='year',
        y='unit_value',
        title=f"unit value for product with OLS overlayed (red): {product_codes[0]}",
        trendline='ols',
        trendline_color_override="red",
    )
    fig_unitvaluetrend.update_traces(mode="lines")
    fig_unitvaluetrend.show()

    # Chart of the detrended unit_value and the % change in effective tariff
    fig_detrend_tariff = px.line(
        detrend_df_rebased_filtered,
        x='year',
        y='unit_value_detrend',
        title=f"unit value for product - Detrended: {product_codes[0]}"
    )

    # fig_detrend_tariff.add_trace(go.Scatter(
    #     x=detrended_df['year'],
    #     y=detrended_df['effective_tariff_pct_change'], 
    #     mode='lines',
    #     name='effective_tariff_pct_change' 
    # ))

    fig_detrend_tariff.show()
    return


@app.cell
def _():
    # Having applied the detrend to all the data, performed some indexation, etc. Let's now look at the unit values at the disaggregated level again. Create the table, plot the scatter
    return


@app.cell
def _(detrend_lf_filtered):
    detrend_lf_filtered.head(10).collect()
    return


@app.cell
def _(
    analyze_tariff_changes,
    detrend_lf_filtered,
    year_gap,
    year_unit_value_end_gap,
):
    tariff_changes_df_detrend_filtered = analyze_tariff_changes(
        detrend_lf_filtered,
        start_year="1996",
        end_year="2023",
        year_gap=year_gap,
        year_unit_value_end_gap=year_unit_value_end_gap,
        years_before_tariff_change_unit_value=0,
        reporter_countries=['156'], # China exporter, 
        partner_countries=['840'],
        tariff_col_name='effective_tariff',
        price_col_name='unit_value_detrend',
    )

    tariff_changes_df_detrend_filtered.head()
    return (tariff_changes_df_detrend_filtered,)


@app.cell
def _(remove_outliers_by_percentile, tariff_changes_df_detrend_filtered):
    # --- 3. REMOVE OUTLIERS --- 
    # Should probably run this earlier.
    tariff_changes_df_detrend_filtered_outliers = remove_outliers_by_percentile(
        tariff_changes_df_detrend_filtered,
        column_names=['unit_value_perc_change_extended', 'tariff_difference', "tariff_perc_change"],
        upper_p=0.99,
        lower_p=0.01,
    )
    return (tariff_changes_df_detrend_filtered_outliers,)


@app.cell
def _(tariff_changes_df_detrend_filtered_outliers):
    tariff_changes_df_detrend_filtered_outliers.head()
    return


@app.cell
def _(
    partner_countries,
    px,
    reporter_countries,
    tariff_changes_df_detrend_filtered_outliers,
    year_gap,
    year_unit_value_end_gap,
):
    # --- 4. CREATE INITIAL TOTAL SCATTER ---
    scatter_fig_detrend = px.scatter(
        tariff_changes_df_detrend_filtered_outliers,
        x="tariff_difference",
        # x="tariff_perc_change",
        # y="unit_value_perc_change_extended",
        y="unit_value_difference",
        title=f"""
            {year_gap}-year abs change in tariff % vs {year_unit_value_end_gap}-year abs change in unit value {partner_countries} (importer) & {reporter_countries} (exporter)
        """,
        hover_data=['tariff_change_range', "unit_value_change_range", 'product_code'],
        color='tariff_change_range',
        # trendline='ols'
    )

    scatter_fig_detrend.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Plot the distribution of pass-throughs (sensitivity of price to tariff)

        Where are the new tariffs on this distribution?
        """
    )
    return


if __name__ == "__main__":
    app.run()
