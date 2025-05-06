

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

    from tqdm import tqdm
    return functools, mo, operator, pl, px, tqdm


@app.cell
def _(pl):
    def remove_outliers_by_percentile(
        df: pl.DataFrame, 
        column_name: str, 
        lower_p: float = 0.01, 
        upper_p: float = 0.99
    ) -> pl.DataFrame:
        """
        Removes outliers from a specified column in a Polars DataFrame based on percentiles.

        Args:
            df: The input Polars DataFrame.
            column_name: The name of the column from which to remove outliers.
                         This column must be numeric.
            lower_p: The lower percentile (e.g., 0.01 for 1st percentile).
                         Must be between 0 and 1.
            upper_p: The upper percentile (e.g., 0.99 for 99th percentile).
                         Must be between 0 and 1, and greater than lower_p.

        Returns:
            A new Polars DataFrame with outliers removed from the specified column.
            If bounds cannot be determined (e.g., column is all nulls or empty),
            or if calculated lower_bound > upper_bound, the original DataFrame is returned with a warning.
            Errors from Polars (e.g., column not found, non-numeric column for quantile)
            will propagate.
        """

        # Validate percentile inputs for the function's contract
        if not (0 <= lower_p < 1 and 0 <= upper_p < 1 and lower_p < upper_p):
            raise ValueError(
                "Percentiles must be between 0 and 1, and lower_p must be less than upper_p. "
                f"Received: lower_p={lower_p}, upper_p={upper_p}"
            )

        # Attempt to calculate quantile bounds.
        # Polars will raise errors if column_name does not exist or is not numeric.
        try:
            lower_bound = df[column_name].quantile(lower_p, interpolation='linear')
            upper_bound = df[column_name].quantile(upper_p, interpolation='linear')
        except Exception as e:
            print(f"Error calculating quantiles for column '{column_name}': {e}")
            print("Ensure the column exists and is numeric. Returning original DataFrame.")
            return df


        # Handle cases where bounds might be None (e.g., column is empty or all nulls)
        if lower_bound is None or upper_bound is None:
            print(
                f"Warning: Could not determine sensible quantile bounds for '{column_name}' "
                f"(column might be empty, all nulls, or have too few distinct non-null values). "
                "Original DataFrame is returned."
            )
            return df

        # Handle cases where, due to unusual data, lower_bound might exceed upper_bound
        if lower_bound > upper_bound:
            print(
                f"Warning: Calculated lower bound ({lower_bound:.4f}) is greater than upper bound ({upper_bound:.4f}) "
                f"for column '{column_name}'. This can happen with unusual data distributions or percentile choices. "
                "Original DataFrame is returned."
            )
            return df

        print(f"\n--- Outlier Removal for column: '{column_name}' ---")
        print(f"Original DataFrame shape: {df.shape}")
        print(f"Using percentiles: lower={lower_p*100:.1f}th (value: {lower_bound:.4f}), upper={upper_p*100:.1f}th (value: {upper_bound:.4f})")

        # Filter the DataFrame
        df_cleaned = df.filter(
            (pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound)
        )

        removed_count = df.height - df_cleaned.height
        print(f"Number of outliers removed: {removed_count}")
        print(f"Cleaned DataFrame shape: {df_cleaned.shape}")

        return df_cleaned

    return (remove_outliers_by_percentile,)


@app.cell
def _(pl):
    # unified_lf = pl.scan_parquet("data/final/unified_filtered_10000val_100c_sample_1000krows_filter")
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")
    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Effect of tariff change on unit price level

        Can we identify a relationship between the unit price level at different intervals and the tariff application?

        What we're essentially looking for, is the change in unit price on the X axis and the tariff application on the Y axis, at different time delays.

        * If there is a change in the tariff level between t and t+x
        * What is the change in unit price between those

        So first, we need to find those elements where there's a change in the tariff level between t and t+x
        """
    )
    return


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
def _():
    # specific_df = saved_data_list[1110]
    # df_pd = specific_df.to_pandas()

    # # # title = f"Product {product}; country {country_name}"
    # # title='test'

    # # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # # fig.add_trace(
    # #     go.Scatter(x=df_pd['year'], y=df_pd['unit_value'], name='unit_value', mode='lines'),
    # #     secondary_y=False,
    # # )

    # # fig.add_trace(
    # #     go.Scatter(x=df_pd['year'], y=df_pd['effective_tariff'], name='effective_tariff', mode='lines'),
    # #     secondary_y=True,
    # # )

    # # fig.update_layout(
    # #     title_text=title
    # # )

    # # fig.update_yaxes(title_text="unit_value", secondary_y=False)
    # # fig.update_yaxes(title_text="effective_tariff", secondary_y=True)
    # # fig.update_xaxes(title_text="year")

    # # fig.show()

    # fig = px.line(
    #     df_pd,
    #     x='year',
    #     y=['unit_value', 'effective_tariff'],
    #     # title=f"Product {product}; country {country_name}",
    #     title="Test"
    # )
    # fig.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Regression approaches
        - Simple scatter plot of unitprice change vs tariff change at different intervals
        - Estimate overall parameters using the above!
        - More complex panel using some sort of panel OLS with HDFE
        - Other?

        ## 1. Simple scatter plot of different subsets of the data
        """
    )
    return


@app.cell
def _():
    # 1. Filter the subset
    # 2. Identify all cases of tariff increase or decrease
    # 3. Extract the change in tariffs to the change in price
    # 4. Store in a list along with some metadata
    # 5. Plot on a scatter. Use the metadata to group products, countries, etc.
    # 6. Apply this methodology to the 2018 US-China tariffs on specific goods.
    # 7. Estimate the coefficient. Estimate statistical significance.
    return


@app.cell
def _():
    # Config for subsequent analysis
    start_year = "2000" # Starting year for analysis
    end_year = "2023" # End year for analysis
    year_gap = 1 # Gap to identify and calculate the tariff change
    year_unit_value_end_gap = 2 # year gap to calculate the change in unit value to

    reporter_countries = ['156'] # China exporter
    partner_countries = ['840'] # USA importer
    # product_codes = ['01'] # Can use the aggregated code values...
    product_codes = None
    return (
        end_year,
        partner_countries,
        product_codes,
        reporter_countries,
        start_year,
        year_gap,
        year_unit_value_end_gap,
    )


@app.cell
def _(end_year, start_year, year_gap):
    # Create a list of year-on-year changes 
    year_range_end_excluded = [
        str(year)
        for year in range(int(start_year), int(end_year) + 1 - year_gap)
    ]

    year_pairs = []
    for year in year_range_end_excluded:
        year_pairs.append(
            [
                year, 
                str(int(year) + year_gap)
            ]
        )
    return (year_pairs,)


@app.cell
def _(
    functools,
    operator,
    partner_countries,
    pl,
    product_codes,
    reporter_countries,
    tqdm,
    unified_lf,
    year_pairs,
    year_unit_value_end_gap,
):
    # --- 1. FILTER DATA --- 
    if reporter_countries:
        filtered_lf = unified_lf.filter(
            pl.col('reporter_country').is_in(reporter_countries)
        )

    if partner_countries:
        filtered_lf = filtered_lf.filter(
            pl.col('partner_country').is_in(partner_countries)
        )

    if product_codes:
        conditions = [
            pl.col('product_code').str.slice(0, len(p)) == p
            for p in product_codes
        ]
        combined_condition = functools.reduce(operator.or_, conditions) # Combine conditions with an or
        filtered_lf = filtered_lf.filter(combined_condition)

    # Create a unit value column
    filtered_lf = filtered_lf.with_columns(
        (pl.col('value') / pl.col('quantity')).alias('unit_value')
    )

    # print("Length post filter:", filtered_lf.select(pl.len()).collect().item())

    # Now find all values within this set where there was a change in tariff between the years and get the magnitude of that change.
    group_cols = ['reporter_country', 'partner_country', 'product_code'] # This is our UUID
    extracted_dfs_list = []

    for y1, y2 in tqdm(year_pairs):

        # The year at which we want to stop measuring the price change
        year_before_imposition = str(int(y1) - 1)
        year_unit_value_end = str(int(y1) + int(year_unit_value_end_gap))
        relevant_years = sorted(list(set([year_before_imposition, y1, y2, year_unit_value_end])))

        # --- 2. FIND ROWS WHERE THE EFFECTIVE_TARIFF CHANGED --- 
        changed_groups_lf = filtered_lf.filter(
            pl.col('year').is_in(relevant_years)
        ).group_by(
            group_cols, maintain_order=True
        ).agg(
            # Tariff difference calculated between y1 and y2
            (
                pl.col('effective_tariff').filter(pl.col('year') == y2).first() -
                pl.col('effective_tariff').filter(pl.col('year') == y1).first()
            ).alias('tariff_difference'),
            # Unit value difference calculated between y1 and year_unit_value_end (y1+2)
            (
                (
                    pl.col('unit_value').filter(pl.col('year') == year_unit_value_end).first() -
                    pl.col('unit_value').filter(pl.col('year') == year_before_imposition).first()
                ) / pl.col('unit_value').filter(pl.col('year') == year_before_imposition).first()
            ).alias('unit_value_perc_change_extended'),
            # Value difference
            (
                pl.col('value').filter(pl.col('year') == year_unit_value_end).first() -
                pl.col('value').filter(pl.col('year') == year_before_imposition).first()
            ).alias('value_difference_extended'),
            # Quantity difference
            (
                pl.col('quantity').filter(pl.col('year') == year_unit_value_end).first() -
                pl.col('quantity').filter(pl.col('year') == year_before_imposition).first()
            ).alias('quantity_difference_extended'),
            # Contemporaneous unit value difference (between y1 and y2) for comparison
            (
                (
                    pl.col('unit_value').filter(pl.col('year') == y2).first() -
                    pl.col('unit_value').filter(pl.col('year') == y1).first()
                ) / pl.col('unit_value').filter(pl.col('year') == y1).first()
            ).alias('unit_value_perc_tariff_period')

        ).filter(
            (pl.col('tariff_difference').is_not_null()) &
            (pl.col('tariff_difference') != 0.0) &
            (pl.col('unit_value_perc_change_extended').is_not_null())

        ).with_columns(
            pl.lit(y1).alias('year_period_start'),
            pl.lit(y2).alias('year_tariff_change_observed'),
            pl.lit(year_unit_value_end).alias('year_unit_value_extended_end')
        )

        # changed_groups_lf = filtered_lf.filter(
        #     pl.col('year').is_in([y1, y2])
        # ).group_by(
        #     group_cols
        # ).agg(
        #     (pl.col('effective_tariff').last() - pl.col('effective_tariff').first()).alias('tariff_difference'),
        #     (pl.col('value').last() - pl.col('value').first()).alias('value_difference'),
        #     (pl.col('quantity').last() - pl.col('quantity').first()).alias('quantity_difference'),
        #     (pl.col('unit_value').last() - pl.col('unit_value').first()).alias('unit_value_difference'),
        # ).filter(
        #     pl.col('tariff_difference') != 0.0
        # ).with_columns(
        #     pl.lit(y1).alias('start_year'),
        #     pl.lit(y2).alias('end_year'),
        # )

        extracted_dfs_list.append(changed_groups_lf.collect())
    return (extracted_dfs_list,)


@app.cell
def _(extracted_dfs_list, pl):
    combined_df = pl.concat(extracted_dfs_list)
    combined_df.sample(10)
    return (combined_df,)


@app.cell
def _(combined_df, remove_outliers_by_percentile):
    normalised_combined_df = remove_outliers_by_percentile(
        combined_df,
        column_name='unit_value_perc_change_extended',
    )
    normalised_combined_df = remove_outliers_by_percentile(
        normalised_combined_df,
        column_name='tariff_difference',
    )
    return (normalised_combined_df,)


@app.cell
def _(normalised_combined_df, px, year_gap, year_unit_value_end_gap):
    scatter_fig = px.scatter(
        normalised_combined_df,
        x="tariff_difference",
        y="unit_value_perc_change_extended",
        title=f"""
            {year_gap}-year change in tariffs (abs in %) vs {year_unit_value_end_gap}-year change in unit value (% change) over same period between US (importer) & China (exporter)
        """,
        hover_data=['year_period_start', "year_tariff_change_observed", 'product_code'],
        color='year_period_start',
        trendline='ols'
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

        1. For each product line
        2. Find the line of best fit for unit value growth across all cases
        3. Treat this as the underlying common price growth for the product line.
        4. Remove this from individual countries' data.
        """
    )
    return


@app.cell
def _(pl, unified_lf):
    detrend_lf = unified_lf.with_columns(
        (pl.col('value') / pl.col('quantity')).alias('unit_value')
    ).group_by(
        ['product_code', 'year'], maintain_order=True
    ).agg(
        pl.sum('value'),
        pl.sum('quantity'),
        pl.mean('unit_value'),
    )

    # detrend_lf = detrend_lf.explode(['unit_value', 'value', 'quantity', 'year'])
    test_code = '870899'
    detrend_lf.filter(pl.col('product_code') == test_code).collect()
    return (detrend_lf,)


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

        elif not y_clean_pd.empty():
            slope = 0.0

        else:
            slope = 0.0
            intercept = 0.0


        return pl.DataFrame({"slope": [slope], "intercept": [intercept]})

    udf_output_schema_sm = {"slope": pl.Float64, "intercept": pl.Float64}
    return calculate_trend_params_statsmodels, udf_output_schema_sm


@app.cell
def _(
    calculate_trend_params_statsmodels,
    detrend_lf,
    pl,
    udf_output_schema_sm,
):
    from functools import partial

    detrend_calc_func = partial(
        calculate_trend_params_statsmodels,
        target_col_name="unit_value"
    )

    trend_params_lf = detrend_lf.group_by(
        "product_code", maintain_order=True
    ).map_groups(
        detrend_calc_func, schema=udf_output_schema_sm
    ).with_columns(
        pl.lit('T')
    )

    trend_params_lf.head().collect()
    return


@app.cell
def _():
    # Detrend, having calculated the slope and y intercept for each product code
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
