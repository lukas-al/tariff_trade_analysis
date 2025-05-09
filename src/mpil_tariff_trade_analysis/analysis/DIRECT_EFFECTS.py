

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from tqdm import tqdm
    return go, make_subplots, mo, pl, px, tqdm


@app.cell
def _(mo):
    mo.md(
        r"""
        # Visualise the direct effects of tariffs on bilateral trade prices
        Simple: create charts which explain the direct effects of tariffs on trade

        ## Vis 1: 
        Scatter plot(s) of the change in tariff against the contemporanous, lagged 1 yr, lagged 2 yr change in bilateral unit price. For all products, for whatever subset of the dataset.

        ## Vis 2: 
        The same, but using the de-trended data

        ## Vis 3: 
        Visualising the detrending of the data. 

        1. Plot the global trend against the series.
        2. Plot all the individual country time series and the global trend. Highlight the US' trade and the UK's with China, for example.
        3. View how the de-trending affects the series.

        ## Vis 4: 
        A histogram of all the 'effective passthroughs after N years' of tariff change to bilateral trade price.

        ## Vis 5:
        Conclusion: is this statistically significant?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Functions
        Extracted functions to do the analysis
        """
    )
    return


@app.cell
def _(pl):
    def remove_outliers_by_percentile_lazy(
        ldf: pl.LazyFrame,
        column_names: list[str],
        lower_p: float = 0.01,
        upper_p: float = 0.99
    ) -> pl.LazyFrame:
        """
        Removes outliers from specified columns in a Polars LazyFrame based on percentiles
        without collecting the LazyFrame. Outlier bounds for each column are determined
        from its distribution in the original input LazyFrame. Filters are then applied
        sequentially.

        Args:
            ldf: The input Polars LazyFrame.
            column_names: A list of names of columns from which to remove outliers.
                          These columns must be numeric.
            lower_p: The lower percentile (e.g., 0.01 for 1st percentile).
                     Must be between 0 and 1.
            upper_p: The upper percentile (e.g., 0.99 for 99th percentile).
                     Must be between 0 and 1, and greater than lower_p.

        Returns:
            A new Polars LazyFrame with outlier filtering applied to the specified columns.
        """
        if not (0 <= lower_p < 1 and 0 <= upper_p < 1 and lower_p < upper_p):
            raise ValueError(
                "Percentiles must be between 0 and 1, and lower_p must be less than upper_p. "
                f"Received: lower_p={lower_p}, upper_p={upper_p}"
            )

        ldf_cleaned = ldf

        for column_name in column_names:
            try:
                lower_bound = ldf.select(pl.col(column_name).quantile(lower_p)).collect().item()
                upper_bound = ldf.select(pl.col(column_name).quantile(upper_p)).collect().item()
                ldf_cleaned = ldf_cleaned.filter(
                    (pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound)
                )
            except Exception:
                print(f"Warning: Skipping outlier removal for column '{column_name}'. Ensure it is numeric.")

        return ldf_cleaned
    return (remove_outliers_by_percentile_lazy,)


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
    return


@app.cell
def _(pl, tqdm):
    def extract_tariff_changes(
        filtered_lf: pl.LazyFrame,
        start_year: str,
        end_year: str,
        year_gap: int,
        year_unit_value_end_gap: int,
        years_before_tariff_change_unit_value: int = 0,
        tariff_col_name: str = 'effective_tariff',
        price_col_name: str = 'unit_value'
    ) -> pl.DataFrame:
        """
        Filters a Polars LazyFrame, identifies tariff changes over a configured period,
        extracts changes in unit value, value, and quantity, and returns these instances.

        Args:
            filtered_lf: The input Polars LazyFrame containing trade data.
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

        Returns:
            A Polars DataFrame containing instances of tariff changes and their
            relevant information.
        """

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
                        .filter(pl.col("year") == y2) # Average tariff in y2
                        .mean()
                        - pl.col(tariff_col_name)
                        .filter(pl.col("year") == y1) # Average tariff in y1
                        .mean()
                    ).alias("tariff_change"),
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
                    # Unit value g
                    (
                         pl.col(price_col_name)
                        .filter(pl.col("year") == year_unit_value_end)
                        .mean()
                        - pl.col(price_col_name)
                        .filter(pl.col("year") == year_before_imposition)
                        .mean()
                    ).alias(price_col_name+"_change"),
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
                    ) * 100).alias(price_col_name+"_perc_change"), # price_col_name+"_perc_change"
                    # Value difference calculated between year_before_imposition and year_unit_value_end
                    (
                        pl.col("value")
                        .filter(pl.col("year") == year_unit_value_end)
                        .sum()
                        - pl.col("value")
                        .filter(pl.col("year") == year_before_imposition)
                        .sum()
                    ).alias("value_change"),
                    # Quantity difference calculated between year_before_imposition and year_unit_value_end
                    (
                        pl.col("quantity")
                        .filter(pl.col("year") == year_unit_value_end)
                        .sum()
                        - pl.col("quantity")
                        .filter(pl.col("year") == year_before_imposition)
                        .sum()
                    ).alias("quantity_change"),
                )
                .filter(
                    (pl.col("tariff_change").is_not_null())
                    & (pl.col("tariff_change") != 0.0)
                    & (pl.col(price_col_name+"_change").is_not_null())
                )
                .with_columns(
                    pl.lit(y1+"-"+y2).alias("tariff_change_range"),
                    pl.lit(year_before_imposition+"-"+year_unit_value_end).alias(
                        "unit_value_change_range"
                    ),
                )
            )

            changed_groups_df = changed_groups_lf.collect()
            print(f"    Identified {changed_groups_df.height} cases of tariff change between {y1}-{y2}")
        
            extracted_dfs_list.append(changed_groups_df)
    
        combined_df = pl.concat(extracted_dfs_list)
        return combined_df
    return (extract_tariff_changes,)


@app.cell
def _(mo):
    mo.md(r"""# Config""")
    return


@app.cell
def _():
    price_change_start = 0 # Number of years before the first year of the tariff change that we start measuring prices.
    price_change_end = 2 # Number of years after the last year of the tariff change that we stop measuring prices.
    tariff_change_gap = 1 # Gap between years that we're measuring the start of tariff changes

    reporter_countries = ["156"] # Reporter countries to filter to [China]
    partner_countries = ["840"] # Partner countries to filter to [USA]
    product_codes = None # Products to filter to

    # data_to_load = "data/final/unified_filtered_10000minval_top100countries/" 
    data_to_load = "data/final/unified_trade_tariff_partitioned/"
    return (
        data_to_load,
        partner_countries,
        price_change_end,
        price_change_start,
        product_codes,
        reporter_countries,
        tariff_change_gap,
    )


@app.cell
def _(data_to_load, pl):
    unified_lf = pl.scan_parquet(data_to_load)

    print(unified_lf.head().collect())
    unified_lf.head().collect() # If we're in notebook
    return (unified_lf,)


@app.cell
def _(
    functools,
    operator,
    partner_countries,
    pl,
    product_codes,
    remove_outliers_by_percentile_lazy,
    reporter_countries,
    unified_lf,
):
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

    filtered_lf = remove_outliers_by_percentile_lazy(
        ldf=filtered_lf,
        column_names=[
            "effective_tariff", 
            "unit_value",
            "unit_value_detrended",
        ],
        lower_p=0.01,
        upper_p=0.99,
    )
    return (filtered_lf,)


@app.cell
def _(filtered_lf, pl):
    print(f"--- LDF post filtering: ---")
    print(f"   Post filtering: {filtered_lf.select(pl.len()).collect().item()} Rows left")

    filtered_lf_head = filtered_lf.head().collect()

    print(filtered_lf_head) # If in cmdline
    filtered_lf_head # If interactive
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Vis 1
        Scatter plot(s) of the change in tariff against the contemporanous, lagged 1 yr, lagged 2 yr change in bilateral unit price. For all products, for whatever subset of the dataset.
        """
    )
    return


@app.cell
def _(
    extract_tariff_changes,
    filtered_lf,
    price_change_end,
    price_change_start,
    tariff_change_gap,
):
    # First we need to extract the rows we're interested in
    print("--- Extracting cases of change in tariff ---")

    tariff_changes_df = extract_tariff_changes(
        filtered_lf=filtered_lf,
        start_year='1999',
        end_year='2023',
        years_before_tariff_change_unit_value=price_change_start,
        year_unit_value_end_gap=price_change_end,
        year_gap=tariff_change_gap,
        tariff_col_name="effective_tariff",
        price_col_name="unit_value",
    )

    print(f"--- Identified {tariff_changes_df.height} cases of tariff change in sample ---")
    return (tariff_changes_df,)


@app.cell
def _(tariff_changes_df):
    print(tariff_changes_df.head()) # If we're working from cmdline
    tariff_changes_df.head() # If we're in notebook mode
    return


@app.cell
def _(
    price_change_end,
    price_change_start,
    px,
    tariff_change_gap,
    tariff_changes_df,
):
    # Draw our chart
    scatter_filtered = px.scatter(
        tariff_changes_df,
        x="tariff_change",
        # x="tariff_perc_change",
        # y="unit_value_perc_change",
        y="unit_value_change",
        title=f"""
            {tariff_change_gap}-year abs change in tariff vs {price_change_start}-{price_change_end}-year abs change in unit value US (importer) & China (exporter)
        """,
        hover_data=['tariff_change_range', "unit_value_change_range", 'product_code'],
        color='tariff_change_range',
        # trendline='ols'
    )

    scatter_filtered.write_html('charts/direct_effects/scatter_US-CHINA.html')
    scatter_filtered.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Vis 2
        The same, but using the de-trended data
        """
    )
    return


@app.cell
def _(
    extract_tariff_changes,
    filtered_lf,
    price_change_end,
    price_change_start,
    tariff_change_gap,
):
    # First we need to extract the rows we're interested in
    print("--- Extracting cases of change in tariff ---")

    tariff_changes_df_detrend = extract_tariff_changes(
        filtered_lf=filtered_lf,
        start_year='1999',
        end_year='2023',
        years_before_tariff_change_unit_value=price_change_start,
        year_unit_value_end_gap=price_change_end,
        year_gap=tariff_change_gap,
        tariff_col_name="effective_tariff",
        price_col_name="unit_value_detrended",
    )

    print(f"--- Identified {tariff_changes_df_detrend.height} cases of tariff change in sample ---")
    return (tariff_changes_df_detrend,)


@app.cell
def _(tariff_changes_df_detrend):
    print(tariff_changes_df_detrend.head()) # If we're working from cmdline
    tariff_changes_df_detrend.head() # If we're in notebook mode
    return


@app.cell
def _(
    price_change_end,
    price_change_start,
    px,
    tariff_change_gap,
    tariff_changes_df_detrend,
):
    # Draw our chart
    scatter_filtered_detrend = px.scatter(
        tariff_changes_df_detrend,
        x="tariff_change",
        # x="tariff_perc_change",
        # y="unit_value_perc_change",
        y="unit_value_detrended_change",
        title=f"""
            t0 to t+{tariff_change_gap} year abs change in tariff vs t+{price_change_start} to t+{price_change_end} year abs change in unit value US (importer) & China (exporter)
        """,
        hover_data=['tariff_change_range', "unit_value_change_range", 'product_code'],
        color='tariff_change_range',
        # trendline='ols'
    )

    scatter_filtered_detrend.write_html('charts/direct_effects/scatter_detrend_US-CHINA.html')
    scatter_filtered_detrend.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Vis 3

        Visualising the detrending of the data.

        1. Plot the global trend against the series.
        2. Plot all the individual country time series and the global trend. Highlight the US' trade and the UK's with China, for example.
        3. View how the de-trending affects the series.
        """
    )
    return


@app.cell
def _(filtered_lf, pl):
    # Select a product. Plot the global trend against that series.
    unique_products = filtered_lf.select(pl.col('product_code')).unique().collect().sort('product_code')['product_code']
    product_of_interest = [unique_products[2]]

    # product_of_interest = ["940600"]# Hard fix for now since unique has a random order

    print("Inspecting product:", product_of_interest)
    return (product_of_interest,)


@app.cell
def _(filtered_lf, pl, product_of_interest):
    product_df = filtered_lf.filter(pl.col('product_code').is_in(product_of_interest)).collect()

    # In case we've selected more than one reporter/partner_country/product_code
    product_df = product_df.group_by(['year', 'product_code']).agg(
        pl.sum('quantity'),
        pl.sum('value'),
        pl.mean('effective_tariff'),
        pl.mean('unit_value_detrended'),
        pl.mean('unit_value'),
        pl.mean('product_global_trend'),
        weighted_effective_tariff = (pl.col('effective_tariff') * pl.col('value')).sum() / pl.col('value').sum(),
        weighted_unit_value = (pl.col('unit_value') * pl.col('value')).sum() / pl.col('value').sum(),
        weighted_unit_value_detrended = (pl.col('unit_value_detrended') * pl.col('value')).sum() / pl.col('value').sum(),
    ).sort('year')

    print("Grouped filtered_df and weighted trade volumes. In case N*i-j* > 1\n", product_df.head())
    product_df.head()
    return (product_df,)


@app.cell
def _(product_df, product_of_interest, px):
    # Line chart 1: Product vs global trend
    global_trend_vs_series = px.line(
        product_df,
        x="year",
        y=["weighted_unit_value", "product_global_trend"],
        color="product_code",
        title="Global trend (dotted) vs product price (solid)",
    )

    # Hack to make the global trend appear as dotted
    global_trend_vs_series.for_each_trace(
        lambda trace: trace.update(line_dash="dot") if not trace.showlegend else ()
    )

    global_trend_vs_series.write_html(
        f"charts/direct_effects/global_trend_vs_product_{product_of_interest}.html"
    )
    global_trend_vs_series.show()
    return


@app.cell
def _(go, make_subplots, pl, product_df, product_of_interest, px):
    def _():
        # Get unique product codes and assign colors
        unique_product_codes = product_df['product_code'].unique().to_list()
        colors = px.colors.qualitative.Plotly

        # Create a figure with 2 rows
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(
                                "Global Trend vs Product Price & Detrended Value (Bar)",
                                            "Weighted Unit Value Detrended (Line)"),
                            row_heights=[0.7, 0.3]
                           )

        # --- Chart 1: Product vs global trend (lines) and detrended (bars) ---
        for i, product in enumerate(unique_product_codes):
            product_specific_df = product_df.filter(pl.col('product_code') == product)
            color = colors[i % len(colors)]

            # Line for weighted_unit_value
            fig.add_trace(
                go.Scatter(
                    x=product_specific_df['year'],
                    y=product_specific_df['weighted_unit_value'],
                    name=f'{product} - Weighted Value',
                    legendgroup=f'group-{product}',
                    line=dict(color=color),
                    showlegend=True
                ),
                row=1, col=1
            )

            # Line for product_global_trend (dotted)
            fig.add_trace(
                go.Scatter(
                    x=product_specific_df['year'],
                    y=product_specific_df['product_global_trend'],
                    name=f'{product} - Global Trend',
                    legendgroup=f'group-{product}',
                    line=dict(dash='dot', color=color),
                    showlegend=True
                ),
                row=1, col=1
            )

            # Bar for weighted_unit_value_detrended
            fig.add_trace(
                go.Bar(
                    x=product_specific_df['year'],
                    y=product_specific_df['weighted_unit_value_detrended'],
                    name=f'{product} - Detrended (Bar)',
                    legendgroup=f'group-{product}',
                    marker_color=color,
                    opacity=0.6, # Optional: make bars slightly transparent
                    showlegend=True
                ),
                row=1, col=1
            )

        # --- Chart 2: Line of weighted_unit_value_detrended ---
        for i, product in enumerate(unique_product_codes):
            product_specific_df = product_df.filter(pl.col('product_code') == product)
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=product_specific_df['year'],
                    y=product_specific_df['weighted_unit_value_detrended'],
                    name=f'{product} - Detrended (Line)',
                    legendgroup=f'group-{product}-detrended', # Use a different legend group if needed
                    line=dict(color=color),
                    showlegend=False # Avoid duplicate legend entries if names are similar
                ),
                row=2, col=1
            )


        # Update layout
        fig.update_layout(
            height=700,
            title_text="Analysis of detrending effect on individual products",
            barmode='group' # For the bars in the first subplot
        )
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Detrended Value", row=2, col=1)
        fig.write_html(f'charts/direct_effects/detrending_effect_{product_of_interest}.html')
        return fig.show()


    _()
    return


@app.cell
def _(go, pl, product_of_interest, unified_lf):
    ### Line chart - the global lines for the countries

    num_top_ij = 50

    # First filter for the right year
    product_specific_lf = unified_lf.filter(
        pl.col('product_code').is_in(product_of_interest)
    )

    # Calculate total value for each reporter-partner pair
    top_pairs_ranked_lf = product_specific_lf.group_by(['reporter_country', 'partner_country']).agg(
        total_value_for_pair = pl.sum('value')
    ).sort(
        'total_value_for_pair', descending=True
    ).head(num_top_ij)

    top_pairs_keys_lf = top_pairs_ranked_lf.select(['reporter_country', 'partner_country'])

    # Use an inner join to filter the data
    top_pairs_data_lf = product_specific_lf.join(
        top_pairs_keys_lf,
        on=['reporter_country', 'partner_country'],
        how='inner'
    ).group_by(
        ['year', 'reporter_country', 'partner_country']
    ).agg(
        # unit_value = pl.sum('value') / pl.sum('quantity')
        pl.mean('unit_value')
    ).sort('year')

    top_pairs_data_df = top_pairs_data_lf.collect()
    product_specific_df = product_specific_lf.select(
        ['product_global_trend', 'year']
    ).group_by('year').agg(
        pl.mean('product_global_trend') # Should only be one value anyway
    ).sort('year').collect()

    global_trend_vs_individual_ijk = go.Figure()

    unique_pairs_to_plot = top_pairs_data_df.select(['reporter_country', 'partner_country']).unique()

    for pair_row in unique_pairs_to_plot.iter_rows(named=True):
        reporter = pair_row['reporter_country']
        partner = pair_row['partner_country']
    
        pair_specific_data = top_pairs_data_df.filter(
            (pl.col('reporter_country') == reporter) & (pl.col('partner_country') == partner)
        ).sort('year')
    
        global_trend_vs_individual_ijk.add_trace(
            go.Scatter(
                x=pair_specific_data['year'],
                y=pair_specific_data['unit_value'],
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.4)'),
                showlegend=False,
                hoverinfo='skip'
                # name=f'{reporter}-{partner}',
                # hovertemplate='Year: %{x}<br>Unit Value: %{y:.2f}<extra></extra>'
            )
        )

    global_trend_vs_individual_ijk.add_trace(
        go.Scatter(
            x=product_specific_df['year'],
            y=product_specific_df['product_global_trend'],
            mode='lines',
            line=dict(color='red', width=2.5),
            name='Global Trend'
        )
    )

    title_product_str = product_of_interest[0] if product_of_interest and len(product_of_interest) == 1 else "Selected Product"
    global_trend_vs_individual_ijk.update_layout(
        title=f'Top {num_top_ij} Importer-Exporter Pair Trends vs. Global Trend for Product: {title_product_str}',
        xaxis_title='Year',
        yaxis_title='Unit Value',
        showlegend=False # To show the legend for the 'Global Trend' line
    )

    global_trend_vs_individual_ijk.write_html(f'charts/direct_effects/global_trend_overlayed_{product_of_interest}.html')
    global_trend_vs_individual_ijk.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Vis 4
        A histogram of all the 'effective passthroughs after N years' of tariff change to bilateral trade price.

        Effective passthrough = (dT / dP)
        """
    )
    return


@app.cell
def _(tariff_changes_df_detrend):
    tariff_changes_df_detrend
    return


@app.cell
def _(pl, tariff_changes_df_detrend):
    effective_passthroughs_df = tariff_changes_df_detrend.with_columns(
        (pl.col("unit_value_detrended_change") / pl.col("tariff_change")).alias('effective_passthrough')
    ).select('effective_passthrough').drop_nulls() # drop nulls that may result from division by zero or other issues

    # Extract the column as a Polars Series
    effective_passthroughs_series = effective_passthroughs_df.get_column('effective_passthrough')
    return (effective_passthroughs_df,)


@app.cell
def _(effective_passthroughs, effective_passthroughs_df, px):

    # Calculate mean and median
    mean_val = effective_passthroughs.mean()
    median_val = effective_passthroughs.median()

    # 1. Create a histogram with lines for the mean and median
    fig_hist_lines = px.histogram(
        effective_passthroughs_df,
        x='effective_passthrough',
        title='Histogram of Effective Passthrough with Mean and Median Lines'
    )

    fig_hist_lines.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )

    fig_hist_lines.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="bottom right"
    )

    print("Showing histogram with mean and median lines:")
    fig_hist_lines.show()

    # 2. Create a histogram with an overlaid box plot
    # Plotly Express can create a histogram with a marginal box plot.
    fig_hist_box = px.histogram(
        effective_passthroughs_df,
        x='effective_passthrough',
        marginal='box',  # Add a box plot on the margin
        title='Histogram of Effective Passthrough with Overlaid Box Plot'
    )

    print("Showing histogram with an overlaid box plot:")
    fig_hist_box.write_html('charts/direct_effects/effective_passthrough_hist.html')
    fig_hist_box.show()

    return mean_val, median_val


@app.cell
def _(effective_passthroughs_df, mean_val, median_val, px):
    # 1. Create a histogram with lines for the mean and median, with log y-axis
    fig_hist_lines_log_y = px.histogram(
        effective_passthroughs_df,
        x='effective_passthrough',
        title='Histogram of Effective Passthrough (Log Y-axis) with Mean and Median Lines',
        log_y=True  # Set y-axis to logarithmic scale
    )

    fig_hist_lines_log_y.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )

    fig_hist_lines_log_y.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="bottom right"
    )

    print("Showing histogram with mean/median lines and log y-axis:")
    fig_hist_lines_log_y.write_html('charts/direct_effects/effective_passthrough_hist_logY.html')
    fig_hist_lines_log_y.show()
    return


if __name__ == "__main__":
    app.run()
