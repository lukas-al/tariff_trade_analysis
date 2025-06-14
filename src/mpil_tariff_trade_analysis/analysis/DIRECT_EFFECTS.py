import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse

    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    return argparse, go, make_subplots, mo, pl, px


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
    # Hack to avoid rendering in dashboard view
    _ = 42
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Functions
        Extracted functions to do the analysis
        """
    )
    # Hack to avoid rendering in dashboard view
    _ = 42
    return


@app.cell(hide_code=True)
def _(pl):
    def remove_outliers_by_percentile_lazy(
        ldf: pl.LazyFrame,
        column_names: list[str],
        lower_p: float = 0.01,
        upper_p: float = 0.99,
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
                f"Percentiles must be between 0 and 1, and lower_p must be less than upper_p. Received: lower_p={lower_p}, upper_p={upper_p}"
            )

        ldf_cleaned = ldf

        for column_name in column_names:
            try:
                lower_bound = ldf.select(pl.col(column_name).quantile(lower_p)).collect().item()
                upper_bound = ldf.select(pl.col(column_name).quantile(upper_p)).collect().item()
                ldf_cleaned = ldf_cleaned.filter((pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound))
            except Exception:
                print(f"Warning: Skipping outlier removal for column '{column_name}'. Ensure it is numeric.")

        return ldf_cleaned

    return (remove_outliers_by_percentile_lazy,)


@app.cell(hide_code=True)
def _(pl):
    def remove_outliers_by_percentile(
        df: pl.DataFrame,
        column_names: list[str],  # Modified: now accepts a list of column names
        lower_p: float = 0.01,
        upper_p: float = 0.99,
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
                f"Percentiles must be between 0 and 1, and lower_p must be less than upper_p. Received: lower_p={lower_p}, upper_p={upper_p}"
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
                col_data_for_quantile = df.get_column(column_name)  # Ensures column exists before quantile
                lower_bound = col_data_for_quantile.quantile(lower_p, interpolation="linear")
                upper_bound = col_data_for_quantile.quantile(upper_p, interpolation="linear")
            except Exception as e:  # Catches PolarsErrors like ColumnNotFoundError or non-numeric errors
                print(f"Error calculating quantiles for column '{column_name}': {e}")
                print("Ensure the column exists and is numeric. Skipping outlier removal for this column.")
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
            print(
                f"Using percentiles for '{column_name}': lower={lower_p * 100:.1f}th (value: {lower_bound:.4f}), upper={upper_p * 100:.1f}th (value: {upper_bound:.4f})"
            )

            height_before_filter_for_this_col = df_cleaned.height

            # Apply filter to the currently processed DataFrame
            df_cleaned = df_cleaned.filter((pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound))

            removed_count_this_column = height_before_filter_for_this_col - df_cleaned.height

            if removed_count_this_column > 0:
                print(f"Number of outliers removed based on '{column_name}': {removed_count_this_column}")
            else:
                print(f"No outliers removed based on '{column_name}'.")
            print(f"Shape after filtering based on '{column_name}': {df_cleaned.shape}")

        final_shape = df_cleaned.shape
        total_rows_removed = initial_shape[0] - final_shape[0]

        print("\n--- Outlier Removal Summary ---")
        print(f"Initial DataFrame shape: {initial_shape}")
        print(f"Final DataFrame shape after all successful filters: {final_shape}")
        print(f"Total rows removed: {total_rows_removed}")

        return df_cleaned

    return


@app.cell
def _(mo, pl):
    def extract_tariff_changes(
        filtered_lf: pl.LazyFrame,
        start_year: str,
        end_year: str,
        year_gap: int,
        year_unit_value_end_gap: int,
        tariff_col_name: str,
        price_col_name: str,
        years_before_tariff_change_unit_value: int = 0,
    ) -> pl.DataFrame:
        """
        Filters a Polars LazyFrame, identifies tariff changes over a configured period,
        extracts changes in unit value, value, and quantity, and returns these instances.

        Args:
            filtered_lf: The input Polars LazyFrame containing trade data.
                        Expected columns: 'reporter_country', 'partner_country',
                                          'product_code', 'year', 'average_tariff_official',
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
        year_pairs = [(str(y1_int), str(y1_int + year_gap)) for y1_int in range(int(start_year), int(end_year) + 1 - year_gap)]

        group_cols = ["reporter_country", "partner_country", "product_code"]
        extracted_dfs_list = []

        for y1, y2 in mo.status.progress_bar(year_pairs, title="Processing year pairs"):
            # The years between which we want to measure the price change
            year_before_imposition = str(int(y1) - years_before_tariff_change_unit_value)
            year_unit_value_end = str(int(y1) + int(year_unit_value_end_gap))

            # Ensure relevant_years are distinct and sorted, useful for filtering
            relevant_years = sorted(list(set([year_before_imposition, y1, y2, year_unit_value_end])))

            changed_groups_lf = (
                filtered_lf.filter(pl.col("year").is_in(relevant_years))
                .group_by(group_cols, maintain_order=True)
                .agg(
                    # Tariff difference calculated between y1 and y2
                    (
                        pl.col(tariff_col_name)
                        .filter(pl.col("year") == y2)  # Average tariff in y2
                        .mean()
                        - pl.col(tariff_col_name)
                        .filter(pl.col("year") == y1)  # Average tariff in y1
                        .mean()
                    ).alias(tariff_col_name + "_change"),
                    # Tariff percentage change calculated between y1 and y2
                    (
                        (pl.col(tariff_col_name).filter(pl.col("year") == y2).mean() - pl.col(tariff_col_name).filter(pl.col("year") == y1).mean())
                        / pl.col(tariff_col_name).filter(pl.col("year") == y1).mean()
                        # * 100
                    ).alias(tariff_col_name + "_perc_change"),
                    # Unit value g
                    (
                        pl.col(price_col_name).filter(pl.col("year") == year_unit_value_end).mean()
                        - pl.col(price_col_name).filter(pl.col("year") == year_before_imposition).mean()
                    ).alias(price_col_name + "_change"),
                    # Unit value percentage change calculated between year_before_imposition and year_unit_value_end
                    (
                        (
                            (
                                pl.col(price_col_name).filter(pl.col("year") == year_unit_value_end).mean()
                                - pl.col(price_col_name).filter(pl.col("year") == year_before_imposition).mean()
                            )
                            / pl.col(price_col_name).filter(pl.col("year") == year_before_imposition).mean()
                        )
                        * 100
                    ).alias(price_col_name + "_perc_change"),  # price_col_name+"_perc_change"
                    # Value difference calculated between year_before_imposition and year_unit_value_end
                    (
                        pl.col("value").filter(pl.col("year") == year_unit_value_end).sum()
                        - pl.col("value").filter(pl.col("year") == year_before_imposition).sum()
                    ).alias("value_change"),
                    # Quantity difference calculated between year_before_imposition and year_unit_value_end
                    (
                        pl.col("quantity").filter(pl.col("year") == year_unit_value_end).sum()
                        - pl.col("quantity").filter(pl.col("year") == year_before_imposition).sum()
                    ).alias("quantity_change"),
                )
                .filter(
                    (pl.col(tariff_col_name + "_change").is_not_null())
                    & (pl.col(tariff_col_name + "_change").is_not_nan())
                    & (pl.col(tariff_col_name + "_change") != 0.0)
                    & (pl.col(price_col_name + "_change").is_not_null())
                )
                .with_columns(
                    pl.lit(y1 + "-" + y2).alias(tariff_col_name + "_change_range"),
                    pl.lit(year_before_imposition + "-" + year_unit_value_end).alias(price_col_name + "_change_range"),
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
    # Hack to avoid rendering in dashboard view
    _ = 42
    return


@app.cell
def _(argparse):
    parser = argparse.ArgumentParser(description="Marimo visualise unified")

    # Add your parameters/arguments here
    parser.add_argument(
        "--fullfat",
        action="store_true",
        help="Using this flag will run on all the data",
    )
    args = parser.parse_args()

    args.fullfat

    # HARDCODE
    args.fullfat = True
    return (args,)


@app.cell
def _(args):
    price_change_start = 0  # Number of years before the first year of the tariff change that we start measuring prices.
    price_change_end = 2  # Number of years after the last year of the tariff change that we stop measuring prices.
    tariff_change_gap = 1  # Gap between years that we're measuring the start of tariff changes

    reporter_countries = ["156"]  # Reporter countries to filter to [China]
    partner_countries = ["840"]  # Partner countries to filter to [USA]
    product_codes = None  # Products to filter to

    if args.fullfat:
        data_to_load = "data/final/unified_trade_tariff_partitioned/"
        print("Running across all the data")
    else:
        data_to_load = "data/final/unified_filtered_10000minval_top100countries/"
        print("Running across subset of the data")

    # tariff_col_name = 'effective_tariff_arcsinh'
    # price_col_name = 'unit_value_arcsinh'
    tariff_col_name = "average_tariff_official"
    price_col_name = "unit_value"
    return (
        data_to_load,
        partner_countries,
        price_change_end,
        price_change_start,
        price_col_name,
        product_codes,
        reporter_countries,
        tariff_change_gap,
        tariff_col_name,
    )


@app.cell
def _():
    # # --- Configuration Defaults ---
    # _DEFAULT_PRICE_CHANGE_START = 0
    # _DEFAULT_PRICE_CHANGE_END = 2
    # _DEFAULT_TARIFF_CHANGE_GAP = 1
    # _DEFAULT_REPORTER_COUNTRIES = ["156"]  # China
    # _DEFAULT_PARTNER_COUNTRIES = ["840"]  # USA
    # _DEFAULT_PRODUCT_CODES = None
    # _DEFAULT_FULLFAT = True
    # _DEFAULT_ARCSINH = False

    # # --- UI Elements ---
    # # (Defined exactly as before)
    # use_arcsinh = mo.ui.checkbox(value=_DEFAULT_ARCSINH, label=" ")
    # full_fat_data = mo.ui.checkbox(value=_DEFAULT_FULLFAT, label=" ")

    # price_change_start_input = mo.ui.number(
    #     start=-5,
    #     stop=5,
    #     step=1,
    #     value=_DEFAULT_PRICE_CHANGE_START,
    #     # label="Years before first tariff change:"
    # )
    # price_change_end_input = mo.ui.number(
    #     start=0,
    #     stop=10,
    #     step=1,
    #     value=_DEFAULT_PRICE_CHANGE_END,
    #     # label="Years after last tariff change:"
    # )
    # tariff_change_gap_input = mo.ui.number(
    #     start=0,
    #     stop=5,
    #     step=1,
    #     value=_DEFAULT_TARIFF_CHANGE_GAP,
    #     # label="Gap between tariff change years:"
    # )

    # reporter_countries_input = mo.ui.text(
    #     value=", ".join(_DEFAULT_REPORTER_COUNTRIES),
    #     # label="Reporter Countries (comma-separated):"
    # )
    # partner_countries_input = mo.ui.text(
    #     value=", ".join(_DEFAULT_PARTNER_COUNTRIES),
    #     # label="Partner Countries (comma-separated):"
    # )
    # product_codes_input = mo.ui.text(
    #     value="",  # Start empty for None default
    #     # label="Product Codes (comma-separated, blank for all):"
    # )

    # # --- Create and Display Form using .batch and .form ---
    # mo.md("""
    # ## Data Analysis Configuration

    # Configure the parameters for the trade and tariff analysis.
    # """)

    # config_form_output = (
    #     mo.md("""
    #         # CONFIG - Will not run dynamically in .html mode.
    #         **Data Selection:**

    #         - Use full dataset (uncheck for testing purposes): <br> {full_fat_data}
    #         - Reporter Countries (comma-separated, blank for all): <br> {reporter_countries_input}
    #         - Partner Countries (comma-separated, blank for all): <br> {partner_countries_input}
    #         - Product Codes (comma-separated, blank for all): <br> {product_codes_input}

    #         **Time Window:**

    #         - Years before first tariff change: <br> {price_change_start_input}
    #         - Years after last tariff change: <br> {price_change_end_input}
    #         - Gap between tariff change: <br> {tariff_change_gap_input}

    #         **Data Transformation:**

    #         - Use inverse hyperbolic sine transformation: <br> {use_arcsinh}
    # """)
    #     .batch(
    #         full_fat_data=full_fat_data,
    #         reporter_countries_input=reporter_countries_input,
    #         partner_countries_input=partner_countries_input,
    #         product_codes_input=product_codes_input,
    #         price_change_start_input=price_change_start_input,
    #         price_change_end_input=price_change_end_input,
    #         tariff_change_gap_input=tariff_change_gap_input,
    #         use_arcsinh=use_arcsinh,
    #     )
    #     .form(
    #         bordered=True,
    #     )
    # )

    # config_form_output
    return


@app.cell
def _():
    # price_change_start = config_form_output.value["price_change_start_input"]
    # price_change_end = config_form_output.value["price_change_end_input"]
    # tariff_change_gap = config_form_output.value["tariff_change_gap_input"]

    # # Helper to parse comma-separated strings into lists
    # def _parse_codes(text_value):
    #     codes = [code.strip() for code in text_value.split(",") if code.strip()]
    #     return codes if codes else None

    # reporter_countries = _parse_codes(
    #     config_form_output.value["reporter_countries_input"]
    # )
    # partner_countries = _parse_codes(
    #     config_form_output.value["partner_countries_input"]
    # )
    # product_codes = _parse_codes(config_form_output.value["product_codes_input"])

    # # Determine data path based on checkbox
    # if config_form_output.value["full_fat_data"]:
    #     data_to_load = "data/final/unified_trade_tariff_partitioned/"
    #     data_desc = "Full dataset"
    # else:
    #     data_to_load = "data/final/unified_filtered_10000minval_top100countries/"
    #     data_desc = "Subset of data"

    # # Determine column names based on checkbox
    # if config_form_output.value["use_arcsinh"]:
    #     tariff_col_name = "effective_tariff_arcsinh"
    #     price_col_name = "unit_value_arcsinh"
    #     transform_desc = "arcsinh transformed"
    # else:
    #     tariff_col_name = "average_tariff_official"
    #     price_col_name = "unit_value"
    #     transform_desc = "standard"

    # # --- Output Display ---

    # # Display the current configuration values derived from the UI
    # mo.md(f"""
    # ### Current Configuration:

    # * **Data Source:** `{data_desc}` (`{data_to_load}`)
    # * **Reporter Countries:** `{reporter_countries}`
    # * **Partner Countries:** `{partner_countries}`
    # * **Product Codes:** `{product_codes}`
    # * **Price Change Start Offset:** `{price_change_start}` years
    # * **Price Change End Offset:** `{price_change_end}` years
    # * **Tariff Change Gap:** `{tariff_change_gap}` years
    # * **Tariff Column:** `{tariff_col_name}` ({transform_desc})
    # * **Price Column:** `{price_col_name}` ({transform_desc})
    # """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Preliminary Results: Direct effects of tariffs on trade

    ## Recap:
    I've been looking into bilateral (between reporter [exporter] country **i** and partner [importer] country **j**) trade values and volumes for product **k**, denominated in the [Harmonised System](https://en.wikipedia.org/wiki/Harmonized_System) nomenclature. 

    The data is annual, running from 1995 to 2023, and is sourced from the [CEPII research institute](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37). I have merged this with a dataset of tariffs, performing some simple calculations to obtain the % effectivbe tariff for each **i-j-k** triple per year. The tariff data is sourced from [WITS](https://wits.worldbank.org/).

    It is worth noting that BACI reports trade as free-on-board (FOB) and removes the cost-insurance-freight (CIF) components. 

    See below a snapshot of the entire merged dataset
    """
    )
    return


@app.cell
def _(data_to_load, pl):
    unified_lf = pl.scan_parquet(data_to_load)

    print(unified_lf.head().collect())
    unified_lf.head(10).collect()  # If we're in notebook
    return (unified_lf,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Filtering
    For the purposes of this report, I have filtered the data to only contain reports from China (country code 156) and the USA (country code 840).

    This significantly reduces the size of the dataset. See below a snapshot of the filtered data.
    """
    )
    return


@app.cell
def _(
    functools,
    operator,
    partner_countries,
    pl,
    price_col_name,
    product_codes,
    remove_outliers_by_percentile_lazy,
    reporter_countries,
    tariff_col_name,
    unified_lf,
):
    filtered_lf = unified_lf
    if reporter_countries:
        filtered_lf = filtered_lf.filter(pl.col("reporter_country").is_in(reporter_countries))

    if partner_countries:
        filtered_lf = filtered_lf.filter(pl.col("partner_country").is_in(partner_countries))

    if product_codes:
        conditions = [pl.col("product_code").str.slice(0, len(p)) == p for p in product_codes]
        # Combine conditions with an OR
        combined_condition = functools.reduce(operator.or_, conditions)
        filtered_lf = filtered_lf.filter(combined_condition)

    filtered_lf = remove_outliers_by_percentile_lazy(
        ldf=filtered_lf,
        column_names=[
            tariff_col_name,
            price_col_name,
            price_col_name + "_detrended",
        ],
        lower_p=0.01,
        upper_p=0.99,
    )
    return (filtered_lf,)


@app.cell
def _(filtered_lf, pl):
    print("--- LDF post filtering: ---")
    print(f"   Post filtering: {filtered_lf.select(pl.len()).collect().item()} Rows left")

    filtered_lf_head = filtered_lf.head().collect()

    print(filtered_lf_head)  # If in cmdline
    filtered_lf_head  # If interactive
    return


@app.cell
def _(filtered_lf, mo, pl, unified_lf):
    pre_filter = mo.stat(
        value=unified_lf.select(pl.len()).collect().item(),
        label="Number of rows pre-filter",
    )

    post_filter = mo.stat(
        value=filtered_lf.select(pl.len()).collect().item(),
        label="Number of rows post-filter",
    )

    mo.hstack([pre_filter, post_filter], justify="center", gap="2rem")
    return


@app.cell
def _(
    extract_tariff_changes,
    filtered_lf,
    price_change_end,
    price_change_start,
    price_col_name,
    tariff_change_gap,
    tariff_col_name,
):
    # First we need to extract the rows we're interested in
    print("--- Extracting cases of change in tariff ---")

    tariff_changes_df = extract_tariff_changes(
        filtered_lf=filtered_lf,
        start_year="1999",
        end_year="2023",
        years_before_tariff_change_unit_value=price_change_start,
        year_unit_value_end_gap=price_change_end,
        year_gap=tariff_change_gap,
        tariff_col_name=tariff_col_name,
        price_col_name=price_col_name,
    )

    print(f"--- Identified {tariff_changes_df.height} cases of tariff change in sample ---")
    return (tariff_changes_df,)


@app.cell
def _(mo, tariff_changes_df):
    txt = mo.md(
        r"""
        ## Vis 1

        The first exercise is to look at the direct effect of tariff impositions on the bilateral trade price, meaning the total value / total volume traded in a year. 

        After filtering to contain only instances where the effective tariff rate changed between two consecutive years, below is a scatter plot of the change in effective tariff rate (t - t+1) against the change in unit price (t - t+2). 

        *To note: unit price is denominated in dimensioned units, such as $/kg, $/mm, etc. Depending on the product*

        See below a sample of the resulting dataframe.
        """
    )
    num_idents = mo.stat(value=tariff_changes_df.height, label="Number of tariff changes identified")

    mo.hstack([txt, num_idents], justify="start", gap="2rem")
    return


@app.cell
def _():
    # mo.md(rf"""We identify {tariff_changes_df.height} instances of tariff changes in the dataset. See a sample of the dataset below.""")
    return


@app.cell
def _(tariff_changes_df):
    print(tariff_changes_df.head())  # If we're working from cmdline
    tariff_changes_df.head()  # If we're in notebook mode
    return


@app.cell
def _(mo):
    mo.md(r"""And below, the chart""")
    return


@app.cell
def _(
    partner_countries,
    price_change_end,
    price_change_start,
    price_col_name,
    px,
    reporter_countries,
    tariff_change_gap,
    tariff_changes_df,
    tariff_col_name,
):
    # Draw our chart
    scatter_filtered = px.scatter(
        tariff_changes_df,
        x=tariff_col_name + "_change",
        # x="tariff_perc_change",
        # y="unit_value_perc_change",
        y=price_col_name + "_change",
        title=f"""
            t0 to t+{tariff_change_gap} abs change in tariff vs t+{price_change_start} to t+{price_change_end} year abs change in unit value between {partner_countries} (importer) & {reporter_countries} (exporter)
        """,
        hover_data=[
            tariff_col_name + "_change_range",
            price_col_name + "_change_range",
            "product_code",
        ],
        color=tariff_col_name + "_change_range",
        # trendline='ols'
    )

    scatter_filtered.write_html("charts/direct_effects/scatter_US-CHINA.html")
    scatter_filtered
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Vis 2
    Probably unsuprisingly, there appears to be little visible signal. To reduce at least *some* noise in this analysis, 
    the next thing I tried was to 'de-trend' the unit price for each *i-j-k* triple. To do so I calculate the trade-value weighted average **global** unit price for each product *k* for each year, and subtract that from each *i-j-k*'s price. Leaving the 'idiosyncratic' price movement for each bilateral trading relationship, for each product.

    Again, a sample of the data.
    """
    )
    return


@app.cell
def _(
    extract_tariff_changes,
    filtered_lf,
    price_change_end,
    price_change_start,
    price_col_name,
    tariff_change_gap,
    tariff_col_name,
):
    # First we need to extract the rows we're interested in
    print("--- Extracting cases of change in tariff ---")

    tariff_changes_df_detrend = extract_tariff_changes(
        filtered_lf=filtered_lf,
        start_year="1999",
        end_year="2023",
        years_before_tariff_change_unit_value=price_change_start,
        year_unit_value_end_gap=price_change_end,
        year_gap=tariff_change_gap,
        tariff_col_name=tariff_col_name,
        price_col_name=price_col_name + "_detrended",
    )

    print(f"--- Identified {tariff_changes_df_detrend.height} cases of tariff change in sample ---")
    return (tariff_changes_df_detrend,)


@app.cell
def _(tariff_changes_df_detrend):
    print(tariff_changes_df_detrend.head())  # If we're working from cmdline
    tariff_changes_df_detrend.tail()  # If we're in notebook mode
    return


@app.cell
def _(mo):
    mo.md(r"""And same chart as before, using the detrended prices""")
    return


@app.cell
def _(
    partner_countries,
    price_change_end,
    price_change_start,
    price_col_name,
    px,
    reporter_countries,
    tariff_change_gap,
    tariff_changes_df_detrend,
    tariff_col_name,
):
    # Draw our chart
    scatter_filtered_detrend = px.scatter(
        tariff_changes_df_detrend,
        # x="effective_tariff_change",
        x=tariff_col_name + "_change",
        y=price_col_name + "_detrended_change",
        # y="unit_value_detrended_change",
        title=f"Detrended t0 to t+{tariff_change_gap} year change in tariff vs t+{price_change_start} to t+{price_change_end} year change in unit value between {partner_countries} (importer) (USA) & {reporter_countries} (exporter) (CHINA)",
        hover_data=[
            tariff_col_name + "_change_range",
            price_col_name + "_detrended_change_range",
            "product_code",
        ],
        color=tariff_col_name + "_change_range",
        # trendline='ols'
    )

    scatter_filtered_detrend.write_html("charts/direct_effects/scatter_detrend_US-CHINA.html")
    scatter_filtered_detrend
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Vis 3

    This section contains some charts demonstrating the effect of de-trending the data. I have isolated this to a single product code, 845011. These are 'Fully Automatic Washing Machines'.
    """
    )
    return


@app.cell
def _(filtered_lf, pl):
    # Select a product. Plot the global trend against that series.
    unique_products = filtered_lf.select(pl.col("product_code")).unique().collect().sort("product_code")["product_code"]
    product_of_interest = [unique_products[100]]

    product_of_interest = ["845011"]  # AUTOMATIC WASHING MACHINES

    print("Inspecting product:", product_of_interest)
    return (product_of_interest,)


@app.cell
def _(mo):
    mo.md(r"""First, what the raw data looks like:""")
    return


@app.cell
def _(filtered_lf, pl, price_col_name, product_of_interest, tariff_col_name):
    product_df = filtered_lf.filter(pl.col("product_code").is_in(product_of_interest)).collect()

    # In case we've selected more than one reporter/partner_country/product_code
    product_df = (
        product_df.group_by(["year", "product_code"])
        .agg(
            pl.sum("quantity"),
            pl.sum("value"),
            pl.mean(tariff_col_name),
            pl.mean(price_col_name + "_detrended"),
            pl.mean(price_col_name),
            pl.mean("price_global_trend"),
            weighted_effective_tariff=(pl.col(tariff_col_name) * pl.col("value")).sum() / pl.col("value").sum(),
            weighted_unit_value_arcsinh=(pl.col(price_col_name) * pl.col("value")).sum() / pl.col("value").sum(),
            weighted_unit_value_arcsinh_detrended=(pl.col(price_col_name + "_detrended") * pl.col("value")).sum() / pl.col("value").sum(),
            weighted_unit_value=(pl.col("unit_value") * pl.col("value")).sum() / pl.col("value").sum(),
            weighted_unit_value_detrended=(pl.col("unit_value_detrended") * pl.col("value")).sum() / pl.col("value").sum(),
        )
        .sort("year")
    )

    print(
        "Grouped filtered_df and weighted trade volumes. In case N*i-j* > 1\n",
        product_df.head(),
    )
    product_df
    return (product_df,)


@app.cell
def _(mo):
    mo.md(text="First we visualise the global trend for the product, against the China-US prices")
    return


@app.cell
def _(product_df, product_of_interest, px):
    # Line chart 1: Product vs global trend
    global_trend_vs_series = px.line(
        product_df,
        x="year",
        y=["weighted_unit_value", "price_global_trend"],
        color="product_code",
        title="Global trend (dotted) vs bilateral price (China-US) (solid)",
    )

    # Hack to make the global trend appear as dotted
    global_trend_vs_series.for_each_trace(lambda trace: trace.update(line_dash="dot") if not trace.showlegend else ())

    global_trend_vs_series.write_html(f"charts/direct_effects/global_trend_vs_product_{product_of_interest}.html")

    global_trend_vs_series
    return


@app.cell
def _(mo):
    mo.md(text="And below, a view of the residual and resulting series.")
    return


@app.cell
def _(go, make_subplots, pl, product_df, product_of_interest, px):
    def _():
        # Get unique product codes and assign colors
        unique_product_codes = product_df["product_code"].unique().to_list()
        colors = px.colors.qualitative.Plotly

        # Create a figure with 2 rows
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Global Trend vs Product Price & Detrended Value (Bar)",
                "Weighted Unit Value Detrended (Line)",
            ),
            # row_heights=[0.7, 0.3]
        )

        # --- Chart 1: Product vs global trend (lines) and detrended (bars) ---
        for i, product in enumerate(unique_product_codes):
            product_specific_df = product_df.filter(pl.col("product_code") == product)
            color = colors[i % len(colors)]

            # Line for weighted_unit_value
            fig.add_trace(
                go.Scatter(
                    x=product_specific_df["year"].cast(pl.Int32),
                    y=product_specific_df["weighted_unit_value"],
                    name=f"{product} - Weighted Value",
                    legendgroup=f"group-{product}",
                    line=dict(color=color),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Line for product_global_trend (dotted)
            fig.add_trace(
                go.Scatter(
                    x=product_specific_df["year"].cast(pl.Int32),
                    y=product_specific_df["price_global_trend"],
                    name=f"{product} - Global Trend",
                    legendgroup=f"group-{product}",
                    line=dict(dash="dot", color=color),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Bar for weighted_unit_value_detrended
            fig.add_trace(
                go.Bar(
                    x=product_specific_df["year"].cast(pl.Int32),
                    y=product_specific_df["weighted_unit_value_detrended"],
                    name=f"{product} - Detrended (Bar)",
                    legendgroup=f"group-{product}",
                    marker_color=color,
                    opacity=0.6,  # Optional: make bars slightly transparent
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # --- Chart 2: Line of weighted_unit_value_detrended ---
        for i, product in enumerate(unique_product_codes):
            product_specific_df = product_df.filter(pl.col("product_code") == product)
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=product_specific_df["year"].cast(pl.Int32),
                    y=product_specific_df["weighted_unit_value_detrended"],
                    name=f"{product} - Detrended (Line)",
                    legendgroup=f"group-{product}-detrended",  # Use a different legend group if needed
                    line=dict(color=color),
                    showlegend=False,  # Avoid duplicate legend entries if names are similar
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Analysis of detrending effect on individual products",
            barmode="group",  # For the bars in the first subplot
        )
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Detrended Value", row=2, col=1)
        fig.write_html(f"charts/direct_effects/detrending_effect_{product_of_interest}.html")
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md(text="The below chart places this in the context of the global bilateral trade prices. The red line is the weighted average")
    return


@app.cell
def _(go, pl, product_of_interest, unified_lf):
    ### Line chart - the global lines for the countries

    num_top_ij = 75

    # First filter for the right year
    product_specific_lf = unified_lf.filter(pl.col("product_code").is_in(product_of_interest))

    # Calculate total value for each reporter-partner pair
    top_pairs_ranked_lf = (
        product_specific_lf.group_by(["reporter_country", "partner_country"])
        .agg(total_value_for_pair=pl.sum("value"))
        .sort("total_value_for_pair", descending=True)
        .head(num_top_ij)
    )

    top_pairs_keys_lf = top_pairs_ranked_lf.select(["reporter_country", "partner_country"])

    # Use an inner join to filter the data
    top_pairs_data_lf = (
        product_specific_lf.join(
            top_pairs_keys_lf,
            on=["reporter_country", "partner_country"],
            how="inner",
        )
        .group_by(["year", "reporter_country", "partner_country"])
        .agg(
            # unit_value = pl.sum('value') / pl.sum('quantity')
            pl.mean("unit_value")
        )
        .sort("year")
    )

    top_pairs_data_df = top_pairs_data_lf.collect()
    product_specific_df = (
        product_specific_lf.select(["price_global_trend", "year"])
        .group_by("year")
        .agg(
            pl.mean("price_global_trend")  # Should only be one value anyway
        )
        .sort("year")
        .collect()
    )

    global_trend_vs_individual_ijk = go.Figure()

    unique_pairs_to_plot = top_pairs_data_df.select(["reporter_country", "partner_country"]).unique()

    for pair_row in unique_pairs_to_plot.iter_rows(named=True):
        reporter = pair_row["reporter_country"]
        partner = pair_row["partner_country"]

        pair_specific_data = top_pairs_data_df.filter((pl.col("reporter_country") == reporter) & (pl.col("partner_country") == partner)).sort("year")

        global_trend_vs_individual_ijk.add_trace(
            go.Scatter(
                x=pair_specific_data["year"].cast(pl.Int32),
                y=pair_specific_data["unit_value"],
                mode="lines",
                line=dict(color="rgba(200, 200, 200, 0.4)"),
                showlegend=False,
                hoverinfo="skip",
                # name=f'{reporter}-{partner}',
                # hovertemplate='Year: %{x}<br>Unit Value: %{y:.2f}<extra></extra>'
            )
        )

    global_trend_vs_individual_ijk.add_trace(
        go.Scatter(
            x=product_specific_df["year"].cast(pl.Int32),
            y=product_specific_df["price_global_trend"],
            mode="lines",
            line=dict(color="red", width=2.5),
            name="Global Trend",
        )
    )

    title_product_str = product_of_interest[0] if product_of_interest and len(product_of_interest) == 1 else "Selected Product"
    global_trend_vs_individual_ijk.update_layout(
        title=f"Top {num_top_ij} Importer-Exporter Pair Trends vs. Global Trend for Product: {title_product_str}",
        xaxis_title="Year",
        yaxis_title="Unit Value",
        showlegend=False,  # To show the legend for the 'Global Trend' line
    )

    global_trend_vs_individual_ijk.write_html(f"charts/direct_effects/global_trend_overlayed_{product_of_interest}.html")
    global_trend_vs_individual_ijk
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Vis 4

    Turning to those scatters, there is little visible correlation. A priori I would've expected any signal to be small, which is why I find the approach of looking at >250 million **i-j-k** triples (as compared to specific case studies) interesting. When the sample is large, a small measured impact may still be significant. 

    In that spirit, the next visualisation is a histogram of all the 'effective passthroughs after N years' of tariff change to bilateral trade price. Where effective passthrough is the (change in tariff) ÷ (change in price).

    Below the summary frame are variations on this same theme.
    """
    )
    return


@app.cell
def _(tariff_changes_df_detrend):
    tariff_changes_df_detrend.describe()
    return


@app.cell
def _(pl, price_col_name, tariff_changes_df_detrend, tariff_col_name):
    effective_passthroughs_df = tariff_changes_df_detrend.with_columns(
        (pl.col(price_col_name + "_detrended_change") / pl.col(tariff_col_name + "_change")).alias("effective_passthrough")
    ).select("effective_passthrough")

    # Calculate the mean of the finite values
    mean_val = effective_passthroughs_df["effective_passthrough"].mean()
    median_val = effective_passthroughs_df["effective_passthrough"].median()
    return effective_passthroughs_df, mean_val, median_val


@app.cell
def _(effective_passthroughs_df, mean_val, median_val, px):
    ""  # 1. Create a histogram with lines for the mean and median

    fig_hist_lines = px.histogram(
        effective_passthroughs_df,
        x="effective_passthrough",
        title="Histogram of Effective Passthrough with Mean and Median Lines",
    )

    fig_hist_lines.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
    )

    fig_hist_lines.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="bottom right",
    )

    print("Showing histogram with mean and median lines:")
    # fig_hist_lines

    # 2. Create a histogram with an overlaid box plot
    # Plotly Express can create a histogram with a marginal box plot.
    fig_hist_box = px.histogram(
        effective_passthroughs_df,
        x="effective_passthrough",
        marginal="box",  # Add a box plot on the margin
        title="Histogram of Effective Passthrough with Overlaid Box Plot",
    )

    print("Showing histogram with an overlaid box plot:")
    fig_hist_box.write_html("charts/direct_effects/effective_passthrough_hist.html")
    fig_hist_box
    return (fig_hist_lines,)


@app.cell
def _(fig_hist_lines):
    fig_hist_lines
    return


@app.cell
def _(effective_passthroughs_df, mean_val, median_val, px):
    # 1. Create a histogram with lines for the mean and median, with log y-axis
    fig_hist_lines_log_y = px.histogram(
        effective_passthroughs_df,
        x="effective_passthrough",
        title="Histogram of Effective Passthrough (Log Y-axis) with Mean and Median Lines",
        log_y=True,  # Set y-axis to logarithmic scale
    )

    fig_hist_lines_log_y.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
    )

    fig_hist_lines_log_y.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="bottom right",
    )

    print("Showing histogram with mean/median lines and log y-axis:")
    fig_hist_lines_log_y.write_html("charts/direct_effects/effective_passthrough_hist_logY.html")
    fig_hist_lines_log_y
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Vis 5

    Visualisation 5 shows the response of prices to a tariff impulse. The price change lines have been averaged into quantiles according to the size / direction of their corresponding tariff change.
    """
    )
    return


@app.cell
def _(mo, pl):
    def extract_tariff_changes_impulse(
        filtered_lf: pl.LazyFrame,
        start_year: str,
        end_year: str,
        year_gap: int,
        series_length: int,
        tariff_col_name: str,
        price_col_name: str,
        years_before_tariff_change: int = 0,
    ):
        """
        Identifies tariff changes for product lines and extracts a time series of prices
        around the time of each identified change.

        Args:
            filtered_lf: LazyFrame with trade data. Expected columns: 'reporter_country',
                         'partner_country', 'product_code', 'year', tariff_col_name, price_col_name.
                         The 'year' column in filtered_lf is assumed to be of string type.
            start_year: Start year for detecting tariff changes (e.g., "2000").
            end_year: End year for detecting tariff changes (e.g., "2023").
            year_gap: Gap in years to identify tariff change (compares year Y with Y + year_gap).
            series_length: Number of years for the price series, starting from the impulse year (y1).
                           For example, if series_length is 3, data for y1, y1+1, y1+2 are included.
                           A series_length of 1 includes only data for the impulse year y1.
            tariff_col_name: Name of the column containing tariff information.
            price_col_name: Name of the column containing price information.
            years_before_tariff_change: Number of years of price data to include *before* the
                                        impulse year (y1). Relative time 0 is the impulse year.

        Returns:
            A Polars DataFrame with price time series for each product line following a tariff change.
            Columns include the group_cols ('reporter_country', 'partner_country', 'product_code'),
            'impulse_year', 'tariff_change_at_impulse', 'year' (actual year of price data),
            the price_col_name, and 'time_relative_to_impulse'.
            Returns an empty DataFrame with the correct schema if no tariff changes are found
            or if no relevant price series data is available.
        """
        group_cols = ["reporter_country", "partner_country", "product_code"]

        # --- Define Schema for Empty Result ---
        # Infer dtypes from input filtered_lf.schema or use sensible defaults.
        price_dtype = filtered_lf.schema.get(price_col_name, pl.Float64)
        group_cols_dtypes = {}
        for gc in group_cols:
            dtype = filtered_lf.schema.get(gc)
            if dtype is None:  # Fallback if column somehow not in schema (though operations would fail earlier)
                dtype = pl.Utf8
            group_cols_dtypes[gc] = dtype

        final_output_schema = {
            **group_cols_dtypes,
            "impulse_year": pl.Int32,
            "tariff_change_at_impulse": pl.Float64,
            "year": pl.Int32,
            price_col_name: price_dtype,
            "time_relative_to_impulse": pl.Int32,
        }

        # --- Part 1: Identify all impulse events (tariff changes) ---
        # y1 is the first year in the pair (y1, y1+year_gap) being compared for tariff change.
        # y1 becomes the 'impulse_year'.
        change_detection_year_pairs = [(str(y1_int), str(y1_int + year_gap)) for y1_int in range(int(start_year), int(end_year) + 1 - year_gap)]

        if not change_detection_year_pairs:
            return pl.DataFrame(schema=final_output_schema)

        list_of_impulse_event_lfs = []
        for y1_str, y2_str in mo.status.progress_bar(change_detection_year_pairs, title="Identifying tariff change events"):
            y1_val = int(y1_str)  # This is the impulse_year

            current_event_lf = (
                filtered_lf.filter(pl.col("year").is_in([y1_str, y2_str]))  # Assumes 'year' in filtered_lf is string
                .group_by(group_cols, maintain_order=True)
                .agg(
                    pl.col(tariff_col_name).filter(pl.col("year") == y1_str).mean().alias("tariff_y1"),
                    pl.col(tariff_col_name).filter(pl.col("year") == y2_str).mean().alias("tariff_y2"),
                )
                .with_columns((pl.col("tariff_y2") - pl.col("tariff_y1")).alias("tariff_change_value"))
                .filter(
                    pl.col("tariff_change_value").is_not_null() & (pl.col("tariff_change_value") != 0.0) & pl.col("tariff_change_value").is_not_nan()
                )
                .select(group_cols + ["tariff_change_value"])
                .with_columns(pl.lit(y1_val).cast(pl.Int32).alias("impulse_year"))
            )
            list_of_impulse_event_lfs.append(current_event_lf)

        # If list_of_impulse_event_lfs is empty (e.g. if change_detection_year_pairs was empty initially, though checked)
        # or if all LFs within are empty (no tariff changes found at all).
        if not list_of_impulse_event_lfs:  # This specific check might be redundant given earlier check.
            return pl.DataFrame(schema=final_output_schema)

        all_impulse_events_lf = pl.concat(list_of_impulse_event_lfs, how="diagonal_relaxed")

        # --- Part 2: Join with original data to extract price series ---
        # Create a version of filtered_lf with 'year' cast to Int32 for numeric operations.
        data_with_int_year_lf = filtered_lf.with_columns(pl.col("year").cast(pl.Int32).alias("year_as_int"))

        # Join identified impulse events with the main data.
        # An empty all_impulse_events_lf will result in an empty series_lf.
        series_lf = data_with_int_year_lf.join(all_impulse_events_lf, on=group_cols, how="inner")

        # Define the window for the time series relative to each impulse_year.
        # time_relative_to_impulse = 0 is the impulse_year itself.
        # Series includes data points from -years_before_tariff_change up to series_length-1.
        min_relative_offset = -years_before_tariff_change
        # If series_length is 1, max_offset is 0 (only the impulse_year data point).
        max_relative_offset = series_length - 1

        # If the calculated window is invalid (e.g., series_length <= 0), no data points are requested.
        if min_relative_offset > max_relative_offset:
            return pl.DataFrame(schema=final_output_schema)

        series_lf = series_lf.with_columns((pl.col("year_as_int") - pl.col("impulse_year")).alias("time_relative_to_impulse")).filter(
            pl.col("time_relative_to_impulse").is_between(min_relative_offset, max_relative_offset, closed="both")
        )

        # Select and arrange final columns for the output.
        final_selection_cols = group_cols + [
            "impulse_year",
            pl.col("tariff_change_value").alias("tariff_change_at_impulse"),
            pl.col("year_as_int").alias("year"),  # The actual year of the price data point.
            price_col_name,
            "time_relative_to_impulse",
            "value",
        ]

        result_lf = series_lf.select(final_selection_cols).sort(group_cols + ["impulse_year", "time_relative_to_impulse"])

        result_df = result_lf.collect()

        # If, after all operations, the resulting DataFrame is empty, return one with the defined schema.
        if result_df.is_empty():
            return pl.DataFrame(schema=final_output_schema)

        return result_df

    return


@app.cell
def _(mo, pl):
    def extract_tariff_changes_impulse_diff(  # Renamed slightly for clarity
        filtered_lf: pl.LazyFrame,
        start_year: str,
        end_year: str,
        year_gap: int,
        series_length: int,
        tariff_col_name: str,
        price_col_name: str,
        years_before_tariff_change: int = 0,
    ):
        """
        Identifies tariff changes and extracts the *difference* in price relative
        to the price at the impulse time (t=0) for each time point in the series.

        Args:
            filtered_lf: LazyFrame with trade data. Expected columns: 'reporter_country',
                         'partner_country', 'product_code', 'year', tariff_col_name,
                         price_col_name, and 'value'.
                         The 'year' column is assumed to be string type.
            start_year: Start year for detecting tariff changes (e.g., "2000").
            end_year: End year for detecting tariff changes (e.g., "2023").
            year_gap: Gap in years to identify tariff change (y1 vs y1 + year_gap).
            series_length: Number of years for the price series, starting from the
                           impulse year (y1). e.g., series_length=3 includes data
                           relative times 0, 1, 2 if years_before_tariff_change=0.
            tariff_col_name: Name of the column containing tariff information.
            price_col_name: Name of the column containing price information.
            years_before_tariff_change: Number of years of price data to include *before*
                                        the impulse year (y1). Relative time 0 is
                                        the impulse year (y1).

        Returns:
            A Polars DataFrame containing the time series of price *differences*
            (price[t] - price[t=0]) for each product line following a tariff change.
            Columns include group_cols, 'impulse_year', 'tariff_change_at_impulse',
            'year', 'time_relative_to_impulse', 'value', and a new column
            '{price_col_name}_diff_from_t0'. Only includes series where the price
            at t=0 is available. Returns an empty DataFrame with the correct schema
            if no valid series are found.
        """
        group_cols = ["reporter_country", "partner_country", "product_code"]
        # Define the name for the new difference column
        price_diff_col_name = f"{price_col_name}_diff_from_t0"

        # --- Define Schema for Empty Result ---
        price_dtype = filtered_lf.schema.get(price_col_name, pl.Float64)
        value_dtype = filtered_lf.schema.get("value", pl.Float64)  # Assuming 'value' column exists
        group_cols_dtypes = {}
        for gc in group_cols:
            dtype = filtered_lf.schema.get(gc)
            if dtype is None:
                dtype = pl.Utf8
            group_cols_dtypes[gc] = dtype

        # Schema now includes the difference column instead of the original price
        final_output_schema = {
            **group_cols_dtypes,
            "impulse_year": pl.Int32,
            "tariff_change_at_impulse": pl.Float64,
            "year": pl.Int32,
            price_diff_col_name: price_dtype,  # Difference likely has same dtype
            "time_relative_to_impulse": pl.Int32,
            "value": value_dtype,  # Include value if it's part of output
        }

        # --- Part 1: Identify all impulse events (tariff changes) ---
        change_detection_year_pairs = [(str(y1_int), str(y1_int + year_gap)) for y1_int in range(int(start_year), int(end_year) + 1 - year_gap)]

        if not change_detection_year_pairs:
            return pl.DataFrame(schema=final_output_schema)

        list_of_impulse_event_lfs = []
        for y1_str, y2_str in mo.status.progress_bar(change_detection_year_pairs, title="Identifying tariff change events"):
            y1_val = int(y1_str)

            current_event_lf = (
                filtered_lf.filter(pl.col("year").is_in([y1_str, y2_str]))
                .group_by(group_cols, maintain_order=True)
                .agg(
                    pl.col(tariff_col_name).filter(pl.col("year") == y1_str).mean().alias("tariff_y1"),
                    pl.col(tariff_col_name).filter(pl.col("year") == y2_str).mean().alias("tariff_y2"),
                )
                .with_columns((pl.col("tariff_y2") - pl.col("tariff_y1")).alias("tariff_change_value"))
                .filter(
                    pl.col("tariff_change_value").is_not_null() & (pl.col("tariff_change_value") != 0.0) & pl.col("tariff_change_value").is_not_nan()
                )
                .select(group_cols + ["tariff_change_value"])
                .with_columns(pl.lit(y1_val).cast(pl.Int32).alias("impulse_year"))
            )
            list_of_impulse_event_lfs.append(current_event_lf)

        if not list_of_impulse_event_lfs:
            return pl.DataFrame(schema=final_output_schema)

        all_impulse_events_lf = pl.concat(list_of_impulse_event_lfs, how="diagonal_relaxed")

        # --- Part 2: Join, extract series, and calculate price difference ---
        data_with_int_year_lf = filtered_lf.with_columns(pl.col("year").cast(pl.Int32).alias("year_as_int"))

        series_lf = data_with_int_year_lf.join(all_impulse_events_lf, on=group_cols, how="inner")

        min_relative_offset = -years_before_tariff_change
        max_relative_offset = series_length - 1

        if min_relative_offset > max_relative_offset:
            return pl.DataFrame(schema=final_output_schema)

        # Calculate relative time and filter time window
        series_lf = series_lf.with_columns((pl.col("year_as_int") - pl.col("impulse_year")).alias("time_relative_to_impulse")).filter(
            pl.col("time_relative_to_impulse").is_between(min_relative_offset, max_relative_offset, closed="both")
        )

        # --- Calculate Difference from t=0 ---
        event_id_cols = group_cols + ["impulse_year"]

        # Add price at t=0 ('price_t0') column using a window function
        # This calculates price_t0 for each event series
        series_lf = series_lf.with_columns(
            pl.col(price_col_name)
            .filter(pl.col("time_relative_to_impulse") == 0)
            .first()  # Get the price where relative time is exactly 0
            .over(event_id_cols)  # Broadcast it over the entire series group
            .alias("price_t0")
        )

        # Filter out series where price_t0 could not be determined (i.e., no data at t=0)
        series_lf = series_lf.filter(pl.col("price_t0").is_not_null())

        # Calculate the difference: price[t] - price[t=0]
        series_lf = series_lf.with_columns((pl.col(price_col_name) - pl.col("price_t0")).alias(price_diff_col_name))

        # --- Select Final Columns ---
        # Include the new difference column, exclude the original price and temporary price_t0
        final_selection_cols = group_cols + [
            "impulse_year",
            pl.col("tariff_change_value").alias("tariff_change_at_impulse"),
            pl.col("year_as_int").alias("year"),
            price_diff_col_name,  # The calculated difference column
            "time_relative_to_impulse",
            "value",  # Include 'value' column as per original select list
        ]

        result_lf = series_lf.select(final_selection_cols).sort(group_cols + ["impulse_year", "time_relative_to_impulse"])

        result_df = result_lf.collect()

        if result_df.is_empty():
            # Return empty df with the *correct updated schema*
            return pl.DataFrame(schema=final_output_schema)

        return result_df

    return (extract_tariff_changes_impulse_diff,)


@app.cell
def _(
    extract_tariff_changes_impulse_diff,
    filtered_lf,
    price_col_name,
    tariff_col_name,
):
    tariff_price_impulse_df = extract_tariff_changes_impulse_diff(
        filtered_lf=filtered_lf,
        start_year="1999",
        end_year="2023",
        year_gap=1,
        series_length=4,
        tariff_col_name=tariff_col_name,
        price_col_name=price_col_name + "_detrended",
        years_before_tariff_change=0,
    )
    return (tariff_price_impulse_df,)


@app.cell
def _(tariff_price_impulse_df):
    tariff_price_impulse_df.head()
    return


@app.cell(hide_code=True)
def _():
    # import math # For adjusting opacity heuristic

    # def plot_rebased_price_evolution_plotly(
    #     impulse_df: pl.DataFrame,
    #     price_col_name: str,
    #     group_cols: list = ["reporter_country", "partner_country", "product_code"],
    #     max_series_to_plot: int | None = None,
    #     value_col_name: str = "value" # Column to determine trade volume for ranking
    # ):
    #     """
    #     Prepares data and plots the rebased price evolution after tariff changes using Plotly.
    #     Optionally plots only the top N series based on total trade volume.

    #     Args:
    #         impulse_df: Polars DataFrame potentially from extract_tariff_changes_impulse.
    #                     Expected columns include group_cols, 'impulse_year',
    #                     'time_relative_to_impulse', the price_col_name, and
    #                     (if max_series_to_plot is used) the value_col_name.
    #         price_col_name: Name of the column containing the price information.
    #         group_cols: List of columns defining a unique product line (excluding impulse_year).
    #         max_series_to_plot: If set to a positive integer N, only the top N series
    #                             ranked by total trade volume (sum of value_col_name over
    #                             the series' time points) will be plotted. If None or <= 0,
    #                             all available series passing filters are plotted.
    #         value_col_name: Name of the column containing trade values, used for ranking
    #                         series when max_series_to_plot is active. Defaults to "value".

    #     Returns:
    #         plotly.graph_objects.Figure: The generated Plotly figure object.
    #                                      Returns an empty figure if no data can be plotted.
    #     """
    #     # --- 1. Input Validation and Initial Checks ---
    #     func_name = "plot_rebased_price_evolution_plotly" # For messages
    #     if isinstance(impulse_df, pl.LazyFrame):
    #         # Ensure we have a DataFrame to work with
    #         impulse_df = impulse_df.collect()

    #     if impulse_df.is_empty():
    #         print(f"[{func_name}] Error: Input DataFrame is empty.")
    #         return go.Figure().update_layout(title_text="No Input Data", xaxis={"visible": False}, yaxis={"visible": False})

    #     # Check for essential columns needed always
    #     essential_cols = group_cols + ["impulse_year", "time_relative_to_impulse", price_col_name]
    #     missing_essential = [col for col in essential_cols if col not in impulse_df.columns]
    #     if missing_essential:
    #         print(f"[{func_name}] Error: Input DataFrame is missing essential columns: {missing_essential}")
    #         return go.Figure().update_layout(title_text=f"Missing Columns: {missing_essential}", xaxis={"visible": False}, yaxis={"visible": False})

    #     # Check for value column if filtering by volume is requested
    #     if max_series_to_plot is not None and max_series_to_plot > 0:
    #         if value_col_name not in impulse_df.columns:
    #             print(f"[{func_name}] Error: '{value_col_name}' column needed for top N filtering is missing.")
    #             return go.Figure().update_layout(title_text=f"Missing Column: {value_col_name}", xaxis={"visible": False}, yaxis={"visible": False})

    #     plot_df = impulse_df # Start with the original DF
    #     plot_title_suffix = ""
    #     event_id_cols = group_cols + ["impulse_year"] # Define once

    #     # Estimate total available series *before* any filtering
    #     # Use select + distinct + height for potentially better performance than n_unique
    #     num_series_available = plot_df.select(event_id_cols).n_unique()

    #     # --- 2. Filter Top N Series if requested ---
    #     apply_top_n_filter = (
    #         max_series_to_plot is not None and
    #         max_series_to_plot > 0 and
    #         max_series_to_plot < num_series_available # Only filter if N is smaller than total
    #     )

    #     if apply_top_n_filter:
    #         print(f"[{func_name}] Selecting top {max_series_to_plot} series out of {num_series_available} based on total volume ('{value_col_name}')...")

    #         # Calculate total volume per series and rank
    #         series_volumes = (
    #             plot_df
    #             .group_by(event_id_cols)
    #             .agg(
    #                 pl.sum(value_col_name).fill_null(0).alias("total_volume") # Handle potential nulls in value
    #             )
    #             .sort("total_volume", descending=True)
    #             .limit(max_series_to_plot)
    #         )

    #         # Keep only the identifiers of the top N
    #         top_n_identifiers = series_volumes.select(event_id_cols)

    #         # Filter the main DataFrame
    #         plot_df = plot_df.join(
    #             top_n_identifiers,
    #             on=event_id_cols,
    #             how="inner"
    #         )

    #         num_series_after_filter = plot_df.select(event_id_cols).n_unique()
    #         if num_series_after_filter == 0: # Check if filtering removed everything
    #              print(f"[{func_name}] Warning: After filtering for top {max_series_to_plot} series, no data remained. Cannot plot.")
    #              return go.Figure().update_layout(title_text=f"No data for Top {max_series_to_plot} series", xaxis={"visible": False}, yaxis={"visible": False})

    #         # Adjust suffix only if filtering actually reduced the number of series significantly (or just always add it)
    #         plot_title_suffix = f" (Top {num_series_after_filter} by Volume)"
    #         print(f"[{func_name}] Filtered down to {num_series_after_filter} series for plotting.")

    #     # Create unique key for Plotly (on the final df to be plotted)
    #     str_event_id_exprs = [pl.col(c).cast(pl.Utf8) for c in event_id_cols]
    #     plot_df = plot_df.with_columns(
    #         pl.concat_str(str_event_id_exprs, separator="|").alias("event_key")
    #     )

    #     # # --- 3. Transform to make vis easier ---
    #     # plot_df = plot_df.with_columns(
    #     #     pl.col(price_col_name).arcsinh()
    #     # )

    #     # # --- 4. Plotting with Plotly Express ---
    #     num_series_plotted = plot_df.select(event_id_cols).n_unique()
    #     print(f"[{func_name}] Plotting {num_series_plotted} individual rebased price series using Plotly...")

    #     # Determine opacity: slightly higher if fewer lines are plotted
    #     opacity = min(0.7, max(0.05, 0.7 - 0.15 * math.log10(max(1, num_series_plotted)))) # Adjusted heuristic

    #     # Define columns for hover tooltip (check if value col exists in final df)
    #     hover_data_cols = ["event_key"] + group_cols + ["impulse_year", price_col_name]
    #     value_col_present_in_final_df = value_col_name in plot_df.columns
    #     if value_col_present_in_final_df:
    #         hover_data_cols.append(value_col_name)

    #     fig = px.line(
    #         plot_df,
    #         x="time_relative_to_impulse",
    #         y=price_col_name,
    #         line_group="event_key", # Ensures separate lines
    #         custom_data=hover_data_cols
    #     )

    #     fig.update_traces(
    #         line=dict(width=0.8), # Slightly thicker lines maybe
    #         opacity=opacity
    #     )

    #     # Update layout properties
    #     title = f"Price Evolution After Tariff Change{plot_title_suffix}<br><sup>{price_col_name.replace('_', ' ').title()}</sup>"
    #     xaxis_title = "Years Relative to Tariff Change (t=0 is change year)"
    #     yaxis_title = f"Rebased {price_col_name.replace('_', ' ').title()} " #(Index, t=0 = 100)

    #     fig.update_layout(
    #         title_text=title,
    #         xaxis_title=xaxis_title,
    #         yaxis_title=yaxis_title,
    #         showlegend=False,
    #         # hovermode="x unified",
    #         hovermode='closest',
    #         # dragmode='pan'
    #     )

    #     # Add reference lines
    #     # fig.add_hline(y=100, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")
    #     fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="t=0 (Change Year)", annotation_position="top right")

    #     # Construct hovertemplate based on available data
    #     group_cols_hover = "<br>".join([f"{col.replace('_', ' ').title()}: %{{customdata[{i+1}]}}" for i, col in enumerate(group_cols)])
    #     impulse_year_index = len(group_cols) + 1
    #     original_price_index = len(group_cols) + 2

    #     value_hover_line = ""
    #     if value_col_present_in_final_df:
    #         value_index = len(group_cols) + 3
    #         # Use G for general numeric format, ~ for approx if large number, .0f for integer-like display
    #         value_hover_line = f"{value_col_name.replace('_', ' ').title()}: %{{customdata[{value_index}]:,.0f}}<br>"

    #     hovertemplate = (
    #         f"<b>Event Details</b><br>"
    #         f"{group_cols_hover}<br>"
    #         f"Impulse Year: %{{customdata[{impulse_year_index}]}}<br>"
    #         f"<br>"
    #         f"Time Relative: %{{x}}<br>"
    #         f"Rebased Price: %{{y:.2f}}<br>"
    #         f"Original {price_col_name.replace('_', ' ')}: %{{customdata[{original_price_index}]:.2f}}<br>"
    #         f"{value_hover_line}"
    #         "<extra></extra>" # Hides the trace info box
    #     )
    #     fig.update_traces(hovertemplate=hovertemplate)

    #     return fig
    return


@app.cell(hide_code=True)
def _():
    # fig_impulse = plot_rebased_price_evolution_plotly(
    #     impulse_df=tariff_price_impulse_df,
    #     price_col_name=price_col_name+"_detrended_diff_from_t0",
    #     value_col_name='value',
    #     max_series_to_plot=50,
    # )

    # fig_impulse.show()
    return


@app.cell
def _(go, pl, px):
    def plot_avg_price_difference_by_tariff_bin_plotly(
        results_diff_df: pl.DataFrame,
        price_diff_col_name: str,
        tariff_change_col_name: str = "tariff_change_at_impulse",
        num_bins: int = 5,
        group_cols: list = ["reporter_country", "partner_country", "product_code"],
        also_plot_bin_stats: bool = True,  # Control whether the second plot is generated
    ):
        """
        Plots the average evolution of price differences grouped by bins
        (quantiles) of the tariff change. Includes tariff change stats in hover info
        and optionally plots these stats separately.

        Args:
            results_diff_df: Polars DataFrame from extract_tariff_changes_impulse_diff.
            price_diff_col_name: Name of the column holding price[t] - price[t=0].
            tariff_change_col_name: Name of the column holding the tariff change value.
            num_bins: Number of quantile-based bins for tariff changes.
            group_cols: List of columns defining a unique product line.
            also_plot_bin_stats: If True, generates a second plot showing tariff
                                 change statistics per bin.

        Returns:
            plotly.graph_objects.Figure: The main Plotly figure showing average trends.
                                         Returns an empty figure if no data can be plotted.
                                         Also shows the second plot if requested.
        """
        func_name = "plot_avg_price_difference_by_tariff_bin_plotly"
        # --- Input Validation and Initial Checks ---
        if isinstance(results_diff_df, pl.LazyFrame):
            results_diff_df = results_diff_df.collect()

        if results_diff_df.is_empty():
            print(f"[{func_name}] Error: Input DataFrame is empty.")
            return go.Figure().update_layout(
                title_text="No Input Data",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )

        essential_cols = group_cols + [
            "impulse_year",
            "time_relative_to_impulse",
            price_diff_col_name,
            tariff_change_col_name,
        ]
        missing_essential = [col for col in essential_cols if col not in results_diff_df.columns]
        if missing_essential:
            print(f"[{func_name}] Error: Input DataFrame is missing essential columns: {missing_essential}")
            return go.Figure().update_layout(
                title_text=f"Missing Columns: {missing_essential}",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
        # --- End Validation ---

        # --- Bin Tariff Changes & Calculate Stats ---
        event_id_cols = group_cols + ["impulse_year"]
        tariff_changes_per_event = results_diff_df.select(event_id_cols + [tariff_change_col_name]).unique(subset=event_id_cols, keep="first")

        # Adjust num_bins if fewer unique events exist
        if tariff_changes_per_event.height < num_bins:
            print(
                f"[{func_name}] Warning: Fewer unique events ({tariff_changes_per_event.height}) than requested bins ({num_bins}). Reducing bin count."
            )
            num_bins = max(1, tariff_changes_per_event.height)

        bin_col_name = "tariff_change_bin"  # Define bin column name
        bin_labels = [f"Q{i + 1}" for i in range(num_bins)]  # Default labels

        try:
            binned_events = tariff_changes_per_event.with_columns(
                pl.col(tariff_change_col_name).qcut(num_bins, labels=bin_labels).alias(bin_col_name)
            )
        except Exception as e:
            print(f"[{func_name}] Error binning tariff changes with qcut: {e}. Using pos/neg/zero bins.")
            binned_events = tariff_changes_per_event.with_columns(
                pl.when(pl.col(tariff_change_col_name) < 0)
                .then(pl.lit("Decrease"))
                .when(pl.col(tariff_change_col_name) > 0)
                .then(pl.lit("Increase"))
                .otherwise(pl.lit("No Change"))
                .alias(bin_col_name)
            )
            bin_labels = sorted(binned_events[bin_col_name].unique().to_list())  # Update labels based on fallback

        # Calculate Tariff Stats per Bin using the binned data
        print(f"[{func_name}] Calculating tariff change statistics per bin...")
        bin_tariff_stats_df = binned_events.group_by(bin_col_name).agg(
            pl.mean(tariff_change_col_name).alias("mean_tariff_change_in_bin"),
            pl.median(tariff_change_col_name).alias("median_tariff_change_in_bin"),
            pl.std(tariff_change_col_name).alias("std_tariff_change_in_bin").fill_null(0),
        )

        # --- Join bin info back and Aggregate Price Diff Data ---
        plot_df = results_diff_df.join(
            # Select only needed columns from binned_events to avoid duplicate tariff_change_col_name
            binned_events.select(event_id_cols + [bin_col_name]),
            on=event_id_cols,
            how="inner",
        )

        if plot_df.is_empty():
            print(f"[{func_name}] No data remaining after joining tariff change bins. Cannot plot.")
            return go.Figure().update_layout(
                title_text="No Data After Binning",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )

        print(f"[{func_name}] Aggregating price differences by time and tariff change bin...")
        agg_df = (
            plot_df.group_by(["time_relative_to_impulse", bin_col_name])
            .agg(
                pl.mean(price_diff_col_name).alias("mean_price_diff"),
                pl.median(price_diff_col_name).alias("median_price_diff"),
                pl.std(price_diff_col_name).alias("std_price_diff").fill_null(0),
                pl.len().alias("n_observations"),
            )
            # Join the pre-calculated tariff stats for this bin for hover info
            .join(bin_tariff_stats_df, on=bin_col_name, how="left")
            .sort([bin_col_name, "time_relative_to_impulse"])  # Sort for consistent plotting
        )

        if agg_df.is_empty():
            print(f"[{func_name}] Aggregated data is empty. Cannot plot.")
            return go.Figure().update_layout(
                title_text="No Aggregated Data",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )

        # --- Plot Aggregated Price Diff Data ---
        print(f"[{func_name}] Plotting average trends...")

        fig = px.line(
            agg_df.sort("time_relative_to_impulse"),  # Ensure time sorting for lines
            x="time_relative_to_impulse",
            y="mean_price_diff",
            color=bin_col_name,  # Use the defined bin column name
            markers=True,
            category_orders={bin_col_name: bin_labels},  # Use labels for ordering legend/colors
            custom_data=[  # Add the tariff stats columns to custom data
                "median_price_diff",
                "std_price_diff",
                "n_observations",
                "mean_tariff_change_in_bin",
                "median_tariff_change_in_bin",
                "std_tariff_change_in_bin",
            ],
        )

        # --- Update Layout and Hover ---
        price_base_name = price_diff_col_name.replace("_diff_from_t0", "")
        tariff_change_title = tariff_change_col_name.replace("_", " ").title()
        title = f"Average Price Difference Evolution by Tariff Change Bin<br><sup>Difference = {price_base_name}[t] - {price_base_name}[t=0]</sup>"
        xaxis_title = "Years Relative to Tariff Change (t=0)"
        yaxis_title = f"Avg. Price Difference from t=0 ({price_base_name.replace('_', ' ').title()})"

        fig.update_layout(
            title_text=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title_text=f"{tariff_change_title} Bin",
            hovermode="closest",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.add_vline(x=0, line_dash="dot", line_color="black")

        # Updated hovertemplate including tariff change stats for the bin
        hovertemplate = (
            f"<b>Bin: %{{fullData.name}}</b><br>"  # Get bin name from trace name
            f"<i>Avg {tariff_change_title} in Bin: %{{customdata[3]:.3f}}</i><br>"
            f"<i>Median {tariff_change_title} in Bin: %{{customdata[4]:.3f}}</i><br>"
            f"<i>Std Dev {tariff_change_title} in Bin: %{{customdata[5]:.3f}}</i><br>"
            f"<br>"
            f"Time Relative: %{{x}}<br>"
            f"Mean Price Diff: %{{y:.2f}}<br>"
            f"Median Price Diff: %{{customdata[0]:.2f}}<br>"
            f"Std Dev Price Diff: %{{customdata[1]:.2f}}<br>"
            f"Observations: %{{customdata[2]:,}}<br>"
            "<extra></extra>"
        )
        fig.update_traces(hovertemplate=hovertemplate)

        # fig.show() # Show the main plot

        return fig

    return (plot_avg_price_difference_by_tariff_bin_plotly,)


@app.cell
def _(
    plot_avg_price_difference_by_tariff_bin_plotly,
    price_col_name,
    tariff_price_impulse_df,
):
    fig_impulse_binned = plot_avg_price_difference_by_tariff_bin_plotly(
        results_diff_df=tariff_price_impulse_df,
        price_diff_col_name=price_col_name + "_detrended_diff_from_t0",
        tariff_change_col_name="tariff_change_at_impulse",
        num_bins=4,
    )
    # fig_impulse_binned.write_html(
    #     "charts/direct_effects/binned_price_impulse_response.html"
    # )
    fig_impulse_binned
    return


@app.cell
def _(mo):
    mo.md(r"""# Statistical significance""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from scipy import stats

    return plt, stats


@app.cell
def _(mo, price_col_name, tariff_changes_df_detrend):
    change_column_to_test = price_col_name + "_detrended_change"
    df = tariff_changes_df_detrend.drop_nulls(subset=[change_column_to_test])
    cell1_md = f"""
    ## Statistical Significance Test

    **Testing Column:** `{change_column_to_test}`

    **Number of Valid Samples:** {df.height}
    """
    mo.md(cell1_md)
    return change_column_to_test, df


@app.cell
def _(change_column_to_test, mo):
    mo.md(
        f"""**Figure 1:** Distribution of changes in `{change_column_to_test}`. The plot shows the empirical distribution of the calculated changes, with mean and median values indicated."""
    )
    return


@app.cell
def _(change_column_to_test, df, px):
    mean_val_statsig = df[change_column_to_test].mean()
    median_val_statsig = df[change_column_to_test].median()

    fig_hist = px.histogram(
        df,  # Polars DataFrame
        x=change_column_to_test,
        title=f"Distribution of {change_column_to_test}",
        labels={change_column_to_test: "Change Value"},
        opacity=0.75,
        color_discrete_sequence=["cornflowerblue"],  # Example color
    )

    fig_hist.update_layout(
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        bargap=0.1,  # Gap between bars
        yaxis_title="Frequency",
    )

    if mean_val_statsig is not None:
        fig_hist.add_vline(
            x=mean_val_statsig,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val_statsig:.4f}",
            annotation_position="top right",
        )
    if median_val_statsig is not None:
        fig_hist.add_vline(
            x=median_val_statsig,
            line_width=2,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Median: {median_val_statsig:.4f}",
            annotation_position="bottom right",  # Adjusted for potential overlap
        )

    fig_hist
    return


@app.cell
def _(change_column_to_test, mo):
    mo.md(
        f"""**Figure 2:** Q-Q (Quantile-Quantile) plot against a normal distribution for `{change_column_to_test}`. Deviations from the red dashed line suggest departures from normality."""
    )
    return


@app.cell
def _(change_column_to_test, df, plt, stats):
    data_for_qq_plot = df[change_column_to_test].to_numpy()  # stats.probplot needs numpy array

    stats.probplot(data_for_qq_plot, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {change_column_to_test}")

    plt.gcf()
    return


@app.cell
def _(change_column_to_test, df, mo, pl, stats):
    popmean_to_test = 0
    sample_data_np = df[change_column_to_test].to_numpy()

    ttest_output_md = "### One-Sample t-test Results\n\n"

    if sample_data_np.size > 1:
        onesample_ttest_result = stats.ttest_1samp(
            sample_data_np,
            popmean=popmean_to_test,
        )
        ttest_output_md += f"This test assesses if the mean of the observed changes in `{change_column_to_test}` is significantly different from {popmean_to_test}.\n\n"
        sample_mean = df[change_column_to_test].mean()
        if sample_mean is not None:
            ttest_output_md += f"- **Sample Mean:** `{sample_mean:.6f}`\n"
        ttest_output_md += f"- **T-statistic:** `{onesample_ttest_result.statistic:.4f}`\n"
        ttest_output_md += f"- **P-value:** `{onesample_ttest_result.pvalue:.6f}`\n\n"
        alpha = 0.05
        ttest_output_md += "**Interpretation (α = 0.05):**\n"
        if onesample_ttest_result.pvalue < alpha:
            ttest_output_md += f"<span style='color:green;'>P-value ({onesample_ttest_result.pvalue:.6f}) is less than alpha ({alpha}).</span>\n"
            ttest_output_md += f"<span style='color:green;'>**Reject the null hypothesis:** The mean change in `{change_column_to_test}` is statistically significant (different from {popmean_to_test}).</span>\n"
        else:
            ttest_output_md += f"<span style='color:red;'>P-value ({onesample_ttest_result.pvalue:.6f}) is not less than alpha ({alpha}).</span>\n"
            ttest_output_md += f"<span style='color:red;'>**Fail to reject the null hypothesis:** There is not enough evidence to conclude that the mean change in `{change_column_to_test}` is statistically significant (different from {popmean_to_test}).</span>\n"
    else:
        ttest_output_md += f"Skipped: Not enough data points ({sample_data_np.size}) for the t-test.\n"
    mo.md(ttest_output_md)

    # --- Cell 5: One-Sample Wilcoxon Signed-Rank Test Results ---
    wilcoxon_output_md = "### One-Sample Wilcoxon Signed-Rank Test Results\n\n"
    wilcoxon_output_md += f"This non-parametric test assesses if the *median* of the observed changes in `{change_column_to_test}` is significantly different from {popmean_to_test}. Applied as the data is not normally distributed.\n\n"
    changes_for_wilcoxon_pl = df.filter(pl.col(change_column_to_test) != 0)[change_column_to_test]
    changes_for_wilcoxon_np = changes_for_wilcoxon_pl.to_numpy()

    if changes_for_wilcoxon_np.size > 0:
        wilcoxon_result = stats.wilcoxon(changes_for_wilcoxon_np, zero_method="pratt", alternative="two-sided")
        sample_median = df[change_column_to_test].median()
        if sample_median is not None:
            wilcoxon_output_md += f"- **Sample Median (original data):** `{sample_median:.6f}`\n"
        wilcoxon_output_md += f"- **Statistic (W):** `{wilcoxon_result.statistic:.4f}`\n"
        wilcoxon_output_md += f"- **P-value:** `{wilcoxon_result.pvalue:.6f}`\n\n"
        alpha = 0.05
        wilcoxon_output_md += "**Interpretation (α = 0.05):**\n"
        if wilcoxon_result.pvalue < alpha:
            wilcoxon_output_md += f"<span style='color:green;'>P-value ({wilcoxon_result.pvalue:.6f}) is less than alpha ({alpha}).</span>\n"
            wilcoxon_output_md += f"<span style='color:green;'>**Reject the null hypothesis:** The median change in `{change_column_to_test}` is statistically significant (different from {popmean_to_test}).</span>\n"
        else:
            wilcoxon_output_md += f"<span style='color:red;'>P-value ({wilcoxon_result.pvalue:.6f}) is not less than alpha ({alpha}).</span>\n"
            wilcoxon_output_md += f"<span style='color:red;'>**Fail to reject the null hypothesis:** There is not enough evidence to conclude that the median change in `{change_column_to_test}` is statistically significant (different from {popmean_to_test}).</span>\n"
    else:
        wilcoxon_output_md += "Skipped: No non-zero change values were available for the Wilcoxon test after filtering, or the dataset was empty.\n"
    mo.md(wilcoxon_output_md)
    return


@app.cell
def _():
    # change_column_to_test = price_col_name + "_detrended_change"
    # df = tariff_changes_df_detrend.drop_nulls(subset=[change_column_to_test])

    # print(f"--- Testing column: {change_column_to_test} ---")
    # print(
    #     f"Number of valid samples: {df.height}"
    # )  # df.height for row count in Polars

    # # Visualize the distribution of changes
    # # Convert Polars Series to NumPy array for seaborn/matplotlib
    # data_for_plot = df[change_column_to_test].to_numpy()
    # mean_val_statsig = df[change_column_to_test].mean()
    # median_val_statsig = df[change_column_to_test].median()

    # plt.figure(figsize=(10, 5))
    # sns.histplot(data_for_plot, kde=True)
    # plt.title(f"Distribution of {change_column_to_test}")
    # plt.xlabel("Change Value")
    # plt.ylabel("Frequency")
    # if mean_val_statsig is not None:
    #     plt.axvline(
    #         mean_val_statsig,
    #         color="r",
    #         linestyle="--",
    #         label=f"Mean: {mean_val_statsig:.4f}",
    #     )
    # if median_val_statsig is not None:
    #     plt.axvline(
    #         median_val_statsig,
    #         color="g",
    #         linestyle="-",
    #         label=f"Median: {median_val_statsig:.4f}",
    #     )
    # plt.legend()
    # plt.show()

    # # Optional: Q-Q plot for normality check
    # stats.probplot(data_for_plot, dist="norm", plot=plt)
    # plt.title(f"Q-Q Plot for {change_column_to_test}")
    # plt.show()

    # # --- 3. Perform Statistical Tests ---

    # # --- Test 1: One-Sample t-test ---
    # # Tests if the mean of the changes in the selected column is significantly different from zero.
    # popmean_to_test = 0  # Null hypothesis: the mean change is zero

    # # Convert Polars Series to NumPy array for scipy function
    # sample_data_np = df[change_column_to_test].to_numpy()

    # onesample_ttest_result = stats.ttest_1samp(
    #     sample_data_np,
    #     popmean=popmean_to_test,
    #     # nan_policy='omit' is default in recent scipy, but Polars drop_nulls already handled NaNs
    # )

    # print("\n--- One-Sample t-test Results ---")
    # print(
    #     f"Testing if mean of '{change_column_to_test}' is different from {popmean_to_test}"
    # )
    # sample_mean = df[change_column_to_test].mean()
    # if sample_mean is not None:
    #     print(f"Mean of sample: {sample_mean:.6f}")
    # print(f"T-statistic: {onesample_ttest_result.statistic:.4f}")
    # print(f"P-value: {onesample_ttest_result.pvalue:.4f}")

    # # Interpretation
    # alpha = 0.05
    # if onesample_ttest_result.pvalue < alpha:
    #     print(
    #         f"P-value ({onesample_ttest_result.pvalue:.4f}) is less than alpha ({alpha})."
    #     )
    #     print(
    #         f"Reject the null hypothesis: The mean {change_column_to_test} is statistically significant (different from {popmean_to_test})."
    #     )
    # else:
    #     print(
    #         f"P-value ({onesample_ttest_result.pvalue:.4f}) is not less than alpha ({alpha})."
    #     )
    #     print(
    #         f"Fail to reject the null hypothesis: There is not enough evidence to say the mean {change_column_to_test} is statistically significant (different from {popmean_to_test})."
    #     )

    # # --- Test 2: One-Sample Wilcoxon Signed-Rank Test (Non-parametric alternative) ---
    # # Tests if the median of the changes is significantly different from zero.

    # # Filter out zeros for Wilcoxon, then convert to NumPy
    # changes_for_wilcoxon_pl = df.filter(pl.col(change_column_to_test) != 0)[
    #     change_column_to_test
    # ]
    # changes_for_wilcoxon_np = changes_for_wilcoxon_pl.to_numpy()

    # if changes_for_wilcoxon_np.size > 0:  # Check if the numpy array is not empty
    #     wilcoxon_result = stats.wilcoxon(
    #         changes_for_wilcoxon_np, zero_method="pratt"
    #     )

    #     print("\n--- One-Sample Wilcoxon Signed-Rank Test Results ---")
    #     print(
    #         f"Testing if median of '{change_column_to_test}' is different from {popmean_to_test}"
    #     )
    #     sample_median = df[change_column_to_test].median()
    #     if sample_median is not None:
    #         print(
    #             f"Median of sample (used for context, test is on non-zero diffs): {sample_median:.6f}"
    #         )
    #     print(f"Statistic: {wilcoxon_result.statistic:.4f}")
    #     print(f"P-value: {wilcoxon_result.pvalue:.4f}")

    #     # Interpretation
    #     if wilcoxon_result.pvalue < alpha:
    #         print(
    #             f"P-value ({wilcoxon_result.pvalue:.4f}) is less than alpha ({alpha})."
    #         )
    #         print(
    #             f"Reject the null hypothesis: The median {change_column_to_test} is statistically significant (different from {popmean_to_test})."
    #         )
    #     else:
    #         print(
    #             f"P-value ({wilcoxon_result.pvalue:.4f}) is not less than alpha ({alpha})."
    #         )
    #         print(
    #             f"Fail to reject the null hypothesis: There is not enough evidence to say the median {change_column_to_test} is statistically significant (different from {popmean_to_test})."
    #         )
    # else:
    #     print("\n--- One-Sample Wilcoxon Signed-Rank Test Results ---")
    #     print(
    #         "Skipped: No non-zero change values available for testing after filtering."
    #     )

    # # --- 4. Testing on a Subset ---
    # # Example: Test only for a specific partner country, e.g., 'USA'
    # target_partner = "840"  # USA
    # print(f"\n--- Testing on Subset: partner_country == '{target_partner}' ---")

    # subset_partner = df.filter(pl.col("partner_country") == target_partner)

    # if (
    #     not subset_partner.is_empty()
    #     and subset_partner.select(
    #         pl.col(change_column_to_test).drop_nulls()
    #     ).height
    #     > 1
    # ):
    #     print(f"Number of samples in subset: {subset_partner.height}")

    #     # --- One-Sample t-test on Subset ---
    #     subset_data_np = (
    #         subset_partner[change_column_to_test].drop_nulls().to_numpy()
    #     )  # Ensure NaNs are dropped for subset too
    #     if (
    #         subset_data_np.size > 1
    #     ):  # ttest_1samp needs at least 2 observations if var is estimated
    #         ttest_subset = stats.ttest_1samp(
    #             subset_data_np, popmean=popmean_to_test
    #         )
    #         print("\nOne-Sample t-test Results (Subset):")
    #         subset_mean = subset_partner[change_column_to_test].mean()
    #         if subset_mean is not None:
    #             print(f"Mean of sample (subset): {subset_mean:.6f}")
    #         print(
    #             f"T-statistic: {ttest_subset.statistic:.4f}, P-value: {ttest_subset.pvalue:.4f}"
    #         )
    #         if ttest_subset.pvalue < alpha:
    #             print(
    #                 "Reject H0: Mean change is statistically significant for this subset."
    #             )
    #         else:
    #             print(
    #                 "Fail to reject H0: Mean change is not statistically significant for this subset."
    #             )
    #     else:
    #         print(
    #             "\nOne-Sample t-test for subset skipped: Not enough data points after dropping NaNs."
    #         )

    #     # --- Wilcoxon on Subset ---
    #     subset_changes_for_wilcoxon_pl = subset_partner.filter(
    #         pl.col(change_column_to_test) != 0
    #     )[change_column_to_test]
    #     subset_changes_for_wilcoxon_np = (
    #         subset_changes_for_wilcoxon_pl.drop_nulls().to_numpy()
    #     )

    #     if subset_changes_for_wilcoxon_np.size > 0:
    #         wilcoxon_subset = stats.wilcoxon(
    #             subset_changes_for_wilcoxon_np, zero_method="pratt"
    #         )
    #         print("\nWilcoxon Signed-Rank Test Results (Subset):")
    #         subset_median = subset_partner[change_column_to_test].median()
    #         if subset_median is not None:
    #             print(
    #                 f"Median of sample (subset, for context): {subset_median:.6f}"
    #             )
    #         print(
    #             f"Statistic: {wilcoxon_subset.statistic:.4f}, P-value: {wilcoxon_subset.pvalue:.4f}"
    #         )
    #         if wilcoxon_subset.pvalue < alpha:
    #             print(
    #                 "Reject H0: Median change is statistically significant for this subset."
    #             )
    #         else:
    #             print(
    #                 "Fail to reject H0: Median change is not statistically significant for this subset."
    #             )
    #     else:
    #         print(
    #             "\nWilcoxon test skipped for subset (no non-zero changes or not enough data)."
    #         )

    # else:
    #     print(
    #         f"Not enough data or no data available for the subset reporter_country == '{target_partner}' after initial filtering."
    #     )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Conclusions and next steps
    As far as I understand it, I see the following natural extensions:

    - to perform a panel regression, controlling for fixed effects to see if we can remove some noise.
    - to do a more specific case study.

    Having said that, my prior is to instead do the following:

    - move onto the more policy-relevant question of trade diversion.

    > ➡️ **Please let me know if you have any thoughts regarding these results, my methodology, or the next steps I should take.**
    """
    )
    return


if __name__ == "__main__":
    app.run()
