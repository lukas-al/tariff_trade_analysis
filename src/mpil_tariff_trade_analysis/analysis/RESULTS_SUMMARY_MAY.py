import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import plotly.express as px

    import pycountry
    import pyfixest

    return mo, pd, pl, px, pycountry, pyfixest


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Work Summary
    End of May summary of work and current progress.

    ## Broad outline
    1. Identified dataset of A) bilateral trade values / volumes at the HS6 level for years 1994-2023
    2. Combined with a dataset of B) MFN & Pref tariffs. Included a dataset of C) exceptional Trump 1 Tariffs (list 1, 2, 3, 4a) after identifying that these are excluded from traditional tariff datasets.
        - Have more data available. Inlcuding exceptions, select retaliatory tariffs, and non-tariff barriers.
    4. Identified the direct effect on export prices of tariff imposition.
        -  Prelim Result: little evidence absorption of tariffs by exporters (prices remain broadly the same). This analysis could be extended if interesting.
    5. Re-estimated the elasticity estimates (section 3) from this [Huang-Mix paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5101245)
        -  For the direct effect. Result:
        -  For the indirect effect on R.o.W imports. Result: 
    6. Adjusted the equation to isolate specific third countries, and reran.
        -  For GBR. Result
        -  For Germany. Result
        -  For Canada. Result
        -  For Singapore. Result
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.cell(hide_code=True)
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

        null_count = plot_df[price_diff_col_name].is_null().sum()
        non_null_count = plot_df[price_diff_col_name].is_not_null().sum()

        print(f"Column to aggregate: '{price_diff_col_name}'")
        print(f"Number of null values: {null_count}")
        print(f"Number of non-null values: {non_null_count}")

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

        print("--- Final Aggregated Data to be Plotted ---")
        print(agg_df.to_pandas().to_string())
        print("-----------------------------------------")

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
        title = f"Average Difference Evolution by Tariff Change Bin<br><sup>Difference = {price_base_name}[t] - {price_base_name}[t=0]</sup>"
        xaxis_title = "Years Relative to Tariff Change (t=0)"
        yaxis_title = f"Avg. Difference from t=0 ({price_base_name.replace('_', ' ').title()})"

        fig.update_layout(
            title_text=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title_text=f"{tariff_change_title} Bin",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1),
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
            f"Mean Diff: %{{y:.2f}}<br>"
            f"Median Diff: %{{customdata[0]:.2f}}<br>"
            f"Std Dev Diff: %{{customdata[1]:.2f}}<br>"
            f"Observations: %{{customdata[2]:,}}<br>"
            "<extra></extra>"
        )
        fig.update_traces(hovertemplate=hovertemplate)

        # fig.show()  # Show the main plot

        return fig

    return


@app.cell(hide_code=True)
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

    return


@app.cell(hide_code=True)
def _(pl):
    def prepare_regression_data_HM1(
        filtered_lf: pl.LazyFrame,
        usa_cc: str,
        china_cc: str,
        effect_year_range: range,
    ) -> pl.DataFrame:
        """
        Prepares data for HM regression by selecting columns, creating interaction
        terms, and generating fixed effects variables.
        """
        tariff_us_china_expr = (
            pl.col("average_tariff_official")
            .filter((pl.col("partner_country") == usa_cc) & (pl.col("reporter_country") == china_cc))
            .mean()
            .over(["year", "product_code"])
            .alias("tariff_us_china")
        )

        input_lf = filtered_lf.select(
            pl.col("year"),
            pl.col("partner_country").alias("importer"),
            pl.col("reporter_country").alias("exporter"),
            pl.col("product_code"),
            pl.col("value").log().alias("log_value"),
            pl.col("quantity").log().alias("log_quantity"),
            tariff_us_china_expr,
        ).filter(pl.col("year").is_in(list(effect_year_range)))

        interaction_expressions = [
            pl.when((pl.col("year") == str(year)) & (pl.col("importer") == usa_cc) & (pl.col("exporter") == china_cc))
            .then(pl.col("tariff_us_china"))
            .otherwise(0.0)
            .alias(f"tariff_interaction_{year}")
            for year in effect_year_range
        ]

        fixed_effect_expressions = [
            pl.concat_str(["importer", "product_code", "year"], separator="^").alias("alpha_ipt").cast(pl.Categorical),
            pl.concat_str(["exporter", "product_code", "year"], separator="^").alias("alpha_jpt").cast(pl.Categorical),
            pl.concat_str(["importer", "exporter"], separator="^").alias("alpha_ij").cast(pl.Categorical),
        ]

        final_lf = input_lf.with_columns(*interaction_expressions, *fixed_effect_expressions)

        return final_lf.collect()

    return (prepare_regression_data_HM1,)


@app.cell(hide_code=True)
def _(CHINA_CC, USA_CC, pl):
    def prepare_regression_data_HM2(
        df: pl.LazyFrame,
        period1_start_year: str,
        period1_end_year: str,
        period2_start_year: str,
        period2_end_year: str,
        top_30_importers: list,
    ) -> pl.DataFrame:
        # Convert year strings to int for duration calculation
        p1_start_int = int(period1_start_year)
        p1_end_int = int(period1_end_year)
        p2_start_int = int(period2_start_year)
        p2_end_int = int(period2_end_year)

        period1_duration = p1_end_int - p1_start_int
        period2_duration = p2_end_int - p2_start_int

        if period1_duration <= 0 or period2_duration <= 0:
            raise ValueError("End year must be after start year for both periods.")

        relevant_years = [
            period1_start_year,
            period1_end_year,
            period2_start_year,
            period2_end_year,
        ]

        # 1. Initial Filtering
        unified_lf_without_oil_filtered = (
            df.filter(pl.col("year").is_in(relevant_years)).filter(pl.col("partner_country").is_in(top_30_importers)).filter(pl.col("value") > 0)
        )

        # 2. Reshape data
        agg_expressions = [pl.col("value").filter(pl.col("year") == pl.lit(year)).first().alias(f"val_{year}") for year in relevant_years]
        reshaped_lf = unified_lf_without_oil_filtered.group_by(["reporter_country", "partner_country", "product_code"]).agg(agg_expressions)

        val_p1_start_col = f"val_{period1_start_year}"
        val_p1_end_col = f"val_{period1_end_year}"
        val_p2_start_col = f"val_{period2_start_year}"
        val_p2_end_col = f"val_{period2_end_year}"

        # 3. Calculate growth rates for Period 1
        period1_growth_lf = (
            reshaped_lf.filter(
                pl.col(val_p1_start_col).is_not_null()
                & (pl.col(val_p1_start_col) > 0)
                & pl.col(val_p1_end_col).is_not_null()
                & (pl.col(val_p1_end_col) > 0)
            )
            .with_columns(
                y=(100 * (pl.col(val_p1_end_col).log() - pl.col(val_p1_start_col).log()) / period1_duration),
                PostEvent=pl.lit(0).cast(pl.Int8),  # Renamed from Post2017 for generality
            )
            .select(
                [
                    "reporter_country",
                    "partner_country",
                    "product_code",
                    "y",
                    "PostEvent",
                ]
            )
        )

        # 4. Calculate growth rates for Period 2
        period2_growth_lf = (
            reshaped_lf.filter(
                pl.col(val_p2_start_col).is_not_null()
                & (pl.col(val_p2_start_col) > 0)
                & pl.col(val_p2_end_col).is_not_null()
                & (pl.col(val_p2_end_col) > 0)
            )
            .with_columns(
                y=(100 * (pl.col(val_p2_end_col).log() - pl.col(val_p2_start_col).log()) / period2_duration),
                PostEvent=pl.lit(1).cast(pl.Int8),  # Renamed from Post2017
            )
            .select(
                [
                    "reporter_country",
                    "partner_country",
                    "product_code",
                    "y",
                    "PostEvent",
                ]
            )
        )

        # 5. Combine the two periods
        regression_input_lf = pl.concat([period1_growth_lf, period2_growth_lf], how="vertical")

        # 6. Add exporter dummies
        regression_input_lf = regression_input_lf.with_columns(
            [
                pl.when(pl.col("reporter_country") == pl.lit(CHINA_CC)).then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int8).alias("is_CHN_exporter"),
                pl.when(pl.col("reporter_country") == pl.lit(USA_CC)).then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int8).alias("is_USA_exporter"),
            ]
        )

        regression_df = regression_input_lf.collect()

        # Drop rows with NaN/inf in 'y'
        regression_df = regression_df.drop_nulls(subset=["y"])
        regression_df = regression_df.filter(pl.col("y").is_finite())

        return regression_df

    return (prepare_regression_data_HM2,)


@app.cell(hide_code=True)
def _(pl):
    def get_oil_exporting_countries(lzdf: pl.LazyFrame, oil_export_percentage_threshold: float) -> list[str]:
        """
        Filters a LazyFrame to find countries where oil products (HS code starting with '27')
        constitute a certain percentage of their total export value.

        Args:
            lzdf: The input Polars LazyFrame.
                  Schema should include 'reporter_country', 'product_code', and 'value'.
            oil_export_percentage_threshold: The minimum percentage (0-100) of total exports
                                             that must be oil products for a country to be included.

        Returns:
            A list of reporter_country names that meet the criteria.
        """
        if not 0 <= oil_export_percentage_threshold <= 100:
            raise ValueError("oil_export_percentage_threshold must be between 0 and 100.")

        # Calculate total export value for each country
        total_exports_by_country = lzdf.group_by("reporter_country").agg(pl.sum("value").alias("total_value"))

        # Calculate total oil export value for each country
        # HS codes for oil and mineral fuels are under Chapter 27.
        oil_exports_by_country = (
            lzdf.filter(pl.col("product_code").str.starts_with("27")).group_by("reporter_country").agg(pl.sum("value").alias("oil_value"))
        )

        # Join total exports with oil exports
        country_export_summary = total_exports_by_country.join(
            oil_exports_by_country,
            on="reporter_country",
            how="left",  # Use left join to keep all countries, oil_value will be null if no oil exports
        ).with_columns(
            pl.col("oil_value").fill_null(0.0)  # Fill nulls with 0 for countries with no oil exports
        )

        # Calculate the percentage of oil exports
        country_export_summary = country_export_summary.with_columns(
            ((pl.col("oil_value") / pl.col("total_value")) * 100).alias("oil_export_percentage")
        )

        # Filter countries above the threshold
        filtered_countries = (
            country_export_summary.filter(pl.col("oil_export_percentage") > oil_export_percentage_threshold)
            .select("reporter_country")
            .collect()  # Collect the results into a DataFrame
        )

        return filtered_countries["reporter_country"].to_list()

    return (get_oil_exporting_countries,)


@app.cell(hide_code=True)
def _(CHINA_CC, USA_CC, pl, pycountry):
    def prepare_regression_data_THIRDCOUNTRY(
        df: pl.LazyFrame,
        country_to_isolate: str,  # The country code of the partner_country to isolate
        period1_start_year: str,
        period1_end_year: str,
        period2_start_year: str,
        period2_end_year: str,
        top_30_importers: list,
    ) -> pl.DataFrame:
        # Check if the country to isolate is in the top_30_importers list
        if country_to_isolate not in top_30_importers:
            raise ValueError(f"Country to isolate '{country_to_isolate}' is not in the top_30_importers list.")

        # Convert year strings to int for duration calculation
        p1_start_int = int(period1_start_year)
        p1_end_int = int(period1_end_year)
        p2_start_int = int(period2_start_year)
        p2_end_int = int(period2_end_year)

        period1_duration = p1_end_int - p1_start_int
        period2_duration = p2_end_int - p2_start_int

        if period1_duration <= 0 or period2_duration <= 0:
            raise ValueError("End year must be after start year for both periods.")

        relevant_years = [
            period1_start_year,
            period1_end_year,
            period2_start_year,
            period2_end_year,
        ]

        # 1. Initial Filtering
        unified_lf_without_oil_filtered = (
            df.filter(pl.col("year").is_in(relevant_years)).filter(pl.col("partner_country").is_in(top_30_importers)).filter(pl.col("value") > 0)
        )

        # 2. Reshape data
        agg_expressions = [pl.col("value").filter(pl.col("year") == pl.lit(year)).first().alias(f"val_{year}") for year in relevant_years]
        reshaped_lf = unified_lf_without_oil_filtered.group_by(["reporter_country", "partner_country", "product_code"]).agg(agg_expressions)

        val_p1_start_col = f"val_{period1_start_year}"
        val_p1_end_col = f"val_{period1_end_year}"
        val_p2_start_col = f"val_{period2_start_year}"
        val_p2_end_col = f"val_{period2_end_year}"

        # 3. Calculate growth rates for Period 1
        period1_growth_lf = (
            reshaped_lf.filter(
                pl.col(val_p1_start_col).is_not_null()
                & (pl.col(val_p1_start_col) > 0)
                & pl.col(val_p1_end_col).is_not_null()
                & (pl.col(val_p1_end_col) > 0)
            )
            .with_columns(
                y=(100 * (pl.col(val_p1_end_col).log() - pl.col(val_p1_start_col).log()) / period1_duration),
                PostEvent=pl.lit(0).cast(pl.Int8),
            )
            .select(
                [
                    "reporter_country",
                    "partner_country",
                    "product_code",
                    "y",
                    "PostEvent",
                ]
            )
        )

        # 4. Calculate growth rates for Period 2
        period2_growth_lf = (
            reshaped_lf.filter(
                pl.col(val_p2_start_col).is_not_null()
                & (pl.col(val_p2_start_col) > 0)
                & pl.col(val_p2_end_col).is_not_null()
                & (pl.col(val_p2_end_col) > 0)
            )
            .with_columns(
                y=(100 * (pl.col(val_p2_end_col).log() - pl.col(val_p2_start_col).log()) / period2_duration),
                PostEvent=pl.lit(1).cast(pl.Int8),
            )
            .select(
                [
                    "reporter_country",
                    "partner_country",
                    "product_code",
                    "y",
                    "PostEvent",
                ]
            )
        )

        # 5. Combine the two periods
        regression_input_lf = pl.concat([period1_growth_lf, period2_growth_lf], how="vertical")

        # 6. Add exporter dummies based on the country_to_isolate
        # CHN and USA are now hardcoded
        regression_input_lf = regression_input_lf.with_columns(
            [
                pl.when((pl.col("reporter_country") == pl.lit(CHINA_CC)) & (pl.col("partner_country") == pl.lit(country_to_isolate)))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_to_isolate}_importer"),
                pl.when((pl.col("reporter_country") == pl.lit(USA_CC)) & (pl.col("partner_country") == pl.lit(country_to_isolate)))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_to_isolate}_importer"),
                pl.when((pl.col("reporter_country") == pl.lit(CHINA_CC)) & (pl.col("partner_country") != pl.lit(country_to_isolate)))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_to_isolate}_importer"),
                pl.when((pl.col("reporter_country") == pl.lit(USA_CC)) & (pl.col("partner_country") != pl.lit(country_to_isolate)))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_to_isolate}_importer"),
            ]
        )

        regression_df = regression_input_lf.collect()

        # Drop rows with NaN/inf in 'y'
        regression_df = regression_df.drop_nulls(subset=["y"])
        regression_df = regression_df.filter(pl.col("y").is_finite())

        return regression_df

    return (prepare_regression_data_THIRDCOUNTRY,)


@app.cell(hide_code=True)
def _(CHINA_CC, USA_CC, pycountry, pyfixest):
    def run_regression_THIRDCOUNTRY(data, country_of_interest_code):
        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest_code}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest_code}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest_code}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest_code}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pyfixest.feols(
            fml=formula,
            data=data,
            vcov="hetero",
        )

        print(f" --- Running model, isolating {pycountry.countries.get(numeric=country_of_interest_code).name} --- ")

        return (model_eq2MOD_v0.summary(), model_eq2MOD_v0.coefplot())

    return (run_regression_THIRDCOUNTRY,)


@app.cell
def _(mo):
    mo.md(r"""## Dataset""")
    return


@app.cell
def _(mo, pl):
    # WITS MFN
    wits_mfn = pl.scan_parquet("data/intermediate/WITS_AVEMFN_CLEAN.parquet")

    mo.vstack(
        [
            mo.md("WITS Most Favoured Nation Tariffs. Head 10:"),
            wits_mfn.head(10).collect(),
        ]
    )
    return


@app.cell
def _(mo, pl):
    wits_pref = pl.scan_parquet("data/intermediate/WITS_AVEPref_CLEAN.parquet")

    mo.vstack(
        [
            mo.md("WITS Preferential Group Tariffs. Head 10:"),
            wits_pref.head(10).collect(),
        ]
    )
    return


@app.cell
def _(mo, pl):
    # BACI
    baci_raw = pl.scan_parquet("data/intermediate/BACI_HS92_V202501_CLEAN.parquet")

    mo.vstack(
        [
            mo.md("BACI raw bilateral trade data. Head 10:"),
            baci_raw.head(10).collect(),
        ]
    )
    return


@app.cell
def _(mo, pd):
    # Raw Trump 1 Tariffs
    cm_us_tariffs = pd.read_csv("data/intermediate/cm_us_tariffs.csv")

    mo.vstack(
        [
            mo.md("Trump Tariffs, as in Huang-Mix paper. Head 10:"),
            cm_us_tariffs.head(10),
        ]
    )
    return (cm_us_tariffs,)


@app.cell
def _(mo, pl):
    # Merged
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")

    mo.vstack(
        [
            mo.md("Unified dataset, joined and harmonised. Head 10:"),
            unified_lf.head(10).collect(),
        ]
    )
    return (unified_lf,)


@app.cell
def _(mo):
    mo.md(r"""## Config""")
    return


@app.cell
def _():
    PRICE_COL_NAME = "unit_value"
    VALUE_COL_NAME = "value"
    QUANTITY_COL_NAME = "quantity"
    TARIFF_COL_NAME = "average_tariff_official"

    REPORTER_COUNTRIES = ["156"]  # Reporter countries to filter to [China]
    PARTNER_COUNTRIES = ["840"]  # Partner countries to filter to [USA]
    PRODUCT_CODES = None  # Products to filter to

    # Number of years before the first year of the tariff change that we start measuring prices.
    PRICE_CHANGE_START = 1
    # Number of years after the last year of the tariff change that we stop measuring prices.
    PRICE_CHANGE_END = 2
    # Gap between years that we're measuring the start of tariff changes
    TARIFF_CHANGE_GAP = 1
    return (
        PARTNER_COUNTRIES,
        PRICE_CHANGE_END,
        PRICE_CHANGE_START,
        PRICE_COL_NAME,
        PRODUCT_CODES,
        QUANTITY_COL_NAME,
        REPORTER_COUNTRIES,
        TARIFF_CHANGE_GAP,
        TARIFF_COL_NAME,
        VALUE_COL_NAME,
    )


@app.cell
def _(
    PARTNER_COUNTRIES,
    PRICE_CHANGE_END,
    PRICE_CHANGE_START,
    PRICE_COL_NAME,
    PRODUCT_CODES,
    QUANTITY_COL_NAME,
    REPORTER_COUNTRIES,
    TARIFF_CHANGE_GAP,
    TARIFF_COL_NAME,
    VALUE_COL_NAME,
    mo,
):
    mo.md(
        f"""
    ### Columns to use
    Price col name: *{PRICE_COL_NAME}*

    Value col name: *{VALUE_COL_NAME}*

    Quantity col name: *{QUANTITY_COL_NAME}*

    Tariff Col name: *{TARIFF_COL_NAME}*


    ### Filter criteria
    Reporter Countries: *{REPORTER_COUNTRIES}*

    Partner Countries: *{PARTNER_COUNTRIES}*

    Product Codes: *{PRODUCT_CODES}*

    ### Identification arguments
    *Number of years before the first year of the tariff change that we start measuring prices.*

    Price Change Start: *{PRICE_CHANGE_START}*

    *Number of years after the last year of the tariff change that we stop measuring prices.*

    Price Change End: *{PRICE_CHANGE_END}*

    *Gap between years that we're using to identify tariff changes*

    Tariff Change Gap: *{TARIFF_CHANGE_GAP}*
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Direct Effects

    What are the direct effects on Chinese exports to the US of the Trump 1 Tariffs?
    """
    )
    return


@app.cell
def _(
    PARTNER_COUNTRIES,
    PRICE_COL_NAME,
    PRODUCT_CODES,
    QUANTITY_COL_NAME,
    REPORTER_COUNTRIES,
    TARIFF_COL_NAME,
    VALUE_COL_NAME,
    functools,
    operator,
    pl,
    remove_outliers_by_percentile_lazy,
    unified_lf,
):
    # First we filter the data
    filtered_lf = unified_lf
    if REPORTER_COUNTRIES:
        filtered_lf = filtered_lf.filter(pl.col("reporter_country").is_in(REPORTER_COUNTRIES))

    if PARTNER_COUNTRIES:
        filtered_lf = filtered_lf.filter(pl.col("partner_country").is_in(PARTNER_COUNTRIES))

    if PRODUCT_CODES:
        conditions = [pl.col("product_code").str.slice(0, len(p)) == p for p in PRODUCT_CODES]
        # Combine conditions with an OR
        combined_condition = functools.reduce(operator.or_, conditions)
        filtered_lf = filtered_lf.filter(combined_condition)

    filtered_lf = remove_outliers_by_percentile_lazy(
        ldf=filtered_lf,
        column_names=[
            TARIFF_COL_NAME,
            VALUE_COL_NAME + "_detrended",
            QUANTITY_COL_NAME + "_detrended",
            PRICE_COL_NAME + "_detrended",
        ],
        lower_p=0.01,
        upper_p=0.99,
    )

    # Also apply a simple logarithm to the quantity, value, and price columns
    filtered_lf = filtered_lf.with_columns(
        pl.col(VALUE_COL_NAME + "_detrended").log().alias(VALUE_COL_NAME + "_detrended_log"),
        pl.col(QUANTITY_COL_NAME + "_detrended").log().alias(QUANTITY_COL_NAME + "_detrended_log"),
        pl.col(PRICE_COL_NAME + "_detrended").log().alias(PRICE_COL_NAME + "_detrended_log"),
        pl.col(TARIFF_COL_NAME).log().alias(TARIFF_COL_NAME + "_log"),
    )
    return (filtered_lf,)


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

    mo.vstack(
        [
            mo.md("First we filter down to only the data we're interested in: US imports from China"),
            mo.hstack(
                [
                    pre_filter,
                    post_filter,
                ],
                justify="center",
                gap="2rem",
            ),
            mo.md("Dataframe sample post filtering:"),
            filtered_lf.head().collect(),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Then we identify cases of tariff change, and extract the relevant data.""")
    return


@app.cell
def _(TARIFF_COL_NAME, mo):
    dependent = "value_detrended_log"
    independent = TARIFF_COL_NAME + "_log"
    mo.md(f"In this case, we're considering the change in **{dependent}** following the imposition in tariffs")
    return dependent, independent


@app.cell
def _(
    PRICE_CHANGE_END,
    PRICE_CHANGE_START,
    TARIFF_CHANGE_GAP,
    dependent,
    extract_tariff_changes,
    filtered_lf,
    independent,
):
    # First we need to extract the rows we're interested in
    print("--- Extracting cases of change in tariff ---")

    tariff_changes_df = extract_tariff_changes(
        filtered_lf=filtered_lf,
        start_year="1999",
        end_year="2023",
        years_before_tariff_change_unit_value=PRICE_CHANGE_START,
        year_unit_value_end_gap=PRICE_CHANGE_END,
        year_gap=TARIFF_CHANGE_GAP,
        tariff_col_name=independent,
        price_col_name=dependent,
    )

    print(f"--- Identified {tariff_changes_df.height} cases of tariff change in sample ---")
    return (tariff_changes_df,)


@app.cell
def _(mo):
    mo.md(r"""Which looks as follows:""")
    return


@app.cell
def _(tariff_changes_df):
    tariff_changes_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""Visualised on a simple scatter, we get the following""")
    return


@app.cell
def _(
    PRICE_CHANGE_END,
    PRICE_CHANGE_START,
    TARIFF_CHANGE_GAP,
    dependent,
    independent,
    px,
    tariff_changes_df,
):
    scatter_filtered = px.scatter(
        tariff_changes_df,
        x=independent + "_change",
        y=dependent + "_change",
        title=f"""
            t0 to t+{TARIFF_CHANGE_GAP} abs change in tariff vs t-{PRICE_CHANGE_START} to t+{PRICE_CHANGE_END} year abs change in <br>trade value between US and China
        """,
        hover_data=[
            independent + "_change_range",
            dependent + "_change_range",
            "product_code",
        ],
        color=independent + "_change_range",
        # trendline='ols'
    )

    scatter_filtered.update_layout(legend=dict())
    scatter_filtered
    return


@app.cell
def _(dependent, mo):
    mo.md(
        rf"""
    We clearly observe the Trump Tariffs. We also observe the weakness of any relationship between bilateral trade value and the imposition of tariffs.
    <!-- 
    Another way to visualise this data, is to observe the change in the dependent variable (**{dependent}**) through time following tariff imposition.

    For this visualisation, I bucket the tariff changes into quantiles by their size. -->
    """
    )
    return


@app.cell
def _():
    # tariff_price_impulse_df = extract_tariff_changes_impulse_diff(
    #     filtered_lf=filtered_lf,
    #     start_year="1999",
    #     end_year="2023",
    #     year_gap=1,
    #     series_length=4,
    #     tariff_col_name=TARIFF_COL_NAME + "_log",
    #     price_col_name=QUANTITY_COL_NAME + "_detrended_log",
    #     years_before_tariff_change=0,
    # )
    return


@app.cell
def _():
    # fig_impulse_binned = plot_avg_price_difference_by_tariff_bin_plotly(
    #     results_diff_df=tariff_price_impulse_df,
    #     price_diff_col_name=QUANTITY_COL_NAME + "_detrended_log_diff_from_t0",
    #     tariff_change_col_name="tariff_change_at_impulse",
    #     num_bins=5,
    # )

    # fig_impulse_binned
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This data is obviously highly noisy, with many confounders. To control for some of these, I replicate the direct-effect elasticity panel regression as specified in the Huang-Mix paper, [Trade Wars and Rumors of Trade Wars:
    The Dynamic Effects of the U.S.–China Tariff Hikes](https://drive.google.com/file/d/1DQABGIs2oD2wt9pMN89uppp44Gc6FIm6/view).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Panel Regression 1 (direct effects)

    The Huang-Mix paper is primarily focussed on computing the welfare impacts of tariffs as a trade policy. It focuses on the price impacts of tariffs, including various expectations and uncertainty channels. This is applied through a large general equillibrium DSGE model, including the behaviour of multiple agents. 

    We are interested in just the panel regression used to inform the elasiticty parameters in the model.
    """
    )
    return


@app.cell(hide_code=True)
def _(pycountry):
    # Get the country codes we're interested in for this analysis:
    USA_CC = pycountry.countries.search_fuzzy("USA")[0].numeric
    CHINA_CC = pycountry.countries.search_fuzzy("China")[0].numeric
    BRAZIL_CC = pycountry.countries.search_fuzzy("Brazil")[0].numeric
    IRELAND_CC = pycountry.countries.search_fuzzy("Ireland")[0].numeric
    ITALY_CC = pycountry.countries.search_fuzzy("Italy")[0].numeric
    SOUTHAFRICA_CC = pycountry.countries.search_fuzzy("South Africa")[0].numeric
    UK_CC = pycountry.countries.search_fuzzy("United Kingdom")[0].numeric
    GERMANY_CC = pycountry.countries.search_fuzzy("Germany")[0].numeric
    FRANCE_CC = pycountry.countries.search_fuzzy("France")[0].numeric

    cc_list_incl_ROW = [
        USA_CC,
        CHINA_CC,
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        UK_CC,
        GERMANY_CC,
        FRANCE_CC,
    ]

    cc_list_ROW = [
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        UK_CC,
        GERMANY_CC,
        FRANCE_CC,
    ]

    cc_list = [USA_CC, CHINA_CC]

    TOTAL_YEAR_RANGE = [str(y) for y in range(1999, 2024)]
    EFFECT_YEAR_RANGE = [str(y) for y in range(2018, 2024)]
    return (
        CHINA_CC,
        EFFECT_YEAR_RANGE,
        FRANCE_CC,
        GERMANY_CC,
        ITALY_CC,
        UK_CC,
        USA_CC,
        cc_list_incl_ROW,
    )


@app.cell(hide_code=True)
def _(cc_list_incl_ROW, pl, unified_lf):
    # Filter data again
    filtered_lf_HM = unified_lf.filter((pl.col("reporter_country").is_in(cc_list_incl_ROW) & pl.col("partner_country").is_in(cc_list_incl_ROW),))
    return (filtered_lf_HM,)


@app.cell(hide_code=True)
def _(
    CHINA_CC,
    EFFECT_YEAR_RANGE,
    USA_CC,
    filtered_lf_HM,
    mo,
    prepare_regression_data_HM1,
):
    # Create our input df and clean it slightly
    input_df_HM1 = prepare_regression_data_HM1(
        filtered_lf=filtered_lf_HM,
        china_cc=CHINA_CC,
        usa_cc=USA_CC,
        effect_year_range=EFFECT_YEAR_RANGE,
    )

    clean_input_df_HM1 = input_df_HM1.drop_nans(subset=["log_value", "tariff_us_china"])

    mo.vstack(
        [
            mo.md("We create the following input dataset, as sampled below:"),
            clean_input_df_HM1.head(),
            mo.md("Some summary statistics:"),
            clean_input_df_HM1.describe(),
        ]
    )
    return (clean_input_df_HM1,)


@app.cell(hide_code=True)
def _(EFFECT_YEAR_RANGE, mo):
    regressors_HM1 = " + ".join([f"tariff_interaction_{year}" for year in EFFECT_YEAR_RANGE])
    regression_formula_HM1 = f"log_value ~ {regressors_HM1} | alpha_ipt + alpha_jpt + alpha_ij"

    mo.vstack(
        [
            mo.md("We specify the regression formula as follows:"),
            mo.md(f"{regression_formula_HM1}"),
        ]
    )
    return (regression_formula_HM1,)


@app.cell
def _(clean_input_df_HM1, mo, pyfixest, regression_formula_HM1):
    with mo.persistent_cache(name="model_cache"):
        model_HM1 = pyfixest.feols(regression_formula_HM1, clean_input_df_HM1)
    return (model_HM1,)


@app.cell
def _(mo):
    mo.md(r"""From which we obtain the following results:""")
    return


@app.cell
def _(model_HM1):
    model_HM1.summary()
    return


@app.cell
def _(model_HM1):
    model_HM1.coefplot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Interpretation
    In the first full year following the imposition of tariffs, there was, on average, a 1.5% reduction in log(trade values) over all products imported from China by the US. 

    The size of this effect increases over time. We do not control for further other tariff impositions or exclusions which could affect this result in later years (2019 onwards).

    #### Conclusions
    We performed this exercise to compare our results against the original paper. They are qualitatively similar, and the quantitative differences can be reasonably explained via differences in methodology.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Panel Regression 2 (indirect effects on third countries)

    A more relevant part of the HM paper is the regression used to estimate the impact of the Trump 1 Tariffs on imports of the rest-of-world (RoW) from China.

    We replicate this specification below. Some more steps are required, including the exclusion of primarily oil exporting countries from the sample, and a filter to the top 30 countries by import value in 2017.
    """
    )
    return


@app.cell(hide_code=True)
def _(get_oil_exporting_countries, mo, pl, pycountry, unified_lf):
    # Identify oil countries -> those with >40% of their exports in hs code 27
    oil_country_list = get_oil_exporting_countries(unified_lf, 40)

    without_oil_unified_lf = unified_lf.filter(~pl.col("reporter_country").is_in(oil_country_list))

    oil_country_list_pycountries = [
        pycountry.countries.get(numeric=country) for country in oil_country_list if pycountry.countries.get(numeric=country)
    ]

    mo.vstack(
        [
            mo.md("We define oil exporters as those with >40% of their exports in hs code 27. This list is as follows:"),
            [country.name for country in oil_country_list_pycountries],
        ]
    )
    return (without_oil_unified_lf,)


@app.cell(hide_code=True)
def _(pl, without_oil_unified_lf):
    country_trade_values_2017 = (
        without_oil_unified_lf.group_by(["year", "partner_country"])
        .agg(
            pl.sum("value"),
            pl.sum("quantity"),
        )
        .filter(pl.col("year") == "2017")
        .sort(by="value", descending=True)
    ).collect()

    # Filter to our top 30 importers
    top_30_importers = country_trade_values_2017.sort(by="value", descending=True).select("partner_country").to_series().to_list()[:29]
    return (top_30_importers,)


@app.cell(hide_code=True)
def _(prepare_regression_data_HM2, top_30_importers, without_oil_unified_lf):
    regression_df_HM2 = prepare_regression_data_HM2(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
    )
    return (regression_df_HM2,)


@app.cell(hide_code=True)
def _():
    regression_formula_HM2 = "y ~ is_CHN_exporter:PostEvent + is_USA_exporter:PostEvent | reporter_country + C(Po stEvent)"
    return (regression_formula_HM2,)


@app.cell(hide_code=True)
def _(mo, pyfixest, regression_df_HM2, regression_formula_HM2):
    with mo.persistent_cache(name="model_cache"):
        model_HM2 = pyfixest.feols(
            fml=regression_formula_HM2,
            data=regression_df_HM2,
            vcov="hetero",
        )
    return (model_HM2,)


@app.cell(hide_code=True)
def _(mo, model_HM2, regression_df_HM2, regression_formula_HM2):
    mo.vstack(
        [
            mo.md("The regression formula is as follows:"),
            regression_formula_HM2,
            mo.md("A sample of the input:"),
            regression_df_HM2.head(),
            mo.md("And the results:"),
            model_HM2.summary(),
            model_HM2.coefplot(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Interpretation
    The average change in growth rates by the RoW between 2012-2016 and 2018-2022, was X% for imports from the US and Y% for imports from China.

    Again, this results differs quantitatively from those in the original HM paper, but is qualitatively similar.

    ### Extension 1: Expanding sample to include 2023
    """
    )
    return


@app.cell(hide_code=True)
def _(
    mo,
    prepare_regression_data_HM2,
    pyfixest,
    regression_formula_HM2,
    top_30_importers,
    without_oil_unified_lf,
):
    regression_df_HM2_v2 = prepare_regression_data_HM2(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2023",
        top_30_importers=top_30_importers,
    )

    with mo.persistent_cache(name="model_cache"):
        model_HM2_v2 = pyfixest.feols(
            fml=regression_formula_HM2,
            data=regression_df_HM2_v2,
            vcov="hetero",
        )

    mo.vstack(
        [
            mo.md("Expanding our sample beyond the paper's specification, up to 2023, we get the following result:"),
            model_HM2_v2.summary(),
            model_HM2_v2.coefplot(),
        ]
    )
    return (regression_df_HM2_v2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When we expand our sample to include more up-to-date data (including 2023), we get different results. This time, they differ qualitatively as well (a different sign on imports from the US).

    This might suggest that the fixed effects are not controlling fully for the effects of COVID?

    ### Extension 2: Filtering to include only tariffed products
    We filter the sample to include only tariffed products.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    cm_us_tariffs,
    mo,
    pl,
    pyfixest,
    regression_df_HM2_v2,
    regression_formula_HM2,
):
    regression_df_HM2_v3 = regression_df_HM2_v2.filter(pl.col("product_code").is_in(pl.Series(cm_us_tariffs["product_code"]).cast(pl.Utf8)))

    with mo.persistent_cache(name="model_cache"):
        model_HM2_v3 = pyfixest.feols(
            fml=regression_formula_HM2,
            data=regression_df_HM2_v3,
            vcov="hetero",
        )

    mo.vstack(
        [
            mo.md("Data filtered to include only products affected by Trump Tariffs:"),
            regression_df_HM2_v3.describe(),
            mo.md("Results when filtered to include only products affected by the tariffs:"),
            model_HM2_v3.summary(),
            model_HM2_v3.coefplot(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Extension 3: filtering to include only products not in trump tariffs""")
    return


@app.cell(hide_code=True)
def _(
    cm_us_tariffs,
    mo,
    pl,
    pyfixest,
    regression_df_HM2_v2,
    regression_formula_HM2,
):
    regression_df_HM2_v4 = regression_df_HM2_v2.filter(~pl.col("product_code").is_in(pl.Series(cm_us_tariffs["product_code"]).cast(pl.Utf8)))

    with mo.persistent_cache(name="model_cache"):
        model_HM2_v4 = pyfixest.feols(
            fml=regression_formula_HM2,
            data=regression_df_HM2_v4,
            vcov="hetero",
        )

    mo.vstack(
        [
            mo.md("Data filtered to include only products **not affected** by Trump Tariffs:"),
            regression_df_HM2_v4.describe(),
            mo.md("Results when filtered to include only products **not affected** by the tariffs:"),
            model_HM2_v4.summary(),
            model_HM2_v4.coefplot(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Conclusion

    These results are counterintuitive and point to some underlying issue in my data, or perhaps uncontrolled effects. They are similar to the Huang-Mix results, who reach for general equillibrium effects as an explanation.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Panel regression 3: Indirect effects on third countries, isolating specific countries

    We modify the panel regression from the previous section to isolate out specific countries, controlling for importer, exporter, and time period fixed effects.
    """
    )
    return


@app.cell(hide_code=True)
def _(CHINA_CC, UK_CC, USA_CC, mo, pycountry):
    country_of_interest = UK_CC

    formula = (
        f"y ~ "
        f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
        f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
        f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
        f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
        f"| reporter_country + C(PostEvent) + partner_country"
    )

    mo.md(f"Formula is as such, assuming country of interest is the UK:<br> **{formula}**")
    return


@app.cell(hide_code=True)
def _(
    UK_CC,
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    regression_df_MOD_UK = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=UK_CC,
    )

    summary_uk, coeffplot_uk = run_regression_THIRDCOUNTRY(
        country_of_interest_code=UK_CC,
        data=regression_df_MOD_UK,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for UK:"),
            summary_uk,
            coeffplot_uk,
        ]
    )
    return


@app.cell
def _(
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    pycountry,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    JAPAN_CC = pycountry.countries.search_fuzzy("Japan")[0].numeric

    regression_df_MOD_JPN = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=JAPAN_CC,
    )

    summary_jpn, coeffplot_jpn = run_regression_THIRDCOUNTRY(
        country_of_interest_code=JAPAN_CC,
        data=regression_df_MOD_JPN,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for Japan:"),
            summary_jpn,
            coeffplot_jpn,
        ]
    )
    return


@app.cell
def _(
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    pycountry,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    AUSTRALIA_CC = pycountry.countries.search_fuzzy("Australia")[0].numeric

    regression_df_MOD_OZ = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=AUSTRALIA_CC,
    )

    summary_OZ, coeffplot_OZ = run_regression_THIRDCOUNTRY(
        country_of_interest_code=AUSTRALIA_CC,
        data=regression_df_MOD_OZ,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for Australia:"),
            summary_OZ,
            coeffplot_OZ,
        ]
    )
    return


@app.cell
def _(
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    pycountry,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    SINGAPORE_CC = pycountry.countries.search_fuzzy("Singapore")[0].numeric

    regression_df_MOD_sin = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=SINGAPORE_CC,
    )

    summary_sin, coeffplot_sin = run_regression_THIRDCOUNTRY(
        country_of_interest_code=SINGAPORE_CC,
        data=regression_df_MOD_sin,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for Singapore:"),
            summary_sin,
            coeffplot_sin,
        ]
    )
    return


@app.cell
def _(
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    pycountry,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    CANADA_CC = pycountry.countries.search_fuzzy("CANADA")[0].numeric

    regression_df_MOD_can = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=CANADA_CC,
    )

    summary_can, coeffplot_can = run_regression_THIRDCOUNTRY(
        country_of_interest_code=CANADA_CC,
        data=regression_df_MOD_can,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for Canada:"),
            summary_can,
            coeffplot_can,
        ]
    )
    return


@app.cell
def _(
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    pycountry,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    INDIA_CC = pycountry.countries.search_fuzzy("India")[0].numeric

    regression_df_MOD_in = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=INDIA_CC,
    )

    summary_in, coeffplot_in = run_regression_THIRDCOUNTRY(
        country_of_interest_code=INDIA_CC,
        data=regression_df_MOD_in,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for India:"),
            summary_in,
            coeffplot_in,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    GERMANY_CC,
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    regression_df_MOD_GER = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=GERMANY_CC,
    )

    summary_ger, coeffplot_ger = run_regression_THIRDCOUNTRY(
        country_of_interest_code=GERMANY_CC,
        data=regression_df_MOD_GER,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for Germany:"),
            summary_ger,
            coeffplot_ger,
        ]
    )
    return


@app.cell
def _(
    FRANCE_CC,
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    regression_df_MOD_FR = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=FRANCE_CC,
    )

    summary_fr, coeffplot_fr = run_regression_THIRDCOUNTRY(
        country_of_interest_code=FRANCE_CC,
        data=regression_df_MOD_FR,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for France:"),
            summary_fr,
            coeffplot_fr,
        ]
    )
    return


@app.cell
def _(
    ITALY_CC,
    mo,
    prepare_regression_data_THIRDCOUNTRY,
    run_regression_THIRDCOUNTRY,
    top_30_importers,
    without_oil_unified_lf,
):
    regression_df_MOD_IT = prepare_regression_data_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=ITALY_CC,
    )

    summary_IT, coeffplot_IT = run_regression_THIRDCOUNTRY(
        country_of_interest_code=ITALY_CC,
        data=regression_df_MOD_IT,
    )

    mo.vstack(
        [
            mo.md("### Results isolating for Italy:"),
            summary_IT,
            coeffplot_IT,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
