import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pycountry
    import pyfixest as pf
    import numpy as np
    import pickle
    import plotly.express as px
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, pd, pf, pl, px, pycountry


@app.cell
def _(pl):
    def fill_column_grouped_sorted(
        lazy_df: pl.LazyFrame,
        column_to_fill: str,
        group_by_cols: list[str],
        sort_by_col: str,
    ) -> pl.LazyFrame:
        """
        Performs a forward fill then a backward fill on a specified column
        within groups, sorted by another column.

        Args:
            lazy_df: The input Polars LazyFrame.
            column_to_fill: The name of the column to apply fill operations on.
            group_by_cols: A list of column names to group by.
            sort_by_col: The name of the column to sort by before filling.

        Returns:
            A new Polars LazyFrame with the specified column filled.
        """
        temp_ffill_col_name = f"__temp_ffill_{column_to_fill}"

        lf_sorted = lazy_df.sort(sort_by_col)

        lf_ffilled = lf_sorted.with_columns(
            pl.col(column_to_fill)
            .forward_fill()
            .over(group_by_cols)
            .alias(temp_ffill_col_name)
        )

        lf_bfilled_final = lf_ffilled.with_columns(
            pl.col(temp_ffill_col_name)
            .backward_fill()
            .over(group_by_cols)
            .alias(column_to_fill)  # Overwrite original column
        )

        return lf_bfilled_final.drop(temp_ffill_col_name)
    return (fill_column_grouped_sorted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Huang-Mix Regression replication
    Replicating the Huang-Mix regression, limiting the rest-of-world to just the UK. 

    The Huang-Mix paper, [Trade Wars and Rumors of Trade Wars:
    The Dynamic Effects of the U.S.–China Tariff Hikes](https://drive.google.com/file/d/1DQABGIs2oD2wt9pMN89uppp44Gc6FIm6/view) is primarily focussed on computing the welfare impacts of tariffs as a trade policy. It focuses on the price impacts of tariffs, including various expectations and uncertainty channels. This is applied through a large general equillibrium DSGE model, including the behaviour of multiple agents. 

    We am not interested in replicated in the entirety of the paper. Instead, we are interested in just the panel regression used to inform the elasiticty parameters in the model. Where the paper calculates the redirection of trade values / volumes to the ROW, we're interested in isolating the effect to just the UK. 

    To answer, empirically, the following question:
    > What was the impact of the 2017-20 US-China trade war on UK imports from China.

    ## Steps

    ### Replicate original spec
    We initially want to replicate their original specification, using our data. This is to ensure our data is good and that we're following the correct process. For a first stab we'd do the regression at the bottom of page 7 - essentially the direct effect (in quantities) for US-China. This means we need to:

    1. replicate the input data as required for the regression.
    2. set up the regression as specified in the original paper.
    3. run the regression.
    4. extract and chart the results.

    ### Implement our spec
    1. Design the input data required for the model
    2. Specify the panel regression as we require, incl. fixed effects. 
    3. Run the regression, estimate the parameters
    4. Chart the results under a range of specifications, fixed effects, etc. 

    ## Regression Specification
    🏗️ TBD
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Replicating the original specification
    Replicating the regression as on page 7 of the paper.
    """
    )
    return


@app.cell
def _(pl):
    raw_data_path = "data/final/unified_trade_tariff_partitioned/"
    # raw_data_path = "data/final/unified_filtered_10000minval_top100countries/"
    unified_lf = pl.scan_parquet(raw_data_path)

    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
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


@app.cell
def _(cc_list_incl_ROW, fill_column_grouped_sorted, pl, unified_lf):
    ### --- 0. Initial filtering to reduce size and ffill / bfill and report only relevant countries
    filtered_lf = unified_lf.filter(
        (
            pl.col("reporter_country").is_in(cc_list_incl_ROW)
            & pl.col("partner_country").is_in(cc_list_incl_ROW),
        )
    )

    filtered_lf = fill_column_grouped_sorted(
        lazy_df=filtered_lf,
        column_to_fill="average_tariff_official",
        group_by_cols=["reporter_country", "partner_country", "product_code"],
        sort_by_col="year",
    )

    print("Input lf post forward/backward fill & filter")
    print(filtered_lf.head().collect())
    return (filtered_lf,)


@app.cell
def _(filtered_lf):
    filtered_lf.collect_schema()
    return


@app.cell
def _():
    # tariff_delta = (
    #     filtered_lf.group_by("year")
    #     .agg(
    #         (pl.col("") - pl.col("official_effective_tariff"))
    #         .mean()
    #         .alias("official_delta")
    #     )
    #     .collect()
    # )

    # tariff_delta
    return


@app.cell
def _(CHINA_CC, USA_CC, filtered_lf, go, make_subplots, pl, unified_lf):
    # OPTIONAL: Vis the effect of ffill
    test_product_code = "511219"

    vis_lf_raw = unified_lf.filter(
        (
            (pl.col("partner_country") == USA_CC)
            & (pl.col("reporter_country") == CHINA_CC)
            & (pl.col("product_code") == test_product_code)
        )
    )

    vis_lf_filtered = filtered_lf.filter(
        (
            (pl.col("partner_country") == USA_CC)
            & (pl.col("reporter_country") == CHINA_CC)
            & (pl.col("product_code") == test_product_code)
        )
    )

    df_plot_raw = vis_lf_raw.collect()
    df_plot_filtered = vis_lf_filtered.collect()

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Raw Effective Tariff",
            "Filtered (ffill) Effective Tariff",
        ),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot_raw["year"],
            y=df_plot_raw["average_tariff"],
            mode="lines+markers",
            name="Raw Tariff",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot_filtered["year"],
            y=df_plot_filtered["average_tariff_official"],
            mode="lines+markers",
            name="Filtered Tariff (ffill + bfill)",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    # Update layout properties
    fig.update_layout(
        height=700,
        title_text=f"Comparison of Raw and Filtered Tariffs<br>Product: {test_product_code}, Reporter: {CHINA_CC}, Partner: {USA_CC}",
        showlegend=False,
    )

    fig.update_yaxes(title_text="Effective Tariff", row=1, col=1)
    fig.update_yaxes(title_text="Effective Tariff", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    fig.show()
    return


@app.cell
def _(CHINA_CC, USA_CC, pl, px, unified_lf):
    # Median tariff
    vis_lf_median = (
        unified_lf.filter(
            pl.col("reporter_country") == CHINA_CC,
            pl.col("partner_country") == USA_CC,
        )
        .group_by(["year"])
        .agg(
            pl.median("average_tariff_official").alias("count_median"),
        )
        .sort("year")
    )

    px.line(vis_lf_median.collect(), x="year", y="count_median")
    return


@app.cell
def _(CHINA_CC, EFFECT_YEAR_RANGE, USA_CC, filtered_lf, pl):
    ### --- 1. Extract the input data as required
    tariff_us_china_expr = (
        pl.col("average_tariff_official")
        .filter(
            (pl.col("partner_country") == USA_CC)
            & (pl.col("reporter_country") == CHINA_CC)
        )
        .mean()  # Take the average tariff if multiple entries exist for the same product
        .over(
            ["year", "product_code"]
        )  # Apply this logic within each year and product_code group
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
    ).filter(pl.col("year").is_in(EFFECT_YEAR_RANGE))

    print("Filtered to select only required data:")
    print(input_lf.head().collect(engine="streaming"))

    ### --- 2. Create the interaction terms
    for year in EFFECT_YEAR_RANGE:
        input_lf = input_lf.with_columns(
            pl.when(
                (pl.col("year") == str(year))
                & (pl.col("importer") == USA_CC)
                & (pl.col("exporter") == CHINA_CC)
            )
            .then(pl.col("tariff_us_china"))
            .otherwise(pl.lit(0.0))
            .alias(f"tariff_interaction_{year}")
        )

    print("Created interaction term")
    print(input_lf.head().collect(engine="streaming"))

    ### --- 3. Create variables for fixed effects
    input_lf = input_lf.with_columns(
        pl.concat_str(
            [
                pl.col("importer"),
                pl.col("product_code"),
                pl.col("year"),
            ],
            separator="^",
        )
        .alias("alpha_ipt")
        .cast(pl.Categorical),
        pl.concat_str(
            [
                pl.col("exporter"),
                pl.col("product_code"),
                pl.col("year"),
            ],
            separator="^",
        )
        .alias("alpha_jpt")
        .cast(pl.Categorical),
        pl.concat_str(
            [
                pl.col("importer"),
                pl.col("exporter"),
            ],
            separator="^",
        )
        .alias("alpha_ij")
        .cast(pl.Categorical),
    )

    print("Created fixed effect terms - these are categorical")
    print(input_lf.head().collect(engine="streaming"))

    # Collecting our final input
    input_df = input_lf.collect()

    print("Final input data description:")
    input_df.describe()
    return (input_df,)


@app.cell
def _(input_df):
    # Deal with some nulls in our key data inputs
    clean_input_df = input_df.drop_nans(subset=["log_value", "tariff_us_china"])

    print("Final describe of input data")
    clean_input_df.describe()
    return (clean_input_df,)


@app.cell
def _(EFFECT_YEAR_RANGE):
    ### --- 3. Define our regression formula

    # Remember the summation term!
    regressors = " + ".join(
        [f"tariff_interaction_{year}" for year in EFFECT_YEAR_RANGE]
    )
    regression_formula = (
        f"log_value ~ {regressors} | alpha_ipt + alpha_jpt + alpha_ij"
    )

    print(f"Regression formula is:\n{regression_formula}")
    return


@app.cell
def _(clean_input_df):
    clean_input_df.head()
    return


@app.cell
def _():
    # # Default config: robust clustered std errors
    # model = pf.feols(regression_formula, clean_input_df)
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(model):
    model.fixef()
    return


@app.cell
def _(model):
    model.etable
    return


@app.cell
def _():
    # try:
    #     if model:
    #         with open(
    #             "src/mpil_tariff_trade_analysis/dev3/model_ols_v2_CM_TARFIFF.pkl",
    #             "wb",
    #         ) as file:
    #             pickle.dump(model, file)
    # except NameError:
    #     print("model not yet defined")
    return


@app.cell
def _():
    # with open(
    #     "src/mpil_tariff_trade_analysis/dev3/model_ols_v2_CM_TARIFF.pkl",
    #     "rb",
    # ) as f:
    #     model_loaded = pickle.load(f)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Interpreting results: 

    RMSE: standard deviation of the regression residuals (the differences between observed and predicted values).

    R2: X% of the variation in log_quantity is explained by the model including all the fixed effects. Commonly high in FE models given the FE explain a lot of the variation

    R2 Within: measures the proportion of the variation within the fixed effect groups (e.g., within each importer-exporter-product unit over time, after accounting for the broader FEs) that is explained by tariff interaction terms. Low value implies very little variance is in log import_quantities is explained by tariff interaction variables.

    ## Replicating Figure 2
    What does my equivalent 'figure 2' look like?
    """
    )
    return


@app.cell
def _(EFFECT_YEAR_RANGE, go, model):
    coefficients = model.coef()
    conf_intervals_beta = model.confint()

    print("Fitted coefficients:, ", coefficients)
    print("Fitted confidence intervals:, ", conf_intervals_beta)

    # Extract relevant beta values and their CIs
    tariff_vars = [f"tariff_interaction_{year}" for year in range(2018, 2024)]

    beta_s_values = abs(coefficients[tariff_vars].values)
    ci_lower_beta_s = -conf_intervals_beta.loc[tariff_vars, "2.5%"].values
    ci_upper_beta_s = -conf_intervals_beta.loc[tariff_vars, "97.5%"].values

    # Calculate Elasticity (E_s = -beta_s) and its CI
    elasticities_mean = beta_s_values
    elasticities_ci_lower = ci_lower_beta_s
    elasticities_ci_upper = ci_upper_beta_s

    ### DRAW CHART

    fig_elasticities = go.Figure()

    fig_elasticities.add_trace(
        go.Scatter(
            x=EFFECT_YEAR_RANGE,
            y=elasticities_ci_upper,
            mode="lines",
            line=dict(width=0),  # Make line invisible
            showlegend=False,
        )
    )

    fig_elasticities.add_trace(
        go.Scatter(
            x=EFFECT_YEAR_RANGE,
            y=elasticities_ci_lower,
            mode="lines",
            line=dict(width=0),  # Make line invisible
            fillcolor="rgba(31, 119, 180, 0.2)",  # Light blue with transparency
            fill="tonexty",  # Fill the area to the previously added trace (upper bound)
            showlegend=False,
            name="95% CI (filled area)",
        )
    )

    fig_elasticities.add_trace(
        go.Scatter(
            x=EFFECT_YEAR_RANGE,
            y=elasticities_mean,
            mode="lines+markers",
            name="Estimated Elasticity",
            line=dict(color="rgb(31, 119, 180)"),
            marker=dict(size=8),
        )
    )

    fig_elasticities.update_layout(
        xaxis_title="Year estimate", yaxis_title="Estimated elasticity"
    )

    fig_elasticities.show()
    return


@app.cell
def _(model):
    model.coefplot()
    return


@app.cell
def _(CHINA_CC, USA_CC, pl, px, unified_lf):
    # Just plot US's China imports over time

    us_china_trade_through_time = unified_lf.filter(
        pl.col("reporter_country") == CHINA_CC,
        pl.col("partner_country") == USA_CC,
    )

    us_china_trade_through_time = (
        us_china_trade_through_time.group_by(["year"])
        .agg(pl.mean("value"))
        .sort("year")
    ).collect()

    px.line(us_china_trade_through_time, x="year", y="value")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Regression 2 - Section 3.2
    Replicating the second regression (identifying indirect effects on ROW)

    ## Filtering country list
    In the paper, we have the following:
    > Next, we document third-party countries’ imports from the U.S. and China between 2012 and 2022, with a focus on the top 30 importers. We rank countries by their total goods import values in 2017. Similar to Fajgelbaum et al (2023), we exclude oil economies such as Russia and Saudi Arabia.

    Ok - need to implement that
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Regression 2 helper functions:

    A) Get oil exporters
    B) Prepare the dataframe
    """
    )
    return


@app.cell
def _(CHINA_CC, USA_CC, pl):
    def eq2_filtering_flexible(
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
            df.filter(pl.col("year").is_in(relevant_years))
            .filter(pl.col("partner_country").is_in(top_30_importers))
            .filter(pl.col("value") > 0)
        )

        # 2. Reshape data
        agg_expressions = [
            pl.col("value")
            .filter(pl.col("year") == pl.lit(year))
            .first()
            .alias(f"val_{year}")
            for year in relevant_years
        ]
        reshaped_lf = unified_lf_without_oil_filtered.group_by(
            ["reporter_country", "partner_country", "product_code"]
        ).agg(agg_expressions)

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
                y=(
                    100
                    * (
                        pl.col(val_p1_end_col).log()
                        - pl.col(val_p1_start_col).log()
                    )
                    / period1_duration
                ),
                PostEvent=pl.lit(0).cast(
                    pl.Int8
                ),  # Renamed from Post2017 for generality
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
                y=(
                    100
                    * (
                        pl.col(val_p2_end_col).log()
                        - pl.col(val_p2_start_col).log()
                    )
                    / period2_duration
                ),
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
        regression_input_lf = pl.concat(
            [period1_growth_lf, period2_growth_lf], how="vertical"
        )

        # 6. Add exporter dummies
        regression_input_lf = regression_input_lf.with_columns(
            [
                pl.when(pl.col("reporter_country") == pl.lit(CHINA_CC))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("is_CHN_exporter"),
                pl.when(pl.col("reporter_country") == pl.lit(USA_CC))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("is_USA_exporter"),
            ]
        )

        regression_df = regression_input_lf.collect()

        # Drop rows with NaN/inf in 'y'
        regression_df = regression_df.drop_nulls(subset=["y"])
        regression_df = regression_df.filter(pl.col("y").is_finite())

        return regression_df
    return (eq2_filtering_flexible,)


@app.cell
def _(pl):
    def get_oil_exporting_countries(
        lzdf: pl.LazyFrame, oil_export_percentage_threshold: float
    ) -> list[str]:
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
            raise ValueError(
                "oil_export_percentage_threshold must be between 0 and 100."
            )

        # Calculate total export value for each country
        total_exports_by_country = lzdf.group_by("reporter_country").agg(
            pl.sum("value").alias("total_value")
        )

        # Calculate total oil export value for each country
        # HS codes for oil and mineral fuels are under Chapter 27.
        oil_exports_by_country = (
            lzdf.filter(pl.col("product_code").str.starts_with("27"))
            .group_by("reporter_country")
            .agg(pl.sum("value").alias("oil_value"))
        )

        # Join total exports with oil exports
        country_export_summary = total_exports_by_country.join(
            oil_exports_by_country,
            on="reporter_country",
            how="left",  # Use left join to keep all countries, oil_value will be null if no oil exports
        ).with_columns(
            pl.col("oil_value").fill_null(
                0.0
            )  # Fill nulls with 0 for countries with no oil exports
        )

        # Calculate the percentage of oil exports
        country_export_summary = country_export_summary.with_columns(
            ((pl.col("oil_value") / pl.col("total_value")) * 100).alias(
                "oil_export_percentage"
            )
        )

        # Filter countries above the threshold
        filtered_countries = (
            country_export_summary.filter(
                pl.col("oil_export_percentage") > oil_export_percentage_threshold
            )
            .select("reporter_country")
            .collect()  # Collect the results into a DataFrame
        )

        return filtered_countries["reporter_country"].to_list()
    return (get_oil_exporting_countries,)


@app.cell
def _(get_oil_exporting_countries, pl, unified_lf):
    # Identify oil countries -> those with >40% of their exports in hs code 27
    oil_country_list = get_oil_exporting_countries(unified_lf, 40)

    without_oil_unified_lf = unified_lf.filter(
        ~pl.col("reporter_country").is_in(oil_country_list)
    )
    return (without_oil_unified_lf,)


@app.cell
def _(pl, without_oil_unified_lf):
    # Get country import values in 2017, sorted by value
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
    top_30_importers = (
        country_trade_values_2017.sort(by="value", descending=True)
        .select("partner_country")
        .to_series()
        .to_list()[:29]
    )
    return (top_30_importers,)


@app.cell
def _(eq2_filtering_flexible, pf, top_30_importers, without_oil_unified_lf):
    regression_df_0 = eq2_filtering_flexible(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2016",
        period2_start_year="2017",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
    )
    model_eq2_v0 = pf.feols(
        fml="y ~ is_CHN_exporter:PostEvent + is_USA_exporter:PostEvent | reporter_country + C(PostEvent)",
        data=regression_df_0,
        vcov="hetero",
    )
    model_eq2_v0.summary()
    return


@app.cell
def _(eq2_filtering_flexible, pf, top_30_importers, without_oil_unified_lf):
    regression_df_v1 = eq2_filtering_flexible(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2023",
        top_30_importers=top_30_importers,
    )
    model_eq2_v1 = pf.feols(
        fml="y ~ is_CHN_exporter:PostEvent + is_USA_exporter:PostEvent | reporter_country + C(PostEvent)",
        data=regression_df_v1,
        vcov="hetero",
    )
    model_eq2_v1.summary()
    return


@app.cell
def _(eq2_filtering_flexible, pf, top_30_importers, without_oil_unified_lf):
    regression_df_v2 = eq2_filtering_flexible(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
    )
    model_eq2_v2 = pf.feols(
        fml="y ~ is_CHN_exporter:PostEvent + is_USA_exporter:PostEvent | reporter_country + C(PostEvent)",
        data=regression_df_v2,
        vcov="hetero",
    )
    model_eq2_v2.summary()
    return


@app.cell
def _(mo):
    mo.md(r"""# Modify eq2 for UK and other countries""")
    return


@app.cell(hide_code=True)
def _(CHINA_CC, USA_CC, pl, pycountry):
    def eq2_filtering_flexible_THIRDCOUNTRY(
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
            raise ValueError(
                f"Country to isolate '{country_to_isolate}' is not in the top_30_importers list."
            )

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
            df.filter(pl.col("year").is_in(relevant_years))
            .filter(pl.col("partner_country").is_in(top_30_importers))
            .filter(pl.col("value") > 0)
        )

        # 2. Reshape data
        agg_expressions = [
            pl.col("value")
            .filter(pl.col("year") == pl.lit(year))
            .first()
            .alias(f"val_{year}")
            for year in relevant_years
        ]
        reshaped_lf = unified_lf_without_oil_filtered.group_by(
            ["reporter_country", "partner_country", "product_code"]
        ).agg(agg_expressions)

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
                y=(
                    100
                    * (
                        pl.col(val_p1_end_col).log()
                        - pl.col(val_p1_start_col).log()
                    )
                    / period1_duration
                ),
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
                y=(
                    100
                    * (
                        pl.col(val_p2_end_col).log()
                        - pl.col(val_p2_start_col).log()
                    )
                    / period2_duration
                ),
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
        regression_input_lf = pl.concat(
            [period1_growth_lf, period2_growth_lf], how="vertical"
        )

        # 6. Add exporter dummies based on the country_to_isolate
        # CHN and USA are now hardcoded
        regression_input_lf = regression_input_lf.with_columns(
            [
                pl.when(
                    (pl.col("reporter_country") == pl.lit(CHINA_CC))
                    & (pl.col("partner_country") == pl.lit(country_to_isolate))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(
                    f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_to_isolate}_importer"
                ),
                pl.when(
                    (pl.col("reporter_country") == pl.lit(USA_CC))
                    & (pl.col("partner_country") == pl.lit(country_to_isolate))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(
                    f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_to_isolate}_importer"
                ),
                pl.when(
                    (pl.col("reporter_country") == pl.lit(CHINA_CC))
                    & (pl.col("partner_country") != pl.lit(country_to_isolate))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(
                    f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_to_isolate}_importer"
                ),
                pl.when(
                    (pl.col("reporter_country") == pl.lit(USA_CC))
                    & (pl.col("partner_country") != pl.lit(country_to_isolate))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(
                    f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_to_isolate}_importer"
                ),
            ]
        )

        regression_df = regression_input_lf.collect()

        # Drop rows with NaN/inf in 'y'
        regression_df = regression_df.drop_nulls(subset=["y"])
        regression_df = regression_df.filter(pl.col("y").is_finite())

        return regression_df
    return (eq2_filtering_flexible_THIRDCOUNTRY,)


@app.cell
def _(
    CHINA_CC,
    UK_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    country_of_interest = UK_CC

    regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
        without_oil_unified_lf,
        period1_start_year="2012",
        period1_end_year="2017",
        period2_start_year="2018",
        period2_end_year="2022",
        top_30_importers=top_30_importers,
        country_to_isolate=country_of_interest,
    )

    formula = (
        f"y ~ "
        f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
        f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
        f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
        f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
        f"| reporter_country + C(PostEvent) + partner_country"
    )


    model_eq2MOD_v0 = pf.feols(
        fml=formula,
        data=regression_df_MOD_v0,
        vcov="hetero",
    )

    print(" --- Running model, isolating the UK --- ")
    print(model_eq2MOD_v0.summary())
    model_eq2MOD_v0.coefplot()
    return


@app.cell
def _(
    CHINA_CC,
    GERMANY_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    def _():
        country_of_interest = GERMANY_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating GERMANY --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(
    CHINA_CC,
    FRANCE_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    def _():
        country_of_interest = FRANCE_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating France --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(
    CHINA_CC,
    ITALY_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    def _():
        country_of_interest = ITALY_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating Italy --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(
    CHINA_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    JAPAN_CC = pycountry.countries.search_fuzzy("Japan")[0].numeric


    def _():
        country_of_interest = JAPAN_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating Japan --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(
    CHINA_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    AUSTRALIA_CC = pycountry.countries.search_fuzzy("Australia")[0].numeric


    def _():
        country_of_interest = AUSTRALIA_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating Australia --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(
    CHINA_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    SINGAPORE_CC = pycountry.countries.search_fuzzy("Singapore")[0].numeric


    def _():
        country_of_interest = SINGAPORE_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating Singapore --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(
    CHINA_CC,
    USA_CC,
    eq2_filtering_flexible_THIRDCOUNTRY,
    pf,
    pycountry,
    top_30_importers,
    without_oil_unified_lf,
):
    CANADA_CC = pycountry.countries.search_fuzzy("CANADA")[0].numeric


    def _():
        country_of_interest = CANADA_CC

        regression_df_MOD_v0 = eq2_filtering_flexible_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_30_importers,
            country_to_isolate=country_of_interest,
        )

        formula = (
            f"y ~ "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=CHINA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent + "
            f"is_{pycountry.countries.get(numeric=USA_CC).alpha_3}_exporter_non{country_of_interest}_importer:PostEvent "
            f"| reporter_country + C(PostEvent) + partner_country"
        )

        model_eq2MOD_v0 = pf.feols(
            fml=formula,
            data=regression_df_MOD_v0,
            vcov="hetero",
        )

        print(" --- Running model, isolating Singapore --- ")
        print(model_eq2MOD_v0.summary())
        return model_eq2MOD_v0.coefplot()


    _()
    return


@app.cell
def _(pycountry, top_30_importers):
    for cc in top_30_importers:
        print(pycountry.countries.get(numeric=cc))
    return


@app.cell
def _(mo):
    mo.md(
        r"""## Run equation 2 again, but filtering for only targeted products (i.e. those which had a tariff applied)"""
    )
    return


@app.cell(hide_code=True)
def _(
    CHINA_CC,
    USA_CC,
    pd,
    pf,
    pl,
    regression_df,
    top_30_importers,
    without_oil_unified_lf,
):
    def eq2_filtering(df):
        # 1. Initial Filtering
        # Filter for relevant years (2012-2022 for context, but specifically 2012, 2016, 2017, 2022 for pivot)
        # Filter for top 30 importers

        unified_lf_without_oil_filtered = (
            df.filter(pl.col("year").is_in(["2012", "2016", "2017", "2022"]))
            .filter(pl.col("partner_country").is_in(top_30_importers))
            .filter(
                pl.col("value") > 0  # Filter out non-positive values early
            )
        )

        # 2. Reshape data to have year-specific values in columns using group_by and agg
        reshaped_lf = unified_lf_without_oil_filtered.group_by(
            ["reporter_country", "partner_country", "product_code"]
        ).agg(
            [
                pl.col("value")
                .filter(pl.col("year") == pl.lit("2012"))
                .first()
                .alias("val_2012"),
                pl.col("value")
                .filter(pl.col("year") == pl.lit("2016"))
                .first()
                .alias("val_2016"),
                pl.col("value")
                .filter(pl.col("year") == pl.lit("2017"))
                .first()
                .alias("val_2017"),
                pl.col("value")
                .filter(pl.col("year") == pl.lit("2022"))
                .first()
                .alias("val_2022"),
            ]
        )

        # 3. Calculate growth rates for Period 1 (2012-2016)
        period1_growth_lf = (
            reshaped_lf.filter(
                pl.col("val_2012").is_not_null()
                & (
                    pl.col("val_2012") > 0
                )  # Ensure positivity again after aggregation
                & pl.col("val_2016").is_not_null()
                & (pl.col("val_2016") > 0)
            )
            .with_columns(
                y=100 * (pl.col("val_2016").log() - pl.col("val_2012").log()) / 4,
                Post2017=pl.lit(0).cast(pl.Int8),
            )
            .select(
                [
                    "reporter_country",
                    "partner_country",
                    "product_code",
                    "y",
                    "Post2017",
                ]
            )
        )
        print(period1_growth_lf.head().collect())

        # 4. Calculate growth rates for Period 2 (2017-2022)
        period2_growth_lf = (
            reshaped_lf.filter(
                pl.col("val_2017").is_not_null()
                & (pl.col("val_2017") > 0)
                & pl.col("val_2022").is_not_null()
                & (pl.col("val_2022") > 0)
            )
            .with_columns(
                y=100 * (pl.col("val_2022").log() - pl.col("val_2017").log()) / 5,
                Post2017=pl.lit(1).cast(pl.Int8),
            )
            .select(
                [
                    "reporter_country",
                    "partner_country",
                    "product_code",
                    "y",
                    "Post2017",
                ]
            )
        )

        # 5. Combine the two periods
        regression_input_lf = pl.concat(
            [period1_growth_lf, period2_growth_lf], how="vertical"
        )

        # 6. Add exporter dummies
        regression_input_lf = regression_input_lf.with_columns(
            [
                pl.when(pl.col("reporter_country") == pl.lit(CHINA_CC))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("is_CHN_exporter"),
                pl.when(pl.col("reporter_country") == pl.lit(USA_CC))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("is_USA_exporter"),
            ]
        )

        regression_df = regression_input_lf.collect()

        # Drop rows with NaN/inf in 'y'
        regression_df = regression_df.drop_nulls(subset=["y"])
        regression_df = regression_df.filter(pl.col("y").is_finite())
        return regression_df


    cm_us_tariffs = pd.read_csv("data/intermediate/carter_mix_hs6_tariffs.csv")

    without_oil_unified_lf_only_targeted = without_oil_unified_lf.filter(
        pl.col("product_code").is_in(
            cm_us_tariffs["product_code"].astype(str).to_list()
        )
    )

    eq2_spec2_regression_df = eq2_filtering(without_oil_unified_lf_only_targeted)

    model_eq2_spec2 = pf.feols(
        fml="y ~ is_CHN_exporter:Post2017 + is_USA_exporter:Post2017 | reporter_country + C(Post2017)",
        data=regression_df,
        vcov="hetero",
    )
    model_eq2_spec2.summary()
    return (
        cm_us_tariffs,
        eq2_filtering,
        eq2_spec2_regression_df,
        model_eq2_spec2,
    )


@app.cell
def _(model_eq2_spec2):
    model_eq2_spec2.coefplot()
    return


@app.cell
def _(eq2_spec2_regression_df):
    eq2_spec2_regression_df
    return


@app.cell
def _():
    # print(without_oil_unified_lf_only_targeted.select(pl.len()).collect())
    # print(without_oil_unified_lf.select(pl.len()).collect())
    return


@app.cell
def _(mo):
    mo.md(
        r"""## Run equation 2 again, but filtering for only non-targeted products (i.e. those which didnt have a tariff applied)"""
    )
    return


@app.cell
def _(cm_us_tariffs, eq2_filtering, pf, pl, without_oil_unified_lf):
    non_targeted = without_oil_unified_lf.filter(
        ~pl.col("product_code").is_in(
            cm_us_tariffs["product_code"].astype(str).to_list()
        )
    )

    regression_df_nontariffed = eq2_filtering(non_targeted)


    model_eq2_spec3 = pf.feols(
        fml="y ~ is_CHN_exporter:Post2017 + is_USA_exporter:Post2017 | reporter_country + C(Post2017)",
        data=regression_df_nontariffed,
        vcov="hetero",
    )
    model_eq2_spec3.summary()
    return model_eq2_spec3, regression_df_nontariffed


@app.cell
def _(model_eq2_spec3):
    model_eq2_spec3.coefplot()
    return


@app.cell
def _(regression_df_nontariffed):
    regression_df_nontariffed
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Why does this look wrong?

    Questions:

    - Why is my R^2 so low?
    - Why is my R2 within so low?
    - Why is my RMSE so high?
    """
    )
    return


@app.cell
def _(CHINA_CC, USA_CC, pl, px, unified_lf):
    # Chart of Imports from China by RoW and from USA by RoW

    imports_from_china = (
        (
            unified_lf.filter(
                pl.col("reporter_country") == CHINA_CC,
                pl.col("partner_country") != USA_CC,
            )
            .group_by(["year"])
            .agg(pl.mean("value"))
        )
        .collect()
        .sort("year")
    )

    px.bar(imports_from_china, x="year", y="value")
    return


@app.cell
def _(CHINA_CC, USA_CC, pl, px, unified_lf):
    imports_from_USA = (
        (
            unified_lf.filter(
                pl.col("reporter_country") == USA_CC,
                pl.col("partner_country") != CHINA_CC,
            )
            .group_by(["year"])
            .agg(pl.mean("value"))
        )
        .collect()
        .sort("year")
    )

    px.bar(imports_from_USA, x="year", y="value")
    return


if __name__ == "__main__":
    app.run()
