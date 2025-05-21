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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, pf, pickle, pl, pycountry


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

    cc_list_incl_ROW = [
        USA_CC,
        CHINA_CC,
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        UK_CC,
    ]

    cc_list = [USA_CC, CHINA_CC]

    TOTAL_YEAR_RANGE = [str(y) for y in range(1999, 2024)]
    EFFECT_YEAR_RANGE = [str(y) for y in range(2018, 2024)]
    return CHINA_CC, EFFECT_YEAR_RANGE, USA_CC, cc_list_incl_ROW


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
        column_to_fill="official_effective_tariff",
        group_by_cols=["reporter_country", "partner_country", "product_code"],
        sort_by_col="year",
    )

    print("Input lf post forward/backward fill & filter")
    print(filtered_lf.head().collect())
    return (filtered_lf,)


@app.cell
def _(CHINA_CC, USA_CC, filtered_lf, go, make_subplots, pl, unified_lf):
    # OPTIONAL: Vis the effect of ffill
    test_product_code = "911180"

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
            y=df_plot_raw["official_effective_tariff"],
            mode="lines+markers",
            name="Raw Tariff",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot_filtered["year"],
            y=df_plot_filtered["official_effective_tariff"],
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
def _(CHINA_CC, EFFECT_YEAR_RANGE, USA_CC, filtered_lf, pl):
    ### --- 1. Extract the input data as required
    tariff_us_china_expr = (
        pl.col("official_effective_tariff")
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
    return (regression_formula,)


@app.cell
def _(clean_input_df):
    clean_input_df.head()
    return


@app.cell
def _(clean_input_df, pf, regression_formula):
    # Default config: robust clustered std errors
    # pyfixest is a wrapper of R's fixest package.
    # This takes about 17m to run.
    model = pf.feols(regression_formula, clean_input_df)
    return (model,)


@app.cell
def _(model, pickle):
    try:
        if model:
            with open(
                "src/mpil_tariff_trade_analysis/dev3/model_ols_v2_OFFICIAL_INCLUDED.pkl",
                "wb",
            ) as file:
                pickle.dump(model, file)
    except NameError:
        print("model not yet defined")
    return


@app.cell
def _(pickle):
    with open(
        "src/mpil_tariff_trade_analysis/dev3/model_ols_v2_OFFICIAL_INCLUDED.pkl",
        "rb",
    ) as f:
        model_loaded = pickle.load(f)
    return


@app.cell
def _(model):
    model.summary()
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
def _(model):
    coefficients = model.coef()
    conf_intervals_beta = model.confint()

    print("Fitted coefficients:, ", coefficients)
    print("Fitted confidence intervals:, ", conf_intervals_beta)

    # Extract relevant beta values and their CIs
    tariff_vars = [f"tariff_interaction_{year}" for year in range(2018, 2023)]

    beta_s_values = coefficients[tariff_vars].values
    ci_lower_beta_s = conf_intervals_beta.loc[tariff_vars, "2.5%"].values
    ci_upper_beta_s = conf_intervals_beta.loc[tariff_vars, "97.5%"].values

    # Calculate Elasticity (E_s = -beta_s) and its CI
    elasticities_mean = -beta_s_values
    elasticities_ci_lower = -ci_lower_beta_s
    elasticities_ci_upper = -ci_upper_beta_s
    return elasticities_ci_lower, elasticities_ci_upper, elasticities_mean


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    elasticities_ci_lower,
    elasticities_ci_upper,
    elasticities_mean,
    go,
):
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


if __name__ == "__main__":
    app.run()
