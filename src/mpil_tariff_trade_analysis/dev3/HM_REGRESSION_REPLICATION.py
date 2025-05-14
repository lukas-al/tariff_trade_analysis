import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pycountry
    import pyfixest as pf
    return mo, pf, pl, pycountry


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
    # raw_data_path = "data/final/unified_trade_tariff_partitioned/"
    raw_data_path = "data/final/unified_filtered_10000minval_top100countries/"
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
    return CHINA_CC, USA_CC


@app.cell
def _(unified_lf):
    unique_years = unified_lf.select("year").unique().collect()["year"].to_list()
    return


@app.cell
def _(CHINA_CC, USA_CC, pl, unified_lf):
    ### --- 1. Extract the input data as required
    tariff_us_china_expr = (
        pl.col("effective_tariff")
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

    input_lf = unified_lf.select(
        pl.col("year"),
        pl.col("partner_country").alias("importer"),
        pl.col("reporter_country").alias("exporter"),
        pl.col("product_code"),
        pl.col("value").log().alias("log_value"),
        pl.col("quantity").log().alias("log_quantity"),
        tariff_us_china_expr,
    )

    print("Filtered to select only required data:")
    print(input_lf.head().collect(engine="streaming"))

    ### --- 2. Create the interaction terms
    for year in range(2018, 2023):
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
        ).alias("alpha_ipt"),
        pl.concat_str(
            [
                pl.col("exporter"),
                pl.col("product_code"),
                pl.col("year"),
            ],
            separator="^",
        ).alias("alpha_jpt"),
        pl.concat_str(
            [
                pl.col("importer"),
                pl.col("exporter"),
            ],
            separator="^",
        ).alias("alpha_ij"),
    )

    print("Created fixed effect terms - these are categorical")
    print(input_lf.head().collect(engine="streaming"))

    # Collecting our final input
    input_df = input_lf.collect()

    print("Final input data description:")
    print(input_df.describe())
    return (input_df,)


@app.cell
def _():
    ### --- 3. Define our regression formula

    # Remember the summation term!
    regressors = " + ".join(
        [f"tariff_interaction_{year}" for year in range(2018, 2023)]
    )
    regression_formula = (
        f"log_quantity ~ {regressors} | alpha_ipt + alpha_jpt + alpha_ij"
    )

    print(f"Regression formula is:\n{regression_formula}")
    return (regression_formula,)


@app.cell
def _(input_df, pf, regression_formula):
    model = pf.feols(regression_formula, input_df) # This takes forever
    return (model,)


@app.cell
def _(model):
    model.summary()
    return


if __name__ == "__main__":
    app.run()
