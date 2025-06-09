import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import pycountry
    import pyfixest
    import pickle
    import statsmodels.api as sm
    import plotly.express as px
    return mo, pd, pl, px, pycountry, pyfixest, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Panel Regressions

    1. Run the equation 2 regressions from HM across all countries, showing how the UK's imports changed over time relative to other countries
    2. Update the direct effect equation to consider more carefully the impact on the UK.
    3. Experiment with some alternative specifications for this equation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Helper functions and config""")
    return


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
def _(CHINA_CC, USA_CC, pl, pycountry):
    def prepare_regression_data_THIRDCOUNTRY(
        df: pl.LazyFrame,
        country_to_isolate: str,  # The country code of the partner_country to isolate
        period1_start_year: str,
        period1_end_year: str,
        period2_start_year: str,
        period2_end_year: str,
        top_importers: list,
    ) -> pl.DataFrame:
        """
        Prep the dataframe for the equation 2 regression, modified to isolate
        the effect on specific third countries as well.
        """
        # Check if the country to isolate is in the top_importers list
        if country_to_isolate not in top_importers:
            raise ValueError(
                f"Country to isolate '{country_to_isolate}' is not in the top_importers list."
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
            .filter(pl.col("partner_country").is_in(top_importers))
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
    return (prepare_regression_data_THIRDCOUNTRY,)


@app.cell
def _(CHINA_CC, USA_CC, pycountry, pyfixest):
    def run_regression_THIRDCOUNTRY(data, country_of_interest_code):
        """
        Run the fixest ols regression on the prepared data for a specific third country.
        """
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

        print(
            f" --- Running model, isolating {pycountry.countries.get(numeric=country_of_interest_code).name} --- "
        )

        return (model_eq2MOD_v0.tidy(), model_eq2MOD_v0.coefplot(), model_eq2MOD_v0)
    return (run_regression_THIRDCOUNTRY,)


@app.cell
def _(
    prepare_regression_data_THIRDCOUNTRY,
    run_regression_THIRDCOUNTRY,
    top_importers,
    without_oil_unified_lf,
):
    def run_regression_for_country(country):
        """
        Wrapper function to run the regression for a specific country.
        """
        country_code = country.numeric
        print(f"Running regression for {country.name} ({country.alpha_3})")

        # Prepare the regression data
        regression_data = prepare_regression_data_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_importers=top_importers,
            country_to_isolate=country_code,
        )

        # Run the regression
        summary, coefplot, model = run_regression_THIRDCOUNTRY(
            regression_data, country_code
        )

        return summary, coefplot, model
    return (run_regression_for_country,)


@app.cell
def _(pycountry):
    USA_CC = pycountry.countries.search_fuzzy("USA")[0].numeric
    CHINA_CC = pycountry.countries.search_fuzzy("China")[0].numeric
    UK_CC = pycountry.countries.search_fuzzy("United Kingdom")[0].numeric

    TOTAL_YEAR_RANGE = [str(y) for y in range(1999, 2024)]
    EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]
    return CHINA_CC, EFFECT_YEAR_RANGE, UK_CC, USA_CC


@app.cell
def _(mo):
    mo.md(r"""## Load data and filter it down""")
    return


@app.cell
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")
    return (unified_lf,)


@app.cell
def _(get_oil_exporting_countries, pl, pycountry, unified_lf):
    # Identify oil countries -> those with >50% of their exports in hs code 27
    oil_country_list = get_oil_exporting_countries(unified_lf, 50)

    without_oil_unified_lf = unified_lf.filter(
        ~pl.col("reporter_country").is_in(oil_country_list)
    )

    oil_country_list_pycountries = [
        pycountry.countries.get(numeric=country)
        for country in oil_country_list
        if pycountry.countries.get(numeric=country)
    ]
    return (without_oil_unified_lf,)


@app.cell
def _(pl, without_oil_unified_lf):
    ### --- TOP IMPORTERS ONLY IN 2017 ---
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
    top_importers = (
        country_trade_values_2017.sort(by="value", descending=True)
        .select("partner_country")
        .to_series()
        .to_list()[:30]  # Add one more - to allow us to remove Russia.
    )
    top_importers.remove(
        "643"
    )  # Remove Russia, which was sanctioned due to Ukraine war and so skews the numbers
    print(top_importers)
    return (top_importers,)


@app.cell
def _(pl, without_oil_unified_lf):
    ### --- TOP COUNTRIES FOR IMPORTS AND EXPORTS 2017 ---
    # Filter for trade data from the year 2017.
    trade_2017_lf = without_oil_unified_lf.filter(pl.col("year") == "2017")

    # Create two views of the data: one for exports and one for imports.
    exports_lf = trade_2017_lf.select(
        pl.col("reporter_country").alias("country"), pl.col("value")
    )
    imports_lf = trade_2017_lf.select(
        pl.col("partner_country").alias("country"), pl.col("value")
    )

    # Concatenate the two views to get all trade flows associated with a country.
    all_trade_lf = pl.concat([exports_lf, imports_lf])

    # Calculate total trade for each country and get the top N.
    top_total_trade = (
        all_trade_lf.group_by("country")
        .agg(pl.sum("value").alias("total_trade"))
        .sort("total_trade", descending=True)
        .head(45)
        .collect()
    )

    # Extract the list of country codes.
    top_countries_list = top_total_trade.get_column("country").to_list()

    # Optionally, remove a specific country, e.g., Russia ('643').
    if "643" in top_countries_list:
        top_countries_list.remove("643")

    print(top_countries_list)
    return (top_countries_list,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Equation 2, modified for 3rd country

    Run over every country in the sample (top N by imports in 2017). Plot the estimated coefficients on a scatter. View some countries' results via an etable.
    """
    )
    return


@app.cell
def _(
    CHINA_CC,
    USA_CC,
    mo,
    pd,
    pycountry,
    run_regression_for_country,
    top_importers,
):
    us_export_growth = {}
    china_export_growth = {}

    results = []
    model_objects = {}
    for country_code in mo.status.progress_bar(top_importers):
        if country_code in [CHINA_CC, USA_CC]:
            continue

        country = pycountry.countries.get(numeric=country_code)
        summary, _, model = run_regression_for_country(country)

        results.append(
            {
                "country": country_code,
                "china_export_growth": summary.iloc[0]["Estimate"],
                "us_export_growth": summary.iloc[1]["Estimate"],
                "country_name": pycountry.countries.get(numeric=country_code).name,
            }
        )

        model_objects[country.name] = model

    df = pd.DataFrame(results)
    return df, model_objects


@app.cell
def _(df, px):
    fig_scatter = px.scatter(
        df,
        x="china_export_growth",
        y="us_export_growth",
        hover_data=["country", "country_name"],
        color="country_name",
        # trendline="ols",
        title="Country import growth from China (X) and USA (Y)",
    )
    fig_scatter
    return


@app.cell
def _(df, px):
    fig_scatter_2 = px.scatter(
        df,
        x="china_export_growth",
        y="us_export_growth",
        hover_data=["country", "country_name"],
        # color="country_name",
        trendline="ols",
        title="Country import growth from China (X) and USA (Y) with OLS",
    )
    fig_scatter_2
    return (fig_scatter_2,)


@app.cell
def _(df, fig_scatter_2, px, sm):
    ols_results = px.get_trendline_results(fig_scatter_2).iloc[0]["px_fit_results"]

    # Calculate Y-axis (vertical) residuals
    df["y_residuals"] = ols_results.resid

    # To get X-axis (horizontal) residuals, we fit a new model swapping X and Y
    x_model = sm.OLS(
        df["china_export_growth"], sm.add_constant(df["us_export_growth"])
    ).fit()
    df["x_residuals"] = x_model.resid

    # Plot Y-axis residuals
    fig_residuals_y = px.scatter(
        df,
        x="china_export_growth",
        y="y_residuals",
        hover_data=["country_name"],
        color="country_name",
        title="Residuals of US Export Growth",
    )
    fig_residuals_y.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuals_y.show()

    # Plot X-axis residuals
    fig_residuals_x = px.scatter(
        df,
        x="x_residuals",
        y="us_export_growth",
        hover_data=["country_name"],
        color="country_name",
        title="Residuals of China Export Growth",
    )
    fig_residuals_x.add_vline(x=0, line_dash="dash", line_color="red")
    fig_residuals_x
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The above is basically, how much did their trend growth in US imports, Chinese imports change after the tariff shock. More specifically - how different is this from the basket of comparable economies and the linear relationship of those countries.

    So the further away from the trend line the country is, the larger it's growth in imports from the respective country was, relative to what we'd expect based on the basket of countries here.

    Conclusions - the UK weakened more than we might expect on imports from the US, and grew Chinese imports relatively more than most comparable EU countries, but less than the global average (in essence.)

    There's probably a more elegant way to assess this. And I wonder how much of this is just fast growth economies / more gravity variables.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Below is a pretty big table of all the regressions. Scroll through as desired...""")
    return


@app.cell
def _(model_objects, pyfixest):
    # Take a look at the etable for the countries
    model_list = [model_obj for model_obj in model_objects.values()]
    model_heads = model_objects.keys()
    pyfixest.etable(
        model_list,
        model_heads=model_heads,
        file_name="notebook_outputs/panel_regressions/regression_A_etable.html",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Regression B: New specification of the equation

    A) Modifying the direct effect regression to point at the UK explicitly

    B) Running some alternative specifications (on fixed effects)

    - Use the direct effect regression
    - Change the i=USA in the dummy to be i=RoW
    - Tau stays the same
    - Then run again, with i=UK
    - Keep all the same effects
    - Then run with the jpt fixed effect

    > Keep the US-China tariff (τ USA,CHN,p,s) as the "treatment" variable but apply this treatment to different groups of observations to measure spillover effects.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Prep the data""")
    return


@app.cell
def _(pl, top_countries_list, unified_lf):
    filtered_lf_newspec = unified_lf.filter(
        (
            pl.col("reporter_country").is_in(top_countries_list)
            & pl.col("partner_country").is_in(top_countries_list),
        )
    )
    return (filtered_lf_newspec,)


@app.cell
def _(CHINA_CC, EFFECT_YEAR_RANGE, UK_CC, USA_CC, filtered_lf_newspec, pl):
    ### --- INPUT DATAFRAME FOR SUBSEQUENT REGRESSION ---
    # First identify the US-China tariffs
    tariff_us_china_expr = (
        pl.col("average_tariff_official")
        .filter(
            (pl.col("partner_country") == USA_CC)
            & (pl.col("reporter_country") == CHINA_CC)
        )
        .mean()
        .over(["year", "product_code"])
        .alias("tariff_us_china")
    )

    # Initial filtering
    input_lf = filtered_lf_newspec.select(
        pl.col("year"),
        pl.col("partner_country").alias("importer"),
        pl.col("reporter_country").alias("exporter"),
        pl.col("product_code"),
        # pl.col("value").log().alias("log_value"),
        # pl.col("quantity").log().alias("log_quantity"),
        pl.col("value"),
        pl.col("quantity"),
        tariff_us_china_expr,
    ).filter(pl.col("year").is_in(list(EFFECT_YEAR_RANGE)))

    # Create our interaction expressions - the core of this
    interaction_expressions_US = [
        pl.when(
            (pl.col("year") == str(year))
            & (pl.col("importer") == USA_CC)
            & (pl.col("exporter") == CHINA_CC)
        )
        .then(pl.col("tariff_us_china"))
        .otherwise(0.0)
        .alias(f"tariff_interaction_US_{year}")
        for year in EFFECT_YEAR_RANGE
    ]

    interaction_expressions_ROW = [
        pl.when(
            (pl.col("importer") != USA_CC)  # Key change here
            & (pl.col("exporter") == CHINA_CC)
            & (pl.col("year") == str(year))
        )
        .then(pl.col("tariff_us_china"))
        .otherwise(0.0)
        .alias(f"tariff_interaction_ROW_{year}")
        for year in EFFECT_YEAR_RANGE
    ]

    interaction_expressions_UK = [
        pl.when(
            (pl.col("importer") == UK_CC)  # Key change here
            & (pl.col("exporter") == CHINA_CC)
            & (pl.col("year") == str(year))
        )
        .then(pl.col("tariff_us_china"))
        .otherwise(0.0)
        .alias(f"tariff_interaction_UK_{year}")
        for year in EFFECT_YEAR_RANGE
    ]
    return input_lf, interaction_expressions_UK, interaction_expressions_US


@app.cell
def _(input_lf, interaction_expressions_UK, interaction_expressions_US):
    final_lf_US = input_lf.with_columns(
        *interaction_expressions_US,
    )
    clean_input_df_US = final_lf_US.drop_nans(
        subset=["value", "tariff_us_china"]
    ).collect()

    # final_lf_ROW = input_lf.with_columns(
    #     *interaction_expressions_ROW,
    # )

    final_lf_UK = input_lf.with_columns(
        *interaction_expressions_UK,
    )
    clean_input_df_UK = final_lf_UK.drop_nans(
        subset=["value", "tariff_us_china"]
    ).collect()
    return clean_input_df_UK, clean_input_df_US


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Run US regression

    Crashes in notebook mode - so assume we've run as a script
    """
    )
    return


@app.cell
def _():
    print("Running US direct effect regression, experimenting on fixed effects")
    return


@app.cell
def _():
    ### --- THIS VERSION CRASHES THE NOTEBOOK ---
    # regressors_US = " + ".join(
    #     [f"tariff_interaction_US_{year}" for year in EFFECT_YEAR_RANGE]
    # )

    # regression_formula_US = f"log(value) ~ {regressors_US} | csw(importer^product_code^year, importer^exporter, exporter^product_code^year)"

    # model_US = pyfixest.feols(
    #     regression_formula_US, clean_input_df_US, vcov="hetero"
    # )

    # with open(r"notebooks/dev4/models/us_model.pkl", "wb") as file:
    #     pickle.dump(model_US, file)

    # with open(r"notebooks/dev4/models/us_model.pkl", "rb") as file_US:
    #     model_US_loaded = pickle.load(file_US)

    # pyfixest.etable(model_US_loaded)
    return


@app.cell
def _(EFFECT_YEAR_RANGE, clean_input_df_US, pyfixest):
    regressors_US = " + ".join(
        [f"tariff_interaction_US_{year}" for year in EFFECT_YEAR_RANGE]
    )

    regression_formula_US = f"log(value) ~ {regressors_US} | importer^product_code^year + importer^exporter + exporter^product_code^year"

    model_US = pyfixest.feols(
        regression_formula_US, clean_input_df_US, vcov="hetero"
    )

    pyfixest.etable(
        model_US,
        file_name="notebook_outputs/panel_regressions/regression_B_US_etable.html",
    )
    return model_US, regressors_US


@app.cell
def _(model_US):
    model_US.coefplot()
    return


@app.cell
def _(mo):
    mo.md(r"""#### No JPT""")
    return


@app.cell
def _(clean_input_df_US, pyfixest, regressors_US):
    regression_formula_US_nojpt = f"log(value) ~ {regressors_US} | importer^product_code^year + importer^exporter"

    model_US_nojpt = pyfixest.feols(
        regression_formula_US_nojpt, clean_input_df_US, vcov="hetero"
    )

    pyfixest.etable(
        model_US_nojpt,
        file_name="notebook_outputs/panel_regressions/regression_B_US_nojpt_etable.html",
    )
    return (model_US_nojpt,)


@app.cell
def _(model_US_nojpt):
    model_US_nojpt.coefplot()
    return


@app.cell
def _(mo, regression_formula_UK):
    mo.md(
        rf"""
    #### US regression formula:

    {regression_formula_UK}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Run UK Regression

    Crashes in notebook mode - so assume we've run as a script
    """
    )
    return


@app.cell
def _():
    print("Running UK direct effect regression, experimenting on fixed effects")
    return


@app.cell
def _():
    ### --- THIS VERSION CRASHES THE NOTEBOOK ---
    # regressors_UK = " + ".join(
    #     [f"tariff_interaction_UK_{year}" for year in EFFECT_YEAR_RANGE]
    # )
    # regression_formula_UK = f"log(value) ~ {regressors_UK} | csw(importer^product_code^year, importer^exporter, exporter^product_code^year)"

    # model_UK = pyfixest.feols(
    #     regression_formula_UK, clean_input_df_UK, vcov="hetero"
    # )

    # with open(r"notebooks/dev4/models/UK_model.pkl", "wb") as file_UK:
    #     pickle.dump(model_UK, file_UK)

    # with open(r"notebooks/dev4/models/UK_model.pkl", "rb") as file_UK_load:
    #     model_UK_loaded = pickle.load(file_UK_load)

    # pyfixest.etable(model_UK_loaded)
    return


@app.cell
def _(EFFECT_YEAR_RANGE, clean_input_df_UK, pyfixest):
    regressors_UK = " + ".join(
        [f"tariff_interaction_UK_{year}" for year in EFFECT_YEAR_RANGE]
    )
    regression_formula_UK = f"log(value) ~ {regressors_UK} | importer^product_code^year + importer^exporter + exporter^product_code^year"

    model_UK = pyfixest.feols(
        regression_formula_UK, clean_input_df_UK, vcov="hetero"
    )

    pyfixest.etable(
        model_UK,
        file_name="notebook_outputs/panel_regressions/regression_B_UK_etable.html",
    )
    return model_UK, regression_formula_UK, regressors_UK


@app.cell
def _(model_UK):
    model_UK.coefplot()
    return


@app.cell
def _(clean_input_df_UK, pyfixest, regressors_UK):
    regression_formula_UK_nojpt = f"log(value) ~ {regressors_UK} | importer^product_code^year + importer^exporter"

    model_UK_nojpt = pyfixest.feols(
        regression_formula_UK_nojpt, clean_input_df_UK, vcov="hetero"
    )

    pyfixest.etable(
        model_UK_nojpt,
        file_name="notebook_outputs/panel_regressions/regression_B_UK_nojpt_etable.html",
    )
    return (model_UK_nojpt,)


@app.cell
def _(model_UK_nojpt):
    model_UK_nojpt.coefplot()
    return


@app.cell
def _(mo, regression_formula_UK):
    mo.md(
        rf"""
    #### UK regression formula

    {regression_formula_UK}
    """
    )
    return


if __name__ == "__main__":
    app.run()
