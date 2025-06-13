import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import pyfixest
    import pycountry
    import plotly.express as px
    import statsmodels.api as sm

    return mo, pd, pl, px, pycountry, pyfixest, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Improving the HM regressions, and other iterations

    High-level, we're answering two seperate questions here. 

    The ininitial HM paper is interested in the response of all other countries' imports from China and the US to the US-China trade war 1.

    The first question we answer, with the scatter approach, is the response of individual countries imports relative to the RoW. And we subsequently see how different (on both dimensions) their response was relative to what we might expect. But this is only for the change in import growth rate between 2012-2017 and 2018-2023. 

    The "Daniel approach" extends this to be continous over years, based off the initial direct-effect regression spec, and we experiment with alternative specifications. **We need to do the cross-country comparison with this approach as well.**

    The second question we answer, with the "lukas regression" approach, is:
    > how did the UK's import patterns from China and ROW shift in response to US-China tariffs, controlling for product-specific global trends, exporter-specific trends, and UK-exporter relationships?"


    ## Scatter of different countries' change in imports between the US and China
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Helper functions""")
    return


@app.cell
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


@app.cell
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
        country_code = country.numeric
        print(f"Running regression for {country.name} ({country.alpha_3})")

        # Prepare the regression data
        regression_data = prepare_regression_data_THIRDCOUNTRY(
            without_oil_unified_lf,
            period1_start_year="2012",
            period1_end_year="2017",
            period2_start_year="2018",
            period2_end_year="2022",
            top_30_importers=top_importers,
            country_to_isolate=country_code,
        )

        # Run the regression
        summary, coefplot, model = run_regression_THIRDCOUNTRY(regression_data, country_code)

        return summary, coefplot

    return (run_regression_for_country,)


@app.cell
def _(mo):
    mo.md(r"""## Load data""")
    return


@app.cell
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")
    return (unified_lf,)


@app.cell
def _(mo):
    mo.md(r"""## Config""")
    return


@app.cell
def _(pycountry):
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
    return CHINA_CC, EFFECT_YEAR_RANGE, UK_CC, USA_CC


@app.cell
def _(mo):
    mo.md(
        r"""
    # Regression 3: Indirect effects on third countries, isolating specific countries. Scatter of results. 

    Remove russian federation from sample
    """
    )
    return


@app.cell
def _(get_oil_exporting_countries, pl, pycountry, unified_lf):
    # Identify oil countries -> those with >50% of their exports in hs code 27
    oil_country_list = get_oil_exporting_countries(unified_lf, 50)

    without_oil_unified_lf = unified_lf.filter(~pl.col("reporter_country").is_in(oil_country_list))

    oil_country_list_pycountries = [
        pycountry.countries.get(numeric=country) for country in oil_country_list if pycountry.countries.get(numeric=country)
    ]
    return oil_country_list, without_oil_unified_lf


@app.cell
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
    top_importers = (
        country_trade_values_2017.sort(by="value", descending=True)
        .select("partner_country")
        .to_series()
        .to_list()[:30]  # Add one more - to allow us to remove Russia.
    )
    top_importers.remove("643")  # Remove Russia, which was sanctioned due to Ukraine war and so skews the numbers
    print(top_importers)
    return (top_importers,)


@app.cell
def _(pl, without_oil_unified_lf):
    # Filter for trade data from the year 2017.
    trade_2017_lf = without_oil_unified_lf.filter(pl.col("year") == "2017")

    # Create two views of the data: one for exports and one for imports.
    exports_lf = trade_2017_lf.select(pl.col("reporter_country").alias("country"), pl.col("value"))
    imports_lf = trade_2017_lf.select(pl.col("partner_country").alias("country"), pl.col("value"))

    # Concatenate the two views to get all trade flows associated with a country.
    all_trade_lf = pl.concat([exports_lf, imports_lf])

    # Calculate total trade for each country and get the top N.
    top_total_trade = (
        all_trade_lf.group_by("country").agg(pl.sum("value").alias("total_trade")).sort("total_trade", descending=True).head(45).collect()
    )

    # Extract the list of country codes.
    top_countries_list = top_total_trade.get_column("country").to_list()

    # Optionally, remove a specific country, e.g., Russia ('643').
    if "643" in top_countries_list:
        top_countries_list.remove("643")

    print(top_countries_list)
    return (top_countries_list,)


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
        # color="country_name",
        trendline="ols",
        title="Country import growth from China (X) and USA (Y)",
    )
    fig_scatter
    return (fig_scatter,)


@app.cell
def _(df, fig_scatter, px, sm):
    ols_results = px.get_trendline_results(fig_scatter).iloc[0]["px_fit_results"]

    # Calculate Y-axis (vertical) residuals
    df["y_residuals"] = ols_results.resid

    # To get X-axis (horizontal) residuals, we fit a new model swapping X and Y
    x_model = sm.OLS(df["china_export_growth"], sm.add_constant(df["us_export_growth"])).fit()
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

    There's probably a more elegant way to assess this. And I wonder how much of this is just GDP growth.
    """
    )
    return


@app.cell
def _(model_objects):
    # Taking a look at the etables

    model_objects
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # New specification: Daniel version (with and without jpt FE)

    - Use the direct effect regression
    - Change the i=USA in the dummy to be i=RoW
    - Tau stays the same
    - Then run again, with i=UK
    - Keep all the same effects
    - Then run with the jpt fixed effect

    ### Objective
    Keep the US-China tariff (τ USA,CHN,p,s) as the "treatment" variable but apply this treatment to different groups of observations to measure spillover effects.
    """
    )
    return


@app.cell
def _():
    # # Filter for trade data from the year 2017.
    # trade_2017_lf = without_oil_unified_lf.filter(pl.col("year") == "2017")

    # # Create two views of the data: one for exports and one for imports.
    # exports_lf = trade_2017_lf.select(
    #     pl.col("reporter_country").alias("country"), pl.col("value")
    # )
    # imports_lf = trade_2017_lf.select(
    #     pl.col("partner_country").alias("country"), pl.col("value")
    # )

    # # Concatenate the two views to get all trade flows associated with a country.
    # all_trade_lf = pl.concat([exports_lf, imports_lf])

    # # Calculate total trade for each country and get the top 30.
    # top_30_total_trade = (
    #     all_trade_lf.group_by("country")
    #     .agg(pl.sum("value").alias("total_trade"))
    #     .sort("total_trade", descending=True)
    #     .head(30)
    #     .collect()
    # )

    # # Extract the list of country codes.
    # top_30_countries_list = top_30_total_trade.get_column("country").to_list()

    # # Optionally, remove a specific country, e.g., Russia ('643').
    # if "643" in top_30_countries_list:
    #     print("Removing Russian Federation from sample")
    #     top_30_countries_list.remove("643")

    # print(top_30_countries_list)
    return


@app.cell
def _(pl, top_countries_list, unified_lf):
    filtered_lf_newspec = unified_lf.filter(
        (pl.col("reporter_country").is_in(top_countries_list) & pl.col("partner_country").is_in(top_countries_list),)
    )
    return (filtered_lf_newspec,)


@app.cell
def _(CHINA_CC, EFFECT_YEAR_RANGE, UK_CC, USA_CC, filtered_lf_newspec, pl):
    # Create our input dataframe
    # First identify the US-China tariffs
    tariff_us_china_expr = (
        pl.col("average_tariff_official")
        .filter((pl.col("partner_country") == USA_CC) & (pl.col("reporter_country") == CHINA_CC))
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
        pl.col("value").log().alias("log_value"),
        pl.col("quantity").log().alias("log_quantity"),
        tariff_us_china_expr,
    ).filter(pl.col("year").is_in(list(EFFECT_YEAR_RANGE)))

    # Create our interaction expressions - the core of this
    interaction_expressions_US = [
        pl.when((pl.col("year") == str(year)) & (pl.col("importer") == USA_CC) & (pl.col("exporter") == CHINA_CC))
        .then(pl.col("tariff_us_china"))
        .otherwise(0.0)
        .alias(f"tariff_interaction_US_{year}")
        for year in EFFECT_YEAR_RANGE
    ]

    interaction_expressions_ROW = [
        pl.when(
            (pl.col("importer") != USA_CC)  # The key change is here
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
            (pl.col("importer") == UK_CC)  # The key change is here
            & (pl.col("exporter") == CHINA_CC)
            & (pl.col("year") == str(year))
        )
        .then(pl.col("tariff_us_china"))
        .otherwise(0.0)
        .alias(f"tariff_interaction_UK_{year}")
        for year in EFFECT_YEAR_RANGE
    ]

    # fixed_effect_expressions = [
    #     pl.concat_str(["importer", "product_code", "year"], separator="^")
    #     .alias("alpha_ipt")
    #     .cast(pl.Categorical),
    #     pl.concat_str(["exporter", "product_code", "year"], separator="^")
    #     .alias("alpha_jpt")
    #     .cast(pl.Categorical),
    #     pl.concat_str(["importer", "exporter"], separator="^")
    #     .alias("alpha_ij")
    #     .cast(pl.Categorical),
    # ]
    return (
        input_lf,
        interaction_expressions_ROW,
        interaction_expressions_UK,
        interaction_expressions_US,
        tariff_us_china_expr,
    )


@app.cell
def _(
    input_lf,
    interaction_expressions_ROW,
    interaction_expressions_UK,
    interaction_expressions_US,
):
    final_lf_US = input_lf.with_columns(
        *interaction_expressions_US,  # *fixed_effect_expressions
    )
    final_lf_ROW = input_lf.with_columns(
        *interaction_expressions_ROW,  # *fixed_effect_expressions
    )
    final_lf_UK = input_lf.with_columns(
        *interaction_expressions_UK,  # *fixed_effect_expressions
    )
    return final_lf_ROW, final_lf_UK, final_lf_US


@app.cell
def _(mo):
    mo.md(
        r"""
    ### US-China
    Validate it's what we'd expect
    """
    )
    return


@app.cell
def _(final_lf_US):
    clean_input_df_US = final_lf_US.drop_nans(subset=["log_value", "tariff_us_china"]).collect()
    return (clean_input_df_US,)


@app.cell
def _(clean_input_df_US):
    clean_input_df_US.head()
    return


@app.cell
def _(EFFECT_YEAR_RANGE, mo):
    regressors_US = " + ".join([f"tariff_interaction_US_{year}" for year in EFFECT_YEAR_RANGE])

    # regression_formula_US = (
    #     f"log_value ~ {regressors_US} | alpha_ipt + alpha_jpt + alpha_ij"
    # )

    # csw() cumulatively adds the fixed effects into the regression, allowing us to compare specs -> crashes my marimo notebook...
    regression_formula_US = f"log_value ~ {regressors_US} | csw(importer^product_code^year, importer^exporter, exporter^product_code^year)"

    # regression_formula_US = f"log_value ~ {regressors_US} | importer^product_code^year + exporter^product_code^year + importer^exporter"

    mo.vstack(
        [
            mo.md("We specify the regression formula as follows:"),
            mo.md(f"{regression_formula_US}"),
        ]
    )
    return regression_formula_US, regressors_US


@app.cell
def _(clean_input_df_US, pyfixest, regression_formula_US):
    # with mo.persistent_cache(name="model_cache"):
    model_US = pyfixest.feols(regression_formula_US, clean_input_df_US, vcov="hetero")
    return (model_US,)


@app.cell
def _(model_US, pyfixest):
    pyfixest.etable(model_US)
    return


@app.cell
def _(model_US):
    (
        model_US.fetch_model(0).coefplot(),
        model_US.fetch_model(1).coefplot(),
        model_US.fetch_model(2).coefplot(),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### RoW""")
    return


@app.cell
def _(final_lf_ROW):
    clean_input_df_ROW = final_lf_ROW.drop_nans(subset=["log_value", "tariff_us_china"]).collect()
    return (clean_input_df_ROW,)


@app.cell
def _(EFFECT_YEAR_RANGE, mo):
    regressors_ROW = " + ".join([f"tariff_interaction_ROW_{year}" for year in EFFECT_YEAR_RANGE])
    regression_formula_ROW = f"log_value ~ {regressors_ROW} | alpha_ipt + alpha_jpt + alpha_ij"

    mo.vstack(
        [
            mo.md("We specify the regression formula as follows:"),
            mo.md(f"{regression_formula_ROW}"),
        ]
    )
    return regression_formula_ROW, regressors_ROW


@app.cell
def _(clean_input_df_ROW, mo, pyfixest, regression_formula_ROW):
    with mo.persistent_cache(name="model_cache"):
        model_ROW = pyfixest.feols(regression_formula_ROW, clean_input_df_ROW)
    return (model_ROW,)


@app.cell
def _(model_ROW):
    model_ROW.summary()
    return


@app.cell
def _(model_ROW):
    model_ROW.coefplot()
    return


@app.cell
def _(mo):
    mo.md(r"""### UK""")
    return


@app.cell
def _(final_lf_UK):
    clean_input_df_UK = final_lf_UK.drop_nans(subset=["log_value", "tariff_us_china"]).collect()
    return (clean_input_df_UK,)


@app.cell
def _(EFFECT_YEAR_RANGE, mo):
    regressors_UK = " + ".join([f"tariff_interaction_UK_{year}" for year in EFFECT_YEAR_RANGE])
    regression_formula_UK = f"log_value ~ {regressors_UK} | alpha_ipt + alpha_jpt + alpha_ij"

    mo.vstack(
        [
            mo.md("We specify the regression formula as follows:"),
            mo.md(f"{regression_formula_UK}"),
        ]
    )
    return regression_formula_UK, regressors_UK


@app.cell
def _(clean_input_df_UK, mo, pyfixest, regression_formula_UK):
    with mo.persistent_cache(name="model_cache"):
        model_UK = pyfixest.feols(regression_formula_UK, clean_input_df_UK)
    return (model_UK,)


@app.cell
def _(model_UK):
    model_UK.summary()
    return


@app.cell
def _(model_UK):
    model_UK.coefplot()
    return


@app.cell
def _(mo):
    mo.md(r"""## Daniel version variation: remove jpt fixed effect...""")
    return


@app.cell
def _(CHINA_CC, EFFECT_YEAR_RANGE, UK_CC, USA_CC, filtered_lf_newspec, pl):
    def prep_direct_effect_nojpt(raw_lf):
        # Create our input dataframe
        # First identify the US-China tariffs
        tariff_us_china_expr = (
            pl.col("average_tariff_official")
            .filter((pl.col("partner_country") == USA_CC) & (pl.col("reporter_country") == CHINA_CC))
            .mean()
            .over(["year", "product_code"])
            .alias("tariff_us_china")
        )

        # Initial filtering
        input_lf = raw_lf.select(
            pl.col("year"),
            pl.col("partner_country").alias("importer"),
            pl.col("reporter_country").alias("exporter"),
            pl.col("product_code"),
            pl.col("value").log().alias("log_value"),
            pl.col("quantity").log().alias("log_quantity"),
            tariff_us_china_expr,
        ).filter(pl.col("year").is_in(list(EFFECT_YEAR_RANGE)))

        # Create our interaction expressions - the core of this
        interaction_expressions_US = [
            pl.when((pl.col("year") == str(year)) & (pl.col("importer") == USA_CC) & (pl.col("exporter") == CHINA_CC))
            .then(pl.col("tariff_us_china"))
            .otherwise(0.0)
            .alias(f"tariff_interaction_US_{year}")
            for year in EFFECT_YEAR_RANGE
        ]

        interaction_expressions_ROW = [
            pl.when(
                (pl.col("importer") != USA_CC)  # The key change is here
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
                (pl.col("importer") == UK_CC)  # The key change is here
                & (pl.col("exporter") == CHINA_CC)
                & (pl.col("year") == str(year))
            )
            .then(pl.col("tariff_us_china"))
            .otherwise(0.0)
            .alias(f"tariff_interaction_UK_{year}")
            for year in EFFECT_YEAR_RANGE
        ]

        fixed_effect_expressions = [
            pl.concat_str(["importer", "product_code", "year"], separator="^").alias("alpha_ipt").cast(pl.Categorical),
            # pl.concat_str(["exporter", "product_code", "year"], separator="^")
            # .alias("alpha_jpt")
            # .cast(pl.Categorical),
            pl.concat_str(["importer", "exporter"], separator="^").alias("alpha_ij").cast(pl.Categorical),
        ]

        final_lf_US = input_lf.with_columns(*interaction_expressions_US, *fixed_effect_expressions)
        final_lf_ROW = input_lf.with_columns(*interaction_expressions_ROW, *fixed_effect_expressions)
        final_lf_UK = input_lf.with_columns(*interaction_expressions_UK, *fixed_effect_expressions)

        return final_lf_US, final_lf_ROW, final_lf_UK

    final_lf_nojpt_US, final_lf_nojpt_ROW, final_lf_nojpt_UK = prep_direct_effect_nojpt(filtered_lf_newspec)
    return final_lf_nojpt_ROW, final_lf_nojpt_UK, final_lf_nojpt_US


@app.cell
def _(mo):
    mo.md(r"""### US - no jpt""")
    return


@app.cell
def _(mo):
    mo.md(r"""### RoW - no jpt""")
    return


@app.cell
def _(mo):
    mo.md(r"""### UK - no jpt""")
    return


@app.cell(hide_code=True)
def _(final_lf_nojpt_US):
    clean_input_df_nojpt_US = final_lf_nojpt_US.drop_nans(subset=["log_value", "tariff_us_china"]).collect()
    return (clean_input_df_nojpt_US,)


@app.cell(hide_code=True)
def _(regressors_US):
    # Run the US regression
    regression_formula_US_nojpt = f"log_value ~ {regressors_US} | alpha_ipt + alpha_ij"
    return (regression_formula_US_nojpt,)


@app.cell(hide_code=True)
def _(clean_input_df_nojpt_US, mo, pyfixest, regression_formula_US_nojpt):
    with mo.persistent_cache(name="model_cache"):
        model_US_nojpt = pyfixest.feols(regression_formula_US_nojpt, clean_input_df_nojpt_US)
    return (model_US_nojpt,)


@app.cell
def _(model_US_nojpt):
    model_US_nojpt.summary(), model_US_nojpt.coefplot()
    return


@app.cell
def _(final_lf_nojpt_ROW):
    clean_input_df_nojpt_ROW = final_lf_nojpt_ROW.drop_nans(subset=["log_value", "tariff_us_china"]).collect()
    return (clean_input_df_nojpt_ROW,)


@app.cell
def _(regressors_ROW):
    regression_formula_ROW_nojpt = f"log_value ~ {regressors_ROW} | alpha_ipt + alpha_ij"
    return (regression_formula_ROW_nojpt,)


@app.cell
def _(clean_input_df_nojpt_ROW, mo, pyfixest, regression_formula_ROW_nojpt):
    with mo.persistent_cache(name="model_cache"):
        model_ROW_nojpt = pyfixest.feols(regression_formula_ROW_nojpt, clean_input_df_nojpt_ROW)
    return (model_ROW_nojpt,)


@app.cell
def _(model_ROW_nojpt):
    model_ROW_nojpt.summary(), model_ROW_nojpt.coefplot()
    return


@app.cell
def _(final_lf_nojpt_UK):
    clean_input_df_nojpt_UK = final_lf_nojpt_UK.drop_nans(subset=["log_value", "tariff_us_china"]).collect()
    return (clean_input_df_nojpt_UK,)


@app.cell
def _(regressors_UK):
    regression_formula_UK_nojpt = f"log_value ~ {regressors_UK} | alpha_ipt + alpha_ij"
    return (regression_formula_UK_nojpt,)


@app.cell
def _(clean_input_df_nojpt_UK, mo, pyfixest, regression_formula_UK_nojpt):
    with mo.persistent_cache(name="model_cache"):
        model_UK_nojpt = pyfixest.feols(regression_formula_UK_nojpt, clean_input_df_nojpt_UK)
    return (model_UK_nojpt,)


@app.cell
def _(model_UK_nojpt):
    model_UK_nojpt.summary(), model_UK_nojpt.coefplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # New spec - my version

    The core hypothesis is that an increase in τ (US,CHN,p,t) leads to an increase in UK imports of product p, particularly from China.

    ## Equation spec

    \[
            ln\_M_{UK,j,p,t} = \beta_0 + \beta_1(\tau_{US,CHN,p,t}) \times D(j=CHN) + \beta_2(\tau_{US,CHN,p,t}) \times D(j=ROW) + \alpha_{UK,p,t} + \alpha_{j,p,t} + \alpha_{UK,j} + \epsilon_{UK,j,p,t}
    \]

    * $ln(M_{UK,j,p,t})$: Natural logarithm of UK imports of product $p$ from exporter $j$ in year $t$.

    * $\tau_{US,CHN,p,t}$: The US ad valorem tariff imposed on product $p$ originating from China in year $t$. This variable is effectively active in the years 2018-2019 for the purpose of this study.

    * $D(j=CHN)$: An indicator variable that equals 1 if the exporting country $j$ is China, and 0 otherwise. The coefficient $\beta_1$ captures the direct diversion effect: an increase in US tariffs on Chinese product $p$ is hypothesized to lead to an increase in UK imports of product $p$ from China.

    * $D(j=ROW)$: An indicator variable that equals 1 if the exporting country $j$ is a 'Rest of World' country (i.e., not China, not the US, and not the UK itself, assuming the UK is the importer $i$). The coefficient $\beta_2$ could capture indirect diversion effects. For instance, if ROW countries now face less competition from China in the US market, they might redirect some of their exports to the UK, or UK importers might find ROW suppliers relatively more attractive.

    <!-- 
    * $X_{UK,j,p,t}$: A vector of additional time-varying control variables, if not fully absorbed by the fixed effects. These could include variables like the bilateral real exchange rate between the UK and exporter $j$, or specific UK policies affecting product $p$ from country $j$. However, the high-dimensional fixed effects specified below will absorb many such common controls. -->

    * $\alpha_{UK,p,t}$: Importer-Product-Year fixed effects. These control for all unobserved factors specific to the UK's demand for product $p$ in year $t$. Examples include shifts in UK consumer preferences for product $p$, UK-specific regulations affecting product $p$ that year, or general economic conditions in the UK influencing overall demand for $p$. Their inclusion is critical to ensure that $\beta_1$ does not spuriously capture a general surge in UK demand for product $p$.

    * $\alpha_{j,p,t}$: Exporter-Product-Year fixed effects. These control for all unobserved factors affecting exporter $j$'s global supply conditions for product $p$ in year $t$. This could include productivity shocks in country $j$ for product $p$, changes in $j$'s global export strategy for $p$, or input cost changes specific to $j$ for product $p$. For $j=CHN$, this FE ensures that an observed increase in UK imports of $p$ from China is not merely due to China generally exporting more of $p$ to all destinations worldwide.

    * $\alpha_{UK,j}$: Importer-Exporter (UK-Country Pair) fixed effects. These absorb all time-invariant unobserved characteristics specific to the UK's trade relationship with each exporter $j$. This includes factors like geographical distance, common language, colonial ties, the stable components of existing trade agreements between the UK and $j$, and general historical import propensity.

    * $\epsilon_{UK,j,p,t}$: The idiosyncratic error term. Standard errors should be clustered (e.g., at the exporter-product level or country-pair level) to account for likely serial correlation and heteroskedasticity within these groups.


    ## Implementation
    Given the high dimensionality (many products, countries, and years), these fixed effects are typically "absorbed" rather than included explicitly. This is specifically supported within the pyfixest package.

    ## Interpretation
    The identification of the trade diversion coefficient β1 (for UK imports from China) relies on comparing the change in UK imports of a specific US-tariffed product p from China, relative to several counterfactuals established by the fixed effects:

    #### Across-Exporter Variation (within UK-Product-Year):
    * Comparing UK imports of product p from China to UK imports of the same product p from other countries k (where D(j=CHN=0)) in the same year, conditional on α(UK,p,t) (UK's overall demand for p) and α(k,p,t) (country k's global supply of p).

    #### Across-Product Variation (within UK-China-Year):
    * Comparing UK imports of tariffed product p from China to UK imports of other (less or non-tariffed) products q from China in the same year, conditional on UK demand for p and q and China's global supply of p and q.

    #### Over-Time Variation (within UK-China-Product): 
    Comparing UK imports of product p from China during the tariff period (2018-19) to the pre/post-tariff period, conditional on all fixed effects.

    ### Intepreting the Betas:
    A statistically significant and positive β1 would indicate that, holding constant UK's general demand for product p in a given year, and China's general global supply capacity for product p in that year, and the baseline UK-China trade relationship, an increase in US tariffs on Chinese product p is associated with an increase in UK imports of product p from China. This is the primary measure of trade diversion of Chinese goods to the UK market.

    The interpretation of β2 (for UK imports from ROW) is more nuanced. A positive β2 might suggest that the UK increased its imports of product p from ROW countries. This could occur if these ROW countries became relatively more competitive in the UK market compared to China (if China's prices to the UK rose or availability fell despite diversion efforts), or if ROW countries redirected their own exports from the US market towards other markets like the UK.

    ## Assumptions & Limitations
    #### Strict Exogeneity
    The core assumption is that the error term is uncorrelated with the independent variables τ(US,CHN,p,t) and its interactions, controls, in all time periods, conditional on the unobserved fixed effects. For the US-China tariff variable, this implies that unobserved shocks affecting UK's specific import demand for product p from exporter j are not correlated with the US decision to impose tariffs on China for that product. This is generally plausible given the UK is a third country in the US-China dispute.

    #### Time-Invariant Unobserved Heterogeneity Adequately Controlled
    The model assumes that the included fixed effects adequately capture all relevant unobserved heterogeneity that is either time-invariant like α(UK,j) or common to specific dimensions like α(UK,p,t) or α(j,p,t). If there are time-varying unobserved factors at the UK-exporter-product level that are correlated with both the US-China tariffs (or their anticipation) and UK import decisions, and are not captured by the included FEs, the estimates can be biased. For example, if a global restructuring of supply chains for product p was occurring concurrently with the tariffs for reasons independent of the tariffs (e.g., technological changes, rising labor costs in China unrelated to tariffs) and this also influenced UK sourcing for p from China, the fixed effects might not fully disentangle these influences.

    #### Measurement Error
    Fixed-effects estimation can exacerbate attenuation bias if independent variables are measured with error. Tariff data itself can be complex and prone to errors in application or recording.3 Trade values can also suffer from measurement error.

    Specifically, we are aggregating tariffs at the HS6 level, leading to a systematic overestimation in the number of products tariffed (given these were applied at the HS10 level). We also do not consider exclusions over 2019, 2020. Nor do we consider retaliations, antidumping measures, or more complex quota systems.

    #### Parameter Homogeneity
    Our model assumes that the coefficients  (β1, β2) are constant across all products, exporters, and time periods. It is likely there is within-group heterogeneity, for example in the broader type of products being tariffed (e.g. intermediate, final, consumption). This can be explored further with another panel regression.

    ## Alternative specifications and robustnes
    To check the robustness of our conclusions, we will want to explore alternative specifications of our model, including:

    * **Alternative fixed effect specifications**: Test the sensitivity of β1 and β2 to variations in the fixed effects structure. For example, one might use exporter-year and importer-year fixed effects instead of the more granular exporter-product-year and importer-product-year if there's concern about absorbing too much relevant variation or if product-specific time-varying shocks are deemed less critical than country-specific time-varying shocks.
    * **Sub-sample analysis**: By product type, by exporter characteristics, by time period.
    * **Alternative dependent variable**: Use log quantities, or log unit value instead.
    * **Explicit controls for alternative shocks**: For COVID, Brexit?
    * **Placebo tests**: Create leads of the tariff variable and test against those, to validate the causality of our findings. We would expect a synthetic lead by 2 years of the tariff shock to not have an impact which would show up in the data. If it does, this implies (implausibly) strong anticipation effects
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Implementation: Lukas version""")
    return


@app.cell
def _(unified_lf):
    unified_lf.head().collect()
    return


@app.cell
def _(
    CHINA_CC,
    EFFECT_YEAR_RANGE,
    UK_CC,
    oil_country_list,
    pl,
    tariff_us_china_expr,
    unified_lf,
):
    ## Step 1: Re-Create the requried columns
    # Initial filtering
    uk_imports_lf = unified_lf.select(
        pl.col("year"),
        pl.col("partner_country").alias("importer"),
        pl.col("reporter_country").alias("exporter"),
        pl.col("product_code"),
        pl.col("value").log().alias("log_value"),
        pl.col("quantity").log().alias("log_quantity"),
        tariff_us_china_expr,
    ).filter(pl.col("year").is_in(list(EFFECT_YEAR_RANGE)))

    ## Step 2: Limit to only UK imports from China and remove the oil countries
    uk_imports_lf = uk_imports_lf.filter(
        pl.col("importer") == UK_CC,
        ~pl.col("exporter").is_in(oil_country_list),  # > remove those oil countries!
    )

    ## Step 3: Create the new interaction terms
    uk_imports_lf = uk_imports_lf.with_columns(
        # Interaction 1: (tau_US,CHN) * D(j=CHN)
        pl.when(pl.col("exporter") == CHINA_CC).then(pl.col("tariff_us_china")).otherwise(0.0).alias("tariff_x_china"),
        # Interaction 2: (tau_US,CHN) * D(j=ROW)
        pl.when(pl.col("exporter") != CHINA_CC).then(pl.col("tariff_us_china")).otherwise(0.0).alias("tariff_x_row"),
    ).drop_nulls()

    # ## Step 4: Create the fixed effect columns
    # fixed_effect_expressions = [
    #     pl.concat_str(["importer", "product_code", "year"], separator="^")
    #     .alias("alpha_ipt")
    #     .cast(pl.Categorical),
    #     pl.concat_str(["exporter", "product_code", "year"], separator="^")
    #     .alias("alpha_jpt")
    #     .cast(pl.Categorical),
    #     pl.concat_str(["importer", "exporter"], separator="^")
    #     .alias("alpha_ij")
    #     .cast(pl.Categorical),
    # ]

    uk_imports_pd = uk_imports_lf.collect().to_pandas()
    uk_imports_pd.head()
    return (uk_imports_pd,)


@app.cell
def _(uk_imports_pd):
    uk_imports_pd.describe()
    return


@app.cell
def _():
    lukas_formula = "log_value ~ tariff_x_china + tariff_x_row | year^product_code + exporter^year"
    return (lukas_formula,)


@app.cell
def _(lukas_formula, mo, pyfixest, uk_imports_pd):
    with mo.persistent_cache(name="model_cache"):
        model_lukas_directeffect = pyfixest.feols(fml=lukas_formula, data=uk_imports_pd)
    return (model_lukas_directeffect,)


@app.cell
def _(model_lukas_directeffect):
    model_lukas_directeffect.summary(), model_lukas_directeffect.coefplot()
    return


if __name__ == "__main__":
    app.run()
