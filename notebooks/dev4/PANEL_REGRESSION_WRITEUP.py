import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pyfixest
    import numpy as np
    import polars as pl
    import pandas as pd
    import re

    from typing import Optional
    return Optional, mo, pl, pyfixest, re


@app.cell
def _(mo):
    mo.md(r"""### Code""")
    return


@app.cell
def _(Optional, pl, pyfixest, re):
    def run_direct_effect_regression(
        data: pl.LazyFrame,
        interaction_term_name: str,
        interaction_importers: list[str],
        interaction_exporters: list[str],
        year_range: list[str],
        formula: str,
        vcov: Optional[str | dict] = "hetero",
        filter_expression: Optional[pl.Expr] = None,
    ):
        """
        Runs a direct effect regression based on US-China tariffs.

        This function preserves all columns from the input data and dynamically
        handles the dependent variable specified in the formula.

        Args:
            data: Pre-filtered LazyFrame. Must contain the dependent variable
                  column specified in the formula.
            interaction_term_name: A descriptive name for the interaction.
            interaction_importers: A list of importer country codes for the interaction.
            interaction_exporters: A list of exporter country codes for the interaction.
            year_range: A list of years (as strings) for the analysis.
            formula: The complete regression formula for pyfixest.
            vcov: The variance-covariance matrix specification.
            filter_expression: An optional Polars expression to filter the data
                               after tariff calculation but before regression.

        Returns:
            A tuple containing the model object, the etable, and the coefficient plot.
        """
        USA_CC = "840"
        CHINA_CC = "156"

        try:
            dependent_var_str = formula.split("~")[0].strip()
            dependent_var_col = re.findall(r"\b\w+\b", dependent_var_str)[-1]
        except IndexError:
            raise ValueError(
                f"Could not parse dependent variable from formula: {formula}"
            )

        tariff_expr = (
            pl.col("average_tariff_official")
            .filter(
                (pl.col("partner_country") == USA_CC)
                & (pl.col("reporter_country") == CHINA_CC)
            )
            .mean()
            .over(["year", "product_code"])
            .alias("tariff_us_china")
        )

        input_lf = data.with_columns(
            pl.col("partner_country").alias("importer"),
            pl.col("reporter_country").alias("exporter"),
            tariff_expr,
        ).filter(pl.col("year").is_in(year_range))

        interaction_filter = (pl.col("importer").is_in(interaction_importers)) & (
            pl.col("exporter").is_in(interaction_exporters)
        )

        interaction_expressions = [
            pl.when(interaction_filter & (pl.col("year") == str(year)))
            .then(pl.col("tariff_us_china"))
            .otherwise(0.0)
            .alias(f"{interaction_term_name}_{year}")
            for year in year_range
        ]

        final_lf = input_lf.with_columns(*interaction_expressions)

        # Apply the additional filter expression if provided
        if filter_expression is not None:
            final_lf = final_lf.filter(filter_expression)

        print(
            f"Checking for nulls in dependent variable '{dependent_var_col}' and 'tariff_us_china'."
        )
        clean_df = final_lf.drop_nulls(
            subset=[dependent_var_col, "tariff_us_china"]
        ).collect()

        model = pyfixest.feols(fml=formula, data=clean_df, vcov=vcov)

        etable = pyfixest.etable(model)
        coefplot = model.coefplot(joint=True)

        return model, etable, coefplot
    return (run_direct_effect_regression,)


@app.cell
def _(pl):
    def get_oil_exporting_countries(
        lzdf: pl.LazyFrame, oil_export_percentage_threshold: float
    ) -> list[str]:
        """
        Finds countries where oil products (HS code '27') exceed a certain
        percentage of their total export value.
        """
        print("--- Filtering out oil countries ---")

        total_exports = lzdf.group_by("reporter_country").agg(
            pl.sum("value").alias("total_value")
        )

        oil_exports = (
            lzdf.filter(pl.col("product_code").str.starts_with("27"))
            .group_by("reporter_country")
            .agg(pl.sum("value").alias("oil_value"))
        )

        summary = total_exports.join(
            oil_exports, on="reporter_country", how="left"
        ).with_columns(pl.col("oil_value").fill_null(0.0))

        summary = summary.with_columns(
            ((pl.col("oil_value") / pl.col("total_value")) * 100).alias(
                "oil_export_percentage"
            )
        )

        filtered_countries = summary.filter(
            pl.col("oil_export_percentage") > oil_export_percentage_threshold
        )

        return filtered_countries.collect()["reporter_country"].to_list()
    return (get_oil_exporting_countries,)


@app.cell
def _(get_oil_exporting_countries, pl):
    def prepare_analysis_data(
        source_lf: pl.LazyFrame,
        top_n: int,
        selection_year: str,
        year_range_to_keep: list[str] | None = None,
        selection_method: str = "total_trade",
        oil_export_threshold: float | None = 50.0,
        countries_to_exclude: list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Filters and subsets the main trade dataset to create a sample for analysis.

        Args:
            source_lf: The initial LazyFrame of the full trade dataset.
            top_n: The number of top countries to select (e.g., 40).
            selection_year: The year used to determine the top countries (e.g., "2017").
            year_range_to_keep: A list of years (as strings) to keep in the final
                                dataset. If None, all years are kept.
            selection_method: Method to determine top countries. Can be "total_trade"
                              or "importers".
            oil_export_threshold: The export percentage to classify a country as an
                                  oil exporter. If None, this step is skipped.
            countries_to_exclude: A list of country codes to remove from the
                                  final country list.

        Returns:
            A filtered LazyFrame containing only trade flows between the
            selected group of countries for the specified years.
        """
        print("--- Cleaning data ---")

        lf = source_lf

        if oil_export_threshold is not None:
            oil_countries = get_oil_exporting_countries(lf, oil_export_threshold)
            lf = lf.filter(~pl.col("reporter_country").is_in(oil_countries))

        trade_in_year_lf = lf.filter(pl.col("year") == selection_year)

        if selection_method == "importers":
            top_countries_df = (
                trade_in_year_lf.group_by("partner_country")
                .agg(pl.sum("value").alias("import_value"))
                .sort("import_value", descending=True)
                .head(top_n)
                .collect()
            )
            top_countries_list = top_countries_df["partner_country"].to_list()

        elif selection_method == "total_trade":
            exports_lf = trade_in_year_lf.select(
                pl.col("reporter_country").alias("country"), "value"
            )
            imports_lf = trade_in_year_lf.select(
                pl.col("partner_country").alias("country"), "value"
            )

            top_countries_df = (
                pl.concat([exports_lf, imports_lf])
                .group_by("country")
                .agg(pl.sum("value").alias("total_trade"))
                .sort("total_trade", descending=True)
                .head(top_n)
                .collect()
            )
            top_countries_list = top_countries_df["country"].to_list()
        else:
            raise ValueError(
                "selection_method must be 'importers' or 'total_trade'"
            )

        if countries_to_exclude:
            top_countries_list = [
                c for c in top_countries_list if c not in countries_to_exclude
            ]

        print(f"Final sample includes {len(top_countries_list)} countries.")

        # Filter by the selected countries first
        analysis_lf = lf.filter(
            pl.col("reporter_country").is_in(top_countries_list)
            & pl.col("partner_country").is_in(top_countries_list)
        )

        # Apply the year range filter if provided
        if year_range_to_keep:
            analysis_lf = analysis_lf.filter(
                pl.col("year").is_in(year_range_to_keep)
            )

        print("--- Filtering complete ---")
        return analysis_lf
    return (prepare_analysis_data,)


@app.cell
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")

    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _(prepare_analysis_data, unified_lf):
    # 1. Prepare the analysis data with specific parameters
    analysis_lf = prepare_analysis_data(
        source_lf=unified_lf,
        top_n=40,
        selection_year="2017",
        year_range_to_keep=[str(y) for y in range(2017, 2024)],
        selection_method="total_trade",
        oil_export_threshold=50.0,
        countries_to_exclude=["643", "344"],  # Russia, Hong Kong
    )
    return (analysis_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Analysis 1: Indirect effects of Trade war 1 on the UK

    This notebook explores the effects of the 2018 US-China trade war on UK imports from China. We attempt to find evidence for the existence of dumping, and to extrapolate this evidence to identify an implied effect over trade war 2.

    To answer this question, we construct the following dataset:

    1. Trade values (USD), and trade volumes (SI unit), for bilateral trade between all countries in the world, annually. At the HS6 product level (some 5000 unique product codes). This data is sourced from [CEPII](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37), where they describe it as such:
    > BACI provides data on bilateral trade flows for 200 countries at the product level (5000 products). Products correspond to the "Harmonized System" nomenclature (6 digit code).
    2. The simple average (required as tariffs are often applied at the hs10 level) ad-valorem equivalent tariffs for each of the ~5000 products, through time and for each bilateral trading pair, resulting from both preferential agreements and on a most-favoured nation basis. This is sourced from the [WITS tariff database](https://wits.worldbank.org/), produced by the World Bank.
    3. Specific 'exceptional' tariffs introduced by the US administration during the 2018 US-China trade war. These are sourced from replication materials for [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5101245) by Trang T. Hoang and Carter Mix at the Federal Reserve Board. We validate these against [the US Trade Representative's website](https://ustr.gov/issue-areas/enforcement/section-301-investigations).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Analysis
    We are essentially trying to connect UK imports from China, to the application of tariffs by the US on Chinese goods in the 2018 trade war. We are attempting to determine if there is a causal link as follows:

    1. The US tariffs Chinese goods.
    2. In response to a) the increased effective price for US importers & consumers; b) other political pressures, the US demand for Chinese goods falls.
    3. Chinese exporters respond by reducing their price and identifying new trading relationships with other countries, including the UK.
    4. This results in an increased volume of Chinese goods, at a lower export price, being sent to the UK. This is called 'dumping'.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Methodology""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Approach 1
    To assess this question, first order, we run a causal inference assessment. We construct a panel of bilateral trade between a range of countries, where the treatement is the application of tariffs by the US on China of a specific good. Our observed outcome variable is a) the volume; b) the value; c) the unit value; of exports from China to the UK. 

    We are interested in the effect of the tariffs on UK imports, but include all global bilateral trade from the rest of the world as well. This allows us to implicitly construct control groups across two dimensions: a) imports of those products tariffed by the US from countries other than China which were not tariffed; b) imports from China of products which were not tariffed.

    Relatedly, we employ fixed effects (more flexible dummy variables) to control for variation between our treatment and non-treatment groups as well as confounding effects. Replicating an existing specification, we control for a) variation across time within exporter and specific product; b) variation across time within importer and product; c) variation between importer-exporter pair. This specification is important to get right. Some causal inference theory:
    > If we condition on a common/descendent/mediating effects (e.g. fluctuations in price of goods) of the treatment (tariff application by the US), we cause selection bias, reducing the power of our test to detect the impact of the treatment on the observed outcome variable (UK imports from China).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Specification
    We fit an OLS fixed effect panel regression, using the [pyfixest](https://github.com/py-econometrics/pyfixest) package. 

    Adapting the equation defined in Section 3.1 of [Mix-Huang, 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5101245), we specify the following model:

    $$
    \begin{align*}
    L_{i,j,p,t} &= \beta \cdot \left( \mathbb{I}(i = \mathrm{UK}) \cdot \mathbb{I}(j = \mathrm{China}) \cdot \mathbb{I}(t \in [2017,2024]) \cdot \mathrm{Tariff}_{p,t}^{\mathrm{US-China}} \right) \\
    &\quad + \alpha_{i,p,t} + \lambda_{i,j} + \varepsilon_{i,j,p,t}
    \end{align*}
    $$

    Where the notation is defined as follows:

    * $L_{i,j,p,t}$: Represents the dependent variable, specifically $L_{i,j,p,t} = \log(\mathrm{Value}_{i,j,p,t})$.
        * $\mathrm{Value}_{i,j,p,t}$: The underlying traded value/volume/price of interest for the observation.
    * Indices:
        * $i$: Represents the importer.
        * $j$: Represents the exporter.
        * $p$: Represents the product code.
        * $t$: Represents the year.
    * $\beta$: The coefficient of interest, quantifying the impact of the main regressor term.
    * $\mathbb{I}(\cdot)$: The indicator function, which equals 1 if the condition in the parenthesis is true, and 0 otherwise.
        * $\mathbb{I}(i = \mathrm{UK})$: Equals 1 if the importer $i$ is the United Kingdom, 0 otherwise.
        * $\mathbb{I}(j = \mathrm{China})$: Equals 1 if the exporter $j$ is China, 0 otherwise.
        * $\mathbb{I}(t \in [2017,2024])$: Equals 1 if the year $t$ falls within the range of 2017 to 2024 (inclusive), 0 otherwise.
    * $\mathrm{Tariff}_{p,t}^{\mathrm{US-China}}$: The average official tariff rate between the US and China for product code $p$ in year $t$. (Note: The main regressor specifically conditions on UK importer and China exporter, while this tariff term is specified for US-China. Ensure this is intended and clearly defined in your context; it might represent a benchmark or a proxy if direct UK-China tariffs are not used here.)
    * Fixed Effects: These terms control for unobserved heterogeneity between groups, including various sources of bias due to differences between our treated and untreated groups. They work by essentially de-meaning within a specific group, leaving only heterogeneity within that group.
        * $\alpha_{i,p,t}$: Represents exportert-product-year fixed effects, controlling for any unobserved factors specific to each combination of an exporter $i$, product code $p$, and year $t$. 
        * $\lambda_{i,j}$: Represents importer-exporter fixed effects, controlling for unobserved factors specific to each pair of an importer $i$ and exporter $j$.
        <!-- * $\omega_{j,p,t}$: Represents exporter-product-year fixed effects, controlling for unobserved factors specific to each combination of an exporter $j$, product code $p$, and year $t$. -->
    * $\varepsilon_{i,j,p,t}$: The idiosyncratic error term for importer $i$, exporter $j$, product $p$, and year $t$. It captures all other unobserved factors not accounted for by the regressors or fixed effects.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Interpretation
    Interpreting this model, the coefficient $\beta$ quantifies the estimated percentage change in the specific metric used for $\log(\mathrm{Value}_{i,j,p,t})$ (i.e., import value, quantity, or price) for UK imports from China in response to a one percentage point increase in the $\mathrm{Tariff}_{p,t}^{\mathrm{US-China}}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Extensions and alternate specifications

    Based on this foundation, we vary the specification to test our results for robustness, vary the question being answered, compare against literature, etc. Dimensions which we perform this variation across are:

    - Varying control groups (range & breadth of the sample)
    - Varying fixed effects
    - RoW, US, and UK as the importer 
    - Varying the measurement period of the effect (various baseline years)
    - Placebo tests (TBD)
    - Clustering standard errors with varying specifications

    A range of alternate specifications is contained in APPENDIX [X]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Further Considerations
    It is important to note that the policy response of individual countries, such as the UK, to the US tariffs on China is embedded within our data (i.e. endogenous). For example, the UK implemented industry-specific subsidies alongside various import restrictions. Third countries exhibited a wide-ranging willingness to absorb exports from China in the 2018 trade war. See [this paper for more](https://cepr.org/voxeu/columns/redirecting-chinese-exports-us-evidence-trade-deflection-first-us-china-trade-war).
    > _"There is huge variation across importing nations in the scale of Chinese trade deflection and in their appetite for absorbing extra imports without taking defensive action."_
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Results

    Summarising - we do not identify a clear case of trade diversion from China to the UK, particularly as compared to the rest of the world, as a result of the 2018 US-China trade war.

    More specifically - we do not find evidence to reject the hypothesis, that over 2018, 2019, and 2020 there was a significant difference from 0 for the aggregate cross-elasticity of UK imports from China of specific goods to US tariffs on those goods.

    The rest of this section is structured as follows:

    1. First, we identify the impact on US imports from China of the US tariffs.
    2. Second, we compare this against the impact on the RoW - a select basket of countries. These first two sections serve to identify the efficacy of our methodology.
    3. Third, we look at the impact on UK imports of Chinese goods of the US tariffs.
    4. Fourth, we examine countries which exhibited significant trade diversion
    5. Fifth, we examine how correlated this is to the endogenous protectionist policy response by the relevant country
    6. Finally, we conclude as to what we can apply from this to the current trade war, discussing limitations and conditioning on the policy response.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The impact on US imports from China of the US tariffs.

    We construct three regressions to assess this, each targeting a different dependent variable - Total bilateral trade value in USD, total quantity in SI units, and a unit value price constructed by dividing the total value by total quantity.

    Turning first to total traded value...
    """
    )
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [
            str(y) for y in range(2017, 2024)
        ]  # 1 year less than the range of the data

        regressors = " + ".join(
            f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE
        )
        formula = f"log(value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    In this case, our control group is as follows:

    - Bilateral trade values between non-US and non-China in both tariffed and non-tariffed goods
    - US imports of non-tariffed goods from China
    - US imports of tariffed goods from countries other than China

    Our sample _does_ include some goods which were tariffed 

    - countries, including the US and countries other than China, of goods which were tariffed.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Extrapolating to the current trade war

    In this section we attempt to extrapolate, from historical data, what the effects of the escalation of the trade war over 2025 might be. 

    ### Limitations
    This time is different? What arguments can we motivate for that?
    """
    )
    return


if __name__ == "__main__":
    app.run()
