import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from typing import Optional

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    import polars as pl
    import pycountry
    import pyfixest

    return Optional, go, mo, pd, pl, pycountry, pyfixest, re


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
            raise ValueError(f"Could not parse dependent variable from formula: {formula}")

        tariff_expr = (
            pl.col("average_tariff_official")
            .filter((pl.col("partner_country") == USA_CC) & (pl.col("reporter_country") == CHINA_CC))
            .mean()
            .over(["year", "product_code"])
            .alias("tariff_us_china")
        )

        input_lf = data.with_columns(
            pl.col("partner_country").alias("importer"),
            pl.col("reporter_country").alias("exporter"),
            tariff_expr,
        ).filter(pl.col("year").is_in(year_range))

        interaction_filter = (pl.col("importer").is_in(interaction_importers)) & (pl.col("exporter").is_in(interaction_exporters))

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

        print(f"Checking for nulls in dependent variable '{dependent_var_col}' and 'tariff_us_china'.")
        clean_df = final_lf.drop_nulls(subset=[dependent_var_col, "tariff_us_china"]).collect()

        model = pyfixest.feols(fml=formula, data=clean_df, vcov=vcov)

        etable = pyfixest.etable(model)
        coefplot = model.coefplot(joint=True, plot_backend="matplotlib")

        return model, etable, coefplot

    return (run_direct_effect_regression,)


@app.cell
def _(pl):
    def get_oil_exporting_countries(lzdf: pl.LazyFrame, oil_export_percentage_threshold: float) -> list[str]:
        """
        Finds countries where oil products (HS code '27') exceed a certain
        percentage of their total export value.
        """
        print("--- Filtering out oil countries ---")

        total_exports = lzdf.group_by("reporter_country").agg(pl.sum("value").alias("total_value"))

        oil_exports = lzdf.filter(pl.col("product_code").str.starts_with("27")).group_by("reporter_country").agg(pl.sum("value").alias("oil_value"))

        summary = total_exports.join(oil_exports, on="reporter_country", how="left").with_columns(pl.col("oil_value").fill_null(0.0))

        summary = summary.with_columns(((pl.col("oil_value") / pl.col("total_value")) * 100).alias("oil_export_percentage"))

        filtered_countries = summary.filter(pl.col("oil_export_percentage") > oil_export_percentage_threshold)

        return filtered_countries.collect()["reporter_country"].to_list()

    return (get_oil_exporting_countries,)


@app.cell
def _(get_oil_exporting_countries, pl):
    def prepare_analysis_data(
        source_lf: pl.LazyFrame,
        top_n: int | None = None,
        selection_year: str | None = None,
        year_range_to_keep: list[str] | None = None,
        selection_method: str = "total_trade",
        oil_export_threshold: float | None = 50.0,
        countries_to_exclude: list[str] | None = None,
        countries_to_include: list[str] | None = None,
        product_codes_to_exclude: list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Filters and subsets the main trade dataset to create a sample for analysis.

        Args:
            source_lf: The initial LazyFrame of the full trade dataset.
            top_n: The number of top countries to select (e.g., 40).
            selection_year: The year used to determine the top countries.
            year_range_to_keep: A list of years to keep in the final dataset.
            selection_method: Method to determine top countries ("total_trade" or "importers").
            oil_export_threshold: Export percentage to classify a country as an oil exporter.
            countries_to_exclude: A list of country codes to remove.
            countries_to_include: A specific list of country codes to use for the analysis.
            product_codes_to_exclude: A list of product code prefixes (e.g., HS chapters)
                                      to remove from the dataset.

        Returns:
            A filtered LazyFrame.
        """
        if countries_to_include and (top_n or selection_year):
            raise ValueError("'countries_to_include' cannot be used with 'top_n' or 'selection_year'.")
        if not countries_to_include and not (top_n and selection_year):
            raise ValueError("Either 'countries_to_include' or both 'top_n' and 'selection_year' must be provided.")

        print("--- Cleaning data ---")

        lf = source_lf

        if product_codes_to_exclude:
            print(f"Excluding product codes starting with: {product_codes_to_exclude}")
            exclusion_expr = pl.any_horizontal(pl.col("product_code").str.starts_with(code) for code in product_codes_to_exclude)
            lf = lf.filter(~exclusion_expr)

        if oil_export_threshold is not None:
            oil_countries = get_oil_exporting_countries(lf, oil_export_threshold)
            lf = lf.filter(~pl.col("reporter_country").is_in(oil_countries))

        if countries_to_include:
            top_countries_list = countries_to_include
        else:
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
                exports_lf = trade_in_year_lf.select(pl.col("reporter_country").alias("country"), "value")
                imports_lf = trade_in_year_lf.select(pl.col("partner_country").alias("country"), "value")

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
                raise ValueError("selection_method must be 'importers' or 'total_trade'")

        if countries_to_exclude:
            top_countries_list = [c for c in top_countries_list if c not in countries_to_exclude]

        print(f"Final sample includes {len(top_countries_list)} countries.")

        analysis_lf = lf.filter(pl.col("reporter_country").is_in(top_countries_list) & pl.col("partner_country").is_in(top_countries_list))

        if year_range_to_keep:
            analysis_lf = analysis_lf.filter(pl.col("year").is_in(year_range_to_keep))

        return analysis_lf

    return (prepare_analysis_data,)


@app.cell
def _(go, pd):
    def plot_elasticity(model, keyword):
        coefficients = model.coef()
        conf_intervals = model.confint()

        # Find all coefficient names containing the keyword
        interaction_vars = [v for v in coefficients.index if keyword in v]

        if not interaction_vars:
            raise ValueError(f"No coefficients found with the keyword: '{keyword}'")

        # Build a dataframe with the relevant data
        df = pd.DataFrame(
            {
                "coefficient": coefficients[interaction_vars] * 100,
                "ci_lower": conf_intervals.loc[interaction_vars, "2.5%"] * 100,
                "ci_upper": conf_intervals.loc[interaction_vars, "97.5%"] * 100,
            }
        )

        # Calculate elasticity (E_s = -beta_s) and its CI
        elasticities_mean = df["coefficient"].values
        elasticities_ci_lower = df["ci_upper"].values
        elasticities_ci_upper = df["ci_lower"].values

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=elasticities_ci_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=elasticities_ci_lower,
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(31, 119, 180, 0.2)",
                fill="tonexty",
                showlegend=False,
                name="95% CI",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=elasticities_mean,
                mode="lines+markers",
                name="Estimated elasticity of trade response to tariffs in US-China trade war",
                line=dict(color="rgb(31, 119, 180)"),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            xaxis_title="Year Estimate",
            yaxis_title="Estimated Elasticity",
            hovermode="x unified",
            showlegend=False,
        )

        return fig

    return (plot_elasticity,)


@app.cell
def _(go, pd, px):
    def plot_elasticity_multi(models_dict) -> go.Figure:
        """
        Generates an elasticity plot comparing multiple PyFixest model objects.

        Args:
            models_dict: A dictionary where keys are descriptive names (and keywords
                         for interaction variables) and values are fitted Feols models.

        Returns:
            A Plotly Figure object showing estimated elasticities for all models.
        """
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, (name, model) in enumerate(models_dict.items()):
            color_rgb = colors[i % len(colors)]
            color_rgba = color_rgb.replace("rgb", "rgba").replace(")", ", 0.2)")

            coefficients = model.coef()
            conf_intervals = model.confint()

            # The dictionary key 'name' is used as the keyword
            interaction_vars = [v for v in coefficients.index if name in v]

            if not interaction_vars:
                print(f"Warning: No coefficients found for '{name}'. Skipping.")
                continue

            plot_df = pd.DataFrame(
                {
                    "coefficient": coefficients[interaction_vars],
                    "ci_lower": conf_intervals.loc[interaction_vars, "2.5%"],
                    "ci_upper": conf_intervals.loc[interaction_vars, "97.5%"],
                }
            )

            plot_df["year"] = plot_df.index.str.extract(r"(\d{4})").astype(int)
            plot_df = plot_df.sort_values("year")

            elasticities_mean = abs(plot_df["coefficient"])
            elasticities_ci_lower = -plot_df["ci_upper"]
            elasticities_ci_upper = -plot_df["ci_lower"]

            # Add traces for the current model
            fig.add_trace(
                go.Scatter(
                    x=plot_df["year"],
                    y=elasticities_ci_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_df["year"],
                    y=elasticities_ci_lower,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=color_rgba,
                    fill="tonexty",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_df["year"],
                    y=elasticities_mean,
                    name=name,
                    mode="lines+markers",
                    line=dict(color=color_rgb),
                    marker=dict(size=8),
                )
            )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Estimated Elasticity",
            hovermode="x unified",
            legend_title="Model",
            xaxis=dict(tickmode="linear"),
        )

        return fig

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
    KOREA_CC = pycountry.countries.search_fuzzy("Korea")[0].numeric
    TURKEY_CC = pycountry.countries.search_fuzzy("Turkiye")[0].numeric
    AUSTRALIA_CC = pycountry.countries.search_fuzzy("Australia")[0].numeric
    SAUDI_CC = pycountry.countries.search_fuzzy("Saudi Arabia")[0].numeric
    MEXICO_CC = pycountry.countries.search_fuzzy("Mexico")[0].numeric
    CANADA_CC = pycountry.countries.search_fuzzy("Canada")[0].numeric
    INDONESIA_CC = pycountry.countries.search_fuzzy("Indonesia")[0].numeric
    INDIA_CC = pycountry.countries.search_fuzzy("India")[0].numeric
    return (
        AUSTRALIA_CC,
        BRAZIL_CC,
        CANADA_CC,
        FRANCE_CC,
        GERMANY_CC,
        INDIA_CC,
        INDONESIA_CC,
        IRELAND_CC,
        ITALY_CC,
        KOREA_CC,
        MEXICO_CC,
        SAUDI_CC,
        SOUTHAFRICA_CC,
        TURKEY_CC,
        UK_CC,
    )


@app.cell(hide_code=True)
def _():
    alu_steel_product_codes = [
        # Steel Products
        # 720610 through 721650
        "720610",
        "720690",
        "720711",
        "720712",
        "720719",
        "720720",
        "720810",
        "720825",
        "720826",
        "720827",
        "720836",
        "720837",
        "720838",
        "720839",
        "720840",
        "720851",
        "720852",
        "720853",
        "720854",
        "720890",
        "720915",
        "720916",
        "720917",
        "720918",
        "720925",
        "720926",
        "720927",
        "720928",
        "720990",
        "721011",
        "721012",
        "721020",
        "721030",
        "721041",
        "721049",
        "721050",
        "721061",
        "721069",
        "721070",
        "721090",
        "721113",
        "721114",
        "721119",
        "721123",
        "721129",
        "721190",
        "721210",
        "721220",
        "721230",
        "721240",
        "721250",
        "721260",
        "721310",
        "721320",
        "721391",
        "721399",
        "721410",
        "721420",
        "721430",
        "721491",
        "721499",
        "721510",
        "721550",
        "721590",
        "721610",
        "721621",
        "721622",
        "721631",
        "721632",
        "721633",
        "721640",
        "721650",
        # 721699 through 730110
        "721699",
        "721710",
        "721720",
        "721730",
        "721790",
        "721810",
        "721891",
        "721899",
        "721911",
        "721912",
        "721913",
        "721914",
        "721921",
        "721922",
        "721923",
        "721924",
        "721931",
        "721932",
        "721933",
        "721934",
        "721935",
        "721990",
        "722011",
        "722012",
        "722020",
        "722090",
        "722100",
        "722211",
        "722219",
        "722220",
        "722230",
        "722240",
        "722300",
        "722410",
        "722490",
        "722511",
        "722519",
        "722530",
        "722540",
        "722550",
        "722591",
        "722592",
        "722599",
        "722611",
        "722619",
        "722620",
        "722691",
        "722692",
        "722699",
        "722710",
        "722720",
        "722790",
        "722810",
        "722820",
        "722830",
        "722840",
        "722850",
        "722860",
        "722870",
        "722880",
        "722920",
        "722990",
        "730110",
        # 730210
        "730210",
        # 730240 through 730290
        "730240",
        "730290",
        # 730410 through 730690
        "730411",
        "730419",
        "730422",
        "730423",
        "730424",
        "730429",
        "730431",
        "730439",
        "730441",
        "730449",
        "730451",
        "730459",
        "730490",
        "730511",
        "730512",
        "730519",
        "730520",
        "730531",
        "730539",
        "730590",
        "730611",
        "730619",
        "730621",
        "730629",
        "730630",
        "730640",
        "730650",
        "730661",
        "730669",
        "730690",
        # Aluminum Products
        # 7601 (Unwrought aluminum)
        "760110",
        "760120",
        # 7604 (Aluminum bars, rods, and profiles)
        "760410",
        "760421",
        "760429",
        # 7605 (Aluminum wire)
        "760511",
        "760519",
        "760521",
        "760529",
        # 7606 (Aluminum plates, sheets, and strip)
        "760611",
        "760612",
        "760691",
        "760692",
        # 7607 (Aluminum foil)
        "760711",
        "760719",
        "760720",
        # 7608 (Aluminum tubes and pipes)
        "760810",
        "760820",
        # 7609 (Aluminum tube or pipe fittings)
        "760900",
    ]
    return (alu_steel_product_codes,)


@app.cell
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")

    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _(alu_steel_product_codes, prepare_analysis_data, unified_lf):
    # 1. Prepare the analysis data with specific parameters
    analysis_lf = prepare_analysis_data(
        source_lf=unified_lf,
        top_n=40,
        selection_year="2017",
        year_range_to_keep=[str(y) for y in range(2016, 2024)],  # 2016 - 2023
        selection_method="total_trade",
        oil_export_threshold=50.0,
        countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        product_codes_to_exclude=alu_steel_product_codes,
    )
    return (analysis_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Indirect effects of Trade war 1 on the UK

    This notebook explores the effects of the 2018 US-China trade war on UK imports from China. We attempt to find evidence for the existence of dumping, and to extrapolate this evidence to identify an implied effect over trade war 2.

    ## Rationalle
    We are essentially trying to connect UK imports from China, to the application of tariffs by the US on Chinese goods in the 2018 trade war. We are attempting to determine if there is a causal link as follows:

    1. The US tariffs Chinese goods.
    2. In response to a) the increased effective price for US importers & consumers; b) other political pressures, the US demand for Chinese goods falls.
    3. Chinese exporters respond by reducing their price and identifying new trading relationships with other countries, including the UK.
    4. This results in an increased volume of Chinese goods, at a lower export price, being sent to the UK. This is called 'dumping'.

    ## Data
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
    mo.md(r"""## Methodology""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Approach 1
    To approach this question, we employ a causal inference assessment. We construct a panel of bilateral trade between a range of countries, where the treatement is the application of tariffs by the US on China of a specific good. Our observed outcome variable is a) the volume; b) the value; c) the unit value; of exports from China to the UK. 

    We are interested in the effect of the tariffs on UK imports, but include all global bilateral trade from the rest of the world as well. This means we implicitly construct control groups across two dimensions: a) imports of those products tariffed by the US from countries other than China which were not tariffed; b) imports from China of products which were not tariffed.

    We remove Steel and Aluminium products from the sample, as in Hoang-Mix, to control for confounders associated with the broad-based application of tariffs to all countries by the USA. 

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
    &\quad + \alpha_{j,p,t} + \lambda_{i,j} + \varepsilon_{i,j,p,t}
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
    * $\mathrm{Tariff}_{p,t}^{\mathrm{US-China}}$: The average official tariff rate between the US and China for product code $p$ in year $t$.
    * Fixed Effects: These terms control for unobserved heterogeneity between groups, including various sources of bias due to differences between our treated and untreated groups. They work by essentially de-meaning within a specific group, leaving only heterogeneity within that group.
        * $\alpha_{j,p,t}$: Represents importer-product-year fixed effects, controlling for any unobserved factors specific to each combination of an exporter $i$, product code $p$, and year $t$. 
        * $\lambda_{i,j}$: Represents importer-exporter fixed effects, controlling for unobserved factors specific to each pair of an importer $i$ and exporter $j$.
        <!-- * $\omega_{j,p,t}$: Represents exporter-product-year fixed effects, controlling for unobserved factors specific to each combination of an exporter $j$, product code $p$, and year $t$. -->
    * $\varepsilon_{i,j,p,t}$: The idiosyncratic error term for importer $i$, exporter $j$, product $p$, and year $t$. It captures all other unobserved factors not accounted for by the regressors or fixed effects.

    Following similar examples in the literature, we use unclustered heteroskedasticity-robust standard errors (Eicker-Huber-White).
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
    - Removing specific product lines with global tariffs applied
    - Varying fixed effects
    - RoW, US, and UK as the importer 
    - Varying the measurement period of the effect (various baseline years)
    - Placebo tests (TBD)
    - Clustering standard errors with varying specifications

    A range of alternate specifications for all models is contained in APPENDIX [X]
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Results

    Summarising - we do not identify a clear case of trade diversion from China to the UK, particularly as compared to the rest of the world, as a result of the 2018 US-China trade war.

    More specifically - we do not find evidence to reject the hypothesis, that over 2018, 2019, and 2020 there was a significant difference from 0 for the aggregate cross-elasticity of UK imports from China of specific goods to US tariffs on those goods. If there is any effect it, it is small and hidden at the aggregate level.

    _**UPDATE THIS SECTION**_

    The rest of this section is structured as follows:

    1. First, we identify the impact on US imports from China of the US tariffs.
    2. Second, we compare this against the impact on the RoW - a select basket of countries. These first two sections serve to assess the efficacy of our methodology against commonly-held priors.
    3. Third, we look at the impact on UK imports of Chinese goods of the US tariffs.
    4. Fourth, we examine countries which exhibited significant trade diversion
    5. Fifth, we examine how correlated this is to the endogenous protectionist policy response by the relevant country
    6. Finally, we conclude as to what we can apply from this to the current trade war, discussing limitations and conditioning on the policy response.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The impact on US imports from China of the US tariffs.

    We construct three regressions, each targeting a different dependent variable - Total bilateral trade value in USD, total quantity in SI units, and a unit value price constructed by dividing the total value by total quantity.

    Turning first to total traded value...
    """
    )
    return


@app.cell
def _(analysis_lf, mo, plot_elasticity, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity, model.summary()])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    As expected, the coefficient on the interaction term is negative, indicating that US imports from China of the tariffed goods fell in response to the tariffs. The estimated (partial) elasticity is around -0.009 in 2019, the first year following the tariffs suggesting a 1% increase in the tariff rate leads to a circa 0.9% decrease in US imports from China of those goods.

    This is aligned with our priors, and with various comparable literature [CITE CITE CITE].

    We see significant changes in the value of our estimated parameter over the years subsequent to 2019. We might expect a sustained monotonic reduction in the value of the coefficient, as supply chains continue to disentangle. 

    However, and as we will observe below, the magnitude of the shift and its consistency across countries suggests that the fixed effects may not be fully absorbing the effects of COVID, where supply chains globally and particularly from China were significantly affected. We therefore treat results from 2020 onwards with caution. Section [X] outlines some approaches we take to mitigate the impact of COVID [???].

    Next, looking at the quantity traded, in dimensioned units (which we aggregate over).
    """
    )
    return


@app.cell
def _(analysis_lf, mo, plot_elasticity, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(quantity) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The effect on total traded quantities is similar to that on total traded value.

    Turning next to price, or unit value.
    """
    )
    return


@app.cell
def _(analysis_lf, mo, plot_elasticity, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(unit_value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    A-priori, any detectable change in export prices from China to the US of US Tariffs would arise as a result of Chinese exporters absorbing the costs of tariffs to maintain competitiveness in the US. Other  research has broadly concluded that this phenomenon did not occur [CITE], and we should not expect to see prices decline markedly within our data. We should note that tariffs are collected post-import, and are not included in the total bilateral trade values included in BACI. 

    The observed positive sign on the coefficient in 2019 is therefore interesting, and suggests that far from absorbing the cost of tariffs - Chinese exporters may have increased prices at the margin. Alternatively, given the HS-6 product level is still aggregated, there may be within-group substitution effects in the basket of imported goods.

    There are many more arguments we could motivate from this data - but these are simply empirical results and we have not investigated this result further.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The impact on Chinese exports to the RoW of the US-China trade war

    We repeat the exercise above, but instead target all the RoW, which we define as all countries other than the US, China, Russia, Hong Kong, and the UK.

    A-priori, we would hypothesise the effect of the US-China trade war to be a) an increase in the volume of goods imported from China by the RoW, b) a reduction in those goods' unit price. The price channel here is most important.

    Running a regression targetting value, volume, and price below:
    """
    )
    return


@app.cell(hide_code=True)
def _(
    AUSTRALIA_CC,
    BRAZIL_CC,
    CANADA_CC,
    FRANCE_CC,
    GERMANY_CC,
    INDIA_CC,
    INDONESIA_CC,
    IRELAND_CC,
    ITALY_CC,
    KOREA_CC,
    MEXICO_CC,
    SAUDI_CC,
    SOUTHAFRICA_CC,
    TURKEY_CC,
    UK_CC,
    pl,
    unified_lf,
):
    RoW_list = unified_lf.select(pl.col("reporter_country").unique()).collect()
    RoW_list = RoW_list["reporter_country"].to_list()

    countries_to_remove = [
        "826",
        "156",
        "643",
        "344",
        "840",
    ]  # UK, China, Russia, Hong Kong, USA
    RoW_list = [item for item in RoW_list if item not in countries_to_remove]

    HM_RoW_list = [
        UK_CC,
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        GERMANY_CC,
        FRANCE_CC,
        KOREA_CC,
        TURKEY_CC,
        AUSTRALIA_CC,
        SAUDI_CC,
        MEXICO_CC,
        CANADA_CC,
        INDONESIA_CC,
        INDIA_CC,
    ]

    HM_RoW_list_mini = [
        UK_CC,
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        GERMANY_CC,
        FRANCE_CC,
    ]
    return HM_RoW_list_mini, RoW_list


@app.cell
def _(
    RoW_list,
    analysis_lf,
    mo,
    plot_elasticity,
    run_direct_effect_regression,
):
    def _():
        # 2. Build the formula
        interaction_name = "RoW_from_China"
        importer_list = RoW_list
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(
    RoW_list,
    analysis_lf,
    mo,
    plot_elasticity,
    run_direct_effect_regression,
):
    def _():
        # 2. Build the formula
        interaction_name = "RoW_from_China"
        importer_list = RoW_list
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(quantity) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(
    RoW_list,
    analysis_lf,
    mo,
    plot_elasticity,
    run_direct_effect_regression,
):
    def _():
        # 2. Build the formula
        interaction_name = "RoW_from_China"
        importer_list = RoW_list
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(unit_value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Taking these results together, we see a) prices stay the same in 2018, while export values / volumes from China to RoW fell, b) prices decrease consistently, though non-monotonically over 2019-2023, while export values from China to RoW increase, c) quantities increase consistently over the period 2019-2023, suggesting that Chinese exporters are successfully finding new markets for their goods.

    I'm uncertain how to explain the 2018 effect of the value of world imports from China reducing in 2018, contemporaneously with the application of the first tranche of tariffs in June. Throughout, we have not been detecting a serious tariff effect in 2018, which is possible to explain through rational expectations and delays in re-aligning supply chains. Given the structure of our fixed effects, we are observing a Chinese supply-side shock, which I do not know enough to motivate [HELP?]

    Outside of the effect on volumes/values in 2018, these results are broadly consistent with my priors.

    Turning now to our key question - the indirect impacts on the UK.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The impact on UK imports from China of the US-China trade war
    We run the same regressions as above, but targeting the UK-China trading relationship for the estimation of our parameters.
    """
    )
    return


@app.cell
def _(analysis_lf, mo, plot_elasticity, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(analysis_lf, mo, plot_elasticity, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(quantity) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(analysis_lf, mo, plot_elasticity, run_direct_effect_regression):
    def _():
        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China

        regressors = " + ".join(f"{interaction_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
        formula = f"log(unit_value) ~ {regressors} | importer^year^product_code + importer^exporter"
        print(f"Formula for model:\n{formula}")

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=[str(y) for y in range(2016, 2024)],
            formula=formula,
        )

        fig_elasticity = plot_elasticity(model, keyword=interaction_name)

        # 4. View the results
        return mo.vstack([etable, plot, fig_elasticity])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    [TBD] 

    > _What do we think? Does this constitute evidence of 'dumping' or import redirection? We observe a drop in prices over 2020, but not 2019. It should be noted that tariffs were applied in tranches from 2018-2019, with the number of products increasing._

    I find it hard to try justify any arguments of signficant dumping from China on the UK from this data. But that is to be expected, given the response of UK policy was to apply a) subsidies and b) import restrictions.

    Next we look across a basket of countries to try identify what their effect is like once we isolate them individually.
    """
    )
    return


@app.cell
def _(HM_RoW_list_mini, mo, pycountry):
    mo.md(
        rf"""
    ## Individual countries within RoW

    The list of countries we isolate out is as follows:

    {[pycountry.countries.get(numeric=cc).name for cc in HM_RoW_list_mini]}

    We perform the same test across value, quantity and price.
    """
    )
    return


@app.cell
def _():
    # def _(analysis_lf):
    #     models = {}
    #     model_renamer = {}
    #     for country_code in mo.status.progress_bar(HM_RoW_list_mini):
    #         country_key = pycountry.countries.get(numeric=country_code).alpha_3
    #         interaction_term_name = f"{country_key}_from_China"

    #         regressors = "+".join(f"{interaction_term_name}_{year}" for year in [str(y) for y in range(2017, 2024)])
    #         formula = f"log(value)~{regressors}|importer^year^product_code+importer^exporter"

    #         print(f"Running for {country_key}")
    #         model, _, _ = run_direct_effect_regression(
    #             data=analysis_lf,
    #             interaction_term_name=interaction_term_name,
    #             interaction_importers=[country_code],
    #             interaction_exporters=["156"],  # Chiina
    #             year_range=[str(y) for y in range(2016, 2024)],
    #             formula=formula,
    #         )

    #         model_renamer[formula] = country_key
    #         models[country_key] = model

    #     return models, model_renamer

    # models_value, model_renamer_value = _(analysis_lf)
    return


@app.cell
def _():
    # mo.vstack(
    #     [
    #         mo.md("--- Aggregated Regression Results ---"),
    #         pyfixest.etable(models=models_value.values(), model_heads=models_value.keys()),
    #         mo.md("--- Aggregated Coefficient Plot ---"),
    #         pyfixest.coefplot(
    #             models=models_value.values(),
    #             rename_models=model_renamer_value,
    #         ),
    #     ]
    # )
    return


@app.cell
def _():
    # fig_multi_value = plot_elasticity_multi(models=models_value, keyword="_from_China")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Intuition
    In this case, our control group is as follows:

    - Bilateral trade values between non-US and non-China in both tariffed and non-tariffed goods
    - US imports of non-tariffed goods from China
    - US imports of tariffed goods from countries other than China

    For each of our products.

    Our sample _does_ include some goods which were tariffed 

    - countries, including the US and countries other than China, of goods which were tariffed.
    """
    )
    return


if __name__ == "__main__":
    app.run()
