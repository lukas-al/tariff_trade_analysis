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
    import pycountry
    import re
    import matplotlib.pyplot as plt
    from typing import Optional

    return Optional, mo, pl, pycountry, pyfixest, re


@app.cell
def _(mo):
    mo.md(
        r"""
    # Refine the panel regressions

    1. Run the new regression
    2. Re-run it, with the following:

    - Varying control groups
    - Varying fixed effects
    - RoW rather than the US
    - Varying the measurement period of the effect
    - Experimenting with clustered std errors.
    - Placebo tests

    Summarise the results and understand what the optimal specification is. What's the actual result?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Function

    One simple function to rule them all...
    """
    )
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
        )

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
def _(mo):
    mo.md(
        r"""
    ## Data prep pipeline
    We need to filter down the data, as required for our analysisß
    """
    )
    return


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
def _(mo):
    mo.md(r"""## Run the original regression spec we defined""")
    return


@app.cell
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")

    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _():
    EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]
    return (EFFECT_YEAR_RANGE,)


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    mo,
    prepare_analysis_data,
    run_direct_effect_regression,
    unified_lf,
):
    # 1. Prepare the analysis data with specific parameters
    analysis_lf = prepare_analysis_data(
        source_lf=unified_lf,
        top_n=40,
        selection_year="2017",
        year_range_to_keep=[str(y) for y in range(2016, 2024)],
        selection_method="total_trade",
        oil_export_threshold=50.0,
        countries_to_exclude=["643", "344"],  # Russia, Hong Kong
    )

    # 2. Build the formula
    interaction_name = "UK_from_China"
    importer_list = ["826"]  # UK
    exporter_list = ["156"]  # China
    # 1 year less than the range of the data

    regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
    formula = f"log(value) ~ {regressors} | importer^year^product_code + exporter^year^product_code + importer^exporter"
    print(f"Formula for model:\n{formula}")

    # 3. Run the model
    model, etable, plot = run_direct_effect_regression(
        interaction_term_name=interaction_name,
        interaction_importers=importer_list,
        interaction_exporters=exporter_list,
        year_range=EFFECT_YEAR_RANGE,
        formula=formula,
    )

    # 4. View the results
    mo.vstack([etable, plot, model.summary()])
    return (analysis_lf,)


@app.cell
def _(mo):
    mo.md(r"""# Experiment with alternative specs""")
    return


@app.cell
def _(mo):
    mo.md(r"""### UK - all FEs, whole sample, value""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(value) ~ {regressors} | importer^year^product_code + exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### UK - all FEs, whole sample, unit_value""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | importer^year^product_code + exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### UK - all FEs, whole sample, quantity""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | importer^year^product_code + exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Different fixed effects
    Remove jpt
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### no jpt - value""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### no jpt - unit value""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### no jpt - quantity""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## RoW rather than UK

    no jpt, value

    What is the effect when we consider RoW?
    """
    )
    return


@app.cell
def _(pl, unified_lf):
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
    return (RoW_list,)


@app.cell
def _(mo):
    mo.md(r"""### no jpt, value""")
    return


@app.cell
def _(RoW_list, analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2017, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "RoW_from_China"
        importer_list = RoW_list
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### No jpt, unit_value""")
    return


@app.cell
def _(RoW_list, analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2017, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "RoW_from_China"
        importer_list = RoW_list
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### RoW, no jpt, quantity""")
    return


@app.cell
def _(RoW_list, analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2017, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "RoW_from_China"
        importer_list = RoW_list
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""## US rather than UK""")
    return


@app.cell
def _(mo):
    mo.md(r"""### no ipt - value""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### no ipt - unit value""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | importer^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### no ipt - quantity""")
    return


@app.cell
def _(analysis_lf, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | importer^year^product_code + importer^exporter"
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

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reduce sample - only imports by the UK

    Essentially, vary the control groups by changing the sample of data which we're consuming
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Only imports by the UK - no jpt, value""")
    return


@app.cell
def _(analysis_lf, mo, pl, run_direct_effect_regression):
    def _(analysis_lf):
        # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        print(analysis_lf.head().collect())

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year later than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(value) ~ {regressors} | importer^exporter + product_code^year"
        print(f"Formula for model:\n{formula}")

        # 2.5 - Write the filter expression to leave only UK imports
        filter_expr = pl.col("partner_country") == "826"  # UK"

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
            filter_expression=filter_expr,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])

    _(analysis_lf)
    return


@app.cell
def _(mo):
    mo.md(r"""### Only imports by the UK - no jpt, unit_value""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, pl, run_direct_effect_regression):
    def _(analysis_lf):
        # # 1. Prepare the analysis data with specific parameters
        # analysis_lf = prepare_analysis_data(
        #     source_lf=unified_lf,
        #     top_n=40,
        #     selection_year="2017",
        #     year_range_to_keep=[str(y) for y in range(2016, 2024)],
        #     selection_method="total_trade",
        #     oil_export_threshold=50.0,
        #     countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        # )

        print(analysis_lf.head().collect())

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year later than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | importer^exporter + product_code^year"
        print(f"Formula for model:\n{formula}")

        # 2.5 - Write the filter expression to leave only UK imports in the input data
        filter_expr = pl.col("partner_country") == "826"  # UK"

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
            filter_expression=filter_expr,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### Only imports by the UK - no jpt, quantity""")
    return


@app.cell
def _(mo, pl, prepare_analysis_data, run_direct_effect_regression, unified_lf):
    def _():
        # 1. Prepare the analysis data with specific parameters
        analysis_lf = prepare_analysis_data(
            source_lf=unified_lf,
            top_n=40,
            selection_year="2017",
            year_range_to_keep=[str(y) for y in range(2016, 2024)],
            selection_method="total_trade",
            oil_export_threshold=50.0,
            countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        )

        print(analysis_lf.head().collect())

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year later than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | importer^exporter + product_code^year"
        print(f"Formula for model:\n{formula}")

        # 2.5 - Write the filter expression to leave only UK imports
        filter_expr = pl.col("partner_country") == "826"  # UK"

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
            filter_expression=filter_expr,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Reduce sample - only tariffed products""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Only tariffed products - no jpt, value""")
    return


@app.cell
def _(mo, pl, prepare_analysis_data, run_direct_effect_regression, unified_lf):
    def _():
        # 1. Prepare the analysis data with specific parameters
        analysis_lf = prepare_analysis_data(
            source_lf=unified_lf,
            top_n=40,
            selection_year="2017",
            year_range_to_keep=[str(y) for y in range(2016, 2024)],
            selection_method="total_trade",
            oil_export_threshold=50.0,
            countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        )

        print(analysis_lf.head().collect())

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year later than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(value) ~ {regressors} | importer^exporter + exporter^product_code^year"
        print(f"Formula for model:\n{formula}")

        # 2.5 - Write the filter expression to leave only tariffed goods...
        cm_tariffs = pl.read_csv("data/intermediate/cm_us_tariffs.csv")
        cm_tariffs_list = cm_tariffs["product_code"].cast(pl.Utf8).to_list()
        filter_expr = pl.col("product_code").is_in(cm_tariffs_list)

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
            filter_expression=filter_expr,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""### Only tariffed products - no jpt, unit_value""")
    return


@app.cell
def _(mo, pl, prepare_analysis_data, run_direct_effect_regression, unified_lf):
    def _():
        # 1. Prepare the analysis data with specific parameters
        analysis_lf = prepare_analysis_data(
            source_lf=unified_lf,
            top_n=40,
            selection_year="2017",
            year_range_to_keep=[str(y) for y in range(2016, 2024)],
            selection_method="total_trade",
            oil_export_threshold=50.0,
            countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        )

        print(analysis_lf.head().collect())

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year later than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | importer^exporter + exporter^product_code^year"
        print(f"Formula for model:\n{formula}")

        # 2.5 - Write the filter expression to leave only tariffed goods...
        cm_tariffs = pl.read_csv("data/intermediate/cm_us_tariffs.csv")
        cm_tariffs_list = cm_tariffs["product_code"].cast(pl.Utf8).to_list()
        filter_expr = pl.col("product_code").is_in(cm_tariffs_list)

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
            filter_expression=filter_expr,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""### Only tariffed products - no jpt, quantity""")
    return


@app.cell
def _(mo, pl, prepare_analysis_data, run_direct_effect_regression, unified_lf):
    def _():
        # 1. Prepare the analysis data with specific parameters
        analysis_lf = prepare_analysis_data(
            source_lf=unified_lf,
            top_n=40,
            selection_year="2017",
            year_range_to_keep=[str(y) for y in range(2016, 2024)],
            selection_method="total_trade",
            oil_export_threshold=50.0,
            countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        )

        print(analysis_lf.head().collect())

        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year later than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | importer^exporter + exporter^product_code^year"
        print(f"Formula for model:\n{formula}")

        # 2.5 - Write the filter expression to leave only tariffed goods...
        cm_tariffs = pl.read_csv("data/intermediate/cm_us_tariffs.csv")
        cm_tariffs_list = cm_tariffs["product_code"].cast(pl.Utf8).to_list()
        filter_expr = pl.col("product_code").is_in(cm_tariffs_list)

        # 3. Run the model
        model, etable, plot = run_direct_effect_regression(
            data=analysis_lf,
            interaction_term_name=interaction_name,
            interaction_importers=importer_list,
            interaction_exporters=exporter_list,
            year_range=EFFECT_YEAR_RANGE,
            formula=formula,
            filter_expression=filter_expr,
        )

        # 4. View the results
        return mo.vstack([etable, plot, model.summary()])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Clustered errors

    We must cluster std errors at the level of the treatment assignment or at the level of the unit that has correlated shocks.

    The requirement with clustering standard errors, is to account for serial correlation within groups of the error term. We must cluster these at the grouping where the treatment is applied - which is product. 

    So we start by experimenting with clustered std errors on product. We can then improve this by clustering them on importer-year, since we're interested in imports over time. We can then expand the structure

    Interpreting the outputs, we should expect wider confidence bars, lower t values, and lower signifcance values (higher p's). We should report these alongside the un-clustered structure of our regression. If the results remain broadly the same (signs, etc) then we have a valid robustness check on our results.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## "CRV1": "product_code"

    Takes hours...
    """
    )
    return


@app.cell
def _():
    # def _():
    #     # 1. Prepare the analysis data with specific parameters
    #     analysis_lf = prepare_analysis_data(
    #         source_lf=unified_lf,
    #         top_n=40,
    #         selection_year="2017",
    #         year_range_to_keep=[str(y) for y in range(2016, 2024)],
    #         selection_method="total_trade",
    #         oil_export_threshold=50.0,
    #         countries_to_exclude=["643", "344"],  # Russia, Hong Kong
    #     )

    #     # 2. Build the formula
    #     interaction_name = "UK_from_China"
    #     importer_list = ["826"]  # UK
    #     exporter_list = ["156"]  # China
    #     EFFECT_YEAR_RANGE = [
    #         str(y) for y in range(2017, 2024)
    #     ]  # 1 year less than the range of the data

    #     regressors = " + ".join(
    #         f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE
    #     )
    #     formula = f"log(value) ~ {regressors} | exporter^year^product_code + importer^exporter"
    #     print(f"Formula for model:\n{formula}")

    #     # 3. Run the model
    #     model, etable, plot = run_direct_effect_regression(
    #         data=analysis_lf,
    #         interaction_term_name=interaction_name,
    #         interaction_importers=importer_list,
    #         interaction_exporters=exporter_list,
    #         year_range=EFFECT_YEAR_RANGE,
    #         formula=formula,
    #         vcov={"CRV1": "product_code"},
    #     )

    #     # 4. View the results
    #     return mo.vstack([etable, plot, model.summary()])

    # _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Limiting country & product set
    To replicate the Hoang-Mix paper, we need to limit the set of countries we're considering. 

    The reason this is important, is when we have the global control set, we're including countries which _also_ had changes in tariffs. While my data includes a lot of these, it's possible some tariffs aren't included given WITS' limitations. This would reduce the size our detected effect.

    > Based on the research paper "Trade Wars and Rumors of Trade Wars," the "rest of world" sample includes seven countries: Brazil, France, Germany, Ireland, Italy, South Africa, and the United Kingdom.
    > The authors selected these countries by first taking the top 20 countries from which the U.S. or China imported the most in 2017. From this list, they removed any country that has ever had a preferential trade agreement with either the U.S. or China.

    Similarly, the Hoang-Mix paper doesn't include category 27 goods - steel & aluminium related ones. This is intentional, given those tariffs were applied across all countries - confounding our control sample.

    > The authors removed products for which the U.S. applied more widespread tariffs, specifically mentioning steel and washing machines, from their analysis. This was done to isolate the effect of the U.S.-China bilateral tariffs.

    In the following section, we replicate that approach directly. This will inform our final approach which we present in the results.
    """
    )
    return


@app.cell(hide_code=True)
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

    HM_COUNTRY_LIST = [
        USA_CC,
        CHINA_CC,
        UK_CC,
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        GERMANY_CC,
        FRANCE_CC,
    ]

    HM_ROW_LIST = [
        UK_CC,
        BRAZIL_CC,
        IRELAND_CC,
        ITALY_CC,
        SOUTHAFRICA_CC,
        GERMANY_CC,
        FRANCE_CC,
    ]

    HM_COUNTRY_LIST_EXPANDED = [
        USA_CC,
        CHINA_CC,
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

    HM_ROW_LIST_EXPANDED = [
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
    return HM_COUNTRY_LIST, HM_COUNTRY_LIST_EXPANDED, HM_ROW_LIST


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
def _(
    HM_COUNTRY_LIST,
    alu_steel_product_codes,
    prepare_analysis_data,
    unified_lf,
):
    analysis_lf_HMREP = prepare_analysis_data(
        source_lf=unified_lf,
        # top_n=40,
        # selection_year="2017",
        year_range_to_keep=[str(y) for y in range(2016, 2024)],
        selection_method="total_trade",
        oil_export_threshold=50.0,
        countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        countries_to_include=HM_COUNTRY_LIST,
        product_codes_to_exclude=alu_steel_product_codes,
    )
    return (analysis_lf_HMREP,)


@app.cell
def _(mo):
    mo.md(r"""## US no jpt, limited countries, removed alu + steel""")
    return


@app.cell
def _(mo):
    mo.md(r"""### value""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### price""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### quantity""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "USA_from_China"
        importer_list = ["840"]  # USA
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""## UK - no jpt, limited countries, removed alu+steel""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Value""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### Price""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### Quantity""")
    return


@app.cell
def _(analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "UK_from_China"
        importer_list = ["826"]  # UK
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""## RoW - no jpt, limited countries, removed alu+steel""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Value""")
    return


@app.cell
def _(HM_ROW_LIST, analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "ROW_from_China"
        importer_list = HM_ROW_LIST
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### Price""")
    return


@app.cell
def _(HM_ROW_LIST, analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "ROW_from_China"
        importer_list = HM_ROW_LIST
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(unit_value) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(r"""### Quantity""")
    return


@app.cell
def _(HM_ROW_LIST, analysis_lf_HMREP, mo, run_direct_effect_regression):
    def _(analysis_lf):
        # 2. Build the formula
        interaction_name = "ROW_from_China"
        importer_list = HM_ROW_LIST
        exporter_list = ["156"]  # China
        EFFECT_YEAR_RANGE = [str(y) for y in range(2017, 2024)]  # 1 year less than the range of the data

        regressors = " + ".join(f"{interaction_name}_{year}" for year in EFFECT_YEAR_RANGE)
        formula = f"log(quantity) ~ {regressors} | exporter^year^product_code + importer^exporter"
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

    _(analysis_lf_HMREP)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Individual countries within RoW

    We now isolate individual countries out, as we have the UK and US, from this RoW breakdown. Which countries were more exposed to Chinese dumping, and does that correlate with how protectionist they were?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## HM sample - value""")
    return


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    HM_ROW_LIST,
    analysis_lf_HMREP,
    mo,
    pycountry,
    run_direct_effect_regression,
):
    def _(analysis_lf_HMREP):
        models = {}
        model_renamer = {}
        for country_code in mo.status.progress_bar(HM_ROW_LIST):
            # Use pycountry to get a descriptive key (e.g., "USA")
            country_key = pycountry.countries.get(numeric=country_code).alpha_3
            interaction_term_name = f"{country_key}_from_China"

            # Dynamically build the formula for the current country
            regressors = "+".join(f"{interaction_term_name}_{year}" for year in EFFECT_YEAR_RANGE)
            formula = f"log(value)~{regressors}|exporter^year^product_code+importer^exporter"

            print(f"Running for {country_key}")
            model, _, _ = run_direct_effect_regression(
                data=analysis_lf_HMREP,
                interaction_term_name=interaction_term_name,
                interaction_importers=[country_code],
                interaction_exporters=["156"],  # Chiina
                year_range=[str(y) for y in range(2017, 2024)],
                formula=formula,
            )

            model_renamer[formula] = country_key
            models[country_key] = model

        return models, model_renamer

    models_value, model_renamer_value = _(analysis_lf_HMREP)
    return model_renamer_value, models_value


@app.cell
def _(mo, model_renamer_value, models_value, pyfixest):
    mo.vstack(
        [
            mo.md("--- Aggregated Regression Results ---"),
            pyfixest.etable(models=models_value.values(), model_heads=models_value.keys()),
            mo.md("--- Aggregated Coefficient Plot ---"),
            pyfixest.coefplot(
                models=models_value.values(),
                rename_models=model_renamer_value,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## HM sample - unit_value""")
    return


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    HM_ROW_LIST,
    analysis_lf_HMREP,
    mo,
    pycountry,
    run_direct_effect_regression,
):
    def _(analysis_lf_HMREP):
        models = {}
        model_renamer = {}
        for country_code in mo.status.progress_bar(HM_ROW_LIST):
            # Use pycountry to get a descriptive key (e.g., "USA")
            country_key = pycountry.countries.get(numeric=country_code).alpha_3
            interaction_term_name = f"{country_key}_from_China"

            # Dynamically build the formula for the current country
            regressors = "+".join(f"{interaction_term_name}_{year}" for year in EFFECT_YEAR_RANGE)
            formula = f"log(unit_value)~{regressors}|exporter^year^product_code+importer^exporter"

            print(f"Running for {country_key}")
            model, _, _ = run_direct_effect_regression(
                data=analysis_lf_HMREP,
                interaction_term_name=interaction_term_name,
                interaction_importers=[country_code],
                interaction_exporters=["156"],  # Chiina
                year_range=[str(y) for y in range(2017, 2024)],
                formula=formula,
            )

            model_renamer[formula] = country_key
            models[country_key] = model

        return models, model_renamer

    models_price, model_renamer_price = _(analysis_lf_HMREP)
    return model_renamer_price, models_price


@app.cell
def _(mo, model_renamer_price, models_price, pyfixest):
    mo.vstack(
        [
            mo.md("--- Aggregated Regression Results ---"),
            pyfixest.etable(models=models_price.values(), model_heads=models_price.keys()),
            mo.md("--- Aggregated Coefficient Plot ---"),
            pyfixest.coefplot(
                models=models_price.values(),
                rename_models=model_renamer_price,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## HM sample - quantity""")
    return


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    HM_ROW_LIST,
    analysis_lf_HMREP,
    mo,
    pycountry,
    run_direct_effect_regression,
):
    def _(analysis_lf_HMREP):
        models = {}
        model_renamer = {}
        for country_code in mo.status.progress_bar(HM_ROW_LIST):
            # Use pycountry to get a descriptive key (e.g., "USA")
            country_key = pycountry.countries.get(numeric=country_code).alpha_3
            interaction_term_name = f"{country_key}_from_China"

            # Dynamically build the formula for the current country
            regressors = "+".join(f"{interaction_term_name}_{year}" for year in EFFECT_YEAR_RANGE)
            formula = f"log(quantity)~{regressors}|exporter^year^product_code+importer^exporter"

            print(f"Running for {country_key}")
            model, _, _ = run_direct_effect_regression(
                data=analysis_lf_HMREP,
                interaction_term_name=interaction_term_name,
                interaction_importers=[country_code],
                interaction_exporters=["156"],  # Chiina
                year_range=[str(y) for y in range(2017, 2024)],
                formula=formula,
            )

            model_renamer[formula] = country_key
            models[country_key] = model

        return models, model_renamer

    models_qty, model_renamer_qty = _(analysis_lf_HMREP)
    return model_renamer_qty, models_qty


@app.cell
def _(mo, model_renamer_qty, models_qty, pyfixest):
    mo.vstack(
        [
            mo.md("--- x Regression Results ---"),
            pyfixest.etable(models=models_qty.values(), model_heads=models_qty.keys()),
            mo.md("--- Aggregated Coefficient Plot ---"),
            pyfixest.coefplot(
                models=models_qty.values(),
                rename_models=model_renamer_qty,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""## Expanded RoW sample - to align with [this voxEU paper](https://cepr.org/voxeu/columns/redirecting-chinese-exports-us-evidence-trade-deflection-first-us-china-trade-war)"""
    )
    return


@app.cell
def _(
    HM_COUNTRY_LIST_EXPANDED,
    alu_steel_product_codes,
    prepare_analysis_data,
    unified_lf,
):
    analysis_lf_HMREP_EXP = prepare_analysis_data(
        source_lf=unified_lf,
        year_range_to_keep=[str(y) for y in range(2016, 2024)],
        selection_method="total_trade",
        oil_export_threshold=50.0,
        countries_to_exclude=["643", "344"],  # Russia, Hong Kong
        countries_to_include=HM_COUNTRY_LIST_EXPANDED,
        product_codes_to_exclude=alu_steel_product_codes,
    )
    return (analysis_lf_HMREP_EXP,)


@app.cell
def _(mo):
    mo.md(r"""## Expanded RoW sample - value""")
    return


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    HM_ROW_LIST,
    analysis_lf_HMREP_EXP,
    mo,
    pycountry,
    pyfixest,
    run_direct_effect_regression,
):
    def _(analysis_lf_HMREP_EXP):
        models = {}
        model_renamer = {}
        for country_code in mo.status.progress_bar(HM_ROW_LIST):
            # Use pycountry to get a descriptive key (e.g., "USA")
            country_key = pycountry.countries.get(numeric=country_code).alpha_3
            interaction_term_name = f"{country_key}_from_China"

            # Dynamically build the formula for the current country
            regressors = "+".join(f"{interaction_term_name}_{year}" for year in EFFECT_YEAR_RANGE)
            formula = f"log(value)~{regressors}|exporter^year^product_code+importer^exporter"

            print(f"Running for {country_key}")
            model, _, _ = run_direct_effect_regression(
                data=analysis_lf_HMREP_EXP,
                interaction_term_name=interaction_term_name,
                interaction_importers=[country_code],
                interaction_exporters=["156"],  # Chiina
                year_range=[str(y) for y in range(2017, 2024)],
                formula=formula,
            )

            model_renamer[formula] = country_key
            models[country_key] = model

        return mo.vstack(
            [
                mo.md("--- x Regression Results ---"),
                pyfixest.etable(models=models.values(), model_heads=models.keys()),
                mo.md("--- Aggregated Coefficient Plot ---"),
                pyfixest.coefplot(
                    models=models.values(),
                    rename_models=models,
                ),
            ]
        )

    _(analysis_lf_HMREP_EXP)
    return


@app.cell
def _(mo):
    mo.md(r"""## Expanded RoW Sample - UnitValue""")
    return


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    HM_ROW_LIST,
    analysis_lf_HMREP_EXP,
    mo,
    pycountry,
    pyfixest,
    run_direct_effect_regression,
):
    def _(analysis_lf_HMREP_EXP):
        models = {}
        model_renamer = {}
        for country_code in mo.status.progress_bar(HM_ROW_LIST):
            # Use pycountry to get a descriptive key (e.g., "USA")
            country_key = pycountry.countries.get(numeric=country_code).alpha_3
            interaction_term_name = f"{country_key}_from_China"

            # Dynamically build the formula for the current country
            regressors = "+".join(f"{interaction_term_name}_{year}" for year in EFFECT_YEAR_RANGE)
            formula = f"log(unit_value)~{regressors}|exporter^year^product_code+importer^exporter"

            print(f"Running for {country_key}")
            model, _, _ = run_direct_effect_regression(
                data=analysis_lf_HMREP_EXP,
                interaction_term_name=interaction_term_name,
                interaction_importers=[country_code],
                interaction_exporters=["156"],  # Chiina
                year_range=[str(y) for y in range(2017, 2024)],
                formula=formula,
            )

            model_renamer[formula] = country_key
            models[country_key] = model

        return mo.vstack(
            [
                mo.md("--- x Regression Results ---"),
                pyfixest.etable(models=models.values(), model_heads=models.keys()),
                mo.md("--- Aggregated Coefficient Plot ---"),
                pyfixest.coefplot(
                    models=models.values(),
                    rename_models=models,
                ),
            ]
        )

    _(analysis_lf_HMREP_EXP)
    return


@app.cell
def _(mo):
    mo.md(r"""## Expanded RoW Sample: quantity""")
    return


@app.cell
def _(
    EFFECT_YEAR_RANGE,
    HM_ROW_LIST,
    analysis_lf_HMREP_EXP,
    mo,
    pycountry,
    pyfixest,
    run_direct_effect_regression,
):
    def _(analysis_lf_HMREP_EXP):
        models = {}
        model_renamer = {}
        for country_code in mo.status.progress_bar(HM_ROW_LIST):
            # Use pycountry to get a descriptive key (e.g., "USA")
            country_key = pycountry.countries.get(numeric=country_code).alpha_3
            interaction_term_name = f"{country_key}_from_China"

            # Dynamically build the formula for the current country
            regressors = "+".join(f"{interaction_term_name}_{year}" for year in EFFECT_YEAR_RANGE)
            formula = f"log(quantity)~{regressors}|exporter^year^product_code+importer^exporter"

            print(f"Running for {country_key}")
            model, _, _ = run_direct_effect_regression(
                data=analysis_lf_HMREP_EXP,
                interaction_term_name=interaction_term_name,
                interaction_importers=[country_code],
                interaction_exporters=["156"],  # Chiina
                year_range=[str(y) for y in range(2017, 2024)],
                formula=formula,
            )

            model_renamer[formula] = country_key
            models[country_key] = model

        return mo.vstack(
            [
                mo.md("--- x Regression Results ---"),
                pyfixest.etable(models=models.values(), model_heads=models.keys()),
                mo.md("--- Aggregated Coefficient Plot ---"),
                pyfixest.coefplot(
                    models=models.values(),
                    rename_models=models,
                ),
            ]
        )

    _(analysis_lf_HMREP_EXP)
    return


if __name__ == "__main__":
    app.run()
