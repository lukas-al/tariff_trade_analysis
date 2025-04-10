import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # CEPII BACI Dataset
        BACI provides data on bilateral trade flows for 200 countries at the product level (5000 products). Products correspond to the "Harmonized System" nomenclature (6 digit code). The dataset page is available [here](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37).


        Paper on it's manufacture available [here](https://www.cepii.fr/CEPII/en/publications/wp/abstract.asp?NoDoc=2726)

        FAQs available [here](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/FAQ_BACI.html)

        Documentation available [here](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/DescriptionBACI.html)

        The dataset was last updated in January 2025.

        ---

        List of Variables:

        - t: year
        - i: exporter
        - j: importer
        - k: product
        - v: value
        - q: quantity
        """
    )
    return


@app.cell
def _():
    import duckdb
    import marimo as mo
    import polars as pl

    import mpil_tariff_trade_analysis as mtta

    return duckdb, mo, mtta, pl


@app.cell
def _():
    from mpil_tariff_trade_analysis.etl.baci import (
        aggregate_baci,
        baci_to_parquet,
        baci_to_parquet_incremental,
    )

    hs = "HS92"
    release = "202501"

    # Convert BACI data to parquet format for further processing
    try:
        baci_to_parquet_incremental(
            hs,
            release,
            input_folder="data/raw",
            output_folder="data/final",
        )
    except PermissionError as e:
        print(f"Parquet file already exists: \n {e}")
    return (
        aggregate_baci,
        baci_to_parquet,
        baci_to_parquet_incremental,
        hs,
        release,
    )


@app.cell
def _(aggregate_baci, hs, release):
    # Aggregate BACI data by country
    aggregate_baci(
        input=f"data/final/BACI_{hs}_V{release}",
        output=f"data/final/BACI_{hs}_V{release}-2digit.parquet",
        aggregation="2digit",  # 2digit #4digit
    )
    return


@app.cell
def _(duckdb, hs, release):
    # View some summary statistics
    duckdb.sql(
        f"SELECT * FROM read_parquet('data/final/BACI_{hs}_V{release}-total.parquet')"
    ).show()
    return


@app.cell
def _(hs, pl, release):
    query = (
        pl.scan_parquet(f"data/final/BACI_{hs}_V{release}")
        # .collect()
        .describe()
    )
    return (query,)


if __name__ == "__main__":
    app.run()
