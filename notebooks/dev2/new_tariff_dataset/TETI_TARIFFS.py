import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # Incorporate the Teti tariff database

    Available [here](https://feodorateti.github.io/data.html), the Teti tariff dataset is a more refined version of the WITS tariff data commonly used in empirical analysis of the effects of tariffs on trade. 

    WITS has many issues, as outlined in the [accompanying paper](https://feodorateti.github.io/docs/Teti_MissingTariffs_2024_12_19.pdf). Feodora Teti resolves these by merging a range of tariff databases and implementing a more powerfull interpolation algorithm than the WITS version.

    This notebook develops how I would replace the WITS data with Teti's world tariff data.
    """
    )
    return


@app.cell
def _(pl):
    raw_tariffs = pl.scan_csv("data/raw/Pairs vbeta1 Dec 2024/tariffsPairs_88_21_vbeta1-2024-12.csv")

    raw_tariffs.head(1000).collect()
    return (raw_tariffs,)


@app.cell
def _(pl, raw_tariffs):
    print(f"Num rows in dataset: {raw_tariffs.select(pl.len()).collect().item()}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Finding
    The data available online is aggregated. I need the disaggregated data. Have emailed to ask for it. 
    """
    )
    return


if __name__ == "__main__":
    app.run()
