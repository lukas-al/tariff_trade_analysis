import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pprint import pprint
    return mo, pl, pprint


@app.cell
def _(mo):
    mo.md(r"""# Explore our unified dataset""")
    return


@app.cell
def _(pl):
    unified_data = pl.scan_parquet(
        'data/final/unified_trade_tariff_partitioned/',
    )
    return (unified_data,)


@app.cell
def _(unified_data):
    unified_data.describe()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
