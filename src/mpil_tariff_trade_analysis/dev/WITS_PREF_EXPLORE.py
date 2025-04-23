import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import pandas as pd

    from ydata_profiling import ProfileReport
    return ProfileReport, alt, mo, pd, pl


@app.cell
def _(pl):
    # Load the wits pref data. Do we have issues in it?

    lf = pl.scan_parquet("data/intermediate/WITS_AVEPref.parquet")

    lf.describe()
    return (lf,)


if __name__ == "__main__":
    app.run()
