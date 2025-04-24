import marimo

__generated_with = "0.12.8"
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
        # WITS PIPELINE
        Implement the WITS pipeline from start to finish
        """
    )
    return


if __name__ == "__main__":
    app.run()
