import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # MATCH DATA ACROSS WITS & BACI
        Combine into a single unified table

        Table structure:

        | Date | Source | Target | HS Code | Quantity | Value | Effective Tariff (AVE) |
        |------|--------|--------|---------|----------|-------|------------------------|
        |   X   |    X    |    X    |    X    |     X     |   X    |            X            |

        ## How?
        1. Iterate over the BACI clean dataset
        2. For each date, for each i, j, k, attempt to match the triple against WITS
        3. Append to a table
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
