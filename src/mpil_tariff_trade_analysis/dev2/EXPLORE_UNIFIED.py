

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pprint import pprint
    return mo, pl


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
def _(mo):
    mo.md(
        r"""
        # What are we interested in?
        1. Empirical effect on goods imports of tariff impositions

        What is the average size of the effect (at y0, y1, y2, y5) of a tariff imposition on the corresponding product line?

        Price, volume, value. How much substitution to lower-tariff countries is there?

        Considerations:

        - Do we want all data points to count the same? Do we want to preferentially weight larger import volumes? Can we just remove very small imports.
        - Different categories have different dynamics, e.g. commodities vs specialist goods.

        2. How many 'tariff shocks' exist?

        For countries with import volumes above a certain size, how many 'tariff shocks' (some metric similar to the US imposition) can we identify?

        Considertations:

        - What's a tariff shock
        - What countries count, volumes & values, etc.

        3. Trade Deflection - case study across countries and time

        When tariffs cause imports to be deflected to other countries, how do those exporters react? Are there patterns in how they choose countries to redirect to? (Open economies, nearby, etc).

        Considerations:

        - Need to control for the imposition of reciprocal tariffs to understand effect sizes - conditional on no reciprocation, etc.

        4. Second-order impacts

        Can we calculate the impact of cost passthroughs in supply chains? E.g. if export prices for good X from China increase by 10%, the direct impact is that import prices for that good in the UK from China also increase 10%. But if the good is used to produce X,Y,Z products in Germany, what is the impact on prices of goods which the UK imports from Germany?

        Would need to use inter-country I/O tables or some such.

        5. Non-linearities in cost passthrough

        Are manufacturers more likely to pass through cost increases to consumers due to tariffs? Has this ratio stayed the same? Is there reason to think we're in a more sensitive regime?

        # Considerations:
        1. Start with UK, China, US
        2. Use 2018 case study
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Whats the actual todo:
        1. Validate the dataset
        2. Create a factsheet which I can share with MPIL
        3. Answer some questions
        3. Add services and NTBs?
        """
    )
    return


@app.cell
def _(unified_data):
    unified_data.describe()
    return


@app.cell
def _():
    # Check those nulls! Why do we still have them? Are they just 0 tariffs?
    return


if __name__ == "__main__":
    app.run()
