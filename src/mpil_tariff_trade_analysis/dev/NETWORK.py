import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import networkx as nx
    import polars as pl
    return mo, nx, pl


@app.cell
def _(pl):
    DATA_PARQUET_FOLDER = "data/final/unified_trade_tariff_partitioned/"

    lf = pl.scan_parquet(DATA_PARQUET_FOLDER)

    lf.describe()
    return DATA_PARQUET_FOLDER, lf


@app.cell
def _(mo):
    mo.md(
        r"""
        # Validate the dataset

        Baseline: we should have no null values for effective tariff and source, target, value, volume, and HS code.

        - What is common between the existence of nulls?
        - Are there duplicates / double counting?
        - Can we fill them in by looking back over the dataset?
        - Is there an error in the merge logic?

        """
    )
    return


@app.cell
def _(lf, pl):
    null_rows = lf.filter(pl.col("effective_tariff_rate").is_null())
    return (null_rows,)


@app.cell
def _():
    print(f"{(305506345-157709739)/305506345 * 100}% of rows are empty")
    return


@app.cell
def _(null_rows, pl):
    # Materialise our view of a sample of the nulls

    height = null_rows.select(pl.len()).collect().item()
    sample_sz = 10000
    null_sample_pd = null_rows.gather_every(height // sample_sz).collect().to_pandas()
    return height, null_sample_pd, sample_sz


@app.cell
def _(null_sample_pd):
    null_sample_pd.head()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Looking at the first row of this sample, 218 is Finland and 826 is the UK. 

        I can only find data on the WTO website from 2020 onwards.

        Are all the empty rows just early years? How do I want to subset it to get the largest (and longest) sample of the most relevant countries?
        """
    )
    return


@app.cell
def _(null_sample_pd):
    # See what dates are common in this sample
    import altair as alt

    alt.theme.enable("carbong100")

    # Create a bar chart that shows the count of each unique category
    chart = alt.Chart(null_sample_pd).mark_bar().encode(
        x=alt.X('Year:N', title='Category'),  # Nominal type for categorical data
        y=alt.Y('count()', title='Frequency')       # Count the occurrences
    ).properties(
        title='Distribution of Categorical Values'
    ).interactive()

    # Display the chart (In a Jupyter Notebook, simply placing "chart" on a line will render it)
    chart
    return alt, chart


@app.cell
def _(mo):
    mo.md(
        r"""
        This chart provides evidence that the nulls are broadly distributed across time. This points towards a more pervasive issue.

        Next step: validate that the matching code is actually identifying the right data and merging it correctly.


        """
    )
    return


@app.cell
def _(null_sample_pd):
    null_sample_pd.sample(10, random_state=42)[["Source", "Target", "HS_Code", "Quantity", "Value", "Year"]]

    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Process

        1. For each row,
        2. Validate the trade values and volumes using the WTO interface
        3. Valide the tariff measures using WTO & WITS

        ### Row 6252

        - [ ] Trade values: I can only find $7829 in Comtrade directly. I can only find 250kg in quantity.
        - [X] Using the WTO STATS database I can find the value. Important that my approach baselines EXPORTS and theirs baselines IMPORTS.
        - [X] Tariffs - I can find them for Georgia (target) and they're 0.0 (MFN)

        There is no availability of Romania (reporter / source) in the WTO online database for trade / tariffs (https://ttd.wto.org/en/analysis/bilateral-trade-relations?member1=&member2=)

        **MISSING TARIFF VALUE IS IDENTIFIED ONLINE AS 0 (MFN)**

        ### Row 4684
        Germany to Sri Lanka

        - [ ] I've found the value and volume. Sri Lanka has a 2.5% MFN on this line, and yet it hasn't been picked up in the merge. We have conclusive evidence of either of the following: A) Data is missing in the WITS database and/or B) the matching logic is flawed. 
        """
    )
    return


@app.cell
def _(lf, pl):
    # Validating 

    lf.filter(
        (pl.col("Source") == "276") &
        (pl.col("Target") == "144") &
        (pl.col("Year") == 2008) &
        (pl.col("HS_Code") == "310520")
    ).collect()

    # The data point exists, but 
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Identifying the error

        We have evidence now that either A) data is missing in the WITS database and/or B) the matching logic is flawed.

        We start by trying to validate whether the WITS data is missing examples.
        """
    )
    return


@app.cell
def _(pl):
    # Import the wits data
    wits_mfn_lf = pl.scan_parquet("data/intermediate/WITS_AVEMFN.parquet")
    print(wits_mfn_lf.collect_schema())

    wits_mfn_lf.describe()
    return (wits_mfn_lf,)


@app.cell
def _(pl, wits_mfn_lf):
    wits_mfn_lf.filter(
        (pl.col("reporter_country") == "144") &
        (pl.col("product_code") == "310520") &
        (pl.col("year") == 2008)
    ).collect()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Issue identified
        I can find the MFN tariff from Sri Lanka in our WITS database.
        However, Germany doesn't have an MFN tariff in the WITS database since it's part of the EU (I assume). The EU has a 6.5% tariff rate here, which is super significant.

        So, given this is an export from Germany to Sri Lanka, we should have picked up this 2.5% MFN tariff. 

        We therefore need to construct a specific test case for this example. 

        We will need to construct further test cases to validate the integrity of the dataset.
        """
    )
    return


if __name__ == "__main__":
    app.run()
