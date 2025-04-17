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
        # Initial exploration


        Validate the dataset

        Baseline: we should have no null values for effective tariff and source, target, value, volume, and HS code.

        - What is common between the existence of nulls?
        - Are there duplicates / double counting?
        - Can we fill them in by looking back over the dataset?
        - Is there an error in the merge logic?

        > The code below does some initial exploration used to determine issues with the data as constructed
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


@app.cell
def _(mo):
    mo.md(
        r"""
        # Things to fix 1: Reconciling BACI country coding

        The BACI Dataset provides 'country_codes.csv' file which contains the mapping between the 3-digit ISO numeric code used by CEPII to more standard names (e.g. '251' maps to 'France').

        I had assumed that since this is an ISO standard, it would be the same between BACI and WITS. This is unfortunately not the case. For example, WITS codes France as 250 (according to the ISO standard I found online) whereas CEPII-BACI uses 251. This is an issue.

        I therefore need to write a script to reconcile between the two sets of ISO country codes

        To note: BACI codes TAIWAN as 490 (Asia, O.N.I). The ISO standard (and WITS) code it using 158
        """
    )
    return


@app.cell
def _(lf, pl):
    # WITS unique country codes
    baci_cc = set(
        lf.select(pl.col('Target')).unique().collect().to_series().to_list()
    ) | set(
        lf.select(pl.col('Source')).unique().collect().to_series().to_list()
    )

    print(f"Number of unique countries in BACI: {len(baci_cc)}")
    print(baci_cc)
    return (baci_cc,)


@app.cell
def _(pl, wits_mfn_lf):
    # Wits data
    mfn_cc = set(wits_mfn_lf.select(pl.col("reporter_country")).unique().collect().to_series().to_list())

    print(f"Number of unique countries in WITS MFN: {len(mfn_cc)}")
    print(mfn_cc)
    return (mfn_cc,)


@app.cell
def _(pl):
    # Wits data
    wits_pref_lf = pl.scan_parquet("data/intermediate/WITS_AVEPref.parquet")
    wits_pref_lf.collect_schema()


    pref_cc = set(
        wits_pref_lf.select(pl.col('reporter_country')).unique().collect().to_series().to_list()
    ) | set(
        wits_pref_lf.select(pl.col('partner_country')).unique().collect().to_series().to_list()
    )

    print(f"Number of unique countries in WITS Pref: {len(pref_cc)}")
    print(pref_cc)
    return pref_cc, wits_pref_lf


@app.cell
def _(lf, pl):
    # Create a test dataset for the main BACI join
    test_h = lf.select(pl.len()).collect().item()
    test_ssz = 10000
    test_lf = lf.gather_every(test_h // test_ssz).collect().to_pandas()

    test_lf
    return test_h, test_lf, test_ssz


@app.cell
def _():
    # Now I need to map between the wits codes and the BACI codes to a common layer.
    import pandas as pd
    # First load the reference for BACI
    baci_country_ref = pd.read_csv(
        "data/raw/BACI_HS92_V202501/country_codes_V202501.csv"
    )

    # Load reference for wits
    wits_country_ref = pd.read_csv(
        "data/raw/wits_country_codes.csv"
    )

    # Load the pref groups for wits
    pref_groups_ref = pd.read_csv(
        "data/raw/WITS_pref_groups/WITS_pref_groups.csv",
        encoding="ISO-8859-1"
    )
    return baci_country_ref, pd, pref_groups_ref, wits_country_ref


@app.cell
def _(baci_country_ref, pref_groups_ref, wits_country_ref):
    print(baci_country_ref.head())
    print("----")
    print(wits_country_ref.head())
    print("----")
    print(pref_groups_ref.head())
    return


@app.cell
def _(baci_country_ref, wits_country_ref):
    # I need to map between all of these. I'd like to use the WITS codes, since they adhere to ISO more effectively (I think?)

    # First, are there any country codes which aren't in both? If so I'll need to fuzzy match. 

    print(
        f"Num of non-intersecting country names: {len(set(baci_country_ref['country_name']) ^ set(wits_country_ref['Country Name']))}"
    )
    print(
        f"Num of non-intersecting iso3 codes: {len(set(baci_country_ref['country_iso3']) ^ set(wits_country_ref['ISO3']))}"
    )

    return


@app.cell
def _():


    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Things to fix 2: EU and other trade blocks. 

        While countries within the EU report their bilateral trade volumes and values, they report tariffs at the block level. Meaning I can't see MFN etc tariffs for someone like Germany to the US right now!

        So this needs to get fixed. I think the pref tariffs might also suffer from the same, given the EU is a trading block. I also need to check whether the EU is the only block that does this.

        **There is a chance that fixing problem 1 will solve this, if we have a bunch of coding issues.**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Things to fix 3: Validate joining over export/import duties

        I'm not certain I'm joining import duties (which is what WITS reports) with the BACI dataset (which reports exports) correctly. In fact, I need to validate that this is the case.

        This could be fixed by both of the above, or not done.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
