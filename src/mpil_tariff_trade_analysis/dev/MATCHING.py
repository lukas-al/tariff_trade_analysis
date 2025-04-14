import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    return mo, pd, pl


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
        1. Load BACI and WITS (MFN, Pref) datasets.
        2. Rename WITS columns for clarity and consistency before joining.
        3. Left join BACI with MFN tariffs on year, reporter, partner, product.
        4. Left join the result with Preferential tariffs on the same keys.
        5. Calculate the final 'effective_tariff' using `coalesce`, prioritizing preferential tariffs.
        6. Select and rename columns for the final output.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Preparation: Load Data""")
    return


@app.cell(hide_code=True)
def _(pl):
    # Load BACI dataset
    # Using scan for lazy execution until .collect() or .fetch()
    baci = pl.scan_parquet("data/final/BACI_HS92_V202501")

    print("BACI Schema:")
    print(baci.collect_schema())
    print("\nBACI Head:")
    baci.head(5).collect()
    return (baci,)


@app.cell(hide_code=True)
def _(pl):
    # Load WITS MFN dataset
    # !! IMPORTANT: Check the actual column names in your file !!
    # Assuming: ReporterCode, PartnerCode, ProductCode, Year, Value
    avemfn = pl.scan_parquet("data/final/WITS_AVEMFN.parquet")

    print("WITS MFN Schema:")
    print(avemfn.collect_schema())
    print("\nWITS MFN Head:")
    avemfn.head(5).collect()
    return (avemfn,)


@app.cell(hide_code=True)
def _(pl):
    # Load WITS Preferential dataset
    # !! IMPORTANT: Check the actual column names in your file !!
    # Assuming: ReporterCode, PartnerCode, ProductCode, Year, Value
    avepref = pl.scan_parquet("data/final/WITS_AVEPref.parquet")

    print("WITS Pref Schema:")
    print(avepref.collect_schema())
    print("\nWITS Pref Head:")
    avepref.head(5).collect()
    return (avepref,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Preparation: Pref groups to individual countries
        Explode trading (preferential) groups in pref tariff data.

        Map the RegionCode

        """
    )
    return


@app.cell
def _(pd):
    pref_groups = pd.read_csv("data/raw/WITS_pref_groups/WITS_pref_groups.csv", encoding="iso-8859-1")
    return (pref_groups,)


@app.cell
def _(pref_groups):
    pref_group_df = (
        pref_groups
        .groupby("RegionCode")
        .agg(set)
        .reset_index()
        .rename(columns={"RegionCode": "region_code", "Partner":"partner"})
        [['region_code', 'partner']]
    )

    pref_group_df
    return (pref_group_df,)


@app.cell
def _(mo):
    mo.md(r"""# Data Joining""")
    return


@app.cell(hide_code=True)
def _(avemfn):
    # Rename WITS MFN columns for clarity and to match BACI join keys
    # !! ADJUST these column names based on avemfn.collect_schema() output !!
    # Example assumes WITS columns are named 'Year', 'ReporterCode', etc.
    renamed_avemfn = avemfn.rename({
        "year": "t",
        "reporter_country": "i",
        "product_code": "k",
        "tariff_rate": "mfn_tariff_rate",
        "min_rate": "mfn_min_tariff_rate",
        "max_rate": "mfn_max_tariff_rate",
        "tariff_type": "tariff_type"

    }).select(
        "t", "i", "k", "mfn_tariff_rate", "mfn_min_tariff_rate", "mfn_max_tariff_rate", "tariff_type"
    )

    print("Renamed MFN Schema:")
    print(renamed_avemfn.collect_schema())
    print("\nRenamed MFN Head:")
    renamed_avemfn.head(5).collect()
    return (renamed_avemfn,)


@app.cell(hide_code=True)
def _(avepref):
    # Rename WITS Preferential columns
    # !! ADJUST these column names based on avepref.collect_schema() output !!
    # Example assumes WITS columns are named 'Year', 'ReporterCode', etc.
    renamed_avepref = avepref.rename({
        "year": "t",
        "reporter_country": "i",
        "partner_country": "j",
        "product_code": "k",
        "tariff_rate": "pref_tariff_rate",
        "min_rate": "pref_min_tariff_rate",
        "max_rate": "pref_max_tariff_rate",
    }).select(
        "t", "i", "j", "k", "pref_tariff_rate", "pref_min_tariff_rate", "pref_max_tariff_rate"
    )

    print("Renamed Pref Schema:")
    print(renamed_avepref.collect_schema())
    print("\nRenamed Pref Head:")
    renamed_avepref.head(5).collect()
    return (renamed_avepref,)


@app.cell
def _(baci, mo, pl, renamed_avemfn, renamed_avepref):
    # Define the join keys
    # Ensure the data types of these keys are compatible across dataframes
    join_keys = ["t", "i", "j", "k"]

    # 1. Left join BACI with MFN tariffs
    # Keep all rows from BACI, add MFN tariff where match found
    joined_mfn = baci.join(
        renamed_avemfn,
        on=join_keys,
        how="left"
    )

    # 2. Left join the result with Preferential tariffs
    # Keep all rows from the previous join, add Pref tariff where match found
    joined_all = joined_mfn.join(
        renamed_avepref,
        on=join_keys,
        how="left"
    )

    # 3. Calculate the final effective tariff
    # Use preferential tariff if available (not null), otherwise use MFN tariff
    final_table = joined_all.with_columns(
        pl.coalesce(pl.col("pref_tariff"), pl.col("mfn_tariff")).alias("effective_tariff")
    )

    mo.md(f"Joined table schema: `{final_table.schema}`") # Use .schema for LazyFrames
    return final_table, join_keys, joined_all, joined_mfn


@app.cell
def _(final_table, mo, pl):
    # Select and arrange final columns
    # !! ADJUST column names 'v' and 'q' based on your BACI schema output !!
    # Example assumes BACI columns are 't', 'i', 'j', 'k', 'v', 'q'
    final_unified_table = final_table.select(
        pl.col("t").alias("Year"),
        pl.col("i").alias("Source"),      # Reporter country code
        pl.col("j").alias("Target"),      # Partner country code
        pl.col("k").alias("HS_Code"),     # Product code (HS92)
        pl.col("q").alias("Quantity"),    # Assuming 'q' is Quantity in BACI
        pl.col("v").alias("Value"),       # Assuming 'v' is Value in BACI
        pl.col("mfn_tariff"),             # MFN tariff rate (can be null)
        pl.col("pref_tariff"),            # Preferential tariff rate (can be null)
        pl.col("effective_tariff")        # Calculated effective tariff
    )

    mo.md("### Final Unified Table (First 100 rows)")
    final_unified_table.head(100).collect() # Use collect() to view the result
    # You might want to save this result later:
    # final_unified_table.collect().write_parquet("data/final/unified_trade_tariff.parquet")
    return (final_unified_table,)


@app.cell
def _(mo):
    mo.md(r"""# Final Selection""")
    # This cell title might be redundant now, consider removing or merging
    # with the cell above if it just displays the final table.
    return


if __name__ == "__main__":
    app.run()
