src/mpil_tariff_trade_analysis/dev/MATCHING.py
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
        2. Load WITS preferential group mapping.
        3. Convert group mapping to Polars and prepare for expansion.
        4. Rename WITS columns for clarity and consistency before joining.
        5. Expand preferential tariff data from partner groups to individual countries.
        6. Left join BACI with MFN tariffs on year, reporter, product.
        7. Left join the result with *expanded* Preferential tariffs on year, reporter, partner, product.
        8. Calculate the final 'effective_tariff_rate' using `coalesce`, prioritizing preferential tariffs.
        9. Select and rename columns for the final output.
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
    # Assuming: year, reporter_country, product_code, tariff_rate, etc.
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
    # Assuming: year, reporter_country, partner_country, product_code, tariff_rate, etc.
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
        Load and process the mapping file for preferential trading groups.
        """
    )
    return


@app.cell
def _(pd):
    # Load the raw CSV mapping file
    pref_groups = pd.read_csv("data/raw/WITS_pref_groups/WITS_pref_groups.csv", encoding="iso-8859-1")
    return (pref_groups,)


@app.cell
def _(pd, pl, pref_groups):
    # Original Pandas aggregation
    pref_group_df_pd = (
        pref_groups
        .groupby("RegionCode")
        # Convert set to list here for easier Polars conversion
        .agg(lambda x: list(set(x)))
        .reset_index()
        .rename(columns={"RegionCode": "region_code", "Partner":"partner_list"})
        [['region_code', 'partner_list']]
    )

    # Convert to Polars DataFrame
    pref_group_pl = pl.from_pandas(pref_group_df_pd)

    print("Preferential Group Mapping (Polars):")
    print(pref_group_pl.schema)
    print(pref_group_pl.head())

    return pref_group_pl, # pref_group_df_pd # Keep pandas version if needed elsewhere


@app.cell
def _(mo):
    mo.md(r"""# Preparation: Rename Columns""")
    return


@app.cell(hide_code=True)
def _(avemfn, pl):
    # Rename WITS MFN columns for clarity and to match BACI join keys
    # !! ADJUST these column names based on avemfn.collect_schema() output !!
    renamed_avemfn = avemfn.rename({
        "year": "t",
        "reporter_country": "i",
        "product_code": "k",
        "tariff_rate": "mfn_tariff_rate",
        "min_rate": "mfn_min_tariff_rate",
        "max_rate": "mfn_max_tariff_rate",
        "tariff_type": "tariff_type"

    }).select(
        # Select only needed columns for the MFN join (no partner 'j')
        "t", "i", "k", "mfn_tariff_rate", "mfn_min_tariff_rate", "mfn_max_tariff_rate", "tariff_type"
    )

    print("Renamed MFN Schema:")
    print(renamed_avemfn.collect_schema())
    print("\nRenamed MFN Head:")
    renamed_avemfn.head(5).collect()
    return (renamed_avemfn,)


@app.cell(hide_code=True)
def _(avepref, pl):
    # Rename WITS Preferential columns
    # !! ADJUST these column names based on avepref.collect_schema() output !!
    renamed_avepref = avepref.rename({
        "year": "t",
        "reporter_country": "i",
        "partner_country": "j", # This might be a group code or individual country
        "product_code": "k",
        "tariff_rate": "pref_tariff_rate",
        "min_rate": "pref_min_tariff_rate",
        "max_rate": "pref_max_tariff_rate",
    }).select(
        "t", "i", "j", "k", "pref_tariff_rate", "pref_min_tariff_rate", "pref_max_tariff_rate"
    )

    print("Renamed Pref Schema (Before Expansion):")
    print(renamed_avepref.collect_schema())
    print("\nRenamed Pref Head (Before Expansion):")
    renamed_avepref.head(5).collect()
    return (renamed_avepref,)


@app.cell
def _(mo):
    mo.md(r"""## Expand Preferential Tariff Partner Groups""")
    return


@app.cell
def _(pl, pref_group_pl, renamed_avepref):
    # Ensure consistent data types for joining keys if necessary
    # Example: Cast both join keys to Utf8 if they differ
    # pref_group_pl = pref_group_pl.with_columns(pl.col("region_code").cast(pl.Utf8))
    # renamed_avepref = renamed_avepref.with_columns(pl.col("j").cast(pl.Utf8))

    # Left join avepref with the group mapping
    # Rows in avepref where 'j' is a group code will get a list in 'partner_list'
    # Rows where 'j' is an individual country will have null in 'partner_list'
    joined_pref_mapping = renamed_avepref.join(
        pref_group_pl,
        left_on="j",        # Partner code (can be group or individual)
        right_on="region_code", # Group code from mapping
        how="left"
    )

    # Create the final partner list: use the exploded list if 'j' was a group,
    # otherwise use the original 'j' value (put into a list for explode compatibility)
    expanded_pref = joined_pref_mapping.with_columns(
        pl.when(pl.col("partner_list").is_not_null())
        .then(pl.col("partner_list")) # Use the list from mapping
        .otherwise(pl.lit([pl.col("j")])) # Use original 'j' as a single-item list
        .alias("final_partner_list")
    ).explode(
        "final_partner_list" # Explode the list into separate rows
    ).rename(
        {"final_partner_list": "j_individual"} # Rename the exploded column
    ).select(
        # Select all original avepref columns, but replace 'j' logic
        pl.col("t"),
        pl.col("i"),
        pl.col("j_individual").alias("j"), # Use the new individual partner code as 'j'
        pl.col("k"),
        pl.col("pref_tariff_rate"),
        pl.col("pref_min_tariff_rate"),
        pl.col("pref_max_tariff_rate"),
        # Add other columns from renamed_avepref if needed
    )

    print("Expanded Preferential Tariff Schema:")
    print(expanded_pref.collect_schema())
    print("\nExpanded Preferential Tariff Head:")
    print(expanded_pref.head(10).collect()) # Show more rows to see potential expansion

    # Check if any nulls were introduced in 'j' unexpectedly
    null_j_count = expanded_pref.filter(pl.col("j").is_null()).select(pl.count()).collect().item()
    print(f"\nNumber of rows with null 'j' after expansion: {null_j_count}")


    return expanded_pref, joined_pref_mapping # Return the final expanded table


@app.cell
def _(mo):
    mo.md(r"""# Data Joining""")
    return


@app.cell
def _(baci, expanded_pref, mo, pl, renamed_avemfn):
    # Define the join keys
    # MFN join keys (assuming MFN tariff depends only on reporter 'i', not partner 'j')
    mfn_join_keys = ["t", "i", "k"]
    # Preferential join keys (depends on reporter 'i' AND individual partner 'j')
    pref_join_keys = ["t", "i", "j", "k"]

    # Ensure join key types are compatible
    # Example: Cast keys in baci if they are not Int64 like in WITS data
    # baci = baci.with_columns([
    #     pl.col("t").cast(pl.Int64),
    #     pl.col("i").cast(pl.Int64),
    #     pl.col("j").cast(pl.Int64),
    #     pl.col("k").cast(pl.Utf8) # Assuming k is product code string
    # ])
    # renamed_avemfn = renamed_avemfn.with_columns(pl.col("k").cast(pl.Utf8))
    # expanded_pref = expanded_pref.with_columns(pl.col("k").cast(pl.Utf8))


    # 1. Left join BACI with MFN tariffs
    # Keep all rows from BACI, add MFN tariff where t, i, k match
    joined_mfn = baci.join(
        renamed_avemfn,
        on=mfn_join_keys,
        how="left"
    )

    # 2. Left join the result with *Expanded* Preferential tariffs
    # Keep all rows from the previous join, add Pref tariff where t, i, j, k match
    joined_all = joined_mfn.join(
        expanded_pref, # Use the expanded preferential data
        on=pref_join_keys,
        how="left"
    )

    # 3. Calculate the final effective tariff
    # Use preferential tariff rate if available (not null), otherwise use MFN tariff rate
    final_table = joined_all.with_columns(
        pl.coalesce(
            pl.col("pref_tariff_rate"), pl.col("mfn_tariff_rate")
        ).alias("effective_tariff_rate")
    )

    mo.md(f"Joined table schema: `{final_table.collect_schema()}`") # Use .collect_schema() for LazyFrames
    print("\nJoined Table Head (before final selection):")
    print(final_table.head().collect())

    return final_table, joined_all, joined_mfn, mfn_join_keys, pl, pref_join_keys # Return pl for next cell


@app.cell
def _(mo):
    mo.md(r"""# Final Selection""")
    return


@app.cell
def _(final_table, mo, pl):
    # Select and arrange final columns
    # !! ADJUST column names 'v' and 'q' based on your BACI schema output !!
    # Example assumes BACI columns are 't', 'i', 'j', 'k', 'v', 'q'
    final_unified_table = final_table.select(
        pl.col("t").alias("Year"),
        pl.col("i").alias("Source"),      # Reporter country code
        pl.col("j").alias("Target"),      # Partner country code (now individual)
        pl.col("k").alias("HS_Code"),     # Product code (HS92)
        pl.col("q").alias("Quantity"),    # Assuming 'q' is Quantity in BACI
        pl.col("v").alias("Value"),       # Assuming 'v' is Value in BACI
        # Include relevant tariff columns for inspection
        pl.col("mfn_tariff_rate"),
        pl.col("pref_tariff_rate"),
        pl.col("effective_tariff_rate") # Use the coalesced rate
        # Optionally add min/max rates if needed
        # pl.col("mfn_min_tariff_rate"),
        # pl.col("mfn_max_tariff_rate"),
        # pl.col("pref_min_tariff_rate"),
        # pl.col("pref_max_tariff_rate"),
    )

    mo.md("### Final Unified Table (First 100 rows)")
    final_unified_table.head(100).collect() # Use collect() to view the result
    # You might want to save this result later:
    # final_unified_table.collect().write_parquet("data/final/unified_trade_tariff.parquet")
    return (final_unified_table,)


if __name__ == "__main__":
    app.run()

