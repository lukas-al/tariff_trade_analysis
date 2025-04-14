import marimo

__generated_with = "0.12.8"
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
    # Load BACI dataset
    import polars as pl

    baci = pl.scan_parquet("data/final/BACI_HS92_V202501")

    baci.collect_schema()
    baci.head(100).collect()
    return baci, pl


@app.cell
def _(pl):
    # Load WITS datasets

    avemfn = pl.scan_parquet("data/final/WITS_AVEMFN.parquet")

    avemfn.collect_schema()
    avemfn.collect().head(100)
    return (avemfn,)


@app.cell
def _(pl):
    avepref = pl.scan_parquet("data/final/WITS_AVEPref.parquet")
    avepref.collect_schema()

    avepref.head(100).collect()
    return (avepref,)


@app.cell
def _(pd):
    # pref_beneficiary_groups = pl.scan_csv("data/raw/WITS_pref_groups/WITS_pref_groups.csv")
    # pref_beneficiary_groups.collect_schema()

    # pref_beneficiary_groups.head(100).collect()

    pref_groups = pd.read_csv("data/raw/WITS_pref_groups/WITS_pref_groups.csv", encoding="ISO-8859-1")
    return (pref_groups,)


@app.cell
def _(baci):
    unique_years = baci.sort(by="t").select("t").unique().collect().to_pandas()["t"].to_list()
    unique_sources = baci.sort(by="t").select("i").unique().collect().to_pandas()["i"].to_list()
    return unique_sources, unique_years


@app.cell
def _(baci, pl, unique_sources, unique_years):
    from tqdm.auto import tqdm

    # Create one large nested dictionary -> we can flatten this at the end.
    data_dict = {}

    pbar = tqdm(unique_years)
    for year in pbar:
        pbar.set_postfix(item=year)

        # Create a new empty dict to store the unique source in
        data_dict[year] = {}

        for source in unique_sources:
            # Create a new empty dict to store the unique source in
            data_dict[year][source] = {}

            # Get the collection of trades which correspond to this source and date
            trade_collection = baci.filter(
                (pl.col("i") == source) & (pl.col("t") == year)
            ).collect()

            for trade in trade_collection.iter_rows():
                print(trade)
                data_dict[year][source] = {
                    "target": trade[2],
                    "hs_code": trade[3],
                    "value": trade[4],
                    "quantity": trade[5],
                }
                break

            break

        break
    return data_dict, pbar, source, tqdm, trade, trade_collection, year


@app.cell
def _(data_dict):
    data_dict
    return


if __name__ == "__main__":
    app.run()
