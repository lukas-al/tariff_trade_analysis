

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pyarrow.parquet as pq
    import pyarrow as pa
    return mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
        # CREATE UNIFIED DATASET
        Join trade values and volumes with tariff amounts. Left join on BACI. Simple operation.
        """
    )
    return


@app.cell
def _(pl):
    # Load the requisite dataframes
    ave_pref = pl.scan_parquet("data/intermediate/WITS_AVEPref_CLEAN.parquet")
    ave_mfn = pl.scan_parquet("data/intermediate/WITS_AVEMFN_CLEAN.parquet")
    baci = pl.scan_parquet("data/intermediate/BACI_HS92_V202501_CLEAN.parquet")
    return ave_mfn, ave_pref, baci


@app.cell
def _(mo):
    mo.md(r"""## Format the datasets to make them joinable""")
    return


@app.cell
def _(ave_pref, pl):
    # Inspect them
    ave_pref_clean = ave_pref.with_columns(
        pl.col("tariff_rate").str.strip_chars().cast(pl.Float32),
        pl.col("min_rate").str.strip_chars().cast(pl.Float32),
        pl.col("max_rate").str.strip_chars().cast(pl.Float32),
    ).drop(['hs_revision', 'tariff_type'])

    ave_pref_clean = ave_pref_clean.rename(
        {
            'tariff_rate': 'tariff_rate_pref',
            'min_rate': 'min_rate_pref',
            'max_rate': 'max_rate_pref',
        }
    )

    ave_pref_clean.head().collect()
    return (ave_pref_clean,)


@app.cell
def _(ave_mfn, pl):
    ave_mfn_clean = ave_mfn.with_columns(
        pl.col("tariff_rate").str.strip_chars().cast(pl.Float32),
        pl.col("min_rate").str.strip_chars().cast(pl.Float32),
        pl.col("max_rate").str.strip_chars().cast(pl.Float32),
    ).drop(['hs_revision', 'tariff_type'])

    ave_mfn_clean = ave_mfn_clean.rename(
        {
            'tariff_rate': 'tariff_rate_mfn',
            'min_rate': 'min_rate_mfn',
            'max_rate': 'max_rate_mfn',
        }
    )


    ave_mfn_clean.head().collect()
    return (ave_mfn_clean,)


@app.cell
def _(baci, pl):
    baci_clean = baci.rename({
        "t": "year",
        "i": "reporter_country", # Exporter
        "j": "partner_country", # Importer
        "k": "product_code",
        'v': 'volume',
        'q': 'quantity',
    }).with_columns(
        pl.col('year').cast(pl.Utf8)
    )

    baci_clean.head().collect()
    return (baci_clean,)


@app.cell
def _(mo):
    mo.md(r"""# Operate over the BACI dataset in chunks""")
    return


@app.cell
def _(ave_mfn_clean, ave_pref_clean, baci_clean, pl):
    # First get the chunk values
    from tqdm.auto import tqdm
    import time
    import gc

    start_time = time.time()

    # unique_years = baci_clean.select("year").unique().collect().to_series().to_list()
    # unique_years.sort()

    # MANUAL OVERRIDE
    unique_years = ['1995', '1996', '1997', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    # unique_years = ['2013']

    print(f"Unique years in dataset:\n{unique_years}")

    schema = None
    for i, year in enumerate(unique_years):
        print(f"--- Processing Chunk {i + 1}/{len(unique_years)}: Year = {year} ---")

        filtered_baci = baci_clean.filter(pl.col('year')==year)
        filtered_avepref = ave_pref_clean.filter(pl.col('year')==year).select(
            'partner_country',
            'product_code',
            'tariff_rate_pref',
            'min_rate_pref',
            'max_rate_pref',
        )
        filtered_avemfn = ave_mfn_clean.filter(pl.col('year')==year).select(
            'reporter_country',
            'product_code',
            'tariff_rate_mfn',
            'min_rate_mfn',
            'max_rate_mfn',
        ).rename({'reporter_country': 'partner_country'}) # Rename to flip the reporter to be our importer (partner)

        # Join avepref to BACI
        # -> WITS pref tariffs are import duties. The importer in BACI is the 'partner'.
        baci_avepref = filtered_baci.join(
            filtered_avepref,
            how='left',
            on=['partner_country', 'product_code'],
        )

        # Join avemfn to BACI
        baci_all = baci_avepref.join(
            filtered_avemfn,
            how='left',
            on=['partner_country', 'product_code']
        )

        # Coalesce the MFN and AVEPREF tariffs. Pref takes precedent. If neither then None (so we can count)
        unified_baci_lf = baci_all.with_columns(
            pl.when(pl.col('tariff_rate_pref').is_not_null())
            .then(pl.col('tariff_rate_pref'))
            .otherwise(pl.col('tariff_rate_mfn'))
            .alias('effective_tariff')
        )

        print(f"    Joining WITS to BACI")
        print(f"    Coalescing AVEPref and MFN tariffs")

        # ---------------------- OLD VERSION USING PYARROW ----------------------s
        # # Collect the chunk
        # unified_baci = unified_baci_lf.collect(engine="streaming")
        # unified_baci = unified_baci_lf.collect()

        # # Convert to pyarrow
        # unified_baci = unified_baci.to_arrow()

        # # Set a single schema so pyarrow doesn't write chunks idiosyncratically
        # expected_schema = pa.schema([
        #     pa.field('year', pa.string(), nullable=False),
        #     pa.field('reporter_country', pa.string(), nullable=False),
        #     pa.field('partner_country', pa.string(), nullable=False),
        #     pa.field('product_code', pa.string(), nullable=False),
        #     pa.field('volume', pa.float64(), nullable=True),
        #     pa.field('quantity', pa.float64(), nullable=True),
        #     pa.field('tariff_rate_pref', pa.float32(), nullable=True),
        #     pa.field('min_rate_pref', pa.float32(), nullable=True),
        #     pa.field('max_rate_pref', pa.float32(), nullable=True),
        #     pa.field('tariff_rate_mfn', pa.float32(), nullable=True),
        #     pa.field('min_rate_mfn', pa.float32(), nullable=True),
        #     pa.field('max_rate_mfn', pa.float32(), nullable=True),
        #     pa.field('effective_tariff', pa.float32(), nullable=True),
        # ])
        # unified_baci = unified_baci.cast(expected_schema, safe=False)

        # # Write to parquet
        # print(f"    Sinking chunk for {year}")
        # pq.write_to_dataset(
        #     table=unified_baci,
        #     root_path='data/final/unified_trade_tariff_partitioned/',  # PyArrow might prefer string paths
        #     partition_cols=['year'],  # Use 'Year' or configured column
        #     existing_data_behavior="delete_matching",  # Safer than delete_matching if run concurrently
        #     compression='ZSTD',
        #     compression_level=9
        # )
    
        # # IDK if this makes any difference. I think something is going on to kill it silently
        # del unified_baci
        # gc.collect()


        # ---------------------- NEW VERSION USING POLARS ONLY ----------------------
        unified_baci_lf.sink_parquet(
            pl.PartitionByKey(
                base_path='data/final/unified_trade_tariff_partitioned/',
                by=pl.col('year'),
            ),
            mkdir=True,
        )

    
    print(f"Time elapsed = {(time.time() - start_time)/60} mins")
    print('-----COMPLETE-----')
    return


if __name__ == "__main__":
    app.run()
