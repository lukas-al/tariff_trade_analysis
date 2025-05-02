

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import itertools
    import functools
    import operator
    import time
    import math
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pycountry
    import pickle

    from tqdm import tqdm
    return functools, mo, operator, pickle, pl, px, time, tqdm


@app.cell
def _(pl):
    # unified_lf = pl.scan_parquet("data/final/unified_filtered_10000val_100c_sample_1000krows_filter")
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned")
    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Effect of tariff change on unit price level

        Can we identify a relationship between the unit price level at different intervals and the tariff application?

        What we're essentially looking for, is the change in unit price on the X axis and the tariff application on the Y axis, at different time delays.

        * If there is a change in the tariff level between t and t+x
        * What is the change in unit price between those

        So first, we need to find those elements where there's a change in the tariff level between t and t+x
        """
    )
    return


@app.cell
def _(unified_lf):
    # Identify elements where there's change in tariff level between t and t+x
    # This means we need to create time series of each product and country pair. Or we could simply iterate over the whole table? Maybe that's actually fine.

    # Unique list of all countries
    country_list = set(unified_lf.select('reporter_country').unique().collect()['reporter_country'].to_list()).intersection(
        unified_lf.select('partner_country').unique().collect()['partner_country'].to_list()
    )

    # List of all years
    year_list = unified_lf.select('year').unique().collect()['year'].to_list()

    # List of all products
    product_list = unified_lf.select('product_code').unique().collect()['product_code'].to_list()
    return (product_list,)


@app.cell
def _(pl, product_list, time, tqdm, unified_lf):
    data_list = []

    starttime = time.time()
    # for country, product in tqdm(itertools.product(country_list, product_list)):
    for product in tqdm(product_list, desc="iterating over table"):
        country = '840' # Fix to be USA for now
        # country_name = pycountry.countries.get(numeric=country).name
        country_name = "USA"
        product_vals = unified_lf.filter(
            pl.col('partner_country') == country,
            pl.col('product_code') == product
        ).group_by(
            ['year']
        ).agg(
            pl.sum('value'),
            pl.sum('quantity'),
            pl.mean('effective_tariff'),
        ).with_columns(
            (pl.col('value') / pl.col('quantity')).alias('unit_value')
        ).drop(
            'value',
            'quantity'
        ).with_columns(
            pl.lit(country).alias('country_code'),
            pl.lit(product).alias('product_code')
        ).sort('year')
        df = product_vals.collect()

        data_list.append(df)
        # fig = px.line(
        #     df.to_pandas(),
        #     x='year',
        #     y=['unit_value', 'effective_tariff'],
        #     title=f"Product {product}; country {country_name}",
        # )
        # fig.show()

        # break



    print(f"Total time to pass over table is {int(time.time()-starttime)}s")
    return (data_list,)


@app.cell
def _(data_list, pickle):
    with open("src/mpil_tariff_trade_analysis/dev2/visualise/processed_ts_USA_product_averages.pkl", 'wb') as f:
        pickle.dump(data_list, f)
    return


@app.cell
def _(pickle):
    with open("src/mpil_tariff_trade_analysis/dev2/visualise/processed_ts_USA_product_averages.pkl", 'rb') as f:
        saved_data_list = pickle.load(f)
    return (saved_data_list,)


@app.cell
def _(px, saved_data_list):
    specific_df = saved_data_list[1110]
    df_pd = specific_df.to_pandas()

    # # title = f"Product {product}; country {country_name}"
    # title='test'

    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # fig.add_trace(
    #     go.Scatter(x=df_pd['year'], y=df_pd['unit_value'], name='unit_value', mode='lines'),
    #     secondary_y=False,
    # )

    # fig.add_trace(
    #     go.Scatter(x=df_pd['year'], y=df_pd['effective_tariff'], name='effective_tariff', mode='lines'),
    #     secondary_y=True,
    # )

    # fig.update_layout(
    #     title_text=title
    # )

    # fig.update_yaxes(title_text="unit_value", secondary_y=False)
    # fig.update_yaxes(title_text="effective_tariff", secondary_y=True)
    # fig.update_xaxes(title_text="year")

    # fig.show()

    fig = px.line(
        df_pd,
        x='year',
        y=['unit_value', 'effective_tariff'],
        # title=f"Product {product}; country {country_name}",
        title="Test"
    )
    fig.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Regression approaches
        - Simple scatter plot of unitprice change vs tariff change at different intervals
        - Estimate overall parameters using the above!
        - More complex panel using some sort of panel OLS with HDFE
        - Other?

        ## 1. Simple scatter plot of different subsets of the data
        """
    )
    return


@app.cell
def _():
    # 1. Filter the subset
    # 2. Identify all cases of tariff increase or decrease
    # 3. Extract the change in tariffs to the change in price
    # 4. Store in a list along with some metadata
    # 5. Plot on a scatter. Use the metadata to group products, countries, etc.
    # 6. Apply this methodology to the 2018 US-China tariffs on specific goods.
    # 7. Estimate the coefficient. Estimate statistical significance.
    return


@app.cell
def _():
    # Config for subsequent analysis
    start_year = "2000" # Starting year for analysis
    end_year = "2023" # End year for analysis
    year_gap = 1

    reporter_countries = ['156'] # China exporter
    partner_countries = ['840'] # USA importer
    # product_codes = ['01'] # Can use the aggregated code values...
    product_codes = None
    return (
        end_year,
        partner_countries,
        product_codes,
        reporter_countries,
        start_year,
        year_gap,
    )


@app.cell
def _(end_year, start_year, year_gap):
    # For each year in start_year, 
    year_range_end_excluded = [
        str(year)
        for year in range(int(start_year), int(end_year) + 1 - year_gap)
    ]

    year_pairs = []
    for year in year_range_end_excluded:
        year_pairs.append(
            [
                year, 
                str(int(year) + year_gap)
            ]
        )

    print(year_pairs)
    return (year_pairs,)


@app.cell
def _():
    return


@app.cell
def _(
    functools,
    operator,
    partner_countries,
    pl,
    product_codes,
    reporter_countries,
    unified_lf,
    year_pairs,
):
    # --- 1. FILTER DATA --- 
    if reporter_countries:
        filtered_lf = unified_lf.filter(
            pl.col('reporter_country').is_in(reporter_countries)
        )

    if partner_countries:
        filtered_lf = filtered_lf.filter(
            pl.col('partner_country').is_in(partner_countries)
        )

    if product_codes:
        conditions = [
            pl.col('product_code').str.slice(0, len(p)) == p
            for p in product_codes
        ]
        combined_condition = functools.reduce(operator.or_, conditions) # Combine conditions with an or
        filtered_lf = filtered_lf.filter(combined_condition)

    # Create a unit value column
    filtered_lf = filtered_lf.with_columns(
        (pl.col('value') / pl.col('quantity')).alias('unit_value')
    )

    print("Length post filter:", filtered_lf.select(pl.len()).collect().item())

    # Now find all values within this set where there was a change in tariff between the years and get the magnitude of that change.
    group_cols = ['reporter_country', 'partner_country', 'product_code'] # This is our UUID

    for y1, y2 in year_pairs:

        # --- 2. FIND ROWS WHERE THE EFFECTIVE_TARIFF CHANGED --- 
        changed_groups_lf = filtered_lf.filter(
            pl.col('year').is_in([y1, y2])
        ).group_by(
            group_cols
        ).agg(
            (pl.col('effective_tariff').last() - pl.col('effective_tariff').first()).alias('tariff_difference'),
            (pl.col('value').last() - pl.col('value').first()).alias('value_difference'),
            (pl.col('quantity').last() - pl.col('quantity').first()).alias('quantity_difference'),
            (pl.col('unit_value').last() - pl.col('unit_value').first()).alias('unit_value_difference'),
        ).filter(
            pl.col('tariff_difference') > 0.0
        )
            
        print(changed_groups_lf.collect())
   
        break
    return


@app.cell
def _(unified_lf):
    unified_lf.collect_schema()
    return


@app.cell
def _(intermediate_years):
    intermediate_years
    return


if __name__ == "__main__":
    app.run()
