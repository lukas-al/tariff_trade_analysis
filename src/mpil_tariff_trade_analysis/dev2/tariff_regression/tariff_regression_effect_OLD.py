

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle
    import time

    import marimo as mo
    import plotly.express as px
    import polars as pl
    from tqdm import tqdm

    return mo, pickle, pl, px, time, tqdm


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
    country_list = set(
        unified_lf.select("reporter_country").unique().collect()["reporter_country"].to_list()
    ).intersection(
        unified_lf.select("partner_country").unique().collect()["partner_country"].to_list()
    )

    # List of all years
    year_list = unified_lf.select("year").unique().collect()["year"].to_list()

    # List of all products
    product_list = unified_lf.select("product_code").unique().collect()["product_code"].to_list()
    return (product_list,)


@app.cell
def _(pl, product_list, time, tqdm, unified_lf):
    data_list = []

    starttime = time.time()
    # for country, product in tqdm(itertools.product(country_list, product_list)):
    for product in tqdm(product_list, desc="iterating over table"):
        country = "840"  # Fix to be USA for now
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
    with open(
        "src/mpil_tariff_trade_analysis/dev2/visualise/processed_ts_USA_product_averages.pkl", "wb"
    ) as f:
        pickle.dump(data_list, f)
    return


@app.cell
def _(pickle):
    with open(
        "src/mpil_tariff_trade_analysis/dev2/visualise/processed_ts_USA_product_averages.pkl", "rb"
    ) as f:
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
        x="year",
        y=["unit_value", "effective_tariff"],
        # title=f"Product {product}; country {country_name}",
        title="Test",
    )
    fig.show()
    return


@app.cell
def _(unified_lf):
    unified_lf.tail().collect()
    return


if __name__ == "__main__":
    app.run()
