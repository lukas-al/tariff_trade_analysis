

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import time
    import os
    import random
    import shutil
    import plotly.express as px
    import pycountry
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import argparse
    return argparse, go, make_subplots, mo, pl, px, pycountry


@app.cell
def _(argparse):
    parser = argparse.ArgumentParser(description="Marimo visualise unified")

    # Add your parameters/arguments here
    parser.add_argument("--fullfat", action="store_true", help="Using this flag will run on all the data")
    args = parser.parse_args()

    args.fullfat
    return (args,)


@app.cell
def _(mo):
    mo.md(r"""# Explore our unified dataset""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Reduce the size of the dataset...
        We can either:

        - Only consider a subset of countries
        - Remove sub-scale trade volumes
        - Do both.

        Lets limit our analysis to the top 100 countries by trading volume

        We also discard any trades worth less than $100.000 (BACI is denonomited in thousands)
        """
    )
    return


@app.cell
def _(args, pl):
    print("--- CREATING CHARTS ---")
    if not args.fullfat:
        lf = pl.scan_parquet("data/final/unified_filtered_10000val_100c_sample_10000.0krows_filter")
        print('Running in test mode on a sub-sample of the data')
    if args.fullfat:
        lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned/")
        print('Running on full dataset')
    return (lf,)


@app.cell
def _(lf):
    print(lf.head().collect(engine='streaming'))
    return


@app.cell
def _(lf):
    print(lf.collect_schema().names())
    return


@app.cell
def _(mo):
    mo.md(r"""# Validate""")
    return


@app.cell
def _(sample_lf):
    if False: # TESTING ONLY
        sample_df = sample_lf.collect(engine='streaming')

    try: 
        sample_df.sample(n=10, seed=69)
    except NameError:
        pass

    return


@app.cell
def _(mo):
    mo.md(
        """
        1. Germany to Belgium. Lead and such, 2017. Value matches, in EU so tariff 0 matches.
        2. Russia to Austria. Oils petroleum. Can't find value in Comtrade. Tariff matches
        3. Netherlands to Belgium. 2020. Crabs. -> **Can't find this. This is an intra-EU trade, but it's being picked up as 19%. I can't find evidence to support that.**
        4. Morocco to Italy. 2018. Crabs. Tariff matches (Morocco pref. agreement). Can't find the value / volume.

        WTO groups the EU...
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Visualisation""")
    return


@app.cell
def _(lf, pl, pycountry):
    def get_alpha_3(cc):
        country = pycountry.countries.get(numeric=cc)
        return country.alpha_3 if country else None

    def get_country_name(cc):
        country = pycountry.countries.get(numeric=cc)
        return country.name if country else None

    aggregated_df = lf.group_by("reporter_country").agg(
        pl.len().alias("record_count"),  # Count rows per group
        pl.sum("volume").alias("total_value") # Sum the 'volume' (value) per group
    ).with_columns(
        pl.col('reporter_country')
        .map_elements(get_alpha_3, return_dtype=pl.Utf8)
        .alias('reporter_country_alpha3')
    ).with_columns(
        pl.col('reporter_country')
        .map_elements(get_country_name, return_dtype=pl.Utf8)
        .alias('reporter_country_name')
    ).with_columns(
        (pl.col('total_value') * 1000)
    ).collect(engine='streaming')
    return (aggregated_df,)


@app.cell
def _(aggregated_df, px):

    #

    print("\n--- Generating Plotly geomap ---")
    fig = px.scatter_geo(
        aggregated_df.to_pandas(), 
        locations="reporter_country_alpha3",
        locationmode="ISO-3",  
        size="record_count",   
        color="total_value",   
        hover_name="reporter_country_name",
        hover_data={                   
            "reporter_country_alpha3": True,
            "reporter_country": True,
            "record_count": ':,',  
            "total_value": ':,.2f'
        },
        projection="natural earth",
        title="Global Exports: Record Count (Size) and Total Value (Color) (USD) by Reporter Country",
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=50
    )

    # Improve layout
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        geo=dict(
            showland=True,
            landcolor="rgb(217, 217, 217)",
            subunitcolor="rgb(255, 255, 255)"
        )
    )

    print("Displaying and saving map...")
    fig.show()


    fig.write_html('src/mpil_tariff_trade_analysis/dev2/Geo_Scatter.html')
    return


@app.cell
def _(aggregated_df, go, make_subplots):


    print("\n--- Generating Bar Charts ---")

    TOP_N = 20
    label_col = "reporter_country_alpha3"

    top_value_df = aggregated_df.sort("total_value", descending=True).head(TOP_N)
    top_count_df = aggregated_df.sort("record_count", descending=True).head(TOP_N)

    bar_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top {TOP_N} Countries by Total Export Value",
                        f"Top {TOP_N} Countries by Record Count")
    )

    bar_fig.add_trace(
        go.Bar(
            y=top_value_df[label_col].to_list(),
            x=top_value_df["total_value"].to_list(), 
            name="Total Value",
            orientation='h',
            marker_color='#FF6925'
        ),
        row=1, col=1 
    )

    bar_fig.add_trace(
        go.Bar(
            y=top_count_df[label_col].to_list(),
            x=top_count_df["record_count"].to_list(),
            name="Record Count",
            orientation='h',
            marker_color='#4B6BFF'
        ),
        row=1, col=2
    )

    bar_fig.update_layout(
        title_text=f"Top {TOP_N} Exporting Countries by Value and Record Count",
        height=max(600, TOP_N * 30), 
        showlegend=False,
        yaxis1=dict(autorange="reversed"),
        yaxis2=dict(autorange="reversed"),
        bargap=0.15
    )

    bar_fig.update_xaxes(title_text="Total Export Value ($)", row=1, col=1)
    bar_fig.update_xaxes(title_text="Number of Records", row=1, col=2)
    bar_fig.update_yaxes(title_text="Country", row=1, col=1)
    bar_fig.update_yaxes(title_text="Country", row=1, col=2)


    bar_fig.show()
    bar_fig.write_html("src/mpil_tariff_trade_analysis/dev2/Bar_plot.html")
    return


if __name__ == "__main__":
    app.run()
