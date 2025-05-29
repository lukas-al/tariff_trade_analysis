

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import os
    import random
    import shutil
    import time

    import marimo as mo
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import pycountry
    from plotly.subplots import make_subplots
    return argparse, go, make_subplots, mo, pl, plotly, px, pycountry


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
        lf = pl.scan_parquet("data/final/unified_filtered_10000val_100c_sample_1000krows_filter")
        print('Running in test mode on a sub-sample of the data')
    if args.fullfat:
        lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned/")
        print('   Running on full dataset!')
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
def _():
    # if False: # TESTING ONLY
    #     sample_df = sample_lf.collect(engine='streaming')

    # try: 
    #     sample_df.sample(n=10, seed=69)
    # except NameError:
    #     pass

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
        try:
            country = pycountry.countries.get(numeric=cc)
        except:
            return None

        return country.alpha_3 if country else None

    def get_country_name(cc):
        try:
            country = pycountry.countries.get(numeric=cc)
        except:
            return None
        return country.name if country else None

    aggregated_df = lf.group_by("reporter_country").agg(
        pl.len().alias("record_count"),  # Count rows per group
        pl.sum("value").alias("total_value") # Sum the 'volume' (value) per group
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
    return aggregated_df, get_alpha_3


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


@app.cell
def _(get_alpha_3, go, lf, make_subplots, pl):
    def _():
        # IN FOCUS - US TRADE PARTNERS

        US_NUMERIC_CODE_STR = "840"
        TOP_N_PARTNERS = 20

        us_partner_agg_lf = lf.filter(
            pl.col("reporter_country") == US_NUMERIC_CODE_STR,
        ).group_by("partner_country").agg(
            pl.sum("value").alias("total_value"),
            pl.len().alias("record_count")
        ).with_columns(
            (pl.col('total_value') * 1000)
        )

        us_partner_df = us_partner_agg_lf.collect(engine='streaming')

        # --- Convert Partner Numeric Codes to Alpha-3 ---
        print("Converting partner country codes to ISO alpha-3...")
        us_partner_df = us_partner_df.with_columns(
            pl.col("partner_country")
              .map_elements(get_alpha_3, return_dtype=pl.Utf8, skip_nulls=False) # Apply conversion
              .alias("partner_country_alpha3")
        )

        # --- Prepare Data for Plotting ---
        print("Preparing data for bar charts...")

        # Use alpha-3 code for labels if available and not null, otherwise fallback
        label_col = "partner_country_alpha3"
        # Check if the column exists and has non-null values
        if label_col not in us_partner_df.columns or us_partner_df[label_col].is_null().all():
             print(f"Warning: Column '{label_col}' is missing or all null. Falling back to 'partner_country'.")
             label_col = "partner_country" # Fallback to original numeric code

        # Filter out rows where the chosen label column is null
        us_partner_df = us_partner_df.filter(pl.col(label_col).is_not_null())
        if us_partner_df.is_empty():
            print(f"Error: No data remaining after attempting to map partner country codes.")
            exit()

        # Sort by Total Value and get top N partners
        top_value_partners_df = us_partner_df.sort("total_value", descending=True).head(TOP_N_PARTNERS)

        # Sort by Record Count and get top N partners
        top_count_partners_df = us_partner_df.sort("record_count", descending=True).head(TOP_N_PARTNERS)

        print(f"Top {TOP_N_PARTNERS} US partners by Value:\n{top_value_partners_df.head()}")
        print(f"\nTop {TOP_N_PARTNERS} US partners by Count:\n{top_count_partners_df.head()}")

        # --- Create Bar Charts for US Partners ---
        print("\n--- Generating US Partner Bar Charts ---")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Top {TOP_N_PARTNERS} US Partners by Total Export Value",
                            f"Top {TOP_N_PARTNERS} US Partners by Record Count")
        )

        # Bar Chart for Top Value Partners
        fig.add_trace(
            go.Bar(
                y=top_value_partners_df[label_col].to_list(),
                x=top_value_partners_df["total_value"].to_list(),
                name="Total Value to Partner",
                orientation='h',
                marker_color='rgb(0, 150, 136)' # Teal color
            ),
            row=1, col=1
        )

        # Bar Chart for Top Count Partners
        fig.add_trace(
            go.Bar(
                y=top_count_partners_df[label_col].to_list(),
                x=top_count_partners_df["record_count"].to_list(),
                name="Record Count to Partner",
                orientation='h',
                marker_color='rgb(255, 152, 0)' # Orange color
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"US Exports: Top {TOP_N_PARTNERS} Partners by Value and Record Count",
            height=max(600, TOP_N_PARTNERS * 30), # Adjust height
            showlegend=False,
            yaxis1=dict(autorange="reversed"),
            yaxis2=dict(autorange="reversed"),
            bargap=0.15
        )
        fig.update_xaxes(title_text="Total Export Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Records", row=1, col=2)
        fig.update_yaxes(title_text="artner Country", row=1, col=1)
        fig.update_yaxes(title_text="Partner Country", row=1, col=2)

        fig.write_html("src/mpil_tariff_trade_analysis/dev2/US_exports.html")
        return fig.show()
    # 

    _()
    return


@app.cell
def _(get_alpha_3, go, lf, make_subplots, pl, plotly):
    def create_trade_charts():
        # IN FOCUS - US TRADE PARTNERS

        US_NUMERIC_CODE_STR = "840"
        TOP_N_PARTNERS = 15  # Reduced for clarity in example output/plot

        # === Part 1: Calculate Overall Top Partners (Bar Charts) ===
        print("Calculating overall top partners...")
        us_partner_agg_lf = lf.filter(
            pl.col("reporter_country") == US_NUMERIC_CODE_STR,
        ).group_by("partner_country").agg(
            pl.sum("value").alias("total_value"),
            # Assuming a count mechanism exists or use pl.len() if appropriate
            # If 'record_count' column doesn't exist use pl.len()
            pl.len().alias("record_count") if 'record_count' not in lf.columns else pl.sum("record_count").alias("record_count")
        ).with_columns(
            (pl.col('total_value') * 1000) # Scale value if necessary
        )

        us_partner_df = us_partner_agg_lf.collect() # Use collect(), streaming might not be needed for this size

        # --- Convert Partner Numeric Codes to Alpha-3 ---
        print("Converting partner country codes to ISO alpha-3 for bar charts...")
        us_partner_df = us_partner_df.with_columns(
            pl.col("partner_country")
              .map_elements(get_alpha_3, return_dtype=pl.Utf8, skip_nulls=False) # Apply conversion
              .alias("partner_country_alpha3")
        )

        # --- Prepare Data for Bar Plotting ---
        print("Preparing data for bar charts...")
        label_col = "partner_country_alpha3"
        # Check if the column exists and has non-null values
        if label_col not in us_partner_df.columns or us_partner_df[label_col].is_null().all():
              print(f"Warning: Column '{label_col}' is missing or all null. Falling back to 'partner_country'.")
              label_col = "partner_country" # Fallback to original numeric code

        # Filter out rows where the chosen label column is null
        us_partner_df = us_partner_df.filter(pl.col(label_col).is_not_null())
        if us_partner_df.is_empty():
            print(f"Error: No data remaining after attempting to map partner country codes for bar charts.")
            # exit() # Or handle error appropriately
            return None # Exit function if no data

        # Sort by Total Value and get top N partners
        top_value_partners_df = us_partner_df.sort("total_value", descending=True).head(TOP_N_PARTNERS)

        # Sort by Record Count and get top N partners
        top_count_partners_df = us_partner_df.sort("record_count", descending=True).head(TOP_N_PARTNERS)

        print(f"Top {TOP_N_PARTNERS} US partners by Value:\n{top_value_partners_df.head()}")
        print(f"\nTop {TOP_N_PARTNERS} US partners by Count:\n{top_count_partners_df.head()}")

        # Get the list of top N partner *numeric codes* for filtering time series data
        top_n_partner_codes = top_value_partners_df["partner_country"].to_list()
        print(f"\nTop {TOP_N_PARTNERS} partner numeric codes for time series: {top_n_partner_codes}")

        # === Part 2: Calculate Yearly Volume for Top N Partners (Line Chart) ===
        print("\nCalculating yearly volume for top partners...")

        # Check if 'year' column exists
        if 'year' not in lf.columns:
            print("Error: 'year' column not found in the LazyFrame. Cannot create time series chart.")
            # exit() # Or handle error appropriately
            return None # Exit function if no year data

        # Filter original data for US reporter, top N partners, and aggregate by year/partner
        yearly_volume_lf = lf.filter(
            (pl.col("reporter_country") == US_NUMERIC_CODE_STR) &
            (pl.col("partner_country").is_in(top_n_partner_codes))
        ).group_by(["year", "partner_country"]).agg(
            pl.sum("value").alias("yearly_volume")
        ).sort("year", "partner_country") # Sort for plotting

        yearly_volume_df = yearly_volume_lf.collect() # Collect the results

        # --- Convert Partner Numeric Codes to Alpha-3 for Line Chart ---
        print("Converting partner country codes to ISO alpha-3 for line chart...")
        if not yearly_volume_df.is_empty():
            yearly_volume_df = yearly_volume_df.with_columns(
                pl.col("partner_country")
                  .map_elements(get_alpha_3, return_dtype=pl.Utf8, skip_nulls=False) # Apply conversion
                  .alias("partner_country_alpha3")
            )
            # Filter out nulls *after* mapping, in case some top partners couldn't be mapped
            yearly_volume_df = yearly_volume_df.filter(pl.col("partner_country_alpha3").is_not_null())

        else:
            print("Warning: No yearly volume data found for the top partners.")


        # --- Create Combined Figure ---
        print("\n--- Generating Combined US Partner Charts ---")

        # Create subplots: 2 rows. Row 1 has 2 cols (bars), Row 2 has 1 col spanning both (line)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(f"Top {TOP_N_PARTNERS} Partners by Total Export Value",
                            f"Top {TOP_N_PARTNERS} Partners by Record Count",
                            f"Yearly Export Volume for Top {TOP_N_PARTNERS} Partners by Value"),
            specs=[[{}, {}],           # Row 1: Two normal subplots
                   [{"colspan": 2}, None]] # Row 2: One subplot spanning 2 columns
        )

        # --- Add Bar Charts (Row 1) ---
        # Bar Chart for Top Value Partners
        fig.add_trace(
            go.Bar(
                y=top_value_partners_df[label_col].to_list(),
                x=top_value_partners_df["total_value"].to_list(),
                name="Total Value",
                orientation='h',
                marker_color='rgb(0, 150, 136)' # Teal color
            ),
            row=1, col=1
        )

        # Bar Chart for Top Count Partners
        fig.add_trace(
            go.Bar(
                y=top_count_partners_df[label_col].to_list(),
                x=top_count_partners_df["record_count"].to_list(),
                name="Record Count",
                orientation='h',
                marker_color='rgb(255, 152, 0)' # Orange color
            ),
            row=1, col=2
        )

        # --- Add Line Chart (Row 2) ---
        print("Adding line chart traces...")
        if not yearly_volume_df.is_empty():
            # Use Plotly Express for easier color mapping by country, or loop manually
            # Manual loop approach:
            unique_partners = yearly_volume_df["partner_country_alpha3"].unique().to_list()
            colors = plotly.colors.qualitative.Plotly

            for i, partner_alpha3 in enumerate(unique_partners):
                partner_data = yearly_volume_df.filter(pl.col("partner_country_alpha3") == partner_alpha3)
                fig.add_trace(
                    go.Scatter(
                        x=partner_data["year"].to_list(),
                        y=partner_data["yearly_volume"].to_list(),
                        mode='lines+markers',
                        name=partner_alpha3, # Legend label
                        marker_color=colors[i % len(colors)], # Cycle through colors
                        legendgroup="yearly", # Group legends if needed
                        legendgrouptitle_text="Top Partners (Yearly)"
                    ),
                    row=2, col=1 # Add to the subplot in row 2, column 1 (which spans)
                )
        else:
            print("Skipping line chart generation as no yearly data was prepared.")


        # --- Update Layout ---
        print("Updating figure layout...")
        chart_height = max(600, TOP_N_PARTNERS * 30) # Base height for bars
        total_height = chart_height + 400 # Add space for the line chart + titles/margins

        fig.update_layout(
            title_text=f"US Exports Analysis: Top {TOP_N_PARTNERS} Partners",
            height=total_height,
            showlegend=True, # Show legend for the line chart
            legend=dict(groupclick="toggleitem"), # Allow toggling groups in legend
            yaxis1=dict(autorange="reversed"), # Bar chart Y axis
            yaxis2=dict(autorange="reversed"), # Bar chart Y axis
            # yaxis3 is the line chart's Y-axis (Plotly numbers axes row by row)
            yaxis3_title_text="Yearly Export Value, 000s USD",
            # xaxis3 is the line chart's X-axis
            xaxis3_title_text="Year",
            bargap=0.15
        )
        # Update axes titles specifically
        fig.update_xaxes(title_text="Total Export Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Records", row=1, col=2)
        fig.update_yaxes(title_text="Partner Country", row=1, col=1)
        fig.update_yaxes(title_text="Partner Country", row=1, col=2)

        # Ensure X-axis for line chart shows years properly (treat as category or linear)
        # If years are discrete numbers, linear is fine. If they are strings, category might be better.
        # Assuming 'year' column is numeric:
        fig.update_xaxes(type='linear', dtick=1, row=2, col=1) # Show yearly ticks


        # Save and show
        # output_path = "src/mpil_tariff_trade_analysis/dev2/US_exports_combined.html"
        output_path = "US_exports_combined.html" # Simpler path for testing
        print(f"Saving chart to {output_path}")
        fig.write_html(output_path)
        return fig.show() # Return the figure object

    create_trade_charts()

    return


@app.cell
def _(get_alpha_3, go, lf, make_subplots, pl):
    def _():
        US_NUMERIC_CODE_STR = '840'
        TOP_N_SOURCES = 20

        us_import_agg_lf = lf.filter(
            pl.col("partner_country") == US_NUMERIC_CODE_STR # Filter where US is the partner
        ).group_by("reporter_country").agg( # Group by the reporting country (source of import)
            pl.sum("value").alias("total_value"),
            pl.len().alias("record_count")
        ).with_columns(
            (pl.col('total_value') * 1000)
        )

        # 3. Collect results
        print("Starting US import source aggregation...")
        us_import_source_df = us_import_agg_lf.collect()
        print("US import source aggregation complete.")

        if us_import_source_df.is_empty():
            print(f"Warning: No data found for partner country code {US_NUMERIC_CODE_STR} (US Imports).")
            exit() # Exit if no US import data

        # --- Convert Reporter (Source) Numeric Codes to Alpha-3 ---
        print("Converting source country codes to ISO alpha-3...")
        us_import_source_df = us_import_source_df.with_columns(
            pl.col("reporter_country") # Convert the reporter code now
              .map_elements(get_alpha_3, return_dtype=pl.Utf8, skip_nulls=False)
              .alias("reporter_country_alpha3")
        )

        # --- Prepare Data for Plotting ---
        print("Preparing data for bar charts...")

        # Use alpha-3 code for labels if available and not null, otherwise fallback
        label_col = "reporter_country_alpha3"
        if label_col not in us_import_source_df.columns or us_import_source_df[label_col].is_null().all():
             print(f"Warning: Column '{label_col}' is missing or all null. Falling back to 'reporter_country'.")
             label_col = "reporter_country" # Fallback to original numeric code

        # Filter out rows where the chosen label column is null
        us_import_source_df = us_import_source_df.filter(pl.col(label_col).is_not_null())
        if us_import_source_df.is_empty():
            print(f"Error: No data remaining after attempting to map source country codes.")
            exit()


        # Sort by Total Value and get top N sources
        top_value_sources_df = us_import_source_df.sort("total_value", descending=True).head(TOP_N_SOURCES)

        # Sort by Record Count and get top N sources
        top_count_sources_df = us_import_source_df.sort("record_count", descending=True).head(TOP_N_SOURCES)

        print(f"Top {TOP_N_SOURCES} US import sources by Value:\n{top_value_sources_df.head()}")
        print(f"\nTop {TOP_N_SOURCES} US import sources by Count:\n{top_count_sources_df.head()}")

        # --- Create Bar Charts for US Import Sources ---
        print("\n--- Generating US Import Source Bar Charts ---")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Top {TOP_N_SOURCES} Sources of US Imports by Total Value",
                            f"Top {TOP_N_SOURCES} Sources of US Imports by Record Count")
        )

        # Bar Chart for Top Value Sources
        fig.add_trace(
            go.Bar(
                y=top_value_sources_df[label_col].to_list(), # Source Country on Y-axis
                x=top_value_sources_df["total_value"].to_list(), # Value on X-axis
                name="Total Import Value from Source",
                orientation='h',
                marker_color='rgb(142, 68, 173)' # Purple color
            ),
            row=1, col=1
        )

        # Bar Chart for Top Count Sources
        fig.add_trace(
            go.Bar(
                y=top_count_sources_df[label_col].to_list(), # Source Country on Y-axis
                x=top_count_sources_df["record_count"].to_list(), # Count on X-axis
                name="Record Count from Source",
                orientation='h',
                marker_color='rgb(231, 76, 60)' # Red color
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"US Imports: Top {TOP_N_SOURCES} Source Countries by Value and Record Count",
            height=max(600, TOP_N_SOURCES * 30), # Adjust height
            showlegend=False,
            yaxis1=dict(autorange="reversed"),
            yaxis2=dict(autorange="reversed"),
            bargap=0.15
        )
        fig.update_xaxes(title_text="Total Import Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Records", row=1, col=2)
        fig.update_yaxes(title_text="Source Country", row=1, col=1)
        fig.update_yaxes(title_text="Source Country", row=1, col=2)

        fig.write_html("src/mpil_tariff_trade_analysis/dev2/US_imports.html")
        return fig.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""# Compare the original BACI data""")
    return


@app.cell
def _():
    # baci_original = pl.scan_csv("data/raw/BACI_HS92_V202501/BACI*.csv")

    # baci_usa_exports_lf = baci_original.filter(
    #     pl.col("i") == 842 # Filter where US is the partner (BACI has USA as 842 rather than 840)
    # ).group_by(["j", 't']).agg( # Group by the reporting country (source of import)
    #     pl.sum('v').alias('total_volume'),
    #     pl.len().alias("record_count")
    # ).with_columns(
    #     (pl.col('total_volume') * 1000)
    # ).sort("t", "j")

    # trf_baci_usa_exports = baci_usa_exports_lf.with_columns(
    #     pl.col("j").cast(pl.Utf8).str.zfill(3)
    #       .map_elements(get_alpha_3, return_dtype=pl.Utf8, skip_nulls=False) # Apply conversion
    #       .alias("partner_country_alpha3")
    # )

    # trf_baci_usa_exports_pd = trf_baci_usa_exports.collect(engine='streaming').to_pandas()
    # fig_px_baci = px.line(
    #         trf_baci_usa_exports_pd,
    #         x="t",                      # Column for the x-axis
    #         y="total_volume",           # Column for the y-axis
    #         color="partner_country_alpha3", # Column to define line colors and legend entries
    #         markers=True,               # Show markers on the lines (equiv. to mode='lines+markers')
    #         title="USA Exports Volume by Partner Country - BACI ORIGINAL", # Chart title
    #         labels={                    # Optional: Customize axis/legend labels
    #             "t": "Year",
    #             "total_volume": "Total Export Volume",
    #             "partner_country_alpha3": "Partner Country"
    #         },
    # )

    # fig_px_baci.show()
    return


@app.cell
def _():
    # def _():
    #     baci_intermediate = pl.scan_parquet("data/intermediate/BACI_HS92_V202501_CLEAN.parquet")

    #     baci_usa_exports_lf = baci_intermediate.filter(
    #         pl.col("i") == "840" # Filter where US is the partner (BACI has USA as 842 rather than 840)
    #     ).group_by(["j", 't']).agg( # Group by the reporting country (source of import)
    #         pl.sum('v').alias('total_value'),
    #         pl.len().alias("record_count")
    #     ).with_columns(
    #         (pl.col('total_value') * 1000)
    #     ).sort("t", "j")

    #     trf_baci_usa_exports = baci_usa_exports_lf.with_columns(
    #         pl.col("j").cast(pl.Utf8).str.zfill(3)
    #           .map_elements(get_alpha_3, return_dtype=pl.Utf8, skip_nulls=False) # Apply conversion
    #           .alias("partner_country_alpha3")
    #     )

    #     trf_baci_usa_exports_pd = trf_baci_usa_exports.collect(engine='streaming').to_pandas()
    #     fig_px_baci = px.line(
    #             trf_baci_usa_exports_pd,
    #             x="t",                      # Column for the x-axis
    #             y="total_value",           # Column for the y-axis
    #             color="partner_country_alpha3", # Column to define line colors and legend entries
    #             markers=True,               # Show markers on the lines (equiv. to mode='lines+markers')
    #             title="USA Exports Value by Partner Country - BACI INTERMEDIATE", # Chart title
    #             labels={                    # Optional: Customize axis/legend labels
    #                 "t": "Year",
    #                 "total_value": "Total Export Value",
    #                 "partner_country_alpha3": "Partner Country"
    #             },
    #     )
    #     return fig_px_baci.show()

    # _()
    return


@app.cell
def _(mo):
    mo.md(r"""# Tariffs""")
    return


@app.cell
def _(lf, px):
    def distplot_effective_values(arg_lf):    

        # --- Option 1: Normalized Histogram using Plotly Express ---
        # Collect the data from the LazyFrame column
        data_df = arg_lf.select('average_tariff_official').collect()

        # Create the normalized histogram
        # Plotly Express often works directly with Polars DataFrames/Series
        fig_hist = px.histogram(
            data_df.drop_nulls(),
            x='average_tariff_official',
            histnorm='probability density', # Normalize to represent density
            title='Normalised dist. of effective tariff values',
            nbins=3000,
            log_y=True,
        )


        # --- Displaying the plots (choose one or both) ---
        print("Showing normalized histogram (px.histogram):")
        return fig_hist

    fig_hist = distplot_effective_values(lf)
    fig_hist.write_html("src/mpil_tariff_trade_analysis/dev2/dist_of_effective_tariffs.html")

    return (distplot_effective_values,)


@app.cell
def _(distplot_effective_values, lf, pl):
    # Can we do the same, but remove all inter-EU trades?
    EU_COUNTRY_LIST = ["040", "056", "100", "191", "196", "203", "208", "233", "246", "250", "276", "300", "348", "372", "380", "428", "440", "442", "470", "528", "616", "620", "642", "703", "705", "724", "752", "492"]

    filtered_lf = lf.filter(
        ~(
            (pl.col('reporter_country').is_in(EU_COUNTRY_LIST)) &
            (pl.col('partner_country').is_in(EU_COUNTRY_LIST))
        )
    )

    fig_hist_nonEU = distplot_effective_values(filtered_lf)
    fig_hist_nonEU.write_html("src/mpil_tariff_trade_analysis/dev2/dist_of_effective_tariffs_NON-EU.html")
    return


@app.cell
def _(mo):
    mo.md(r"""# Summary""")
    return


@app.cell
def _(lf):
    print("Writing unified summary to txt")
    lf_desc = lf.describe()
    lf_desc.serialize("src/mpil_tariff_trade_analysis/dev2/full_summary.txt", format='json')
    return


@app.cell
def _():
    print("--- COMPLETE ---")
    return


@app.cell
def _(pl):
    # Full desc:
    full_lf_desc = pl.DataFrame.deserialize('src/mpil_tariff_trade_analysis/dev2/full_summary.txt', format='json')

    full_lf_desc
    return


if __name__ == "__main__":
    app.run()
