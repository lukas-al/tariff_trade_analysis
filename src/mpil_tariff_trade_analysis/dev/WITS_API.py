import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # World Trade Data API
        Use the API to consume, reporter-partner-product triple the range of tariffs through time.
        """
    )
    return


@app.cell
def _():
    from mpil_tariff_trade_analysis.etl.world_trade_data_api import (
        get_indicator,
        get_tariff_reported,
        get_tariff_estimated,
    )

    import mpil_tariff_trade_analysis as mtta
    return get_indicator, get_tariff_estimated, get_tariff_reported, mtta


@app.cell
def _(mtta):
    mtta.etl.world_trade_data_api.get_dataavailability()
    return


@app.cell
def _(mo):
    mo.md(
        """
        Need a list of all countries:

        - CEPI BACI uses ISO 3-digit codes
        - WITS uses a mix of ISO (name) and numeric 3 digit code. Available on the dataavailability endpoint.


        Need a list of all products, at the right level of interest.

        Then run the tariff scrape over each triplet, saving the data to some optimised store

        Then we can align the values, volumes, estimated and reported tariffs into one graph.abs

        Then we can add on the services (value & volume)
        """
    )
    return


@app.cell
def _(wits):
    import itertools
    import os
    import time

    import pandas as pd
    from tqdm.auto import tqdm

    # Function to get tariff data for a given reporter, partner, product triple
    def get_tariff_data(reporter, partner, product):
        try:
            # Get tariff data from WITS API
            df = wits.get_tariff_data(reporter=reporter, partner=partner, product=product)

            # Add metadata columns if data was returned
            if not df.empty:
                df["reporter"] = reporter
                df["partner"] = partner
                df["product"] = product
            return df
        except Exception as e:
            print(f"Error retrieving data for {reporter}-{partner}-{product}: {e}")
            return pd.DataFrame()  # Return empty dataframe on error

    # Main function to retrieve and save tariff data
    def retrieve_and_save_tariff_data(
        reporters, partners, products, output_dir="tariff_data_output"
    ):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate all combinations
        combinations = list(itertools.product(reporters, partners, products))

        # Process combinations with progress bar
        for index, (reporter, partner, product) in enumerate(
            tqdm(combinations, desc="Processing tariff data")
        ):
            # Get tariff data
            tariff_df = get_tariff_data(reporter, partner, product)

            # Save data if not empty
            if not tariff_df.empty:
                filename = f"tariff_{reporter}_{partner}_{product}.parquet"
                filepath = os.path.join(output_dir, filename)
                tariff_df.to_parquet(filepath, index=False)

            # Small delay to avoid API rate limits
            time.sleep(0.5)

    # Example usage
    reporters = ["USA", "CHN", "DEU"]  # Example reporter country codes
    partners = ["CAN", "MEX", "FRA"]  # Example partner country codes
    products = ["1001", "1002", "1003"]  # Example product codes (HS codes)

    # Uncomment to run the retrieval process
    # retrieve_and_save_tariff_data(reporters, partners, products)
    return (
        get_tariff_data,
        itertools,
        os,
        partners,
        pd,
        products,
        reporters,
        retrieve_and_save_tariff_data,
        time,
        tqdm,
    )


if __name__ == "__main__":
    app.run()
