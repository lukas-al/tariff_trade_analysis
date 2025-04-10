import marimo

from mpil_tariff_trade_analysis.utils.logging_config import get_logger, setup_logging

__generated_with = "0.10.12"
app = marimo.App(width="medium")

# Set up logging
logger = get_logger(__name__)


@app.cell
def _():
    import marimo as mo

    logger.info("Application started")
    mo.md("Hello")
    return (mo,)


@app.cell
def test_cell():
    logger.debug("Test cell executed")
    return


if __name__ == "__main__":
    # Initialize logging when app is run directly
    setup_logging()
    logger.info("Starting MPIL Tariff Trade Analysis application")
    app.run()
