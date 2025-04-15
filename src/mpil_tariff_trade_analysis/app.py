import marimo
import os
from pathlib import Path

# Import pipeline functions
from mpil_tariff_trade_analysis.etl.baci import baci_to_parquet # Assuming this is updated to write partitioned parquet
from mpil_tariff_trade_analysis.etl.WITS_cleaner import process_and_save_wits_data
from mpil_tariff_trade_analysis.etl.matching_chunked import run_chunked_matching_pipeline
# Import default paths from matching_logic to ensure consistency
from mpil_tariff_trade_analysis.etl.matching_logic import (
    DEFAULT_BACI_PATH,
    DEFAULT_WITS_MFN_PATH,
    DEFAULT_WITS_PREF_PATH,
    DEFAULT_PREF_GROUPS_PATH,
)
# Import default output path from matching_chunked
from mpil_tariff_trade_analysis.etl.matching_chunked import DEFAULT_CHUNKED_OUTPUT_DIR

from mpil_tariff_trade_analysis.utils.logging_config import get_logger, setup_logging

__generated_with = "0.10.12" # Keep original or update as needed
app = marimo.App(width="medium")

# --- Configuration ---
# Setup logging first
setup_logging()
logger = get_logger(__name__)

# Define data parameters (Consider making these configurable with marimo UI elements later)
HS_CODE = "HS92"
BACI_RELEASE = "V202501"
RAW_DATA_DIR = "data/raw"
INTERMEDIATE_DATA_DIR = "data/intermediate"
FINAL_DATA_DIR = "data/final" # Used by chunked matching output

# Construct expected paths based on conventions and defaults
# Ensure paths are strings for functions that might not handle Path objects
baci_input_folder = str(Path(RAW_DATA_DIR))
# baci_to_parquet expects output_folder, it constructs the full path inside
baci_output_folder = str(Path(INTERMEDIATE_DATA_DIR))
# WITS cleaner paths
wits_mfn_output_path = str(DEFAULT_WITS_MFN_PATH) # data/intermediate/WITS_AVEMFN.parquet
wits_pref_output_path = str(DEFAULT_WITS_PREF_PATH) # data/intermediate/WITS_AVEPref.parquet
wits_raw_dir = str(Path(RAW_DATA_DIR) / "WITS_tariff") # Assuming this structure
# Matching pipeline paths
baci_intermediate_path_for_matching = str(DEFAULT_BACI_PATH) # data/intermediate/BACI_HS92_V202501
pref_groups_path = str(DEFAULT_PREF_GROUPS_PATH) # data/raw/WITS_pref_groups/WITS_pref_groups.csv
matching_output_dir = str(DEFAULT_CHUNKED_OUTPUT_DIR) # data/final/unified_trade_tariff_partitioned


@app.cell
def _(mo):
    logger.info("Pipeline Application Started")
    return mo.md(
        """
        # MPIL Tariff Trade Analysis Pipeline

        This application orchestrates the data processing pipeline:
        1. Process BACI data (CSV to Partitioned Parquet).
        2. Process WITS Tariff data (MFN and Preferential).
        3. Run the Matching Pipeline to unify trade and tariff data.
        """
    ),


@app.cell
def _(
    BACI_RELEASE,
    HS_CODE,
    baci_input_folder,
    baci_output_folder,
    baci_to_parquet,
    mo,
):
    status_md = mo.md("### 1. Processing BACI Data")
    baci_result_path = None # Initialize path variable
    try:
        logger.info(f"Starting BACI processing: HS={HS_CODE}, Release={BACI_RELEASE}")
        logger.info(f"Input folder: {baci_input_folder}, Output folder: {baci_output_folder}")

        # Ensure the function baci_to_parquet is modified to write
        # partitioned parquet to a directory named 'BACI_{hs}_V{release}'
        # inside the output_folder, partitioned by 't'.
        # The function should return the path to the created directory.
        baci_result_path = baci_to_parquet(
            hs=HS_CODE,
            release=BACI_RELEASE,
            input_folder=baci_input_folder,
            output_folder=baci_output_folder,
        )

        if baci_result_path and Path(baci_result_path).exists():
            logger.info(f"BACI processing completed. Output: {baci_result_path}")
            status_md.append(mo.md(f"✅ BACI data processed successfully. Output: `{baci_result_path}`"))
        else:
            # Handle case where function didn't return a valid path or path doesn't exist
            logger.error(f"BACI processing function did not return a valid output path or path does not exist: {baci_result_path}")
            status_md.append(mo.md(f"❌ BACI processing reported success but output path is invalid or missing: `{baci_result_path}`"))
            baci_result_path = None # Ensure it's None on failure

    except Exception as e:
        logger.error(f"BACI processing failed: {e}", exc_info=True)
        status_md.append(mo.md(f"❌ BACI processing failed: {e}"))
        baci_result_path = None # Ensure it's None on failure
        # Optionally raise to stop execution or just log and continue
        # raise e
    return status_md, baci_result_path


@app.cell
def _(
    INTERMEDIATE_DATA_DIR,
    mo,
    process_and_save_wits_data,
    wits_mfn_output_path,
    wits_pref_output_path,
    wits_raw_dir,
):
    status_md = mo.md("### 2. Processing WITS Tariff Data")
    wits_processed = True # Flag to track success

    # Process MFN Tariffs
    try:
        logger.info("Starting WITS MFN Tariff processing.")
        logger.info(f"Raw WITS dir: {wits_raw_dir}")
        logger.info(f"Output path: {wits_mfn_output_path}")
        mfn_result_path = process_and_save_wits_data(
            tariff_type="AVEMFN",
            base_dir=wits_raw_dir,
            output_dir=INTERMEDIATE_DATA_DIR # Function constructs filename internally
        )
        if mfn_result_path and Path(mfn_result_path).exists():
            logger.info("WITS MFN Tariff processing completed.")
            status_md.append(mo.md(f"✅ WITS MFN data processed successfully. Output: `{wits_mfn_output_path}`"))
        else:
            logger.error(f"WITS MFN processing failed or did not produce output at {wits_mfn_output_path}")
            status_md.append(mo.md(f"❌ WITS MFN processing failed or output missing."))
            wits_processed = False

    except Exception as e:
        logger.error(f"WITS MFN processing failed: {e}", exc_info=True)
        status_md.append(mo.md(f"❌ WITS MFN processing failed: {e}"))
        wits_processed = False

    # Process Preferential Tariffs
    try:
        logger.info("Starting WITS Preferential Tariff processing.")
        logger.info(f"Raw WITS dir: {wits_raw_dir}")
        logger.info(f"Output path: {wits_pref_output_path}")
        pref_result_path = process_and_save_wits_data(
            tariff_type="AVEPref",
             base_dir=wits_raw_dir,
            output_dir=INTERMEDIATE_DATA_DIR # Function constructs filename internally
        )
        if pref_result_path and Path(pref_result_path).exists():
            logger.info("WITS Preferential Tariff processing completed.")
            status_md.append(mo.md(f"✅ WITS Preferential data processed successfully. Output: `{wits_pref_output_path}`"))
        else:
            logger.error(f"WITS Preferential processing failed or did not produce output at {wits_pref_output_path}")
            status_md.append(mo.md(f"❌ WITS Preferential processing failed or output missing."))
            wits_processed = False

    except Exception as e:
        logger.error(f"WITS Preferential processing failed: {e}", exc_info=True)
        status_md.append(mo.md(f"❌ WITS Preferential processing failed: {e}"))
        wits_processed = False

    return status_md, wits_processed


@app.cell
def _(
    Path,
    baci_intermediate_path_for_matching,
    baci_result_path, # Get status from BACI cell
    matching_output_dir,
    mo,
    pref_groups_path,
    run_chunked_matching_pipeline,
    wits_mfn_output_path,
    wits_pref_output_path,
    wits_processed, # Get status from WITS cell
):
    status_md = mo.md("### 3. Running Matching Pipeline")

    # Check prerequisites before running
    prereqs_met = True
    if not wits_processed:
         status_md.append(mo.md("Skipping matching pipeline due to errors in WITS processing."))
         logger.warning("Skipping matching pipeline due to errors in WITS processing.")
         prereqs_met = False

    # Check if BACI processing was successful (using the path returned from the BACI cell)
    if not baci_result_path or not Path(baci_result_path).exists():
         # Use the expected path for the error message if the result path is None
         expected_baci_path = baci_intermediate_path_for_matching
         status_md.append(mo.md(f"Skipping matching pipeline: BACI input missing or failed at `{expected_baci_path}`."))
         logger.warning(f"Skipping matching pipeline: BACI input missing or failed at `{expected_baci_path}`.")
         prereqs_met = False
    elif baci_result_path != baci_intermediate_path_for_matching:
        # Log a warning if the actual output path doesn't match the expected default path
        logger.warning(f"BACI output path '{baci_result_path}' differs from expected default '{baci_intermediate_path_for_matching}'. Using actual path.")
        # Use the actual path from the BACI step for the matching pipeline
        baci_input_for_matching = baci_result_path
    else:
        baci_input_for_matching = baci_intermediate_path_for_matching


    if not prereqs_met:
        return status_md # Stop execution of this cell

    # Proceed with matching
    try:
        logger.info("Starting Matching Pipeline.")
        logger.info(f"BACI input: {baci_input_for_matching}")
        logger.info(f"WITS MFN input: {wits_mfn_output_path}")
        logger.info(f"WITS Pref input: {wits_pref_output_path}")
        logger.info(f"Pref Groups input: {pref_groups_path}")
        logger.info(f"Output directory: {matching_output_dir}")

        run_chunked_matching_pipeline(
            baci_path=baci_input_for_matching, # Use the determined BACI path
            wits_mfn_path=wits_mfn_output_path,
            wits_pref_path=wits_pref_output_path,
            pref_groups_path=pref_groups_path,
            output_dir=matching_output_dir,
            # Keep default chunk/partition columns unless needed otherwise
            # chunk_column="t",
            # partition_column="Year",
        )
        logger.info(f"Matching Pipeline completed successfully. Output: {matching_output_dir}")
        status_md.append(mo.md(f"✅ Matching Pipeline completed successfully. Output: `{matching_output_dir}`"))

    except Exception as e:
        logger.error(f"Matching Pipeline failed: {e}", exc_info=True)
        status_md.append(mo.md(f"❌ Matching Pipeline failed: {e}"))
        # raise e # Optional: stop execution

    return status_md


@app.cell
def _(mo):
    logger.info("Pipeline execution finished.")
    return mo.md("--- Pipeline Execution Finished ---")


if __name__ == "__main__":
    # Logging is already set up above
    logger.info("Starting MPIL Tariff Trade Analysis pipeline orchestrator")
    app.run()
