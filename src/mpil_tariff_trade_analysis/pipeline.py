import sys  # Import sys for exit codes
from pathlib import Path

from mpil_tariff_trade_analysis.etl.baci import baci_to_parquet_incremental

# Import pipeline functions
from mpil_tariff_trade_analysis.etl.matching_chunked import (  # Explicitly import needed functions
    DEFAULT_CHUNKED_OUTPUT_DIR,
    run_chunked_matching_pipeline,
)
from mpil_tariff_trade_analysis.etl.matching_logic import (
    DEFAULT_BACI_PATH,
    DEFAULT_PREF_GROUPS_PATH,
    DEFAULT_WITS_MFN_PATH,
    DEFAULT_WITS_PREF_PATH,
)
from mpil_tariff_trade_analysis.etl.WITS_cleaner import process_and_save_wits_data
from mpil_tariff_trade_analysis.utils.logging_config import get_logger, setup_logging

# --- Configuration ---
# Setup logging first
# setup_logging()
# logger = get_logger(__name__)

# # Define data parameters (Consider making these configurable with marimo UI elements later)
# HS_CODE = "HS92"
# BACI_RELEASE = "V202501"
# RAW_DATA_DIR = "data/raw"
# INTERMEDIATE_DATA_DIR = "data/intermediate"
# FINAL_DATA_DIR = "data/final"  # Used by chunked matching output

# # Construct expected paths based on conventions and defaults
# # Ensure paths are strings for functions that might not handle Path objects
# baci_input_folder = str(Path(RAW_DATA_DIR))
# # baci_to_parquet expects output_folder, it constructs the full path inside
# baci_output_folder = str(Path(INTERMEDIATE_DATA_DIR))
# # WITS cleaner paths
# wits_mfn_output_path = str(DEFAULT_WITS_MFN_PATH)  # data/intermediate/WITS_AVEMFN.parquet
# wits_pref_output_path = str(DEFAULT_WITS_PREF_PATH)  # data/intermediate/WITS_AVEPref.parquet
# wits_raw_dir = str(Path(RAW_DATA_DIR) / "WITS_tariff")  # Assuming this structure
# # Matching pipeline paths
# baci_intermediate_path_for_matching = str(DEFAULT_BACI_PATH)  # data/intermediate/BACI_HS92_V202501
# pref_groups_path = str(DEFAULT_PREF_GROUPS_PATH)  # data/raw/WITS_pref_groups/WITS_pref_groups.csv
# matching_output_dir = str(DEFAULT_CHUNKED_OUTPUT_DIR)  # data/final/unified_trade_tariff_partitioned


# --- Global Configuration & Setup ---
setup_logging()
logger = get_logger(__name__)


# Define data parameters and directories directly
def get_config():
    """Returns a dictionary containing configuration parameters."""
    HS_CODE = "HS92"
    BACI_RELEASE = "202501"
    RAW_DATA_DIR = "data/raw"
    INTERMEDIATE_DATA_DIR = "data/intermediate"
    FINAL_DATA_DIR = "data/final"  # Used by chunked matching output

    # Construct expected paths based on conventions and defaults
    baci_input_folder = str(Path(RAW_DATA_DIR))
    baci_output_folder = str(Path(INTERMEDIATE_DATA_DIR))
    wits_mfn_output_path = str(DEFAULT_WITS_MFN_PATH)  # e.g., data/intermediate/WITS_AVEMFN.parquet
    wits_pref_output_path = str(
        DEFAULT_WITS_PREF_PATH
    )  # e.g., data/intermediate/WITS_AVEPref.parquet
    wits_raw_dir = str(Path(RAW_DATA_DIR) / "WITS_tariff")  # Assuming this structure
    baci_intermediate_path_for_matching = str(
        DEFAULT_BACI_PATH
    )  # e.g., data/intermediate/BACI_HS92_V202501
    pref_groups_path = str(
        DEFAULT_PREF_GROUPS_PATH
    )  # e.g., data/raw/WITS_pref_groups/WITS_pref_groups.csv
    matching_output_dir = str(
        DEFAULT_CHUNKED_OUTPUT_DIR
    )  # e.g., data/final/unified_trade_tariff_partitioned

    # Return configuration parameters as a dict so they can be injected into later cells
    return {
        "HS_CODE": HS_CODE,
        "BACI_RELEASE": BACI_RELEASE,
        "RAW_DATA_DIR": RAW_DATA_DIR,
        "INTERMEDIATE_DATA_DIR": INTERMEDIATE_DATA_DIR,
        "FINAL_DATA_DIR": FINAL_DATA_DIR,
        "baci_input_folder": baci_input_folder,
        "baci_output_folder": baci_output_folder,
        "wits_mfn_output_path": wits_mfn_output_path,
        "wits_pref_output_path": wits_pref_output_path,
        "wits_raw_dir": wits_raw_dir,
        "baci_intermediate_path_for_matching": baci_intermediate_path_for_matching,
        "pref_groups_path": pref_groups_path,
        "matching_output_dir": matching_output_dir,
    }


# --- Pipeline Steps as Functions ---


def process_baci_data(config_params):
    """Processes BACI data from CSV to partitioned Parquet."""
    logger.info("--- Step 1: Processing BACI Data ---")
    baci_result_path = None
    try:
        hs_code = config_params["HS_CODE"]
        baci_release = config_params["BACI_RELEASE"]
        input_folder = config_params["baci_input_folder"]
        output_folder = config_params["baci_output_folder"]

        logger.info(f"Starting BACI processing: HS={hs_code}, Release={baci_release}")
        logger.info(f"Input folder: {input_folder}, Output folder: {output_folder}")

        # baci_to_parquet_incremental writes
        # partitioned parquet to a directory named 'BACI_{hs}_V{release}'
        # inside the output_folder, partitioned by 't'.
        # The function should return the path to the created directory.
        baci_result_path = baci_to_parquet_incremental(
            hs=hs_code,
            release=baci_release,
            input_folder=input_folder,
            output_folder=output_folder,
        )

        if baci_result_path and Path(baci_result_path).exists():
            logger.info(f"✅ BACI data processed successfully. Output: `{baci_result_path}`")
        else:
            logger.error(
                f"❌ BACI processing reported success but output path is invalid or missing: `{baci_result_path}`"
            )
            baci_result_path = None  # Ensure it's None on failure

    except Exception as e:
        logger.error(f"❌ BACI processing failed: {e}", exc_info=True)
        baci_result_path = None  # Ensure it's None on failure

    return baci_result_path  # Return the path or None


def process_wits_data(config_params):
    """Processes WITS MFN and Preferential tariff data."""
    logger.info("--- Step 2: Processing WITS Tariff Data ---")
    wits_processed_successfully = True  # Flag to track overall success

    intermediate_dir = config_params["INTERMEDIATE_DATA_DIR"]
    wits_raw_dir = config_params["wits_raw_dir"]
    mfn_output_path = config_params["wits_mfn_output_path"]
    pref_output_path = config_params["wits_pref_output_path"]

    # Process MFN Tariffs
    try:
        logger.info("Starting WITS MFN Tariff processing.")
        logger.info(f"Raw WITS dir: {wits_raw_dir}")
        logger.info(f"Expected output path: {mfn_output_path}")
        mfn_result_path = process_and_save_wits_data(
            tariff_type="AVEMFN",
            base_dir=wits_raw_dir,
            output_dir=intermediate_dir,  # Function constructs filename internally
        )
        # Verify the expected output file was created
        if (
            mfn_result_path
            and Path(mfn_result_path).exists()
            and str(mfn_result_path) == mfn_output_path
        ):
            logger.info(f"✅ WITS MFN data processed successfully. Output: `{mfn_output_path}`")
        else:
            logger.error(
                f"❌ WITS MFN processing failed or did not produce expected output at {mfn_output_path}. Actual result: {mfn_result_path}"
            )
            wits_processed_successfully = False

    except Exception as e:
        logger.error(f"❌ WITS MFN processing failed: {e}", exc_info=True)
        wits_processed_successfully = False

    # Process Preferential Tariffs
    try:
        logger.info("Starting WITS Preferential Tariff processing.")
        logger.info(f"Raw WITS dir: {wits_raw_dir}")
        logger.info(f"Expected output path: {pref_output_path}")
        pref_result_path = process_and_save_wits_data(
            tariff_type="AVEPref",
            base_dir=wits_raw_dir,
            output_dir=intermediate_dir,  # Function constructs filename internally
        )
        # Verify the expected output file was created
        if (
            pref_result_path
            and Path(pref_result_path).exists()
            and str(pref_result_path) == pref_output_path
        ):
            logger.info(
                f"✅ WITS Preferential data processed successfully. Output: `{pref_output_path}`"
            )
        else:
            logger.error(
                f"❌ WITS Preferential processing failed or did not produce expected output at {pref_output_path}. Actual result: {pref_result_path}"
            )
            wits_processed_successfully = False

    except Exception as e:
        logger.error(f"❌ WITS Preferential processing failed: {e}", exc_info=True)
        wits_processed_successfully = False

    return wits_processed_successfully


def run_matching_pipeline(config_params, baci_result_path, wits_processed_status):
    """Runs the matching pipeline to unify trade and tariff data."""
    logger.info("--- Step 3: Running Matching Pipeline ---")

    # Check prerequisites before running
    if not wits_processed_status:
        logger.warning("Skipping matching pipeline due to errors in WITS processing.")
        return False  # Indicate failure

    expected_baci_path = config_params["baci_intermediate_path_for_matching"]
    if not baci_result_path or not Path(baci_result_path).exists():
        logger.warning(
            f"Skipping matching pipeline: BACI input missing or failed. Expected location: `{expected_baci_path}`. Actual result path: {baci_result_path}"
        )
        return False  # Indicate failure
    elif baci_result_path != expected_baci_path:
        logger.warning(
            f"BACI output path '{baci_result_path}' differs from expected default '{expected_baci_path}'. Using actual path for matching."
        )
        # Use the actual path from the BACI step
        baci_input_for_matching = baci_result_path
    else:
        baci_input_for_matching = expected_baci_path  # Use the expected path

    # Proceed with matching
    try:
        wits_mfn_path = config_params["wits_mfn_output_path"]
        wits_pref_path = config_params["wits_pref_output_path"]
        pref_groups_path = config_params["pref_groups_path"]
        matching_output_dir = config_params["matching_output_dir"]

        logger.info("Starting Matching Pipeline.")
        logger.info(f"BACI input: {baci_input_for_matching}")
        logger.info(f"WITS MFN input: {wits_mfn_path}")
        logger.info(f"WITS Pref input: {wits_pref_path}")
        logger.info(f"Pref Groups input: {pref_groups_path}")
        logger.info(f"Output directory: {matching_output_dir}")

        run_chunked_matching_pipeline(
            baci_path=baci_input_for_matching,
            wits_mfn_path=wits_mfn_path,
            wits_pref_path=wits_pref_path,
            pref_groups_path=pref_groups_path,
            output_dir=matching_output_dir,
            # Keep default chunk/partition columns unless needed otherwise
            # chunk_column="t",
            # partition_column="Year",
        )
        logger.info(f"✅ Matching Pipeline completed successfully. Output: `{matching_output_dir}`")
        return True  # Indicate success

    except Exception as e:
        logger.error(f"❌ Matching Pipeline failed: {e}", exc_info=True)
        return False  # Indicate failure


# --- Main Execution ---
def main():
    """Orchestrates the data processing pipeline."""
    logger.info("--- Starting MPIL Tariff Trade Analysis Pipeline ---")

    config_params = get_config()

    # Step 1: Process BACI Data
    baci_output_path = process_baci_data(config_params)
    if not baci_output_path:
        logger.critical("BACI processing failed. Aborting pipeline.")
        sys.exit(1)  # Exit with error code

    # Step 2: Process WITS Data
    wits_success = process_wits_data(config_params)
    if not wits_success:
        logger.critical("WITS processing failed. Aborting pipeline.")
        sys.exit(1)  # Exit with error code

    # Step 3: Run Matching Pipeline
    matching_success = run_matching_pipeline(config_params, baci_output_path, wits_success)
    if not matching_success:
        logger.critical("Matching pipeline failed.")
        sys.exit(1)  # Exit with error code

    logger.info("--- Pipeline Execution Finished Successfully ---")
    sys.exit(0)  # Exit with success code


if __name__ == "__main__":
    main()
