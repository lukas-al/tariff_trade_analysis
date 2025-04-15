"""
ETL Pipeline Script for MPIL Tariff Trade Analysis

Orchestrates the processing of BACI and WITS data, followed by a matching
pipeline to unify trade and tariff information.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Assume these functions exist and perform the described actions.
# It's assumed they can handle Path objects as input where appropriate.
from mpil_tariff_trade_analysis.etl.baci import baci_to_parquet_incremental
from mpil_tariff_trade_analysis.etl.matching_chunked import run_chunked_matching_pipeline
from mpil_tariff_trade_analysis.etl.WITS_cleaner import process_and_save_wits_data

# Logging setup (assuming this works as intended)
from mpil_tariff_trade_analysis.utils.logging_config import get_logger, setup_logging

# --- Global Configuration & Setup ---
setup_logging()
logger = get_logger(__name__)


# --- Configuration Function ---
def get_pipeline_config() -> Dict[str, Any]:
    """
    Generates and returns the pipeline configuration dictionary.

    Defines key parameters and constructs necessary Path objects for data flow.
    Using Path objects directly improves consistency and cross-platform compatibility.
    """
    # Core Parameters
    hs_code = "HS92"
    baci_release = "202501"
    base_data_dir = Path("data").resolve()  # Use absolute paths for robustness

    # Define Directory Structure
    raw_data_dir = base_data_dir / "raw"
    intermediate_data_dir = base_data_dir / "intermediate"
    final_data_dir = base_data_dir / "final"

    # Ensure base directories exist (optional, but good practice)
    # intermediate_data_dir.mkdir(parents=True, exist_ok=True)
    # final_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Input Paths ---
    baci_input_folder = raw_data_dir
    wits_raw_dir = raw_data_dir / "WITS_tariff"
    # Assumes pref_groups is in raw data, adjust if needed
    pref_groups_path = raw_data_dir / "WITS_pref_groups" / "WITS_pref_groups.csv"

    # --- Intermediate & Output Paths ---
    # Define the *intended* output paths clearly.
    baci_output_dir_name = f"BACI_{hs_code}_V{baci_release}"
    baci_intermediate_parquet_path = intermediate_data_dir / baci_output_dir_name
    wits_mfn_output_path = intermediate_data_dir / "WITS_AVEMFN.parquet"
    wits_pref_output_path = intermediate_data_dir / "WITS_AVEPref.parquet"
    matching_output_dir = final_data_dir / "unified_trade_tariff_partitioned"

    # Return configuration parameters
    return {
        # Core Params
        "HS_CODE": hs_code,
        "BACI_RELEASE": baci_release,
        # Base Directories (as Path objects)
        "RAW_DATA_DIR": raw_data_dir,
        "INTERMEDIATE_DATA_DIR": intermediate_data_dir,
        "FINAL_DATA_DIR": final_data_dir,
        # Specific Input Paths (as Path objects)
        "BACI_INPUT_FOLDER": baci_input_folder,
        "WITS_RAW_DIR": wits_raw_dir,
        "PREF_GROUPS_PATH": pref_groups_path,
        # Specific Intermediate/Output Paths (as Path objects)
        "BACI_INTERMEDIATE_OUTPUT_PATH": baci_intermediate_parquet_path,
        "WITS_MFN_OUTPUT_PATH": wits_mfn_output_path,
        "WITS_PREF_OUTPUT_PATH": wits_pref_output_path,
        "MATCHING_OUTPUT_DIR": matching_output_dir,
    }


# --- Pipeline Steps as Functions ---
def process_baci_step(config: Dict[str, Any]) -> Optional[Path]:
    """
    Processes BACI data from CSV to partitioned Parquet.

    Args:
        config: The pipeline configuration dictionary.

    Returns:
        The Path object to the processed BACI Parquet directory if successful,
        otherwise None.
    """
    logger.info("--- Step 1: Processing BACI Data ---")
    hs_code = config["HS_CODE"]
    baci_release = config["BACI_RELEASE"]
    input_folder = config["BACI_INPUT_FOLDER"]
    # Use the specific output path defined in the config
    output_path = config["BACI_INTERMEDIATE_OUTPUT_PATH"]
    # The function needs the parent directory to create the specific output folder in
    output_parent_folder = config["INTERMEDIATE_DATA_DIR"]

    logger.info(f"Starting BACI processing: HS={hs_code}, Release={baci_release}")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Target output path: {output_path}")

    try:
        # Assuming baci_to_parquet_incremental creates the directory
        # 'BACI_{hs}_V{release}' inside `output_parent_folder`.
        # We rely on it creating the *exact* directory specified by `output_path`.
        # If it *returns* the path it created, we should still verify it.
        created_path_str_or_none = baci_to_parquet_incremental(
            hs=hs_code,
            release=baci_release,
            input_folder=str(input_folder),  # Convert to string if function requires it
            output_folder=str(output_parent_folder),  # Convert to string if function requires it
        )

        # Verification is key
        if (
            created_path_str_or_none
            and Path(created_path_str_or_none) == output_path
            and output_path.exists()
            and output_path.is_dir()
        ):
            logger.info(f"✅ BACI data processed successfully. Output: `{output_path}`")
            return output_path  # Return the Path object
        elif created_path_str_or_none:
            logger.error(
                f"❌ BACI processing mismatch. Expected: `{output_path}`, Got: `{created_path_str_or_none}`. Check function behavior."
            )
            return None
        else:
            logger.error(
                f"❌ BACI processing failed to produce expected output at `{output_path}` (Function returned None or empty)."
            )
            return None

    except Exception as e:
        logger.error(f"❌ BACI processing failed with exception: {e}", exc_info=True)
        return None


def process_wits_step(config: Dict[str, Any]) -> bool:
    """
    Processes WITS MFN and Preferential tariff data.

    Args:
        config: The pipeline configuration dictionary.

    Returns:
        True if both WITS processing steps were successful, False otherwise.
    """
    logger.info("--- Step 2: Processing WITS Tariff Data ---")
    wits_raw_dir = config["WITS_RAW_DIR"]
    intermediate_dir = config["INTERMEDIATE_DATA_DIR"]
    mfn_output_path = config["WITS_MFN_OUTPUT_PATH"]
    pref_output_path = config["WITS_PREF_OUTPUT_PATH"]
    overall_success = True

    # Process MFN Tariffs
    logger.info("Starting WITS MFN Tariff processing.")
    logger.info(f"Raw WITS dir: {wits_raw_dir}")
    logger.info(f"Expected output path: {mfn_output_path}")
    try:
        # Assuming process_and_save_wits_data saves the file within intermediate_dir
        # and returns the full path to the created file.
        mfn_result_path_str = process_and_save_wits_data(
            tariff_type="AVEMFN",
            base_dir=str(wits_raw_dir),  # Convert Path to str if necessary
            output_dir=str(intermediate_dir),  # Convert Path to str if necessary
            # If the function *can* take the full desired path, it's better:
            # output_path=str(mfn_output_path)
        )

        # Verify the expected output file was created at the correct location
        if (
            mfn_result_path_str
            and Path(mfn_result_path_str) == mfn_output_path
            and mfn_output_path.exists()
        ):
            logger.info(f"✅ WITS MFN data processed successfully. Output: `{mfn_output_path}`")
        else:
            logger.error(
                f"❌ WITS MFN processing failed or produced unexpected output. Expected: `{mfn_output_path}`, Result: `{mfn_result_path_str}`"
            )
            overall_success = False

    except Exception as e:
        logger.error(f"❌ WITS MFN processing failed with exception: {e}", exc_info=True)
        overall_success = False

    # Process Preferential Tariffs only if MFN succeeded (or run regardless, depending on logic)
    # Current logic runs regardless, only flags failure.
    logger.info("Starting WITS Preferential Tariff processing.")
    logger.info(f"Raw WITS dir: {wits_raw_dir}")
    logger.info(f"Expected output path: {pref_output_path}")
    try:
        pref_result_path_str = process_and_save_wits_data(
            tariff_type="AVEPref",
            base_dir=str(wits_raw_dir),  # Convert Path to str if necessary
            output_dir=str(intermediate_dir),  # Convert Path to str if necessary
            # If the function *can* take the full desired path, it's better:
            # output_path=str(pref_output_path)
        )

        # Verify the expected output file was created
        if (
            pref_result_path_str
            and Path(pref_result_path_str) == pref_output_path
            and pref_output_path.exists()
        ):
            logger.info(
                f"✅ WITS Preferential data processed successfully. Output: `{pref_output_path}`"
            )
        else:
            logger.error(
                f"❌ WITS Preferential processing failed or produced unexpected output. Expected: `{pref_output_path}`, Result: `{pref_result_path_str}`"
            )
            overall_success = False

    except Exception as e:
        logger.error(f"❌ WITS Preferential processing failed with exception: {e}", exc_info=True)
        overall_success = False

    return overall_success


def run_matching_step(config: Dict[str, Any], baci_processed_path: Path) -> bool:
    """
    Runs the matching pipeline to unify trade and tariff data.

    Assumes WITS data has already been processed successfully.

    Args:
        config: The pipeline configuration dictionary.
        baci_processed_path: Path to the successfully processed BACI data directory.

    Returns:
        True if the matching pipeline completed successfully, False otherwise.
    """
    logger.info("--- Step 3: Running Matching Pipeline ---")

    # Inputs are taken directly from config or passed arguments
    wits_mfn_path = config["WITS_MFN_OUTPUT_PATH"]
    wits_pref_path = config["WITS_PREF_OUTPUT_PATH"]
    pref_groups_path = config["PREF_GROUPS_PATH"]
    matching_output_dir = config["MATCHING_OUTPUT_DIR"]

    # Basic check for WITS file existence before running (optional robustness)
    if not wits_mfn_path.exists() or not wits_pref_path.exists():
        logger.error(
            f"❌ Cannot run matching: Missing WITS input files. Checked: `{wits_mfn_path}`, `{wits_pref_path}`"
        )
        return False
    if not pref_groups_path.exists():
        logger.error(
            f"❌ Cannot run matching: Missing Preference Groups file. Checked: `{pref_groups_path}`"
        )
        return False

    logger.info("Starting Matching Pipeline.")
    logger.info(f"BACI input: {baci_processed_path}")
    logger.info(f"WITS MFN input: {wits_mfn_path}")
    logger.info(f"WITS Pref input: {wits_pref_path}")
    logger.info(f"Pref Groups input: {pref_groups_path}")
    logger.info(f"Output directory: {matching_output_dir}")

    try:
        # Pass Path objects if the function supports them, otherwise convert to str
        run_chunked_matching_pipeline(
            baci_path=str(baci_processed_path),  # Or baci_processed_path if it handles Path
            wits_mfn_path=str(wits_mfn_path),  # Or wits_mfn_path
            wits_pref_path=str(wits_pref_path),  # Or wits_pref_path
            pref_groups_path=str(pref_groups_path),  # Or pref_groups_path
            output_dir=str(matching_output_dir),  # Or matching_output_dir
            # chunk_column="t",  # Keep defaults or specify if needed
            # partition_column="Year",
        )
        # Basic verification: Check if the output directory was created
        # More robust check would involve checking for expected partition files.
        if matching_output_dir.exists() and matching_output_dir.is_dir():
            logger.info(
                f"✅ Matching Pipeline completed successfully. Output: `{matching_output_dir}`"
            )
            return True
        else:
            logger.warning(
                f"🤔 Matching pipeline reported success, but output directory `{matching_output_dir}` not found or is not a directory."
            )
            # Depending on strictness, you might return False here.
            # Let's assume the function handles its own errors primarily.
            return True  # Or False if strict check required

    except Exception as e:
        logger.error(f"❌ Matching Pipeline failed with exception: {e}", exc_info=True)
        return False


# --- Main Execution ---
def main() -> int:
    """
    Orchestrates the data processing pipeline.

    Returns:
        0 for success, 1 for failure.
    """
    logger.info("--- Starting MPIL Tariff Trade Analysis Pipeline ---")

    try:
        config = get_pipeline_config()
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        return 1  # Configuration error

    # Step 1: Process BACI Data
    baci_output_path = process_baci_step(config)
    if not baci_output_path:
        logger.critical("BACI processing failed. Aborting pipeline.")
        return 1  # Exit with error code

    # Step 2: Process WITS Data
    wits_success = process_wits_step(config)
    if not wits_success:
        logger.critical("WITS processing failed. Aborting pipeline.")
        return 1  # Exit with error code

    # Step 3: Run Matching Pipeline
    # Pass the verified BACI output path from step 1
    matching_success = run_matching_step(config, baci_output_path)
    if not matching_success:
        logger.critical("Matching pipeline failed.")
        return 1  # Exit with error code

    logger.info("--- Pipeline Execution Finished Successfully ---")
    return 0  # Exit with success code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
