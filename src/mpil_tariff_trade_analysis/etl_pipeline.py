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
from pathlib import Path
from typing import Any, Dict, Optional

# Import the new pipeline runner functions
from mpil_tariff_trade_analysis.etl.baci_pipeline import run_baci_cleaning_pipeline
from mpil_tariff_trade_analysis.etl.merge_pipeline import run_merge_pipeline
from mpil_tariff_trade_analysis.etl.wits_mfn_pipeline import run_wits_mfn_cleaning_pipeline
from mpil_tariff_trade_analysis.etl.wits_pref_pipeline import run_wits_pref_cleaning_pipeline

# Logging setup
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

    # Ensure base directories exist (optional, can be handled by pipelines)
    # raw_data_dir.mkdir(parents=True, exist_ok=True)
    # intermediate_data_dir.mkdir(parents=True, exist_ok=True)
    # final_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Input Paths ---
    baci_input_folder = raw_data_dir # Parent dir of BACI_HSXX_VYYYYYY CSVs
    wits_raw_dir = raw_data_dir / "WITS_tariff" # Parent dir of AVEMFN/AVEPref folders
    pref_groups_path = raw_data_dir / "WITS_pref_groups" / "WITS_pref_groups.csv"
    # Optional reference file paths (can be omitted if defaults in iso_remapping are used)
    # baci_ref_codes_path = raw_data_dir / f"BACI_{hs_code}_V{baci_release}" / f"country_codes_V{baci_release}.csv"
    # wits_ref_codes_path = raw_data_dir / "WITS_country_codes.csv"
    hs_mapping_dir = raw_data_dir / "hs_reference"

    # --- Intermediate & Output Paths ---
    # BACI Paths
    baci_intermediate_raw_parquet_name = f"BACI_{hs_code}_V{baci_release}"
    baci_intermediate_raw_parquet_path = intermediate_data_dir / baci_intermediate_raw_parquet_name
    baci_cleaned_filename = f"BACI_{hs_code}_V{baci_release}_cleaned_remapped.parquet"
    baci_cleaned_output_path = intermediate_data_dir / baci_cleaned_filename

    # WITS Paths
    wits_mfn_cleaned_dir = intermediate_data_dir / "cleaned_wits_mfn"
    wits_mfn_cleaned_output_path = wits_mfn_cleaned_dir / "WITS_AVEMFN_cleaned.parquet"
    wits_pref_cleaned_dir = intermediate_data_dir / "cleaned_wits_pref"
    wits_pref_cleaned_output_path = wits_pref_cleaned_dir / "WITS_AVEPref_cleaned_expanded.parquet"

    # Final Merged Output Path
    merged_output_dir = final_data_dir / "unified_trade_tariff_partitioned" # Keep consistent name

    # Return configuration parameters
    config = {
        # Core Params
        "HS_CODE": hs_code,
        "BACI_RELEASE": baci_release,
        # Base Directories
        "RAW_DATA_DIR": raw_data_dir,
        "INTERMEDIATE_DATA_DIR": intermediate_data_dir,
        "FINAL_DATA_DIR": final_data_dir,
        # Specific Input Paths
        "BACI_INPUT_FOLDER": baci_input_folder,
        "WITS_RAW_DIR": wits_raw_dir,
        "PREF_GROUPS_PATH": pref_groups_path,
        "HS_MAPPING_DIR": hs_mapping_dir,
        # Optional Ref Paths
        # "BACI_REF_CODES_PATH": baci_ref_codes_path,
        # "WITS_REF_CODES_PATH": wits_ref_codes_path,
        # Specific Intermediate/Output Paths
        "BACI_INTERMEDIATE_RAW_PARQUET_PATH": baci_intermediate_raw_parquet_path, # Dir for raw parquet
        "BACI_CLEANED_OUTPUT_PATH": baci_cleaned_output_path, # File for cleaned BACI
        "WITS_MFN_CLEANED_OUTPUT_PATH": wits_mfn_cleaned_output_path, # File for cleaned MFN
        "WITS_PREF_CLEANED_OUTPUT_PATH": wits_pref_cleaned_output_path, # File for cleaned/expanded Pref
        "MERGED_OUTPUT_DIR": merged_output_dir, # Dir for final partitioned output
        # Merge Pipeline Params (can be overridden)
        "CHUNK_COLUMN_NAME": "t",
        "PARTITION_COLUMN": "Year",
    }
    # Log the configuration being used
    logger.info("Pipeline Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    return config


# --- Pipeline Steps as Functions ---

def run_baci_cleaning_step(config: Dict[str, Any]) -> Optional[Path]:
    """Runs the BACI cleaning pipeline."""
    logger.info("--- Step 1: Running BACI Cleaning Pipeline ---")
    try:
        cleaned_baci_path = run_baci_cleaning_pipeline(config)
        if cleaned_baci_path:
            logger.info(f"✅ BACI cleaning finished. Output: {cleaned_baci_path}")
            return cleaned_baci_path
        else:
            logger.error("❌ BACI cleaning pipeline failed.")
            return None
    except Exception as e:
        logger.error(f"❌ BACI cleaning pipeline failed with exception: {e}", exc_info=True)
        return None


def run_wits_mfn_cleaning_step(config: Dict[str, Any]) -> Optional[Path]:
    """Runs the WITS MFN cleaning pipeline."""
    logger.info("--- Step 2: Running WITS MFN Cleaning Pipeline ---")
    try:
        cleaned_mfn_path = run_wits_mfn_cleaning_pipeline(config)
        if cleaned_mfn_path:
            logger.info(f"✅ WITS MFN cleaning finished. Output: {cleaned_mfn_path}")
            return cleaned_mfn_path
        else:
            logger.error("❌ WITS MFN cleaning pipeline failed.")
            return None
    except Exception as e:
        logger.error(f"❌ WITS MFN cleaning pipeline failed with exception: {e}", exc_info=True)
        return None


def run_wits_pref_cleaning_step(config: Dict[str, Any]) -> Optional[Path]:
    """Runs the WITS Preferential cleaning and expansion pipeline."""
    logger.info("--- Step 3: Running WITS Preferential Cleaning & Expansion Pipeline ---")
    try:
        # This pipeline now needs PREF_GROUPS_PATH from config
        if "PREF_GROUPS_PATH" not in config or not config["PREF_GROUPS_PATH"].exists():
             logger.error(f"❌ Cannot run WITS Pref cleaning: Missing PREF_GROUPS_PATH in config or file not found at {config.get('PREF_GROUPS_PATH')}")
             return None

        cleaned_pref_path = run_wits_pref_cleaning_pipeline(config)
        if cleaned_pref_path:
            logger.info(f"✅ WITS Pref cleaning & expansion finished. Output: {cleaned_pref_path}")
            return cleaned_pref_path
        else:
            logger.error("❌ WITS Pref cleaning & expansion pipeline failed.")
            return None
    except Exception as e:
        logger.error(f"❌ WITS Pref cleaning & expansion pipeline failed with exception: {e}", exc_info=True)
        return None


def run_merge_step(
    config: Dict[str, Any],
    cleaned_baci_path: Path,
    cleaned_mfn_path: Path,
    cleaned_pref_path: Path,
) -> bool:
    """Runs the merge pipeline using the cleaned data."""
    logger.info("--- Step 4: Running Data Merging Pipeline ---")

    # Basic check for cleaned input file existence
    if not cleaned_baci_path or not cleaned_baci_path.exists():
        logger.error(f"❌ Cannot run merge: Missing cleaned BACI input file. Checked: `{cleaned_baci_path}`")
        return False
    if not cleaned_mfn_path or not cleaned_mfn_path.exists():
        logger.error(f"❌ Cannot run merge: Missing cleaned MFN input file. Checked: `{cleaned_mfn_path}`")
        return False
    if not cleaned_pref_path or not cleaned_pref_path.exists():
        logger.error(f"❌ Cannot run merge: Missing cleaned Pref input file. Checked: `{cleaned_pref_path}`")
        return False

    try:
        success = run_merge_pipeline(
            config=config,
            cleaned_baci_path=cleaned_baci_path,
            cleaned_mfn_path=cleaned_mfn_path,
            cleaned_pref_path=cleaned_pref_path,
        )
        if success:
            logger.info(f"✅ Merging pipeline completed successfully. Output: `{config.get('MERGED_OUTPUT_DIR')}`")
            return True
        else:
            logger.error("❌ Merging pipeline failed.")
            return False
    except Exception as e:
        logger.error(f"❌ Merging pipeline failed with exception: {e}", exc_info=True)
        return False


# --- Main Execution ---
def main() -> int:
    """
    Orchestrates the data processing pipeline: Clean BACI, Clean MFN, Clean Pref, Merge.

    Returns:
        0 for success, 1 for failure.
    """
    logger.info("--- Starting MPIL Tariff Trade Analysis ETL Pipeline ---")

    try:
        config = get_pipeline_config()
    except Exception as e:
        logger.critical(f"❌ Failed to load configuration: {e}", exc_info=True)
        return 1  # Configuration error

    # Step 1: Clean BACI Data
    cleaned_baci_path = run_baci_cleaning_step(config)
    if not cleaned_baci_path:
        logger.critical("Pipeline aborted due to BACI cleaning failure.")
        return 1

    # Step 2: Clean WITS MFN Data
    cleaned_mfn_path = run_wits_mfn_cleaning_step(config)
    if not cleaned_mfn_path:
        logger.critical("Pipeline aborted due to WITS MFN cleaning failure.")
        return 1

    # Step 3: Clean WITS Pref Data (includes expansion)
    cleaned_pref_path = run_wits_pref_cleaning_step(config)
    if not cleaned_pref_path:
        logger.critical("Pipeline aborted due to WITS Pref cleaning failure.")
        return 1

    # Step 4: Run Merging Pipeline
    merge_success = run_merge_step(
        config, cleaned_baci_path, cleaned_mfn_path, cleaned_pref_path
    )
    if not merge_success:
        logger.critical("Pipeline aborted due to merging failure.")
        return 1

    logger.info("--- ETL Pipeline Execution Finished Successfully ---")
    return 0  # Exit with success code


if __name__ == "__main__":
    import sys
    exit_code = main()
    logger.info(f"Pipeline finished with exit code {exit_code}.")
    sys.exit(exit_code)
