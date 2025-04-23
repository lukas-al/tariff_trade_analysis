"""
Pipeline for cleaning and preparing WITS MFN tariff data.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

# Local imports
from mpil_tariff_trade_analysis.etl.WITS_cleaner import (
    DEFAULT_WITS_BASE_DIR,
    consolidate_wits_tariff_data,
    vectorized_hs_translation,
)
from mpil_tariff_trade_analysis.utils.iso_remapping import remap_codes_and_explode
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)


def rename_wits_mfn_cleaned(filepath: Path) -> pl.LazyFrame:
    """
    Renames columns in the cleaned WITS MFN tariff dataframe.
    Assumes country codes are already remapped to *_iso_numeric.
    """
    logger.debug("Renaming cleaned WITS MFN columns.")

    rename_map = {
        "year": "t",
        "reporter_country_iso_numeric": "i",
        "product_code": "k",
        "tariff_rate": "mfn_tariff_rate",
        "min_rate": "mfn_min_tariff_rate",
        "max_rate": "mfn_max_tariff_rate",
        "tariff_type": "tariff_type",
    }

    # Select only the columns we need after renaming
    select_cols = [
        "t",
        "i",  # Final 'i' column after handling list
        "k",
        "mfn_tariff_rate",
        "mfn_min_tariff_rate",
        "mfn_max_tariff_rate",
        "tariff_type",
    ]

    # Load the df
    df = pl.scan_parquet(filepath)

    # Check if all keys in rename_map exist in df.columns
    missing_cols = [col for col in rename_map.keys() if col not in df.columns]
    if missing_cols:
        logger.error(
            f"Cannot rename WITS MFN: Missing expected input columns: {missing_cols}. Available: {df.columns}"
        )
        raise ValueError(f"Missing columns required for renaming: {missing_cols}")

    # Rename and handle the list column 'i_list'
    renamed_df = df.rename(rename_map).select(select_cols)

    logger.debug(f"Renamed WITS MFN Schema: {renamed_df.collect_schema()}")

    return renamed_df


def validate_cleaned_wits_mfn(lf: pl.LazyFrame) -> bool:
    """
    Performs basic validation on the cleaned and renamed WITS MFN LazyFrame.
    """
    expected_cols = {
        "t",
        "i",  # Should be single string after list handling
        "k",
        "mfn_tariff_rate",
        "mfn_min_tariff_rate",
        "mfn_max_tariff_rate",
        "tariff_type",
    }
    actual_cols = set(lf.columns)

    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        logger.error(
            f"Validation Error: Missing expected columns in cleaned WITS MFN data. "
            f"Missing: {missing}, Found: {actual_cols}"
        )
        return False

    # Add more checks:
    # - Data types (t: Utf8 from load_wits, i: Utf8, k: Utf8, rates: Utf8 from load_wits)
    schema = lf.collect_schema()
    if schema.get("t") != pl.Utf8:
        logger.error(f"Validation Error: MFN 't' column type is {schema.get('t')}, expected Utf8.")
        return False
    if schema.get("i") != pl.Utf8:
        logger.error(f"Validation Error: MFN 'i' column type is {schema.get('i')}, expected Utf8.")
        return False
    if schema.get("k") != pl.Utf8:
        logger.error(f"Validation Error: MFN 'k' column type is {schema.get('k')}, expected Utf8.")
        return False
    # Rates are loaded as Utf8 to handle non-numeric, check later during merge if needed
    if schema.get("mfn_tariff_rate") != pl.Utf8:
        logger.warning(
            f"Validation Warning: MFN 'mfn_tariff_rate' column type is {schema.get('mfn_tariff_rate')}, expected Utf8."
        )

    # - Check tariff rates are within a plausible range (e.g., 0-1000, allowing for specific duties)
    #   Requires casting to numeric, handle errors. Example (consider sampling):
    #   numeric_rates = lf.select(pl.col('mfn_tariff_rate').cast(pl.Float64, strict=False)).collect()
    #   if numeric_rates['mfn_tariff_rate'].max() > 10000: # Arbitrary high threshold
    #       logger.warning("Validation Warning: Found potentially very high MFN tariff rates.")

    # - Check for non-numeric values in rate columns if they should be numeric (they are Utf8 now)

    logger.info("Cleaned WITS MFN data basic validation passed.")
    return True


def run_wits_mfn_cleaning_pipeline(config: Dict[str, Any]) -> Optional[Path]:
    """
    Orchestrates the cleaning of WITS MFN data.

    Args:
        config: The pipeline configuration dictionary. Expected keys:
            - WITS_RAW_DIR (Path to base dir containing AVEMFN/AVEPref folders)
            - WITS_MFN_CLEANED_OUTPUT_PATH (Path for the final cleaned MFN parquet file)
            - Optional: BACI_REF_CODES_PATH, WITS_REF_CODES_PATH (for country mapping)
            - Optional: HS_MAPPING_DIR (for HS translation)

    Returns:
        The Path object to the cleaned WITS MFN Parquet file if successful,
        otherwise None.
    """
    logger.info("--- Starting WITS MFN Cleaning Pipeline ---")
    base_dir = config.get("WITS_RAW_DIR", Path(DEFAULT_WITS_BASE_DIR))
    output_path = config["WITS_MFN_CLEANED_OUTPUT_PATH"]
    # hs_mapping_dir = config.get(
    #     "HS_MAPPING_DIR", "data/raw/hs_reference"
    # )  # Default from WITS_cleaner
    # baci_codes_path = config.get("BACI_REF_CODES_PATH", None)
    # wits_codes_path = config.get("WITS_REF_CODES_PATH", None)

    # Ensure output directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_path.parent}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_path.parent}: {e}")
        return None

    # --- Step 1: Load, Translate HS, Remap Countries ---
    logger.info(f"Loading raw WITS MFN data from base directory: {base_dir}")
    try:
        # Loads the data into a single dataframe.
        consolidated_mfn_path = consolidate_wits_tariff_data(
            tariff_type="AVEMFN",
            base_dir=str(base_dir),
        )

        if not (consolidated_mfn_path and consolidated_mfn_path.exists()):
            logger.error("❌ Consolidating WITS MFN data returned None.")
            raise

        logger.info("✅ WITS MFN data consolidated")

    except Exception as e:
        logger.error(f"❌ Failed during WITS MFN data loading/preprocessing: {e}", exc_info=True)
        return None

    # --- Step 2: TRANSLATE HS CODES TO H0 ---
    logger.info("Translating HS codes...")

    try:
        h0_translated_path = vectorized_hs_translation(consolidated_mfn_path)

        if not (h0_translated_path and h0_translated_path.exists()):
            logger.error("❌ H0 remapping wits MFN failed - path doesn't exist")

    except Exception as e:
        logger.error(f"❌ Failed during WITS MFN data H0 transformation: {e}", exc_info=True)
        return None

    # --- Step 3: Reconcile country names ---
    logger.info("Reconciling country names for WITS MFN")

    reconciled_cc_path = h0_translated_path.stem + "reconciled.parquet"
    try:
        returned_rcc_path = remap_codes_and_explode(
            input_path=h0_translated_path,
            output_path=reconciled_cc_path,
            code_columns_to_remap=["reporter_country"],
            output_column_names=["reporter_country_iso_numeric"],
            use_hive_partitioning=False,
            filter_failed_mappings=True,
        )

        if not (returned_rcc_path or returned_rcc_path.exists()):  # type: ignore
            logger.error("❌ Reconciling country codes failed - path doesn't exist ")

    except Exception as e:
        logger.error(
            f"❌ Failed during WITS MFN data country code reconciliation: {e}", exc_info=True
        )
        return None

    logger.info("✅ Cleaned WITS MFN data processed successful.")

    # --- Step 4: Rename the file and write it to the final save point
    logger.info("Renaming columns for consistency and handling list types.")

    try:
        renamed_df = rename_wits_mfn_cleaned(returned_rcc_path)  # type: ignore
        logger.info("✅ Columns renamed.")
    except Exception as e:
        logger.error(f"❌ Failed during column renaming: {e}", exc_info=True)
        return None

    # --- Step 4: Save Cleaned Data ---
    logger.info(f"Saving cleaned WITS MFN data to: {output_path}")
    try:
        # Collect and save the final dataframe
        renamed_df.collect().write_parquet(output_path)
        logger.info("✅ Cleaned WITS MFN data saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save cleaned WITS MFN data: {e}", exc_info=True)
        return None

    logger.info("--- WITS MFN Cleaning Pipeline Finished Successfully ---")
    return output_path

    # return reconciled_codes_path

    # # --- Step 3: Rename Columns & Handle List Columns ---
    # logger.info("Renaming columns for consistency and handling list types.")
    # try:
    #     renamed_df = rename_wits_mfn_cleaned(loaded_df)
    #     logger.info("✅ Columns renamed.")
    # except Exception as e:
    #     logger.error(f"❌ Failed during column renaming: {e}", exc_info=True)
    #     return None

    # # --- Step 3: Validation ---
    # logger.info("Validating cleaned WITS MFN data.")
    # if not validate_cleaned_wits_mfn(renamed_df):
    #     logger.error("❌ Cleaned WITS MFN data validation failed.")
    #     # Consider deleting output file
    #     # output_path.unlink(missing_ok=True)
    #     return None
    # logger.info("✅ Cleaned WITS MFN data validation successful.")

    # # --- Step 4: Save Cleaned Data ---
    # logger.info(f"Saving cleaned WITS MFN data to: {output_path}")
    # try:
    #     # Collect and save the final dataframe
    #     renamed_df.collect().write_parquet(output_path)
    #     logger.info("✅ Cleaned WITS MFN data saved successfully.")
    # except Exception as e:
    #     logger.error(f"❌ Failed to save cleaned WITS MFN data: {e}", exc_info=True)
    #     return None

    # logger.info("--- WITS MFN Cleaning Pipeline Finished Successfully ---")
    # return output_path


# Example of how to run (optional, for direct execution)
if __name__ == "__main__":
    from mpil_tariff_trade_analysis.utils.logging_config import setup_logging

    setup_logging(log_level="DEBUG")

    # --- Dummy Configuration for Testing ---
    TEST_BASE_DATA_DIR = Path("data").resolve()
    TEST_RAW_DATA_DIR = TEST_BASE_DATA_DIR / "raw"
    TEST_INTERMEDIATE_DATA_DIR = TEST_BASE_DATA_DIR / "intermediate"

    test_config = {
        "WITS_RAW_DIR": TEST_RAW_DATA_DIR / "WITS_tariff",
        "WITS_MFN_CLEANED_OUTPUT_PATH": TEST_INTERMEDIATE_DATA_DIR
        / "cleaned_wits_mfn"
        / "WITS_AVEMFN_cleaned.parquet",
        # Optional paths if not using defaults:
        # "HS_MAPPING_DIR": TEST_RAW_DATA_DIR / "hs_reference",
        # "BACI_REF_CODES_PATH": TEST_RAW_DATA_DIR / "BACI_HS92_V202501" / "country_codes_V202501.csv",
        # "WITS_REF_CODES_PATH": TEST_RAW_DATA_DIR / "WITS_country_codes.csv",
    }
    logger.info(f"Running WITS MFN cleaning pipeline with test config: {test_config}")

    result_path = run_wits_mfn_cleaning_pipeline(test_config)

    if result_path:
        logger.info(
            f"WITS MFN cleaning pipeline test completed successfully. Output: {result_path}"
        )
        sys.exit(0)
    else:
        logger.error("WITS MFN cleaning pipeline test failed.")
        sys.exit(1)
