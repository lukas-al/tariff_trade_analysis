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
# Import the refactored function from the correct module (assuming it's in etl.baci)
from mpil_tariff_trade_analysis.etl.baci import remap_codes_and_explode
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)


def rename_wits_mfn_cleaned(input_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Renames columns in the cleaned WITS MFN tariff LazyFrame.
    Assumes country codes are already remapped to ISO numeric (single value per row).

    Args:
        input_lf: The input LazyFrame after country code remapping.

    Returns:
        A LazyFrame with renamed columns.
    """
    logger.debug("Renaming cleaned WITS MFN columns.")

    # The input column from remap_codes_and_explode will be named based on
    # the 'output_column_names' parameter used in its call.
    # Let's assume it was called with output_column_names=["reporter_country_iso_numeric"]
    # If remap_codes_and_explode replaced the original column, the name might still be 'reporter_country'
    # Check the schema of the input_lf if unsure. Let's assume the output name is used.
    input_reporter_col = "reporter_country_iso_numeric" # Adjust if needed based on remap_codes_and_explode call

    # Check if the expected input column exists
    if input_reporter_col not in input_lf.columns:
         logger.error(
             f"Cannot rename WITS MFN: Missing expected input reporter column: '{input_reporter_col}'. "
             f"Available columns: {input_lf.columns}"
         )
         # Raise an error or handle appropriately
         raise ValueError(f"Missing expected column '{input_reporter_col}' for renaming.")


    rename_map = {
        "year": "t",
        input_reporter_col: "i", # Map the remapped column to 'i'
        "product_code": "k",
        "tariff_rate": "mfn_tariff_rate",
        "min_rate": "mfn_min_tariff_rate",
        "max_rate": "mfn_max_tariff_rate",
        # "tariff_type": "tariff_type", # Keep tariff_type if needed
    }

    # Select only the columns we need after renaming
    select_cols = [
        "t",
        "i",
        "k",
        "mfn_tariff_rate",
        "mfn_min_tariff_rate",
        "mfn_max_tariff_rate",
        "tariff_type", # Include tariff_type if it exists and is needed
    ]

    # Ensure all columns to be selected exist after potential renaming
    available_cols_after_rename = list(input_lf.columns)
    for old, new in rename_map.items():
        if old in available_cols_after_rename:
            available_cols_after_rename.remove(old)
        if new not in available_cols_after_rename:
             available_cols_after_rename.append(new)
        # Special case: if input_reporter_col is same as 'i'
        if old == new and old == input_reporter_col:
             if input_reporter_col not in available_cols_after_rename:
                  available_cols_after_rename.append(input_reporter_col)


    final_select_cols = [col for col in select_cols if col in available_cols_after_rename or col in rename_map.values()]
    # Ensure tariff_type is included if present
    if "tariff_type" in input_lf.columns and "tariff_type" not in final_select_cols:
         final_select_cols.append("tariff_type")


    # Rename and select
    renamed_df = input_lf.rename(rename_map).select(final_select_cols)

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
        extra = actual_cols - expected_cols
        logger.error(
            f"Validation Error: Column mismatch in cleaned WITS MFN data. "
            f"Missing: {missing or 'None'}, Extra: {extra or 'None'}, Found: {actual_cols}"
        )
        return False

    # Add more checks:
    # - Data types (t: Utf8, i: Utf8, k: Utf8, rates: Utf8)
    try:
        schema = lf.limit(0).collect_schema() # More efficient way to get schema
        if schema.get("t") != pl.Utf8:
            logger.error(f"Validation Error: MFN 't' column type is {schema.get('t')}, expected Utf8.")
            # return False # Allow flexibility if year becomes numeric later
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
        if schema.get("tariff_type") != pl.Utf8:
             logger.error(f"Validation Error: MFN 'tariff_type' column type is {schema.get('tariff_type')}, expected Utf8.")
             return False

    except Exception as e:
        logger.error(f"Validation Error: Could not verify schema. Error: {e}")
        return False


    # - Check for nulls in key columns (t, i, k)
    null_check = lf.select(
        pl.col("t").is_null().sum().alias("t_nulls"),
        pl.col("i").is_null().sum().alias("i_nulls"),
        pl.col("k").is_null().sum().alias("k_nulls"),
    ).collect()

    if null_check["t_nulls"][0] > 0 or null_check["i_nulls"][0] > 0 or null_check["k_nulls"][0] > 0:
        logger.error(f"Validation Error: Null values found in key columns: {null_check.to_dicts()}")
        return False

    logger.info("Cleaned WITS MFN data basic validation passed.")
    return True


def run_wits_mfn_cleaning_pipeline(config: Dict[str, Any]) -> Optional[Path]:
    """
    Orchestrates the cleaning of WITS MFN data using in-memory LazyFrames.

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
    logger.info("--- Starting WITS MFN Cleaning Pipeline (In-Memory) ---")
    base_dir = config.get("WITS_RAW_DIR", Path(DEFAULT_WITS_BASE_DIR))
    output_path = config["WITS_MFN_CLEANED_OUTPUT_PATH"]
    hs_mapping_dir = config.get("HS_MAPPING_DIR", "data/raw/hs_reference") # Default from WITS_cleaner
    baci_codes_path = config.get("BACI_REF_CODES_PATH", None) # Let remap func use default if None
    wits_codes_path = config.get("WITS_REF_CODES_PATH", None) # Let remap func use default if None

    # Ensure output directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_path.parent}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_path.parent}: {e}")
        return None

    current_lf: Optional[pl.LazyFrame] = None # Initialize LazyFrame variable

    # --- Step 1: Consolidate WITS data ---
    logger.info(f"Consolidating raw WITS MFN data from base directory: {base_dir}")
    try:
        current_lf = consolidate_wits_tariff_data(
            tariff_type="AVEMFN",
            base_dir=str(base_dir),
        )
        if current_lf is None:
             raise ValueError("Consolidation returned None.")
        logger.info("✅ WITS MFN data consolidated into LazyFrame.")
        logger.debug(f"Schema after consolidation: {current_lf.collect_schema()}")

    except Exception as e:
        logger.error(f"❌ Failed during WITS MFN data consolidation: {e}", exc_info=True)
        return None

    # --- Step 2: Translate HS codes to H0 ---
    logger.info("Translating HS codes...")
    try:
        current_lf = vectorized_hs_translation(current_lf, mapping_dir=hs_mapping_dir)
        if current_lf is None:
             raise ValueError("HS translation returned None.")
        logger.info("✅ HS codes translated.")
        logger.debug(f"Schema after HS translation: {current_lf.collect_schema()}")

    except Exception as e:
        logger.error(f"❌ Failed during WITS MFN data HS translation: {e}", exc_info=True)
        return None

    # --- Step 3: Reconcile country names ---
    logger.info("Reconciling country names for WITS MFN")
    try:
        # Define the output column name for the remapped reporter country
        remapped_reporter_col_name = "reporter_country_iso_numeric"

        current_lf = remap_codes_and_explode(
            input_lf=current_lf,
            # output_path=reconciled_cc_path, # No output path needed here
            code_columns_to_remap=["reporter_country"],
            output_column_names=[remapped_reporter_col_name], # Specify the output name
            # use_hive_partitioning=False, # Not relevant for LazyFrame input
            filter_failed_mappings=True, # Filter rows that couldn't be mapped
            drop_original_code_columns=True, # Drop the original 'reporter_country'
            baci_codes_path=baci_codes_path, # Pass paths or let defaults be used
            wits_codes_path=wits_codes_path,
        )
        if current_lf is None:
            raise ValueError("Country code remapping returned None.")

        logger.info("✅ Country codes reconciled.")
        logger.debug(f"Schema after country reconciliation: {current_lf.collect_schema()}")

    except Exception as e:
        logger.error(
            f"❌ Failed during WITS MFN data country code reconciliation: {e}", exc_info=True
        )
        return None

    # --- Step 4: Rename columns for consistency ---
    logger.info("Renaming columns for consistency.")
    try:
        current_lf = rename_wits_mfn_cleaned(current_lf)
        if current_lf is None:
             raise ValueError("Renaming function returned None.")
        logger.info("✅ Columns renamed.")
        logger.debug(f"Schema after renaming: {current_lf.collect_schema()}")
    except Exception as e:
        logger.error(f"❌ Failed during column renaming: {e}", exc_info=True)
        return None

    # --- Step 5: Validation ---
    logger.info("Validating final cleaned WITS MFN LazyFrame.")
    if not validate_cleaned_wits_mfn(current_lf):
        logger.error("❌ Cleaned WITS MFN data validation failed.")
        return None
    logger.info("✅ Cleaned WITS MFN data validation successful.")

    # --- Step 6: Save Cleaned Data ---
    logger.info(f"Saving cleaned WITS MFN data to: {output_path}")
    try:
        # Collect and save the final dataframe
        current_lf.collect().write_parquet(output_path)
        logger.info("✅ Cleaned WITS MFN data saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save cleaned WITS MFN data: {e}", exc_info=True)
        # Optionally clean up the potentially partially written file
        # output_path.unlink(missing_ok=True)
        return None

    logger.info("--- WITS MFN Cleaning Pipeline (In-Memory) Finished Successfully ---")
    return output_path


# Example of how to run (optional, for direct execution)
if __name__ == "__main__":
    from mpil_tariff_trade_analysis.utils.logging_config import setup_logging
    import pandas as pd # Import pandas for dummy data creation

    setup_logging(log_level="DEBUG")

    # --- Dummy Configuration for Testing ---
    TEST_BASE_DATA_DIR = Path("data").resolve()
    TEST_RAW_DATA_DIR = TEST_BASE_DATA_DIR / "raw"
    TEST_INTERMEDIATE_DATA_DIR = TEST_BASE_DATA_DIR / "intermediate"

    # Ensure test directories exist or create dummy ones if needed for the test run
    (TEST_RAW_DATA_DIR / "WITS_tariff" / "AVEMFN").mkdir(parents=True, exist_ok=True)
    # Add dummy CSV files if needed for consolidate_wits_tariff_data to run without error
    # Example: Create a dummy CSV file
    dummy_csv_dir = TEST_RAW_DATA_DIR / "WITS_tariff" / "AVEMFN" / "AVEMFN_H1_840_2000_U2"
    dummy_csv_dir.mkdir(parents=True, exist_ok=True)
    dummy_csv_path = dummy_csv_dir / "JobID_123.CSV"
    if not dummy_csv_path.exists():
         pd.DataFrame({
             "NomenCode": ["H1"], "Reporter_ISO_N": ["840"], "Year": [2000], "ProductCode": ["010101"],
             "Sum_Of_Rates": ["10"], "Min_Rate": ["5"], "Max_Rate": ["15"], "SimpleAverage": ["10"],
             "Nbr_NA_Lines": [0], "Nbr_Free_Lines": [0], "Nbr_AVE_Lines": [1], "Nbr_Dutiable_Lines": [1],
             "TotalNoOfValidLines": [1], "TotalNoOfLines": [1], "EstCode": ["E"]
         }).to_csv(dummy_csv_path, index=False)


    test_config = {
        "WITS_RAW_DIR": TEST_RAW_DATA_DIR / "WITS_tariff",
        "WITS_MFN_CLEANED_OUTPUT_PATH": TEST_INTERMEDIATE_DATA_DIR
        / "cleaned_wits_mfn"
        / "WITS_AVEMFN_cleaned_inmemory.parquet", # Changed output name slightly for testing
        # Optional paths if not using defaults:
        "HS_MAPPING_DIR": TEST_RAW_DATA_DIR / "hs_reference",
        "BACI_REF_CODES_PATH": TEST_RAW_DATA_DIR / "BACI_HS92_V202501" / "country_codes_V202501.csv",
        "WITS_REF_CODES_PATH": TEST_RAW_DATA_DIR / "WITS_country_codes.csv",
    }
    logger.info(f"Running WITS MFN cleaning pipeline with test config: {test_config}")

    # Make sure reference files exist for the test
    Path(test_config["HS_MAPPING_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(test_config["BACI_REF_CODES_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(test_config["WITS_REF_CODES_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    # Create dummy reference files if they don't exist
    if not Path(test_config["BACI_REF_CODES_PATH"]).exists():
        pd.DataFrame({'country_code':['840'],'country_name':['United States']}).to_csv(test_config["BACI_REF_CODES_PATH"], index=False)
    if not Path(test_config["WITS_REF_CODES_PATH"]).exists():
         pd.DataFrame({'ISO3':['840'],'Country Name':['United States']}).to_csv(test_config["WITS_REF_CODES_PATH"], index=False)
    # Create dummy HS mapping if needed
    hs_map_path = Path(test_config["HS_MAPPING_DIR"]) / "H1_to_H0.CSV"
    if not hs_map_path.exists():
         pd.DataFrame({'H1_Code':['010101'],'Desc':['Dummy'],'H0_Code':['010110']}).to_csv(hs_map_path, index=False, encoding='iso-8859-1')


    result_path = run_wits_mfn_cleaning_pipeline(test_config)

    if result_path:
        logger.info(
            f"WITS MFN cleaning pipeline test completed successfully. Output: {result_path}"
        )
        # Optionally load and check the output file
        try:
            final_df = pl.read_parquet(result_path)
            logger.info(f"Final output schema: {final_df.schema}")
            logger.info(f"Final output shape: {final_df.shape}")
            logger.info(f"Final output head:\n{final_df.head()}")
        except Exception as e:
            logger.error(f"Failed to read or inspect final output file: {e}")

        sys.exit(0)
    else:
        logger.error("WITS MFN cleaning pipeline test failed.")
        sys.exit(1)
