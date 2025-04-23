"""
Pipeline for cleaning and preparing BACI trade data.
"""

from pathlib import Path
from typing import Any, Dict, Optional

# Local imports
from mpil_tariff_trade_analysis.etl.baci import (
    baci_to_parquet_incremental,
    remap_baci_country_codes,
)
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)


# def validate_cleaned_baci(lf: pl.LazyFrame) -> bool:
#     """
#     Performs basic validation on the cleaned BACI LazyFrame.

#     Args:
#         lf: The LazyFrame to validate.

#     Returns:
#         True if validation passes, False otherwise.
#     """
#     # Expect remapped columns from remap_baci_country_codes
#     # The function currently produces 'i_iso_numeric', 'j_iso_numeric'
#     # It also keeps original columns unless drop_original=True was set explicitly
#     # Let's assume drop_original=True was used in remap_baci_country_codes
#     expected_cols_subset = {
#         "t",  # Year (original, should be cast later if needed)
#         "i_iso_numeric",  # Remapped exporter ISO numeric code
#         "j_iso_numeric",  # Remapped importer ISO numeric code
#         "k",  # HS Product code (original)
#         "v",  # Value
#         "q",  # Quantity
#     }
#     actual_cols = set(lf.columns)

#     # Check if the core expected columns are present
#     if not expected_cols_subset.issubset(actual_cols):
#         missing = expected_cols_subset - actual_cols
#         logger.error(
#             f"Validation Error: Missing expected columns in cleaned BACI data. "
#             f"Missing: {missing}, Found: {actual_cols}"
#         )
#         return False

#     # Add more checks here:
#     # - Data type checks (e.g., are i_iso_numeric, j_iso_numeric strings? Is 't' string?)
#     #   remap_baci_country_codes casts 't' to Utf8.
#     schema = lf.collect_schema()
#     if schema.get("t") != pl.Utf8:
#         logger.error(f"Validation Error: BACI 't' column type is {schema.get('t')}, expected Utf8.")
#         return False
#     if schema.get("i_iso_numeric") != pl.List(
#         pl.Utf8
#     ):  # remap_country_code_improved returns List[str]
#         logger.warning(
#             f"Validation Warning: BACI 'i_iso_numeric' column type is {schema.get('i_iso_numeric')}, expected List(Utf8). Check remapping logic."
#         )
#         # This might not be an error if single codes are not wrapped in lists, adjust expectation if needed.
#     if schema.get("j_iso_numeric") != pl.List(pl.Utf8):
#         logger.warning(
#             f"Validation Warning: BACI 'j_iso_numeric' column type is {schema.get('j_iso_numeric')}, expected List(Utf8)."
#         )

#     # - Check for excessive nulls in key columns (i_iso_numeric, j_iso_numeric, k, t)
#     #   Need to collect to check nulls properly on LazyFrame. Be careful with large data.
#     #   Example (consider sampling or doing this on the saved file):
#     #   null_check_df = lf.select([pl.col(c).is_null().sum().alias(f"{c}_nulls") for c in ['t', 'i_iso_numeric', 'j_iso_numeric', 'k']]).collect()
#     #   logger.debug(f"Null counts in key BACI columns: {null_check_df.to_dicts()[0]}")

#     # - Check if HS codes 'k' look like valid HS codes (e.g., length 6 after potential padding)
#     #   Example check (might need collect):
#     #   invalid_k_count = lf.filter(pl.col('k').str.lengths() != 6).limit(1).collect()
#     #   if not invalid_k_count.is_empty():
#     #        logger.error("Validation Error: Found HS codes ('k') not of length 6.")
#     #        return False

#     logger.info("Cleaned BACI data basic validation passed.")
#     return True


def run_baci_cleaning_pipeline(config: Dict[str, Any]) -> Optional[Path]:
    """
    Orchestrates the cleaning of BACI data: CSV -> Parquet -> Remapped Codes.

    Args:
        config: The pipeline configuration dictionary. Expected keys:
            - HS_CODE
            - BACI_RELEASE
            - BACI_INPUT_FOLDER (Path to raw CSVs parent dir)
            - INTERMEDIATE_DATA_DIR (Path to parent dir for intermediate outputs)
            - BACI_INTERMEDIATE_RAW_PARQUET_PATH (Path for initial parquet conversion output dir)
            - BACI_CLEANED_OUTPUT_PATH (Path for final cleaned/remapped output file/dir)
            - BACI_REF_CODES_PATH (Optional, Path to BACI ref codes)
            - WITS_REF_CODES_PATH (Optional, Path to WITS ref codes)

    Returns:
        The Path object to the cleaned BACI Parquet directory/file if successful,
        otherwise None.
    """
    logger.info("--- Starting BACI Cleaning Pipeline ---")
    hs_code = config["HS_CODE"]
    baci_release = config["BACI_RELEASE"]
    input_folder = config["BACI_INPUT_FOLDER"]
    intermediate_parent_dir = config["INTERMEDIATE_DATA_DIR"]
    initial_output_path = config["BACI_INTERMEDIATE_RAW_PARQUET_PATH"]
    cleaned_output_path = config["BACI_CLEANED_OUTPUT_PATH"]

    # --- Step 1: Convert CSV to Partitioned Parquet ---
    logger.info(f"Starting BACI CSV to Parquet conversion: HS={hs_code}, Release={baci_release}")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Target intermediate output directory: {initial_output_path}")

    try:
        created_path_str_or_none = baci_to_parquet_incremental(
            hs=hs_code,
            release=baci_release,
            input_folder=str(input_folder),
            output_folder=str(intermediate_parent_dir),  # Function expects parent dir
        )

        # Verification
        if not (
            created_path_str_or_none
            and Path(created_path_str_or_none) == initial_output_path
            and initial_output_path.exists()
            and initial_output_path.is_dir()
        ):
            logger.error(
                f"❌ BACI CSV to Parquet failed or produced unexpected output. "
                f"Expected dir: `{initial_output_path}`, Got return: `{created_path_str_or_none}`."
            )
            return None
        logger.info(
            f"✅ BACI CSV to Parquet conversion successful. Output dir: `{initial_output_path}`"
        )

    except Exception as e:
        logger.error(f"❌ BACI CSV to Parquet failed with exception: {e}", exc_info=True)
        return None

    # --- Step 2: Remap Country Codes ---
    logger.info("Starting BACI country code remapping.")
    logger.info(f"Input for remapping: {initial_output_path}")
    logger.info(f"Target cleaned output path: {cleaned_output_path}")

    try:
        # remap_baci_country_codes handles reading from the input dir and writing to output path
        final_path_obj = remap_baci_country_codes(
            input_path=initial_output_path,  # Input is the directory from step 1
            output_path=cleaned_output_path,  # Output is a file path (e.g., .../remapped.parquet)
        )

        # Verification: final_path_obj should be the same as cleaned_output_path and exist
        if not (
            final_path_obj
            and final_path_obj == cleaned_output_path
            and cleaned_output_path.exists()
        ):
            logger.error(
                f"❌ BACI country code remapping failed. Expected output not found or mismatch. "
                f"Expected: `{cleaned_output_path}`, Got return value: `{final_path_obj}`."
            )
            return None

        logger.info(f"✅ BACI country code remapping successful. Output: {cleaned_output_path}")

        #     # --- Step 3: Validation ---
        #     logger.info(f"Validating cleaned BACI data at: {cleaned_output_path}")
        #     # remap_baci_country_codes saves a single parquet file
        #     try:
        #         cleaned_lf = pl.scan_parquet(cleaned_output_path)
        #         if not validate_cleaned_baci(cleaned_lf):
        #             logger.error("❌ Cleaned BACI data validation failed.")
        #             # Consider deleting the invalid output file
        #             # cleaned_output_path.unlink(missing_ok=True)
        #             return None
        #         logger.info("✅ Cleaned BACI data validation successful.")
        #     except Exception as e:
        #         logger.error(f"❌ Failed to load or validate cleaned BACI data: {e}", exc_info=True)
        #         return None

        logger.info("--- BACI Cleaning Pipeline Finished Successfully ---")
        return cleaned_output_path  # Return the path to the final cleaned file

    except Exception as e:
        logger.error(f"❌ BACI cleaning pipeline failed with exception: {e}", exc_info=True)
        return None


# # Example of how to run (optional, for direct execution)
# if __name__ == "__main__":
#     from mpil_tariff_trade_analysis.utils.logging_config import setup_logging

#     setup_logging(log_level="DEBUG")  # Setup logging for direct run

#     # --- Dummy Configuration for Testing ---
#     TEST_HS_CODE = "HS92"
#     TEST_BACI_RELEASE = "202501"
#     TEST_BASE_DATA_DIR = Path("data").resolve()
#     TEST_RAW_DATA_DIR = TEST_BASE_DATA_DIR / "raw"
#     TEST_INTERMEDIATE_DATA_DIR = TEST_BASE_DATA_DIR / "intermediate"

#     # Construct paths based on convention used in baci.py and this pipeline
#     baci_intermediate_raw_parquet_name = f"BACI_{TEST_HS_CODE}_V{TEST_BACI_RELEASE}"
#     # Define a specific filename for the remapped output
#     baci_cleaned_filename = f"BACI_{TEST_HS_CODE}_V{TEST_BACI_RELEASE}_cleaned_remapped.parquet"

#     test_config = {
#         "HS_CODE": TEST_HS_CODE,
#         "BACI_RELEASE": TEST_BACI_RELEASE,
#         "BACI_INPUT_FOLDER": TEST_RAW_DATA_DIR,  # Parent dir of BACI_HSXX_VYYYYYY CSVs
#         "INTERMEDIATE_DATA_DIR": TEST_INTERMEDIATE_DATA_DIR,
#         # Path to the directory created by baci_to_parquet_incremental
#         "BACI_INTERMEDIATE_RAW_PARQUET_PATH": TEST_INTERMEDIATE_DATA_DIR
#         / baci_intermediate_raw_parquet_name,
#         # Path to the final cleaned parquet file created by remap_baci_country_codes
#         "BACI_CLEANED_OUTPUT_PATH": TEST_INTERMEDIATE_DATA_DIR / baci_cleaned_filename,
#         # Add paths to reference files if not using defaults
#         # "BACI_REF_CODES_PATH": TEST_RAW_DATA_DIR / "BACI_HS92_V202501" / "country_codes_V202501.csv",
#         # "WITS_REF_CODES_PATH": TEST_RAW_DATA_DIR / "WITS_country_codes.csv",
#     }
#     logger.info(f"Running BACI cleaning pipeline with test config: {test_config}")

#     result_path = run_baci_cleaning_pipeline(test_config)

#     if result_path:
#         logger.info(f"BACI cleaning pipeline test completed successfully. Output: {result_path}")
#         sys.exit(0)
#     else:
#         logger.error("BACI cleaning pipeline test failed.")
#         sys.exit(1)
