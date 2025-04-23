"""
Pipeline for cleaning, expanding, and preparing WITS Preferential tariff data.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

# Import functions moved from matching_logic
from mpil_tariff_trade_analysis.etl.matching_logic import (
    expand_preferential_tariffs,  # Now used here
    load_pref_group_mapping,
)

# Local imports
from mpil_tariff_trade_analysis.etl.WITS_cleaner import (
    DEFAULT_WITS_BASE_DIR,
    consolidate_wits_tariff_data,
)
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)


def rename_wits_pref_expanded(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Renames columns in the cleaned and *expanded* WITS Preferential tariff dataframe.
    Assumes country codes ('i') are already remapped and partner codes ('j')
    represent individual countries after expansion.
    """
    logger.debug("Renaming cleaned & expanded WITS Preferential columns.")
    # Expected columns after expansion: t, i (List[Utf8]), j_individual (Utf8), k, pref_tariff_rate, etc.
    rename_map = {
        "t": "t",  # Keep 't' for consistency
        "i": "i_list",  # Reporter list from remapping
        "j": "j_individual",  # Partner individual code from expansion
        "k": "k",  # HS code (already translated)
        "pref_tariff_rate": "pref_tariff_rate",
        "pref_min_tariff_rate": "pref_min_tariff_rate",
        "pref_max_tariff_rate": "pref_max_tariff_rate",
        # Keep tariff_type if present from load_wits_tariff_data
        # "tariff_type": "tariff_type",
    }
    # Check if tariff_type exists before adding to map/select
    if "tariff_type" in df.columns:
        rename_map["tariff_type"] = "tariff_type"

    # Select only the columns we need after renaming
    select_cols = [
        "t",
        # "i", # Final 'i' column after handling list
        # "j", # Final 'j' column (already individual)
        "k",
        "pref_tariff_rate",
        "pref_min_tariff_rate",
        "pref_max_tariff_rate",
    ]
    if "tariff_type" in rename_map:
        select_cols.append("tariff_type")

    # Check if all keys in rename_map exist in df.columns
    missing_cols = [col for col in rename_map.keys() if col not in df.columns]
    if missing_cols:
        logger.error(
            f"Cannot rename WITS Pref: Missing expected input columns: {missing_cols}. Available: {df.columns}"
        )
        raise ValueError(f"Missing columns required for renaming: {missing_cols}")

    # Rename, handle the list column 'i_list', and use 'j_individual' as 'j'
    renamed_df = (
        df.rename(rename_map)
        .with_columns(
            pl.col("i_list").list.first().alias("i"),  # Take first reporter
            pl.col("j_individual").alias("j"),  # Use expanded partner as 'j'
        )
        .select(select_cols + ["i", "j"])
    )  # Add the new 'i' and 'j' columns

    logger.debug(f"Renamed WITS Preferential Schema: {renamed_df.collect_schema()}")
    return renamed_df


def validate_cleaned_wits_pref(lf: pl.LazyFrame) -> bool:
    """
    Performs basic validation on the cleaned, expanded, and renamed WITS Pref LazyFrame.
    """
    expected_cols = {
        "t",
        "i",  # Single reporter string
        "j",  # Single partner string (individual)
        "k",
        "pref_tariff_rate",
        "pref_min_tariff_rate",
        "pref_max_tariff_rate",
        # "tariff_type", # Optional
    }
    if "tariff_type" in lf.columns:
        expected_cols.add("tariff_type")

    actual_cols = set(lf.columns)

    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        logger.error(
            f"Validation Error: Missing expected columns in cleaned WITS Pref data. "
            f"Missing: {missing}, Found: {actual_cols}"
        )
        return False

    # Add more checks:
    # - Data types (t: Utf8, i: Utf8, j: Utf8, k: Utf8, rates: Utf8)
    schema = lf.collect_schema()
    if schema.get("t") != pl.Utf8:
        logger.error(f"Validation Error: Pref 't' column type is {schema.get('t')}, expected Utf8.")
        return False
    if schema.get("i") != pl.Utf8:
        logger.error(f"Validation Error: Pref 'i' column type is {schema.get('i')}, expected Utf8.")
        return False
    if schema.get("j") != pl.Utf8:
        logger.error(f"Validation Error: Pref 'j' column type is {schema.get('j')}, expected Utf8.")
        return False
    if schema.get("k") != pl.Utf8:
        logger.error(f"Validation Error: Pref 'k' column type is {schema.get('k')}, expected Utf8.")
        return False
    if schema.get("pref_tariff_rate") != pl.Utf8:
        logger.warning(
            f"Validation Warning: Pref 'pref_tariff_rate' column type is {schema.get('pref_tariff_rate')}, expected Utf8."
        )

    # - Check tariff rates are within a plausible range
    # - Check for excessive nulls in key columns (t, i, j, k)

    logger.info("Cleaned WITS Pref data basic validation passed.")
    return True


def run_wits_pref_cleaning_pipeline(config: Dict[str, Any]) -> Optional[Path]:
    """
    Orchestrates the cleaning and expansion of WITS Preferential tariff data.

    Args:
        config: The pipeline configuration dictionary. Expected keys:
            - WITS_RAW_DIR (Path to base dir containing AVEMFN/AVEPref folders)
            - PREF_GROUPS_PATH (Path to the preference groups mapping CSV)
            - WITS_PREF_CLEANED_OUTPUT_PATH (Path for the final cleaned Pref parquet file)
            - Optional: BACI_REF_CODES_PATH, WITS_REF_CODES_PATH (for country mapping)
            - Optional: HS_MAPPING_DIR (for HS translation)

    Returns:
        The Path object to the cleaned WITS Pref Parquet file if successful,
        otherwise None.
    """
    logger.info("--- Starting WITS Preferential Cleaning & Expansion Pipeline ---")
    base_dir = config.get("WITS_RAW_DIR", Path(DEFAULT_WITS_BASE_DIR))
    pref_groups_path = config["PREF_GROUPS_PATH"]
    output_path = config["WITS_PREF_CLEANED_OUTPUT_PATH"]
    hs_mapping_dir = config.get("HS_MAPPING_DIR", "data/raw/hs_reference")
    baci_codes_path = config.get("BACI_REF_CODES_PATH", None)
    wits_codes_path = config.get("WITS_REF_CODES_PATH", None)

    # Ensure output directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_path.parent}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_path.parent}: {e}")
        return None

    # --- Step 1: Load Preference Group Mapping ---
    logger.info(f"Loading preferential groups mapping from: {pref_groups_path}")
    try:
        # Collect mapping as it's needed for expansion
        pref_group_mapping = load_pref_group_mapping(pref_groups_path).collect()
        logger.info(
            f"Preferential group mapping loaded into memory (shape: {pref_group_mapping.shape})."
        )
    except Exception as e:
        logger.error(f"Failed to load preferential group mapping: {e}", exc_info=True)
        return None  # Cannot proceed without mapping

    # --- Step 2: Load Raw Data, Translate HS, Remap Countries ---
    logger.info(f"Loading raw WITS Pref data from base directory: {base_dir}")
    try:
        # Returns LazyFrame with columns like 'year', 'reporter_country_iso_numeric' (List[Utf8]),
        # 'partner_country_iso_numeric' (List[Utf8]), etc.
        loaded_wits_path = consolidate_wits_tariff_data(
            tariff_type="AVEPref",
            base_dir=str(base_dir),
        )

        if loaded_df is None:
            logger.error("❌ Loading WITS Pref data returned None.")
            return None
        logger.info("✅ WITS Pref data loaded, HS translated, countries remapped.")
        logger.debug(f"Schema after loading: {loaded_df.collect_schema()}")

    except Exception as e:
        logger.error(f"❌ Failed during WITS Pref data loading/preprocessing: {e}", exc_info=True)
        return None

    # --- Step 3: Expand Preferential Partner Groups ---
    logger.info("Expanding preferential tariff partner groups.")
    try:
        # The expand_preferential_tariffs function expects specific column names
        # from the *old* rename_wits_pref. We need to adapt it or rename columns *before* calling it.
        # Let's rename temporarily to match the function's expectation, then rename properly later.
        temp_rename_map_in = {
            "year": "t",
            "reporter_country_iso_numeric": "i",  # Function expects 'i' (list is ok here)
            "partner_country_iso_numeric": "j",  # Function expects 'j' (list is ok here)
            "product_code": "k",
            "tariff_rate": "pref_tariff_rate",
            "min_rate": "pref_min_tariff_rate",
            "max_rate": "pref_max_tariff_rate",
        }
        # Check if all keys in temp_rename_map_in exist
        missing_temp_cols = [
            col for col in temp_rename_map_in.keys() if col not in loaded_df.columns
        ]
        if missing_temp_cols:
            raise ValueError(
                f"Missing columns needed for temp rename before expansion: {missing_temp_cols}"
            )

        temp_renamed_df = loaded_df.rename(temp_rename_map_in)

        # Now call the expansion function
        expanded_df = expand_preferential_tariffs(temp_renamed_df, pref_group_mapping.lazy())
        # expanded_df now has columns: t, i (List[Utf8]), j_individual (Utf8), k, pref_tariff_rate, etc.
        logger.info("✅ Preferential partner groups expanded.")
        logger.debug(f"Schema after expansion: {expanded_df.collect_schema()}")

    except Exception as e:
        logger.error(f"❌ Failed during preferential group expansion: {e}", exc_info=True)
        return None

    # --- Step 4: Rename Columns to Final Cleaned State ---
    logger.info("Renaming columns to final cleaned state.")
    try:
        # Pass the *expanded* dataframe to the final renaming function
        final_renamed_df = rename_wits_pref_expanded(expanded_df)
        logger.info("✅ Columns renamed to final state.")
    except Exception as e:
        logger.error(f"❌ Failed during final column renaming: {e}", exc_info=True)
        return None

    # --- Step 5: Validation ---
    logger.info("Validating cleaned and expanded WITS Pref data.")
    if not validate_cleaned_wits_pref(final_renamed_df):
        logger.error("❌ Cleaned WITS Pref data validation failed.")
        # Consider deleting output file
        # output_path.unlink(missing_ok=True)
        return None
    logger.info("✅ Cleaned WITS Pref data validation successful.")

    # --- Step 6: Save Cleaned Data ---
    logger.info(f"Saving cleaned WITS Pref data to: {output_path}")
    try:
        # Collect and save the final dataframe
        final_renamed_df.collect().write_parquet(output_path)
        logger.info("✅ Cleaned WITS Pref data saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save cleaned WITS Pref data: {e}", exc_info=True)
        return None

    logger.info("--- WITS Preferential Cleaning & Expansion Pipeline Finished Successfully ---")
    return output_path


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
        "PREF_GROUPS_PATH": TEST_RAW_DATA_DIR / "WITS_pref_groups" / "WITS_pref_groups.csv",
        "WITS_PREF_CLEANED_OUTPUT_PATH": TEST_INTERMEDIATE_DATA_DIR
        / "cleaned_wits_pref"
        / "WITS_AVEPref_cleaned_expanded.parquet",
        # Optional paths if not using defaults:
        # "HS_MAPPING_DIR": TEST_RAW_DATA_DIR / "hs_reference",
        # "BACI_REF_CODES_PATH": TEST_RAW_DATA_DIR / "BACI_HS92_V202501" / "country_codes_V202501.csv",
        # "WITS_REF_CODES_PATH": TEST_RAW_DATA_DIR / "WITS_country_codes.csv",
    }
    logger.info(f"Running WITS Pref cleaning pipeline with test config: {test_config}")

    # Check if inputs exist
    if not test_config["PREF_GROUPS_PATH"].exists():
        logger.error(f"Pref groups input not found: {test_config['PREF_GROUPS_PATH']}")
        sys.exit(1)
    if not test_config["WITS_RAW_DIR"].exists():
        logger.error(f"WITS raw dir input not found: {test_config['WITS_RAW_DIR']}")
        sys.exit(1)

    result_path = run_wits_pref_cleaning_pipeline(test_config)

    if result_path:
        logger.info(
            f"WITS Pref cleaning pipeline test completed successfully. Output: {result_path}"
        )
        sys.exit(0)
    else:
        logger.error("WITS Pref cleaning pipeline test failed.")
        sys.exit(1)
