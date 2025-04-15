import logging # Added for checking log level
from pathlib import Path

import pandas as pd
import polars as pl

# Assuming your logging setup is accessible via this import path
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration ---
# Define default paths (consider making these configurable, e.g., via arguments)
DEFAULT_BACI_PATH = "data/final/BACI_HS92_V202501"
DEFAULT_WITS_MFN_PATH = "data/final/WITS_AVEMFN.parquet"
DEFAULT_WITS_PREF_PATH = "data/final/WITS_AVEPref.parquet"
DEFAULT_PREF_GROUPS_PATH = "data/raw/WITS_pref_groups/WITS_pref_groups.csv"
DEFAULT_OUTPUT_PATH = "data/final/unified_trade_tariff.parquet"


# --- Helper Functions ---


def load_pref_group_mapping(file_path: str | Path) -> pl.LazyFrame:
    """
    Loads and processes the WITS preferential group mapping file.

    Args:
        file_path: Path to the WITS preferential groups CSV file.

    Returns:
        A Polars LazyFrame containing the mapping from region codes to lists of partner countries.
    """
    logger.info(f"Loading preferential group mapping from: {file_path}")
    try:
        # Load the raw CSV mapping file using Pandas for initial processing
        # Using pandas first as it might handle complex CSV variations, then aggregate
        pref_groups_pd = pd.read_csv(file_path, encoding="iso-8859-1")
        logger.debug(f"Raw preferential groups loaded (pandas shape): {pref_groups_pd.shape}")

        # Aggregate partners into lists per region code
        # Ensure partners are strings and handle potential NaN/None values
        pref_groups_pd['Partner'] = pref_groups_pd['Partner'].astype(str)
        pref_group_df_pd = (
            pref_groups_pd.groupby("RegionCode")['Partner']
            .agg(lambda x: list(set(x.dropna()))) # Use dropna() within lambda, convert set to list
            .reset_index()
            .rename(columns={"RegionCode": "region_code", "Partner": "partner_list"})
        )
        logger.debug(f"Aggregated preferential groups (pandas shape): {pref_group_df_pd.shape}")

        # Convert to Polars LazyFrame
        pref_group_pl = pl.from_pandas(pref_group_df_pd).lazy()
        logger.info("Preferential group mapping loaded and processed.")
        logger.debug(f"Preferential Group Mapping Schema: {pref_group_pl.collect_schema()}")
        if logger.isEnabledFor(logging.DEBUG):
             try:
                 # Use fetch(1) for safer debugging
                 logger.debug(f"Preferential Group Mapping Head (1 row):\n{pref_group_pl.fetch(1)}")
             except Exception as e:
                 logger.warning(f"Could not fetch(1) from pref_group_pl: {e}")

        return pref_group_pl
    except Exception as e:
        logger.error(f"Error loading or processing preferential group mapping: {e}")
        raise


def rename_wits_mfn(df: pl.LazyFrame) -> pl.LazyFrame:
    """Renames columns in the WITS MFN tariff dataframe."""
    logger.debug("Renaming WITS MFN columns.")
    renamed_df = df.rename(
        {
            "year": "t",
            "reporter_country": "i",
            "product_code": "k",
            "tariff_rate": "mfn_tariff_rate",
            "min_rate": "mfn_min_tariff_rate",
            "max_rate": "mfn_max_tariff_rate",
            "tariff_type": "tariff_type", # Keep tariff type for potential debugging
        }
    ).select(
        "t",
        "i",
        "k",
        "mfn_tariff_rate",
        "mfn_min_tariff_rate",
        "mfn_max_tariff_rate",
        "tariff_type",
    )
    logger.debug(f"Renamed WITS MFN Schema: {renamed_df.collect_schema()}")
    return renamed_df


def rename_wits_pref(df: pl.LazyFrame) -> pl.LazyFrame:
    """Renames columns in the WITS Preferential tariff dataframe."""
    logger.debug("Renaming WITS Preferential columns.")
    renamed_df = df.rename(
        {
            "year": "t",
            "reporter_country": "i",
            "partner_country": "j",  # This might be a group code or individual country
            "product_code": "k",
            "tariff_rate": "pref_tariff_rate",
            "min_rate": "pref_min_tariff_rate",
            "max_rate": "pref_max_tariff_rate",
        }
    ).select("t", "i", "j", "k", "pref_tariff_rate", "pref_min_tariff_rate", "pref_max_tariff_rate")
    logger.debug(f"Renamed WITS Preferential Schema: {renamed_df.collect_schema()}")
    return renamed_df


def expand_preferential_tariffs(
    renamed_avepref: pl.LazyFrame, pref_group_mapping: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Expands WITS preferential tariff data from partner groups to individual countries.

    Args:
        renamed_avepref: LazyFrame of renamed WITS preferential tariffs.
        pref_group_mapping: LazyFrame mapping region codes to partner lists.

    Returns:
        A LazyFrame with preferential tariffs expanded to individual partner countries.
    """
    logger.info("Expanding preferential tariff partner groups.")
    logger.debug(f"Input renamed_avepref schema: {renamed_avepref.collect_schema()}")
    logger.debug(f"Input pref_group_mapping schema: {pref_group_mapping.collect_schema()}")

    # --- Explicit Type Casting for Join Keys ---
    # Adjust pl.Utf8 if your codes are numeric (e.g., pl.Int64)
    logger.debug("Casting join keys to pl.Utf8 (adjust if needed).")
    try:
        renamed_avepref = renamed_avepref.with_columns(pl.col("j").cast(pl.Utf8))
        pref_group_mapping = pref_group_mapping.with_columns(pl.col("region_code").cast(pl.Utf8))
        logger.debug(f"Schema after casting renamed_avepref: {renamed_avepref.collect_schema()}")
        logger.debug(f"Schema after casting pref_group_mapping: {pref_group_mapping.collect_schema()}")
    except Exception as e:
        logger.error(f"Failed to cast join keys: {e}")
        raise

    # --- Left join avepref with the group mapping ---
    logger.debug("Attempting left join: renamed_avepref with pref_group_mapping on 'j' == 'region_code'.")
    joined_pref_mapping = renamed_avepref.join(
        pref_group_mapping,
        left_on="j",  # Partner code (can be group or individual)
        right_on="region_code",  # Group code from mapping
        how="left",
    )
    logger.debug("Join completed.")
    logger.debug(f"Schema after joining avepref with group mapping: {joined_pref_mapping.collect_schema()}")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(f"Head(1) after joining avepref with group mapping:\n{joined_pref_mapping.fetch(1)}")
        except Exception as e:
            logger.warning(f"Could not fetch(1) from joined_pref_mapping: {e}")


    # --- Create the final partner list ---
    logger.debug("Creating final partner list using coalesce logic.")
    expanded_pref_pre_explode = joined_pref_mapping.with_columns(
            pl.when(pl.col("partner_list").is_not_null() & (pl.col("partner_list").list.len() > 0)) # Check list not null AND not empty
            .then(pl.col("partner_list"))  # Use the list from mapping
            .otherwise(pl.concat_list(pl.col("j"))) # Use original 'j' as a single-item list if no match or empty list
            .alias("final_partner_list")
        )
    logger.debug("Final partner list column created ('final_partner_list').")
    logger.debug(f"Schema before explode: {expanded_pref_pre_explode.collect_schema()}")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(f"Head(1) before explode (showing 'final_partner_list'):\n{expanded_pref_pre_explode.select(['t', 'i', 'j', 'k', 'partner_list', 'final_partner_list']).fetch(1)}")
        except Exception as e:
            logger.warning(f"Could not fetch(1) before explode: {e}")


    # --- Explode the final partner list ---
    logger.debug("Attempting to explode 'final_partner_list'.")
    try:
        expanded_pref = (
            expanded_pref_pre_explode.explode(
                "final_partner_list"  # Explode the list into separate rows
            )
            .rename(
                {"final_partner_list": "j_individual"}  # Rename the exploded column
            )
            .select(
                pl.col("t"),
                pl.col("i"),
                pl.col("j_individual").alias("j"),  # Use the new individual partner code as 'j'
                pl.col("k"),
                pl.col("pref_tariff_rate"),
                pl.col("pref_min_tariff_rate"),
                pl.col("pref_max_tariff_rate"),
                # Add other columns from renamed_avepref if needed
            )
        )
        logger.debug("Explode operation defined in lazy frame.") # Note: Actual execution happens later
        logger.info("Preferential tariffs expansion plan created.")
        logger.debug(f"Schema AFTER defining explode: {expanded_pref.collect_schema()}")
        if logger.isEnabledFor(logging.DEBUG):
            # This is the critical point where the previous issue occurred
            logger.debug("Attempting to fetch(1) AFTER explode operation definition...")
            try:
                fetched_row = expanded_pref.fetch(1)
                logger.debug(f"Successfully fetched 1 row AFTER explode:\n{fetched_row}")
            except Exception as e:
                # If this fails, it's likely a memory issue or internal Polars error during execution
                logger.error(f"CRITICAL: Failed to fetch(1) AFTER explode definition. Potential OOM or internal error: {e}", exc_info=True)
                # Optionally re-raise or handle differently depending on desired behavior
                raise
        return expanded_pref
    except Exception as e:
        logger.error(f"Error occurred during the definition or initial fetch of the explode operation: {e}", exc_info=True)
        raise


def join_datasets(
    baci: pl.LazyFrame, renamed_avemfn: pl.LazyFrame, expanded_pref: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Joins BACI, MFN, and expanded Preferential tariff data.

    Args:
        baci: LazyFrame of BACI trade data.
        renamed_avemfn: LazyFrame of renamed WITS MFN tariffs.
        expanded_pref: LazyFrame of expanded WITS preferential tariffs.

    Returns:
        A LazyFrame containing the joined data with an 'effective_tariff_rate' column.
    """
    logger.info("Joining BACI, MFN, and Preferential datasets.")
    logger.debug(f"Input BACI schema: {baci.collect_schema()}")
    logger.debug(f"Input renamed_avemfn schema: {renamed_avemfn.collect_schema()}")
    logger.debug(f"Input expanded_pref schema: {expanded_pref.collect_schema()}") # Schema is known even if fetch failed

    # Define join keys
    mfn_join_keys = ["t", "i", "k"]
    pref_join_keys = ["t", "i", "j", "k"]

    # --- Ensure join key types are compatible ---
    # Add casting if necessary, based on actual data types observed
    # Example: Cast all keys to common types (e.g., Int64 for t, i, j; Utf8 for k)
    logger.debug("Casting join key types for final joins (adjust types as needed).")
    try:
        baci = baci.with_columns([
            pl.col("t").cast(pl.Int64), pl.col("i").cast(pl.Utf8), # Assuming country codes are strings now
            pl.col("j").cast(pl.Utf8), pl.col("k").cast(pl.Utf8)
        ])
        renamed_avemfn = renamed_avemfn.with_columns([
            pl.col("t").cast(pl.Int64), pl.col("i").cast(pl.Utf8),
            pl.col("k").cast(pl.Utf8)
        ])
        # expanded_pref 'j' should already be Utf8 from the explode step if casting was done there
        expanded_pref = expanded_pref.with_columns([
            pl.col("t").cast(pl.Int64), pl.col("i").cast(pl.Utf8),
            pl.col("j").cast(pl.Utf8), # Ensure j is Utf8
            pl.col("k").cast(pl.Utf8)
        ])
        logger.debug(f"Schema after casting - BACI: {baci.collect_schema()}")
        logger.debug(f"Schema after casting - MFN: {renamed_avemfn.collect_schema()}")
        logger.debug(f"Schema after casting - Pref: {expanded_pref.collect_schema()}")
    except Exception as e:
        logger.error(f"Failed to cast join keys for final joins: {e}")
        raise


    # 1. Left join BACI with MFN tariffs
    logger.debug(f"Joining BACI with MFN tariffs on {mfn_join_keys}.")
    joined_mfn = baci.join(renamed_avemfn, on=mfn_join_keys, how="left")
    logger.debug(f"Schema after BACI-MFN join: {joined_mfn.collect_schema()}")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(f"Head(1) after BACI-MFN join:\n{joined_mfn.fetch(1)}")
        except Exception as e:
            logger.warning(f"Could not fetch(1) from joined_mfn: {e}")


    # 2. Left join the result with Expanded Preferential tariffs
    logger.debug(f"Joining intermediate result with expanded Preferential tariffs on {pref_join_keys}.")
    joined_all = joined_mfn.join(expanded_pref, on=pref_join_keys, how="left")
    logger.debug(f"Schema after joining with Pref tariffs: {joined_all.collect_schema()}")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(f"Head(1) after joining with Pref tariffs:\n{joined_all.fetch(1)}")
        except Exception as e:
            logger.warning(f"Could not fetch(1) from joined_all: {e}")


    # 3. Calculate the final effective tariff
    logger.debug("Calculating effective tariff rate using coalesce(pref_tariff_rate, mfn_tariff_rate).")
    final_table = joined_all.with_columns(
        pl.coalesce(pl.col("pref_tariff_rate"), pl.col("mfn_tariff_rate")).alias(
            "effective_tariff_rate"
        )
    )
    logger.info("Datasets joined successfully.")
    logger.debug(f"Schema after calculating effective tariff: {final_table.collect_schema()}")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(f"Head(1) after calculating effective tariff:\n{final_table.fetch(1)}")
        except Exception as e:
            logger.warning(f"Could not fetch(1) from final_table (post-coalesce): {e}")
    return final_table


def create_final_table(joined_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Selects and renames columns for the final unified table.

    Args:
        joined_data: LazyFrame containing the joined BACI and WITS data with effective tariff.

    Returns:
        A LazyFrame representing the final unified table structure.
    """
    logger.info("Selecting and renaming columns for the final output table.")
    logger.debug(f"Input schema for final selection: {joined_data.collect_schema()}")
    # !! ADJUST column names 'v' and 'q' based on your actual BACI schema !!
    # Ensure the columns exist before selecting
    available_cols = joined_data.columns
    select_cols = [
        pl.col("t").alias("Year"),
        pl.col("i").alias("Source"),  # Reporter country code
        pl.col("j").alias("Target"),  # Partner country code (individual)
        pl.col("k").alias("HS_Code"),  # Product code (HS92)
    ]
    if "q" in available_cols:
        select_cols.append(pl.col("q").alias("Quantity")) # Assuming 'q' is Quantity in BACI
    else:
        logger.warning("Column 'q' (Quantity) not found in joined data. Skipping.")
    if "v" in available_cols:
         select_cols.append(pl.col("v").alias("Value")) # Assuming 'v' is Value in BACI
    else:
        logger.warning("Column 'v' (Value) not found in joined data. Skipping.")

    select_cols.extend([
        pl.col("mfn_tariff_rate"),
        pl.col("pref_tariff_rate"),
        pl.col("effective_tariff_rate"),  # The coalesced rate
    ])

    # Optionally add min/max rates if needed and available
    # if "mfn_min_tariff_rate" in available_cols: select_cols.append(pl.col("mfn_min_tariff_rate"))
    # if "mfn_max_tariff_rate" in available_cols: select_cols.append(pl.col("mfn_max_tariff_rate"))
    # if "pref_min_tariff_rate" in available_cols: select_cols.append(pl.col("pref_min_tariff_rate"))
    # if "pref_max_tariff_rate" in available_cols: select_cols.append(pl.col("pref_max_tariff_rate"))
    # if "tariff_type" in available_cols: select_cols.append(pl.col("tariff_type")) # Optionally add tariff type

    final_unified_table = joined_data.select(select_cols)

    logger.debug(f"Final table schema after selection/renaming: {final_unified_table.collect_schema()}")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            logger.debug(f"Final table head(1):\n{final_unified_table.fetch(1)}")
        except Exception as e:
            logger.warning(f"Could not fetch(1) from final_unified_table: {e}")
    return final_unified_table


# --- Main Execution ---


def run_matching_pipeline(
    baci_path: str | Path = DEFAULT_BACI_PATH,
    wits_mfn_path: str | Path = DEFAULT_WITS_MFN_PATH,
    wits_pref_path: str | Path = DEFAULT_WITS_PREF_PATH,
    pref_groups_path: str | Path = DEFAULT_PREF_GROUPS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    lazy: bool = True, # Note: Debug logging now uses fetch(1) which triggers some computation
):
    """
    Runs the full data matching pipeline.

    Args:
        baci_path: Path to the BACI data.
        wits_mfn_path: Path to the WITS MFN tariff data.
        wits_pref_path: Path to the WITS Preferential tariff data.
        pref_groups_path: Path to the WITS preferential groups mapping CSV.
        output_path: Path to save the final unified Parquet file.
        lazy: If True, performs final sink lazily. If False, collects the
              entire final result before writing (uses more memory).
              Debug logging inside functions might trigger partial computation via fetch(1).
    """
    logger.info("Starting data matching pipeline...")

    # Load data
    logger.info(f"Loading BACI data from: {baci_path}")
    baci = pl.scan_parquet(baci_path)
    logger.debug(f"Initial BACI schema: {baci.collect_schema()}")

    logger.info(f"Loading WITS MFN data from: {wits_mfn_path}")
    avemfn = pl.scan_parquet(wits_mfn_path)
    logger.debug(f"Initial WITS MFN schema: {avemfn.collect_schema()}")

    logger.info(f"Loading WITS Preferential data from: {wits_pref_path}")
    avepref = pl.scan_parquet(wits_pref_path)
    logger.debug(f"Initial WITS Pref schema: {avepref.collect_schema()}")

    pref_group_mapping = load_pref_group_mapping(pref_groups_path) # Logging inside

    # Rename columns
    renamed_avemfn = rename_wits_mfn(avemfn) # Logging inside
    renamed_avepref = rename_wits_pref(avepref) # Logging inside

    # Expand preferential tariffs
    # This is the function with enhanced debugging
    expanded_pref = expand_preferential_tariffs(renamed_avepref, pref_group_mapping) # Logging inside

    # Join datasets
    joined_data = join_datasets(baci, renamed_avemfn, expanded_pref) # Logging inside

    # Create final table structure
    final_unified_table = create_final_table(joined_data) # Logging inside

    # Execute and save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    logger.info(f"Saving final unified table to: {output_path}")
    try:
        if lazy:
            logger.info("Executing final plan lazily and sinking to parquet...")
            final_unified_table.sink_parquet(output_path)
            logger.info("Lazy execution complete. Data saved.")
        else:
            # Collect the result before writing (uses more memory)
            logger.info("Collecting final result eagerly...")
            result_df = final_unified_table.collect()
            logger.info(f"Eager collection complete. Shape: {result_df.shape}. Writing to parquet...")
            result_df.write_parquet(output_path)
            logger.info("Eager execution complete. Data saved.")
    except Exception as e:
        logger.critical(f"Pipeline failed during final save/collect operation: {e}", exc_info=True)
        raise # Re-raise after logging

    logger.info("Data matching pipeline finished successfully.")


if __name__ == "__main__":
    # This block allows running the script directly
    # You might want to add argument parsing (e.g., using argparse)
    # to specify input/output paths from the command line.

    # --- IMPORTANT: Setup logging BEFORE running the pipeline ---
    # Ensure logging is configured to capture DEBUG messages if needed
    import logging
    from mpil_tariff_trade_analysis.utils.logging_config import setup_logging
    # Set the desired level, e.g., "DEBUG" for detailed output
    # Make sure your setup_logging function configures handlers appropriately
    log_level_to_set = "DEBUG"
    print(f"Attempting to set up logging with level: {log_level_to_set}")
    setup_logging(log_level=log_level_to_set)
    # Re-get the logger instance AFTER setup_logging has run, just in case
    logger = get_logger(__name__)
    logger.info(f"Logging setup complete. Running matching script directly with level {log_level_to_set}.")
    # --- End of logging setup ---

    try:
        # Consider running with lazy=False initially for debugging, if memory allows
        # run_matching_pipeline(lazy=False)
        run_matching_pipeline(lazy=True) # Or keep lazy=True
    except Exception as e:
        # Logger should have already caught this in run_matching_pipeline or sub-functions
        # but we log again here just in case the exception occurred outside that function
        logger.critical(f"Pipeline execution failed in __main__: {e}", exc_info=True)
        # Exit with a non-zero status code to indicate failure
        import sys
        sys.exit(1)

