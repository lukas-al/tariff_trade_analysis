import logging  # Added for checking log level
from pathlib import Path

import pandas as pd
import polars as pl

# Assuming your logging setup is accessible via this import path
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration ---
# Default paths are less relevant here now, but keep for load_pref_group_mapping default if needed
DEFAULT_PREF_GROUPS_PATH = "data/raw/WITS_pref_groups/WITS_pref_groups.csv"


# --- Helper Functions ---
def load_pref_group_mapping(file_path: str | Path = DEFAULT_PREF_GROUPS_PATH) -> pl.LazyFrame:
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
        pref_groups_pd["Partner"] = pref_groups_pd["Partner"].astype(str)
        pref_group_df_pd = (
            pref_groups_pd.groupby("RegionCode")["Partner"]
            .agg(lambda x: list(set(x.dropna())))  # Use dropna() within lambda, convert set to list
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
                raise

        return pref_group_pl
    except Exception as e:
        logger.error(f"Error loading or processing preferential group mapping: {e}")
        raise


# rename_wits_mfn moved to wits_mfn_pipeline.py
# rename_wits_pref moved to wits_pref_pipeline.py


def expand_preferential_tariffs(
    # Expects input df with columns like t, i (list), j (list), k, pref_tariff_rate etc.
    # from load_wits_tariff_data + temp rename
    input_pref_df: pl.LazyFrame,
    pref_group_mapping: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Expands WITS preferential tariff data from partner groups to individual countries.
    Handles list columns for reporter ('i') and partner ('j') resulting from country remapping.

    Args:
        input_pref_df: LazyFrame of WITS preferential tariffs after loading/remapping/HS translation
                       and temporary renaming (expects 't', 'i' (list), 'j' (list), 'k', 'pref_tariff_rate').
        pref_group_mapping: LazyFrame mapping region codes to partner lists (strings).

    Returns:
        A LazyFrame with preferential tariffs expanded to individual partner countries.
        Output schema includes: t, i (list), j_individual (string), k, pref_tariff_rate, ...
    """
    logger.info("Expanding preferential tariff partner groups.")
    logger.debug(f"Input input_pref_df schema: {input_pref_df.collect_schema()}")
    logger.debug(f"Input pref_group_mapping schema: {pref_group_mapping.collect_schema()}")

    # --- Explode the partner list ('j') first ---
    # The 'j' column from load_wits_tariff_data (via create_country_code_mapping_df) is List[Utf8].
    # We need to explode this to handle cases where the original partner code mapped to multiple ISO codes.
    logger.debug("Exploding input partner list column 'j'.")
    try:
        exploded_j_df = input_pref_df.explode("j")
        logger.debug(f"Schema after exploding 'j': {exploded_j_df.collect_schema()}")
    except Exception as e:
        logger.error(f"Failed to explode partner list 'j': {e}", exc_info=True)
        raise

    # --- Explicit Type Casting for Join Keys ---
    logger.debug("Casting join keys (pref_group_mapping: region_code) to pl.Utf8.")
    try:
        pref_group_mapping = pref_group_mapping.with_columns(pl.col("region_code").cast(pl.Utf8))
        logger.debug(
            f"Schema after casting pref_group_mapping: {pref_group_mapping.collect_schema()}"
        )
    except Exception as e:
        logger.error(f"Failed to cast join keys: {e}")
        raise

    # --- Left join avepref (with exploded 'j') with the group mapping ---
    logger.debug(
        "Attempting left join: exploded_j_df with pref_group_mapping on 'j' == 'region_code'."
    )
    joined_pref_mapping = exploded_j_df.join(
        pref_group_mapping,
        left_on="j",  # Partner code (now individual Utf8 after explode)
        right_on="region_code",  # Group code from mapping (Utf8)
        how="left",
    )
    logger.debug("Join completed.")
    logger.debug(f"Schema after joining with group mapping: {joined_pref_mapping.collect_schema()}")

    # --- Create the final partner list ---
    # If 'partner_list' (from group mapping) is not null, use it.
    # Otherwise, use the original 'j' (which is now an individual code after the first explode)
    # Wrap the result in a list for the next explode step.
    logger.debug("Creating final partner list using coalesce logic.")
    expanded_pref_pre_explode2 = joined_pref_mapping.with_columns(
        pl.when(pl.col("partner_list").is_not_null() & (pl.col("partner_list").list.len() > 0))
        .then(pl.col("partner_list"))  # Use the list from mapping
        .otherwise(
            pl.concat_list(pl.col("j"))  # Use original exploded 'j' as a single-item list
        )
        .alias("final_partner_list")
    ).drop(["partner_list", "region_code"])  # Drop intermediate columns

    logger.debug("Final partner list column created ('final_partner_list').")
    logger.debug(f"Schema before second explode: {expanded_pref_pre_explode2.collect_schema()}")

    # --- Explode the final partner list ---
    logger.debug("Attempting to explode 'final_partner_list'.")
    try:
        # Keep all original columns, replace 'j' effectively
        # Select all columns from the input EXCEPT the intermediate ones we dropped
        cols_to_keep = [
            c for c in joined_pref_mapping.columns if c not in ["partner_list", "region_code"]
        ]

        expanded_pref_final = (
            expanded_pref_pre_explode2.explode("final_partner_list")
            .rename({"final_partner_list": "j_individual"})  # This is the final partner string
            # Select original columns (like t, i (list), k, rates) + the new j_individual
            .select(cols_to_keep + ["j_individual"])
            # We don't rename j_individual to 'j' here, let the final rename step handle it
        )
        logger.debug("Second explode operation defined in lazy frame.")
        logger.info("Preferential tariffs expansion plan created.")
        logger.debug(
            f"Schema AFTER defining second explode: {expanded_pref_final.collect_schema()}"
        )

        # Optional fetch for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Attempting to fetch(1) AFTER second explode definition...")
            try:
                fetched_row = expanded_pref_final.fetch(1)
                logger.debug(f"Successfully fetched 1 row AFTER second explode:\n{fetched_row}")
            except Exception as e:
                logger.error(
                    f"CRITICAL: Failed to fetch(1) AFTER second explode definition: {e}",
                    exc_info=True,
                )
                raise
        return expanded_pref_final
    except Exception as e:
        logger.error(
            f"Error occurred during the definition or initial fetch of the second explode operation: {e}",
            exc_info=True,
        )
        raise


def join_datasets(
    baci: pl.LazyFrame, renamed_avemfn: pl.LazyFrame, cleaned_expanded_pref: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Joins cleaned BACI, cleaned MFN, and cleaned/expanded Preferential tariff data.

    Args:
        baci: LazyFrame of cleaned BACI trade data (with columns t, i, j, k, v, q).
              Assumes country codes 'i', 'j' might be lists from remapping.
        renamed_avemfn: LazyFrame of cleaned/renamed WITS MFN tariffs (t, i, k, rates).
                        Assumes 'i' is a single string.
        cleaned_expanded_pref: LazyFrame of cleaned/expanded/renamed WITS preferential tariffs
                               (t, i, j, k, rates). Assumes 'i', 'j' are single strings.

    Returns:
        A LazyFrame containing the joined data with an 'effective_tariff_rate' column.
    """
    logger.info("Joining BACI, MFN, and Preferential datasets.")
    logger.debug(f"Input BACI schema: {baci.collect_schema()}")
    logger.debug(f"Input MFN schema: {renamed_avemfn.collect_schema()}")
    logger.debug(f"Input Pref schema: {cleaned_expanded_pref.collect_schema()}")

    # Define join keys
    # BACI 'i' and 'j' might be lists, MFN 'i' is string, Pref 'i' and 'j' are strings.
    # We need to explode BACI's 'i' and 'j' before joining.
    mfn_join_keys = ["t", "i", "k"]
    pref_join_keys = ["t", "i", "j", "k"]

    # --- Ensure join key types are compatible ---
    # Cast all keys to common types (e.g., Utf8 for t, i, j, k)
    # Tariff rates are likely Utf8 from cleaning, handle later in coalesce.
    logger.debug("Casting join key types to Utf8.")
    try:
        # Explode BACI country lists before casting/joining
        if "i_iso_numeric" in baci.columns and isinstance(
            baci.collect_schema()["i_iso_numeric"], pl.List
        ):
            logger.debug("Exploding BACI 'i_iso_numeric' (exporter list).")
            baci = baci.explode("i_iso_numeric")
        if "j_iso_numeric" in baci.columns and isinstance(
            baci.collect_schema()["j_iso_numeric"], pl.List
        ):
            logger.debug("Exploding BACI 'j_iso_numeric' (importer list).")
            baci = baci.explode("j_iso_numeric")

        # Rename BACI columns to match join keys AFTER exploding
        baci = baci.rename({"i_iso_numeric": "i", "j_iso_numeric": "j"})

        # Cast keys to Utf8
        baci = baci.with_columns(
            [pl.col(c).cast(pl.Utf8) for c in ["t", "i", "j", "k"] if c in baci.columns]
        )
        renamed_avemfn = renamed_avemfn.with_columns(
            [pl.col(c).cast(pl.Utf8) for c in mfn_join_keys if c in renamed_avemfn.columns]
        )
        cleaned_expanded_pref = cleaned_expanded_pref.with_columns(
            [pl.col(c).cast(pl.Utf8) for c in pref_join_keys if c in cleaned_expanded_pref.columns]
        )

        logger.debug(f"Schema after casting/exploding - BACI: {baci.collect_schema()}")
        logger.debug(f"Schema after casting - MFN: {renamed_avemfn.collect_schema()}")
        logger.debug(f"Schema after casting - Pref: {cleaned_expanded_pref.collect_schema()}")

    except Exception as e:
        logger.error(f"Failed to cast/explode join keys: {e}", exc_info=True)
        raise

    # 1. Left join BACI (exploded) with MFN tariffs
    logger.debug(f"Joining BACI with MFN tariffs on {mfn_join_keys}.")
    joined_mfn = baci.join(renamed_avemfn, on=mfn_join_keys, how="left")
    logger.debug(f"Schema after BACI-MFN join: {joined_mfn.collect_schema()}")

    # 2. Left join the result with Cleaned/Expanded Preferential tariffs
    logger.debug(
        f"Joining intermediate result with cleaned/expanded Preferential tariffs on {pref_join_keys}."
    )
    joined_all = joined_mfn.join(cleaned_expanded_pref, on=pref_join_keys, how="left")
    logger.debug(f"Schema after joining with Pref tariffs: {joined_all.collect_schema()}")

    # 3. Calculate the final effective tariff
    # Coalesce needs numeric types. Cast tariff rates first, handling errors.
    logger.debug(
        "Calculating effective tariff rate using coalesce(pref_tariff_rate, mfn_tariff_rate)."
    )
    final_table = (
        joined_all.with_columns(
            pl.col("pref_tariff_rate").cast(pl.Float64, strict=False).alias("pref_numeric"),
            pl.col("mfn_tariff_rate").cast(pl.Float64, strict=False).alias("mfn_numeric"),
        )
        .with_columns(
            pl.coalesce(pl.col("pref_numeric"), pl.col("mfn_numeric")).alias(
                "effective_tariff_rate_numeric"  # Keep numeric version
            )
            # Optionally keep original string versions or cast numeric back to string if needed
            # .cast(pl.Utf8).alias("effective_tariff_rate")
        )
        .drop(["pref_numeric", "mfn_numeric"])
    )  # Drop intermediate numeric columns

    # Rename numeric effective rate to final name
    final_table = final_table.rename({"effective_tariff_rate_numeric": "effective_tariff_rate"})

    logger.info("Datasets joined successfully.")
    logger.debug(f"Schema after calculating effective tariff: {final_table.collect_schema()}")

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

    # Ensure the columns exist before selecting
    available_cols = joined_data.columns
    select_exprs = []  # Use expressions for safe selection

    # Core identifiers (should exist after join)
    select_exprs.extend(
        [
            pl.col("t").alias("Year"),  # t is Utf8
            pl.col("i").alias("Source"),  # i is Utf8 (exploded BACI reporter)
            pl.col("j").alias(
                "Target"
            ),  # j is Utf8 (exploded BACI importer / expanded Pref partner)
            pl.col("k").alias("HS_Code"),  # k is Utf8
        ]
    )

    # BACI Value/Quantity (check existence)
    if "q" in available_cols:
        select_exprs.append(pl.col("q").alias("Quantity"))
    else:
        logger.warning("Column 'q' (Quantity) not found in joined data. Skipping.")
    if "v" in available_cols:
        select_exprs.append(pl.col("v").alias("Value"))
    else:
        logger.warning("Column 'v' (Value) not found in joined data. Skipping.")

    # Tariff Rates (check existence)
    if "mfn_tariff_rate" in available_cols:
        # Keep original string rate for reference
        select_exprs.append(pl.col("mfn_tariff_rate").alias("mfn_tariff_rate_str"))
    if "pref_tariff_rate" in available_cols:
        # Keep original string rate for reference
        select_exprs.append(pl.col("pref_tariff_rate").alias("pref_tariff_rate_str"))
    if "effective_tariff_rate" in available_cols:
        # This is the coalesced numeric rate calculated in join_datasets
        select_exprs.append(pl.col("effective_tariff_rate"))  # Already named correctly
    else:
        # This should not happen if join_datasets worked
        logger.error("Column 'effective_tariff_rate' not found after join. Adding null.")
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias("effective_tariff_rate"))

    # Optional Min/Max Rates (check existence)
    if "mfn_min_tariff_rate" in available_cols:
        select_exprs.append(pl.col("mfn_min_tariff_rate"))
    if "mfn_max_tariff_rate" in available_cols:
        select_exprs.append(pl.col("mfn_max_tariff_rate"))
    if "pref_min_tariff_rate" in available_cols:
        select_exprs.append(pl.col("pref_min_tariff_rate"))
    if "pref_max_tariff_rate" in available_cols:
        select_exprs.append(pl.col("pref_max_tariff_rate"))

    # Optional Tariff Type (check existence)
    if "tariff_type" in available_cols:
        select_exprs.append(pl.col("tariff_type"))

    # Select the columns using the expressions
    final_unified_table = joined_data.select(select_exprs)

    logger.debug(
        f"Final table schema after selection/renaming: {final_unified_table.collect_schema()}"
    )
    return final_unified_table


# --- Main Execution Block Removed ---
# The run_matching_pipeline function and the if __name__ == "__main__": block
# have been moved and adapted into the new matching_chunked.py script
# to handle chunked processing and partitioned output.
