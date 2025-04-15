# src/mpil_tariff_trade_analysis/etl/matching_pandas.py

import logging
from pathlib import Path

import numpy as np  # Often useful with pandas
import pandas as pd

# Assuming your logging setup is accessible via this import path
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration ---
# Keep the same configuration as the Polars version
DEFAULT_BACI_PATH = "data/final/BACI_HS92_V202501"
DEFAULT_WITS_MFN_PATH = "data/final/WITS_AVEMFN.parquet"
DEFAULT_WITS_PREF_PATH = "data/final/WITS_AVEPref.parquet"
DEFAULT_PREF_GROUPS_PATH = "data/raw/WITS_pref_groups/WITS_pref_groups.csv"
DEFAULT_OUTPUT_PATH = "data/final/unified_trade_tariff_pandas.parquet"  # Changed output name


# --- Helper Functions (Pandas Implementation) ---


def load_pref_group_mapping_pd(file_path: str | Path) -> pd.DataFrame:
    """
    Loads and processes the WITS preferential group mapping file using Pandas.

    Args:
        file_path: Path to the WITS preferential groups CSV file.

    Returns:
        A Pandas DataFrame containing the mapping from region codes to lists of partner countries.
    """
    logger.info(f"Loading preferential group mapping from: {file_path} (Pandas)")
    try:
        # Load the raw CSV mapping file using Pandas
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
        logger.info("Preferential group mapping loaded and processed (Pandas).")
        logger.debug(f"Preferential Group Mapping Schema (dtypes):\n{pref_group_df_pd.dtypes}")
        logger.debug(f"Preferential Group Mapping Head (Pandas):\n{pref_group_df_pd.head()}")

        return pref_group_df_pd
    except Exception as e:
        logger.error(f"Error loading or processing preferential group mapping (Pandas): {e}")
        raise


def rename_wits_mfn_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns in the WITS MFN tariff dataframe (Pandas)."""
    logger.debug("Renaming WITS MFN columns (Pandas).")
    try:
        renamed_df = df.rename(
            columns={
                "year": "t",
                "reporter_country": "i",
                "product_code": "k",
                "tariff_rate": "mfn_tariff_rate",
                "min_rate": "mfn_min_tariff_rate",
                "max_rate": "mfn_max_tariff_rate",
                "tariff_type": "tariff_type",
            }
        )
        # Select the desired columns
        selected_df = renamed_df[
            [
                "t",
                "i",
                "k",
                "mfn_tariff_rate",
                "mfn_min_tariff_rate",
                "mfn_max_tariff_rate",
                "tariff_type",
            ]
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning later
        logger.debug(f"Renamed WITS MFN Schema (dtypes):\n{selected_df.dtypes}")
        return selected_df
    except Exception as e:
        logger.error(f"Error renaming WITS MFN columns (Pandas): {e}")
        raise


def rename_wits_pref_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns in the WITS Preferential tariff dataframe (Pandas)."""
    logger.debug("Renaming WITS Preferential columns (Pandas).")
    try:
        renamed_df = df.rename(
            columns={
                "year": "t",
                "reporter_country": "i",
                "partner_country": "j",
                "product_code": "k",
                "tariff_rate": "pref_tariff_rate",
                "min_rate": "pref_min_tariff_rate",
                "max_rate": "pref_max_tariff_rate",
            }
        )
        # Select the desired columns
        selected_df = renamed_df[
            ["t", "i", "j", "k", "pref_tariff_rate", "pref_min_tariff_rate", "pref_max_tariff_rate"]
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning later
        logger.debug(f"Renamed WITS Preferential Schema (dtypes):\n{selected_df.dtypes}")
        return selected_df
    except Exception as e:
        logger.error(f"Error renaming WITS Preferential columns (Pandas): {e}")
        raise


def expand_preferential_tariffs_pd(
    renamed_avepref: pd.DataFrame, pref_group_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Expands WITS preferential tariff data using Pandas.

    Args:
        renamed_avepref: DataFrame of renamed WITS preferential tariffs.
        pref_group_mapping: DataFrame mapping region codes to partner lists.

    Returns:
        A DataFrame with preferential tariffs expanded to individual partner countries.
    """
    logger.info("Expanding preferential tariff partner groups (Pandas).")
    logger.debug(
        f"Input renamed_avepref shape: {renamed_avepref.shape}, dtypes:\n{renamed_avepref.dtypes}"
    )
    logger.debug(
        f"Input pref_group_mapping shape: {pref_group_mapping.shape}, dtypes:\n{pref_group_mapping.dtypes}"
    )

    # --- Ensure join keys are compatible types (e.g., string) ---
    logger.debug("Casting join keys to string type for merge.")
    try:
        renamed_avepref["j"] = renamed_avepref["j"].astype(str)
        pref_group_mapping["region_code"] = pref_group_mapping["region_code"].astype(str)
    except Exception as e:
        logger.error(f"Failed to cast join keys to string (Pandas): {e}")
        raise

    # --- Left merge avepref with the group mapping ---
    logger.debug(
        "Attempting left merge: renamed_avepref with pref_group_mapping on 'j' == 'region_code'."
    )
    try:
        # Using validate='m:1' assuming each 'j' in avepref matches at most one 'region_code'
        joined_pref_mapping = pd.merge(
            renamed_avepref,
            pref_group_mapping,
            left_on="j",
            right_on="region_code",
            how="left",
            validate="many_to_one",
        )
        logger.debug(f"Merge completed. Shape after merge: {joined_pref_mapping.shape}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Head after merge (Pandas):\n{joined_pref_mapping.head()}")
    except Exception as e:
        logger.error(f"Error during merge operation (Pandas): {e}", exc_info=True)
        raise

    # --- Create the final partner list ---
    logger.debug("Creating final partner list using apply/where logic.")
    try:
        # Create a boolean mask for rows where 'partner_list' is usable (not NaN and has items)
        # .isna() checks for NaN. apply checks for list type and length.
        mask = joined_pref_mapping["partner_list"].notna() & joined_pref_mapping[
            "partner_list"
        ].apply(lambda x: isinstance(x, list) and len(x) > 0)

        # Use np.where for conditional assignment (often faster than apply row-wise)
        # If mask is True, use 'partner_list'. Otherwise, create a list containing the original 'j'.
        joined_pref_mapping["final_partner_list"] = np.where(
            mask,
            joined_pref_mapping["partner_list"],
            joined_pref_mapping["j"].apply(lambda x: [x]),  # Ensure the 'else' case is also a list
        )
        logger.debug("Final partner list column created ('final_partner_list').")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Head before explode (showing 'final_partner_list', Pandas):\n{joined_pref_mapping[['t', 'i', 'j', 'k', 'partner_list', 'final_partner_list']].head()}"
            )
    except Exception as e:
        logger.error(f"Error creating final partner list (Pandas): {e}", exc_info=True)
        raise

    # --- Explode the final partner list ---
    logger.debug("Attempting to explode 'final_partner_list'.")
    try:
        # Check memory usage before potentially large explode
        mem_usage_before = joined_pref_mapping.memory_usage(deep=True).sum() / (1024**2)  # MB
        logger.debug(f"Memory usage before explode: {mem_usage_before:.2f} MB")

        expanded_pref = joined_pref_mapping.explode("final_partner_list")

        mem_usage_after = expanded_pref.memory_usage(deep=True).sum() / (1024**2)  # MB
        logger.info(f"Explode operation completed. Shape after explode: {expanded_pref.shape}")
        logger.debug(f"Memory usage after explode: {mem_usage_after:.2f} MB")

        # Rename the exploded column and select final columns
        expanded_pref = expanded_pref.rename(columns={"final_partner_list": "j_individual"})
        expanded_pref = expanded_pref[
            [
                "t",
                "i",
                "j_individual",  # This is the exploded partner code
                "k",
                "pref_tariff_rate",
                "pref_min_tariff_rate",
                "pref_max_tariff_rate",
            ]
        ].rename(columns={"j_individual": "j"})  # Rename back to 'j' for consistency

        logger.debug(f"Schema AFTER explode and selection (dtypes):\n{expanded_pref.dtypes}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Head AFTER explode (Pandas):\n{expanded_pref.head()}")

        return expanded_pref.copy()  # Return a copy to be safe

    except MemoryError:
        logger.error(
            "CRITICAL: MemoryError occurred during explode operation (Pandas). The dataset is likely too large to fit in memory after exploding.",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(f"Error occurred during the explode operation (Pandas): {e}", exc_info=True)
        raise


def join_datasets_pd(
    baci: pd.DataFrame, renamed_avemfn: pd.DataFrame, expanded_pref: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins BACI, MFN, and expanded Preferential tariff data using Pandas.

    Args:
        baci: DataFrame of BACI trade data.
        renamed_avemfn: DataFrame of renamed WITS MFN tariffs.
        expanded_pref: DataFrame of expanded WITS preferential tariffs.

    Returns:
        A DataFrame containing the joined data with an 'effective_tariff_rate' column.
    """
    logger.info("Joining BACI, MFN, and Preferential datasets (Pandas).")
    logger.debug(f"Input BACI shape: {baci.shape}, dtypes:\n{baci.dtypes}")
    logger.debug(
        f"Input renamed_avemfn shape: {renamed_avemfn.shape}, dtypes:\n{renamed_avemfn.dtypes}"
    )
    logger.debug(
        f"Input expanded_pref shape: {expanded_pref.shape}, dtypes:\n{expanded_pref.dtypes}"
    )

    # Define join keys
    mfn_join_keys = ["t", "i", "k"]
    pref_join_keys = ["t", "i", "j", "k"]

    # --- Ensure join key types are compatible ---
    logger.debug("Casting join key types for final merges (Pandas).")
    try:
        # Example: Cast keys to common types (int for t, str for others)
        # Adjust based on actual data needs
        common_key_types = {"t": "int64", "i": "str", "j": "str", "k": "str"}
        for df_ref, keys in [
            (baci, pref_join_keys),
            (renamed_avemfn, mfn_join_keys),
            (expanded_pref, pref_join_keys),
        ]:
            # Use a temporary variable to avoid modifying the original DataFrame if it's passed multiple times
            df = df_ref
            for key in keys:
                if key in df.columns and key in common_key_types:
                    # Check if casting is actually needed to avoid unnecessary operations
                    current_type = str(df[key].dtype)
                    target_type = common_key_types[key]
                    # Handle potential variations like 'Int64' (nullable int) vs 'int64'
                    if target_type == "int64" and current_type.lower().startswith("int"):
                        pass  # Already an integer type
                    elif target_type == "str" and current_type == "object":
                        # Assume object might need casting if target is str, log it
                        logger.debug(
                            f"Casting {key} in DataFrame from {current_type} to {target_type}"
                        )
                        df[key] = df[key].astype(target_type)
                    elif current_type != target_type:
                        logger.debug(
                            f"Casting {key} in DataFrame from {current_type} to {target_type}"
                        )
                        df[key] = df[key].astype(target_type)

        logger.debug(f"Dtypes after casting - BACI:\n{baci.dtypes}")
        logger.debug(f"Dtypes after casting - MFN:\n{renamed_avemfn.dtypes}")
        logger.debug(f"Dtypes after casting - Pref:\n{expanded_pref.dtypes}")
    except Exception as e:
        logger.error(f"Failed to cast join keys for final merges (Pandas): {e}")
        raise

    # 1. Left merge BACI with MFN tariffs
    logger.debug(f"Merging BACI with MFN tariffs on {mfn_join_keys}.")
    try:
        joined_mfn = pd.merge(baci, renamed_avemfn, on=mfn_join_keys, how="left")
        logger.debug(f"Shape after BACI-MFN merge: {joined_mfn.shape}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Head after BACI-MFN merge (Pandas):\n{joined_mfn.head()}")
    except Exception as e:
        logger.error(f"Error during BACI-MFN merge (Pandas): {e}", exc_info=True)
        raise

    # 2. Left merge the result with Expanded Preferential tariffs
    logger.debug(
        f"Merging intermediate result with expanded Preferential tariffs on {pref_join_keys}."
    )
    try:
        joined_all = pd.merge(
            joined_mfn, expanded_pref, on=pref_join_keys, how="left", suffixes=("_mfn", "_pref")
        )  # Add suffixes if any non-key columns overlap besides tariffs
        logger.debug(f"Shape after joining with Pref tariffs: {joined_all.shape}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Head after joining with Pref tariffs (Pandas):\n{joined_all.head()}")
    except Exception as e:
        logger.error(f"Error during merge with Pref tariffs (Pandas): {e}", exc_info=True)
        raise

    # 3. Calculate the final effective tariff (Pandas equivalent of coalesce)
    logger.debug("Calculating effective tariff rate using fillna.")
    try:
        # Use fillna: fill NaNs in pref_tariff_rate with values from mfn_tariff_rate
        joined_all["effective_tariff_rate"] = joined_all["pref_tariff_rate"].fillna(
            joined_all["mfn_tariff_rate"]
        )
        # Alternative: combine_first (handles NaNs in both)
        # joined_all['effective_tariff_rate'] = joined_all['pref_tariff_rate'].combine_first(joined_all['mfn_tariff_rate'])

        logger.info("Datasets joined successfully (Pandas).")
        logger.debug(f"Schema after calculating effective tariff (dtypes):\n{joined_all.dtypes}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Head after calculating effective tariff (Pandas):\n{joined_all[['t', 'i', 'j', 'k', 'mfn_tariff_rate', 'pref_tariff_rate', 'effective_tariff_rate']].head()}"
            )
        return joined_all
    except Exception as e:
        logger.error(f"Error calculating effective tariff (Pandas): {e}", exc_info=True)
        raise


def create_final_table_pd(joined_data: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and renames columns for the final unified table (Pandas).

    Args:
        joined_data: DataFrame containing the joined BACI and WITS data.

    Returns:
        A DataFrame representing the final unified table structure.
    """
    logger.info("Selecting and renaming columns for the final output table (Pandas).")
    logger.debug(f"Input shape for final selection: {joined_data.shape}")
    logger.debug(f"Input columns: {joined_data.columns.tolist()}")

    # Define columns to select and rename in the desired final order
    rename_map = {
        "t": "Year",
        "i": "Source",
        "j": "Target",
        "k": "HS_Code",
        # Add BACI value/quantity if they exist (adjust names 'v', 'q' if needed)
        "q": "Quantity",
        "v": "Value",
        # Keep tariff columns as they are (or rename if desired)
        "mfn_tariff_rate": "mfn_tariff_rate",
        "pref_tariff_rate": "pref_tariff_rate",
        "effective_tariff_rate": "effective_tariff_rate",
        # Optionally include other columns like min/max rates or tariff_type
        "mfn_min_tariff_rate": "mfn_min_tariff_rate",
        "tariff_type": "tariff_type",
    }

    # Filter rename_map to only include columns present in joined_data
    # Keep the original order defined in rename_map
    final_rename_map = {old: new for old, new in rename_map.items() if old in joined_data.columns}
    missing_cols = set(rename_map.keys()) - set(joined_data.columns)
    if missing_cols:
        logger.warning(f"Columns not found in joined data and will be skipped: {missing_cols}")

    # Select and rename
    try:
        # Select columns in the order they appear in final_rename_map keys
        ordered_cols = list(final_rename_map.keys())
        final_unified_table = joined_data[ordered_cols].rename(columns=final_rename_map)

        logger.debug(f"Final table shape after selection/renaming: {final_unified_table.shape}")
        logger.debug(f"Final table schema (dtypes):\n{final_unified_table.dtypes}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Final table head (Pandas):\n{final_unified_table.head()}")
        return final_unified_table
    except Exception as e:
        logger.error(f"Error during final column selection/renaming (Pandas): {e}", exc_info=True)
        raise


# --- Main Execution (Pandas) ---


def run_matching_pipeline_pd(
    baci_path: str | Path = DEFAULT_BACI_PATH,
    wits_mfn_path: str | Path = DEFAULT_WITS_MFN_PATH,
    wits_pref_path: str | Path = DEFAULT_WITS_PREF_PATH,
    pref_groups_path: str | Path = DEFAULT_PREF_GROUPS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
):
    """
    Runs the full data matching pipeline using Pandas.

    Args:
        baci_path: Path to the BACI data (Parquet).
        wits_mfn_path: Path to the WITS MFN tariff data (Parquet).
        wits_pref_path: Path to the WITS Preferential tariff data (Parquet).
        pref_groups_path: Path to the WITS preferential groups mapping CSV.
        output_path: Path to save the final unified Parquet file.
    """
    logger.info("Starting data matching pipeline (Pandas)...")

    # Load data using Pandas
    try:
        logger.info(f"Loading BACI data from: {baci_path}")
        # Specify columns and types if known to save memory
        # Example: baci_pd = pd.read_parquet(baci_path, columns=['t', 'i', 'j', 'k', 'v', 'q'])
        baci_pd = pd.read_parquet(baci_path)
        logger.debug(
            f"Initial BACI shape: {baci_pd.shape}, Memory: {baci_pd.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        logger.info(f"Loading WITS MFN data from: {wits_mfn_path}")
        avemfn_pd = pd.read_parquet(wits_mfn_path)
        logger.debug(
            f"Initial WITS MFN shape: {avemfn_pd.shape}, Memory: {avemfn_pd.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        logger.info(f"Loading WITS Preferential data from: {wits_pref_path}")
        avepref_pd = pd.read_parquet(wits_pref_path)
        logger.debug(
            f"Initial WITS Pref shape: {avepref_pd.shape}, Memory: {avepref_pd.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        pref_group_mapping_pd = load_pref_group_mapping_pd(pref_groups_path)  # Logging inside

    except Exception as e:
        logger.critical(f"Failed to load initial data (Pandas): {e}", exc_info=True)
        raise

    # Rename columns
    renamed_avemfn_pd = rename_wits_mfn_pd(avemfn_pd)  # Logging inside
    # Free up memory if original df is no longer needed
    del avemfn_pd
    logger.debug("Deleted original avemfn_pd dataframe.")

    renamed_avepref_pd = rename_wits_pref_pd(avepref_pd)  # Logging inside
    del avepref_pd
    logger.debug("Deleted original avepref_pd dataframe.")

    # Expand preferential tariffs
    # This is the potentially memory-intensive step
    expanded_pref_pd = expand_preferential_tariffs_pd(
        renamed_avepref_pd, pref_group_mapping_pd
    )  # Logging inside
    # Free up memory
    del renamed_avepref_pd
    del pref_group_mapping_pd
    logger.debug("Deleted renamed_avepref_pd and pref_group_mapping_pd dataframes.")

    # Join datasets
    joined_data_pd = join_datasets_pd(
        baci_pd, renamed_avemfn_pd, expanded_pref_pd
    )  # Logging inside
    # Free up memory
    del baci_pd
    del renamed_avemfn_pd
    del expanded_pref_pd
    logger.debug("Deleted baci_pd, renamed_avemfn_pd, and expanded_pref_pd dataframes.")

    # Create final table structure
    final_unified_table_pd = create_final_table_pd(joined_data_pd)  # Logging inside
    del joined_data_pd
    logger.debug("Deleted joined_data_pd dataframe.")

    # Save the final result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    logger.info(f"Saving final unified table to: {output_path} (Pandas)")
    try:
        # Consider using pyarrow engine for potentially better performance/memory
        final_unified_table_pd.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(
            f"Pandas execution complete. Data saved. Final shape: {final_unified_table_pd.shape}"
        )
    except Exception as e:
        logger.critical(f"Pipeline failed during final save operation (Pandas): {e}", exc_info=True)
        raise

    logger.info("Data matching pipeline (Pandas) finished successfully.")


if __name__ == "__main__":
    # Setup logging
    import logging

    from mpil_tariff_trade_analysis.utils.logging_config import setup_logging

    setup_logging(log_level=logging.DEBUG)  # Use DEBUG for detailed logs
    logger = get_logger(__name__)  # Re-get logger after setup

    logger.info("Running matching_pandas.py script directly.")
    try:
        run_matching_pipeline_pd()
    except Exception as e:
        logger.critical(f"Pipeline execution failed in __main__ (Pandas): {e}", exc_info=True)
        import sys

        sys.exit(1)
