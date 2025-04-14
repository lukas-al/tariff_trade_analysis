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
        pref_groups_pd = pd.read_csv(file_path, encoding="iso-8859-1")

        # Aggregate partners into lists per region code
        pref_group_df_pd = (
            pref_groups_pd.groupby("RegionCode")
            .agg(lambda x: list(set(x)))  # Convert set to list for Polars compatibility
            .reset_index()
            .rename(columns={"RegionCode": "region_code", "Partner": "partner_list"})[
                ["region_code", "partner_list"]
            ]
        )

        # Convert to Polars LazyFrame
        pref_group_pl = pl.from_pandas(pref_group_df_pd).lazy()
        logger.info("Preferential group mapping loaded and processed.")
        logger.debug(f"Preferential Group Mapping Schema: {pref_group_pl.collect_schema()}")
        return pref_group_pl
    except Exception as e:
        logger.error(f"Error loading or processing preferential group mapping: {e}")
        raise


def rename_wits_mfn(df: pl.LazyFrame) -> pl.LazyFrame:
    """Renames columns in the WITS MFN tariff dataframe."""
    logger.debug("Renaming WITS MFN columns.")
    # !! ADJUST column names based on actual schema if needed !!
    return df.rename(
        {
            "year": "t",
            "reporter_country": "i",
            "product_code": "k",
            "tariff_rate": "mfn_tariff_rate",
            "min_rate": "mfn_min_tariff_rate",
            "max_rate": "mfn_max_tariff_rate",
            "tariff_type": "tariff_type",
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


def rename_wits_pref(df: pl.LazyFrame) -> pl.LazyFrame:
    """Renames columns in the WITS Preferential tariff dataframe."""
    logger.debug("Renaming WITS Preferential columns.")
    # !! ADJUST column names based on actual schema if needed !!
    return df.rename(
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
    # Ensure consistent data types for joining keys if necessary (example below)
    # pref_group_mapping = pref_group_mapping.with_columns(pl.col("region_code").cast(pl.Utf8))
    # renamed_avepref = renamed_avepref.with_columns(pl.col("j").cast(pl.Utf8))

    # Left join avepref with the group mapping
    joined_pref_mapping = renamed_avepref.join(
        pref_group_mapping,
        left_on="j",  # Partner code (can be group or individual)
        right_on="region_code",  # Group code from mapping
        how="left",
    )

    # Create the final partner list and explode
    expanded_pref = (
        joined_pref_mapping.with_columns(
            pl.when(pl.col("partner_list").is_not_null())
            .then(pl.col("partner_list"))  # Use the list from mapping
            .otherwise(pl.concat_list(pl.col("j")))  # Use original 'j' as a single-item list
            .alias("final_partner_list")
        )
        .explode(
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
    logger.info("Preferential tariffs expanded.")
    logger.debug(f"Expanded Preferential Tariff Schema: {expanded_pref.collect_schema()}")
    return expanded_pref


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
    # Define join keys
    mfn_join_keys = ["t", "i", "k"]
    pref_join_keys = ["t", "i", "j", "k"]

    # Ensure join key types are compatible (add casting if necessary)
    # Example:
    # baci = baci.with_columns([
    #     pl.col("t").cast(pl.Int64), pl.col("i").cast(pl.Int64),
    #     pl.col("j").cast(pl.Int64), pl.col("k").cast(pl.Utf8)
    # ])
    # renamed_avemfn = renamed_avemfn.with_columns(pl.col("k").cast(pl.Utf8))
    # expanded_pref = expanded_pref.with_columns(pl.col("k").cast(pl.Utf8))

    # 1. Left join BACI with MFN tariffs
    logger.debug("Joining BACI with MFN tariffs.")
    joined_mfn = baci.join(renamed_avemfn, on=mfn_join_keys, how="left")

    # 2. Left join the result with Expanded Preferential tariffs
    logger.debug("Joining intermediate result with expanded Preferential tariffs.")
    joined_all = joined_mfn.join(expanded_pref, on=pref_join_keys, how="left")

    # 3. Calculate the final effective tariff
    logger.debug("Calculating effective tariff rate using coalesce.")
    final_table = joined_all.with_columns(
        pl.coalesce(pl.col("pref_tariff_rate"), pl.col("mfn_tariff_rate")).alias(
            "effective_tariff_rate"
        )
    )
    logger.info("Datasets joined successfully.")
    logger.debug(f"Joined table schema: {final_table.collect_schema()}")
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
    # !! ADJUST column names 'v' and 'q' based on your actual BACI schema !!
    final_unified_table = joined_data.select(
        pl.col("t").alias("Year"),
        pl.col("i").alias("Source"),  # Reporter country code
        pl.col("j").alias("Target"),  # Partner country code (individual)
        pl.col("k").alias("HS_Code"),  # Product code (HS92)
        pl.col("q").alias("Quantity"),  # Assuming 'q' is Quantity in BACI
        pl.col("v").alias("Value"),  # Assuming 'v' is Value in BACI
        pl.col("mfn_tariff_rate"),
        pl.col("pref_tariff_rate"),
        pl.col("effective_tariff_rate"),  # The coalesced rate
        # Optionally add min/max rates if needed
        # pl.col("mfn_min_tariff_rate"),
        # pl.col("mfn_max_tariff_rate"),
        # pl.col("pref_min_tariff_rate"),
        # pl.col("pref_max_tariff_rate"),
    )
    logger.debug(f"Final table schema: {final_unified_table.collect_schema()}")
    return final_unified_table


# --- Main Execution ---


def run_matching_pipeline(
    baci_path: str | Path = DEFAULT_BACI_PATH,
    wits_mfn_path: str | Path = DEFAULT_WITS_MFN_PATH,
    wits_pref_path: str | Path = DEFAULT_WITS_PREF_PATH,
    pref_groups_path: str | Path = DEFAULT_PREF_GROUPS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    lazy: bool = True,
):
    """
    Runs the full data matching pipeline.

    Args:
        baci_path: Path to the BACI data.
        wits_mfn_path: Path to the WITS MFN tariff data.
        wits_pref_path: Path to the WITS Preferential tariff data.
        pref_groups_path: Path to the WITS preferential groups mapping CSV.
        output_path: Path to save the final unified Parquet file.
        lazy: If True, performs operations lazily until the final write.
              If False, collects intermediate results (uses more memory).
    """
    logger.info("Starting data matching pipeline...")

    # Load data
    logger.info(f"Loading BACI data from: {baci_path}")
    baci = pl.scan_parquet(baci_path)
    logger.info(f"Loading WITS MFN data from: {wits_mfn_path}")
    avemfn = pl.scan_parquet(wits_mfn_path)
    logger.info(f"Loading WITS Preferential data from: {wits_pref_path}")
    avepref = pl.scan_parquet(wits_pref_path)
    pref_group_mapping = load_pref_group_mapping(pref_groups_path)

    # Rename columns
    renamed_avemfn = rename_wits_mfn(avemfn)
    renamed_avepref = rename_wits_pref(avepref)

    # Expand preferential tariffs
    expanded_pref = expand_preferential_tariffs(renamed_avepref, pref_group_mapping)

    # Join datasets
    joined_data = join_datasets(baci, renamed_avemfn, expanded_pref)

    # Create final table structure
    final_unified_table = create_final_table(joined_data)

    # Execute and save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    logger.info(f"Saving final unified table to: {output_path}")
    if lazy:
        final_unified_table.sink_parquet(output_path)
        logger.info("Lazy execution complete. Data saved.")
    else:
        # Collect the result before writing (uses more memory)
        result_df = final_unified_table.collect()
        result_df.write_parquet(output_path)
        logger.info(f"Eager execution complete. Data saved. Shape: {result_df.shape}")

    logger.info("Data matching pipeline finished successfully.")


if __name__ == "__main__":
    # This block allows running the script directly
    # You might want to add argument parsing (e.g., using argparse)
    # to specify input/output paths from the command line.
    logger.info("Running matching script directly.")
    run_matching_pipeline()
