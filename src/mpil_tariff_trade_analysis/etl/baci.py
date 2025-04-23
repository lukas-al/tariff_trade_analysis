import glob
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any # Added List, Dict, Any

import duckdb
import polars as pl # Import polars
from tqdm.auto import tqdm

# Local imports
from mpil_tariff_trade_analysis.utils.iso_remapping import (
    DEFAULT_BACI_COUNTRY_CODES_PATH,
    DEFAULT_WITS_COUNTRY_CODES_PATH,
    # We need the underlying functions, not the old wrapper
    load_reference_map,
    create_iso_mapping_table,
    apply_iso_mapping,
)
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)


def baci_to_parquet(hs, release, input_folder="raw", output_folder="intermediate"):
    # Determine input path
    baci_folder = f"BACI_{hs}_V{release}"

    if input_folder is not None:
        input_path = os.path.join(input_folder, baci_folder)
    else:
        input_path = baci_folder

    # Determine output file
    if output_folder is not None:
        output_file = os.path.join(output_folder, f"{baci_folder}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Created output directory: {output_folder}")
    else:
        output_file = f"{baci_folder}"

    logger.info(f"Converting BACI files from {input_path} to {output_file}")

    # Compile all BACI tables into one table
    duckdb.sql(f"COPY( SELECT * FROM read_csv_auto('{input_path}/*.csv') ) TO '{output_file}'")

    # Report result
    output_location = output_folder if output_folder else "project root"
    logger.info(f"'{baci_folder}' successfully saved in '{output_location}'.")


def baci_to_parquet_incremental(hs, release, input_folder="raw", output_folder="intermediate"):
    baci_folder = f"BACI_{hs}_V{release}"
    input_path = os.path.join(input_folder, baci_folder)
    output_file = os.path.join(output_folder, f"{baci_folder}")

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")

    # Remove existing Parquet file if present.
    if os.path.exists(output_file):
        logger.warning(f"Removing existing output file: {output_file}")
        shutil.rmtree(output_file)
        # os.remove(output_file)

    logger.info("Running incremental conversion...")

    # Get all CSV files matching the pattern
    logger.info(f"TEST: {os.path.join(input_path, 'BACI*.csv')}")
    csv_files = glob.glob(os.path.join(input_path, "BACI*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to process")

    for i, csv_file in tqdm(enumerate(csv_files), desc="Processing CSV files"):
        file_basename = os.path.basename(csv_file)
        logger.debug(f"Processing file {i + 1}/{len(csv_files)}: {file_basename}")
        sql_query = f"""
            COPY (
                SELECT *, '{i}' AS partition_col
                FROM read_csv_auto('{csv_file}')
            )
            TO '{output_file}'
            (FORMAT 'parquet', COMPRESSION 'SNAPPY', PARTITION_BY (partition_col), APPEND);
        """
        duckdb.sql(sql_query)

    logger.info(f"'{baci_folder}.parquet' successfully saved in '{output_folder}'.")

    return output_file


def aggregate_baci(input, output, aggregation="country"):
    logger.info(f"Aggregating BACI data with aggregation level: {aggregation}")

    if aggregation == "2digit":
        logger.debug("Using 2-digit HS code aggregation")
        duckdb.sql(
            f"""
            COPY (
                SELECT t, i, j, k2, SUM(v) AS v, SUM(q) AS q
                FROM SELECT t, i, j, SUBSTRING(k, -6, 2) AS k2, v, q
                FROM read_parquet('{input}/*/*.parquet', hive_partitioning=true))
                GROUP BY t, i, j, k2
                ORDER BY t
            ) TO '{output}'
            """
        )

    elif aggregation == "4digit":
        logger.debug("Using 4-digit HS code aggregation")
        duckdb.sql(
            f"""
            COPY (
                SELECT t, i, j, k4, SUM(v) AS v, SUM(q) AS q
                FROM SELECT t, i, j, substring(k, -6, 4) AS k4, v, q 
                FROM read_parquet('{input}/*/*.parquet', hive_partitioning=true)
                GROUP BY t, i, j, k4
                ORDER BY t
            ) TO '{output}'
            """
        )

    else:
        logger.debug("Using country-level aggregation")
        duckdb.sql(
            f"""
            COPY (
                SELECT t, i, j, SUM(v) AS v, SUM(q) AS q
                FROM read_parquet('{input}/*/*.parquet', hive_partitioning=true)
                GROUP BY t, i, j
                ORDER BY t
            ) TO '{output}'
            """
        )

    logger.info(f"Aggregation complete. Results saved to {output}")


# --- Refactored remap_codes_and_explode ---
def remap_codes_and_explode(
    input_lf: pl.LazyFrame, # Changed from input_path
    # output_path: str | Path, # Removed - no longer writes file here
    code_columns_to_remap: List[str],
    output_column_names: List[str],
    year_column_name: Optional[str] = "t",
    # use_hive_partitioning: bool = True, # Removed - not relevant for LazyFrame input
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
    baci_ref_code_col: str = "country_code",
    baci_ref_name_col: str = "country_name",
    wits_ref_code_col: str = "ISO3", # WITS uses ISO3 for numeric codes in its ref file
    wits_ref_name_col: str = "Country Name",
    drop_original_code_columns: bool = True,
    filter_failed_mappings: bool = True,
) -> Optional[pl.LazyFrame]: # Changed return type
    """
    Remaps specified country code columns in a LazyFrame to ISO 3166-1 numeric codes,
    exploding rows where a single input code maps to multiple ISO codes.

    Operates entirely in-memory using Polars LazyFrames.

    Args:
        input_lf: The input Polars LazyFrame.
        code_columns_to_remap: List of column names containing the original country codes.
        output_column_names: List of desired output column names for the remapped ISO codes.
                             Must be the same length as code_columns_to_remap.
        year_column_name: Optional name of the year column for partitioning output (if writing later).
                          Not used for partitioning here, but kept for potential downstream use.
        baci_codes_path: Path to BACI country code reference CSV.
        wits_codes_path: Path to WITS country code reference CSV.
        baci_ref_code_col: Column name for codes in BACI reference.
        baci_ref_name_col: Column name for names in BACI reference.
        wits_ref_code_col: Column name for codes in WITS reference.
        wits_ref_name_col: Column name for names in WITS reference.
        drop_original_code_columns: If True, drop the original code columns after remapping.
        filter_failed_mappings: If True, filter out rows where any code column failed to map.

    Returns:
        A Polars LazyFrame with remapped and potentially exploded country codes,
        or None if an error occurs.
    """
    if len(code_columns_to_remap) != len(output_column_names):
        logger.error("Mismatch between number of input code columns and output column names.")
        raise ValueError("code_columns_to_remap and output_column_names must have the same length.")

    logger.info(
        f"Starting ISO code remapping for columns: {code_columns_to_remap} -> {output_column_names}"
    )

    try:
        # --- 1. Load Reference Data (remains the same) ---
        logger.debug("Loading reference country code maps...")
        baci_map = load_reference_map(baci_codes_path, baci_ref_code_col, baci_ref_name_col)
        wits_map = load_reference_map(wits_codes_path, wits_ref_code_col, wits_ref_name_col)
        logger.debug("Reference maps loaded.")

        # --- 2. Prepare Input LazyFrame ---
        # No need to scan from path, use input_lf directly
        lf = input_lf
        original_schema = lf.schema
        logger.debug(f"Input LazyFrame schema: {original_schema}")

        # --- 3. Create Mapping Table for Unique Codes ---
        unique_codes_list = []
        for col in code_columns_to_remap:
            unique_codes_list.append(lf.select(pl.col(col).unique()))

        # Combine unique codes from all specified columns
        unique_codes_combined_lf = pl.concat(unique_codes_list).unique()
        # The column name here doesn't strictly matter as create_iso_mapping_table works on the Series
        unique_codes_series = unique_codes_combined_lf.collect().to_series()
        logger.info(f"Found {len(unique_codes_series)} unique codes across columns to map.")

        mapping_df = create_iso_mapping_table(
            unique_codes=unique_codes_series,
            baci_codes_path=baci_codes_path,
            wits_codes_path=wits_codes_path,
            baci_code_col=baci_ref_code_col,
            baci_name_col=baci_ref_name_col,
            wits_code_col=wits_ref_code_col,
            wits_name_col=wits_ref_name_col,
        )
        mapping_lf = mapping_df.lazy() # Use lazy version for joins
        logger.debug(f"Created mapping table with {mapping_df.height} entries.")

        # --- 4. Apply Mapping and Explode ---
        current_lf = lf
        temp_output_cols = [] # Store intermediate list column names

        for i, original_col in enumerate(code_columns_to_remap):
            output_col = output_column_names[i]
            temp_list_col = f"_{output_col}_iso_list" # Temporary name for the list column
            temp_output_cols.append(temp_list_col)

            logger.debug(f"Applying mapping for column '{original_col}' -> '{temp_list_col}'")
            current_lf = apply_iso_mapping(
                lf=current_lf,
                code_column_name=original_col,
                mapping_df=mapping_lf, # Pass lazy mapping frame
                output_list_col_name=temp_list_col,
                drop_original=False, # Keep original for now if needed, or set based on param
            )

        # --- 5. Filter Failed Mappings (Optional) ---
        if filter_failed_mappings:
            filter_expr = pl.all_horizontal(
                pl.col(c).list.len() > 0 for c in temp_output_cols
            )
            rows_before = current_lf.select(pl.count()).collect()[0, 0]
            current_lf = current_lf.filter(filter_expr)
            rows_after = current_lf.select(pl.count()).collect()[0, 0]
            if rows_before > rows_after:
                 logger.warning(f"Filtered out {rows_before - rows_after} rows due to failed mappings.")
            else:
                 logger.info("No rows filtered due to failed mappings.")


        # --- 6. Explode Mapped Lists ---
        # Explode needs to happen sequentially or carefully managed if multiple columns explode
        # For simplicity, let's assume we explode one by one if needed.
        # If only one column needs exploding, it's simpler.
        # If multiple might explode, the cross-product can be large.
        logger.debug(f"Exploding list columns: {temp_output_cols}")
        if temp_output_cols:
             current_lf = current_lf.explode(temp_output_cols)
             logger.debug("Explosion complete.")


        # --- 7. Rename and Select Final Columns ---
        rename_dict = {temp_col: final_col for temp_col, final_col in zip(temp_output_cols, output_column_names)}
        final_lf = current_lf.rename(rename_dict)

        # Determine columns to keep
        final_cols = list(original_schema.keys()) # Start with original columns
        if drop_original_code_columns:
            final_cols = [c for c in final_cols if c not in code_columns_to_remap]

        # Add the new output columns (if they weren't replacing originals)
        for out_col in output_column_names:
             if out_col not in final_cols:
                 final_cols.append(out_col)

        # Ensure order or just select the necessary ones
        # We might have intermediate columns, so select explicitly
        select_cols = [col for col in final_lf.columns if col in final_cols or col in output_column_names]
        # Remove duplicates if any column was both original and output
        select_cols = list(dict.fromkeys(select_cols))


        final_lf = final_lf.select(select_cols)


        logger.info(f"Remapping and exploding complete. Final schema: {final_lf.schema}")

        # --- 8. Remove Writing Logic ---
        # logger.info(f"Writing remapped data to: {output_path}")
        # partition_by_col = year_column_name if use_hive_partitioning and year_column_name else None
        # final_lf.collect().write_parquet(
        #     output_path,
        #     # use_pyarrow=True, # Pyarrow often needed for partitioning
        #     # pyarrow_options={"partition_cols": [partition_by_col]} if partition_by_col else None,
        # )
        # logger.info("✅ Remapped data saved successfully.")
        # return Path(output_path) # Return Path object

        return final_lf # Return the final LazyFrame

    except Exception as e:
        logger.error(f"❌ Failed during code remapping and exploding: {e}", exc_info=True)
        return None


# --- remap_baci_country_codes needs adjustment if used ---
# This function now needs to accept the output LazyFrame from remap_codes_and_explode
# and potentially write it if that's its purpose.
# For now, we focus on the WITS pipeline usage.

def remap_baci_country_codes(
    input_path: str | Path,  # Keep input path for initial load if needed
    output_path: str | Path, # Keep output path for final save
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
) -> Optional[Path]:
    """
    Wrapper function to load BACI data, remap exporter ('i') and importer ('j')
    country codes to ISO 3166-1 numeric using the generic `remap_codes_and_explode`
    function, and save the result.

    Args:
        input_path: Path to the input BACI parquet dataset (directory or file).
        output_path: Path to save the final remapped data (single Parquet file).
        baci_codes_path: Optional path to BACI country code reference CSV.
        wits_codes_path: Optional path to WITS country code reference CSV.

    Returns:
        Path object to the output location if successful, otherwise None.
    """
    logger.info("Initiating BACI-specific country code remapping.")
    logger.info(f"Input: {input_path}, Output: {output_path}")

    try:
        # Load the initial BACI data
        # Assuming hive partitioning based on original function call context
        input_lf = pl.scan_parquet(input_path, hive_partitioning=True)

        remapped_lf = remap_codes_and_explode(
            input_lf=input_lf, # Pass the loaded LazyFrame
            # output_path=output_path, # Removed from call
            code_columns_to_remap=["i", "j"],
            output_column_names=["i", "j"], # Output columns replace input
            year_column_name="t",
            # use_hive_partitioning=True, # Removed from call
            baci_codes_path=baci_codes_path,
            wits_codes_path=wits_codes_path,
            drop_original_code_columns=True, # Originals 'i', 'j' are replaced
            filter_failed_mappings=True,
        )

        if remapped_lf is None:
            logger.error("Remapping function returned None.")
            return None

        # Write the final result
        logger.info(f"Writing remapped BACI data to: {output_path}")
        # Decide on partitioning for the output here if needed
        remapped_lf.collect().write_parquet(output_path)
        logger.info("✅ Remapped BACI data saved successfully.")
        return Path(output_path)

    except Exception as e:
        logger.error(f"❌ Failed during BACI remapping wrapper: {e}", exc_info=True)
        return None
