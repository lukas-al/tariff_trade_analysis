import glob
import os
import shutil
from pathlib import Path  # Add Path
from typing import Optional  # Add Optional

import duckdb
import polars as pl  # Add polars
from tqdm.auto import tqdm

# Local imports
from mpil_tariff_trade_analysis.utils.iso_remapping import (
    DEFAULT_BACI_COUNTRY_CODES_PATH,
    DEFAULT_WITS_COUNTRY_CODES_PATH,
    apply_iso_mapping,  # <-- New function
    create_iso_mapping_table,  # <-- New function
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


def remap_baci_country_codes(
    input_path: str | Path,  # Directory of input parquet files
    output_path: str | Path,  # Path for the output parquet file
    # Reference data paths can be overridden if needed
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
) -> Optional[Path]:
    """
    Loads processed BACI data, remaps exporter ('i') and importer ('j') country codes
    to ISO 3166-1 numeric using a vectorized join approach, handles group expansions
    (like EFTA) by exploding the resulting lists, and saves the final result.

    Args:
        input_path: Path to the directory containing the input BACI parquet files
                    (e.g., output of baci_to_parquet_incremental or aggregate_baci).
                    Assumes hive partitioning if applicable.
        output_path: Path to save the final remapped data as a single Parquet file.
                     The output will contain columns 'i_iso_numeric' and 'j_iso_numeric'.
        baci_codes_path: Optional path to BACI country code reference CSV.
        wits_codes_path: Optional path to WITS country code reference CSV.

    Returns:
        Path object to the output location if successful, otherwise None.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    logger.info(
        f"Starting vectorized country code remapping for BACI data from: {input_path}"
    )
    logger.info(f"Output will be saved to: {output_path}")

    try:
        # --- 1. Setup Output Path ---
        output_file = output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured parent directory exists for output file: {output_file.parent}")

        # --- 2. Load Input Data ---
        logger.info(f"Scanning input BACI dataset at: {input_path}...")
        # Use hive partitioning if the input is structured like output_folder/partition_col=val/*.parquet
        # Adjust source path pattern if necessary
        try:
            lf = pl.scan_parquet(input_path / "*/*.parquet", hive_partitioning=True)
            logger.info("Detected hive partitioning structure.")
        except Exception:  # Fallback if hive partitioning fails or not present
            logger.warning(
                "Could not detect hive partitioning, attempting to scan directory directly. "
                "Ensure input path points to the directory containing Parquet files."
            )
            lf = pl.scan_parquet(input_path)

        # --- 3. Get Unique Codes ---
        logger.info("Extracting unique country codes from 'i' and 'j' columns...")
        unique_i_codes = lf.select("i").unique().collect().get_column("i")
        unique_j_codes = lf.select("j").unique().collect().get_column("j")
        # Combine unique codes from both columns, ensuring final uniqueness
        all_unique_codes = pl.concat([unique_i_codes, unique_j_codes]).unique()
        logger.info(f"Found {len(all_unique_codes)} unique codes across 'i' and 'j'.")

        # --- 4. Create Mapping Table ---
        logger.info("Creating ISO mapping table...")
        mapping_df = create_iso_mapping_table(
            unique_codes=all_unique_codes,
            baci_codes_path=baci_codes_path,
            wits_codes_path=wits_codes_path,
            # Assuming default column names in reference files are correct
            # baci_code_col="country_code", baci_name_col="country_name",
            # wits_code_col="ISO3", wits_name_col="Country Name",
        )

        if mapping_df.is_empty():
            logger.error("ISO mapping table creation failed or resulted in an empty table.")
            return None

        # --- 5. Apply Mapping via Joins ---
        logger.info("Applying mapping to 'i' (exporter) column...")
        lf_mapped_i = apply_iso_mapping(
            lf=lf,
            code_column_name="i",
            mapping_df=mapping_df,
            output_list_col_name="i_iso_list",
            drop_original=True,
        )

        logger.info("Applying mapping to 'j' (importer) column...")
        lf_mapped_ij = apply_iso_mapping(
            lf=lf_mapped_i,
            code_column_name="j",
            mapping_df=mapping_df,
            output_list_col_name="j_iso_list",
            drop_original=True,
        )

        # --- 6. Handle Rows with Failed Mappings (Optional Filter) ---
        # Decide how to handle rows where either 'i' or 'j' failed to map
        # Option 1: Filter them out
        original_row_count = lf.select(pl.count()).collect()[0, 0]
        lf_filtered = lf_mapped_ij.filter(
            pl.col("i_iso_list").is_not_null() & pl.col("j_iso_list").is_not_null()
        )
        filtered_row_count = lf_filtered.select(pl.count()).collect()[0, 0]
        if original_row_count > filtered_row_count:
            logger.warning(
                f"Filtered out {original_row_count - filtered_row_count} rows due to failed country code mapping in 'i' or 'j'."
            )
        # Option 2: Keep them (they will have null in the list columns) - lf_filtered = lf_mapped_ij

        # --- 7. Explode Lists for Group Expansion ---
        logger.info("Exploding 'i_iso_list' for group expansion...")
        lf_exploded_i = lf_filtered.explode("i_iso_list")

        logger.info("Exploding 'j_iso_list' for group expansion...")
        lf_exploded_ij = lf_exploded_i.explode("j_iso_list")

        # --- 8. Rename Final Columns ---
        logger.info("Renaming final ISO numeric columns...")
        lf_renamed = lf_exploded_ij.rename(
            {"i_iso_list": "i_iso_numeric", "j_iso_list": "j_iso_numeric"}
        )

        # --- 9. Cast 't' (Year) Column ---
        # Cast 't' (year) column to String/Utf8 before saving if it exists
        if "t" in lf_renamed.columns:
            logger.info("Casting 't' (year) column to String/Utf8...")
            lf_final = lf_renamed.with_columns(pl.col("t").cast(pl.Utf8))
        else:
            logger.warning("Column 't' not found in the remapped DataFrame. Skipping cast.")
            lf_final = lf_renamed

        # --- 10. Save Result ---
        logger.info(f"Saving remapped and exploded data to: {output_file}...")
        # Collect the final LazyFrame and write to the output file path
        lf_final.collect().write_parquet(output_file)

        logger.info("BACI country code remapping completed successfully.")
        return output_file  # Return the path to the saved file

    except (
        pl.exceptions.ComputeError,
        pl.exceptions.PolarsError,
    ) as e:  # Catch broader Polars errors
        logger.exception(f"Polars error during BACI remapping (check paths/data/schema): {e}")
        return None
    except FileNotFoundError as e:
        logger.exception(
            f"File not found during BACI remapping (check input path '{input_path}'): {e}"
        )
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during BACI country code remapping: {e}")
        return None
