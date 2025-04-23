import glob
import os
import shutil
from pathlib import Path  # Add Path
from typing import Optional  # Add Optional

import duckdb
import polars as pl  # Add polars
from tqdm.auto import tqdm

# Local imports
# Updated import: only need create_country_code_mapping_df
from mpil_tariff_trade_analysis.utils.iso_remapping import (
    DEFAULT_BACI_COUNTRY_CODES_PATH,
    DEFAULT_WITS_COUNTRY_CODES_PATH,
    create_country_code_mapping_df,  # <-- Import the updated function
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
    input_path: str | Path,
    output_path: str | Path,
    # Reference data paths can be overridden if needed
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
) -> Optional[Path]:
    """
    Loads processed BACI data, remaps exporter ('i') and importer ('j') country codes
    to ISO 3166-1 numeric using the unified mapping logic (including hardcodes and
    potential row explosion for groups like EFTA), and saves the result.

    Args:
        input_path: Path to the directory containing the input BACI parquet files
                    (e.g., output of baci_to_parquet_incremental or aggregate_baci).
                    Assumes a structure readable by pl.scan_parquet.
        output_path: Path to save the remapped data (e.g., a new directory or single file).
                     The output will contain columns like 'i_iso_numeric', 'j_iso_numeric'.
        baci_codes_path: Optional path to BACI country code reference CSV.
        wits_codes_path: Optional path to WITS country code reference CSV.

    Returns:
        Path object to the output location if successful, otherwise None.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    logger.info(f"Starting country code remapping for BACI data from: {input_path}")
    logger.info(f"Output will be saved to: {output_path}")

    try:
        # Ensure output directory exists
        # Check if output_path looks like a directory path or file path
        if output_path.suffix == "" or "/" in str(output_path.name) or output_path.is_dir():
            # Treat as directory path
            output_dir = output_path
            # Define a default filename if saving to a directory
            output_file = output_dir / "remapped_baci_iso_numeric.parquet"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured output directory exists: {output_dir}")
        else:
            # Treat as file path
            output_file = output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured parent directory exists for output file: {output_file.parent}")

        # Load the dataset as a LazyFrame
        logger.info(f"Scanning input BACI dataset at: {input_path}...")
        lf = pl.scan_parquet(input_path)

        # Define country code columns to remap
        country_cols_to_remap = ["i", "j"]
        logger.debug(f"Columns to remap: {country_cols_to_remap}")

        # Apply the unified mapping and explosion logic from iso_remapping
        # This function now handles the mapping, potential explosion, and column renaming/dropping
        remapped_lf = create_country_code_mapping_df(
            lf=lf,
            code_columns=country_cols_to_remap,
            # baci_codes_path=baci_codes_path,
            # wits_codes_path=wits_codes_path,
            # Assuming default column names in reference files are correct
            # baci_code_col="country_code", baci_name_col="country_name",
            # wits_code_col="ISO3", wits_name_col="Country Name",
            drop_original=True,  # Keep this consistent with the function's default or set explicitly
        )

        # Check if the remapping actually produced results
        if remapped_lf is None:
            logger.error("Country code remapping failed unexpectedly. Check logs in iso_remapping.")
            return None  # Remapping failed

        # Cast 't' (year) column to String/Utf8 before saving if it exists
        if "t" in remapped_lf.columns:
            logger.info("Casting 't' (year) column to String/Utf8...")
            remapped_lf = remapped_lf.with_columns(pl.col("t").cast(pl.Utf8))
        else:
            logger.warning("Column 't' not found in the remapped DataFrame. Skipping cast.")

        # Save the result
        logger.info(f"Saving remapped data to: {output_file}...")
        # Collect the LazyFrame and write to the determined output file path
        remapped_lf.collect().write_parquet(output_file)

        logger.info("BACI country code remapping completed successfully.")
        return output_file  # Return the path to the saved file

    except pl.exceptions.ComputeError as e:
        logger.exception(f"Polars ComputeError during BACI remapping (check paths/data): {e}")
        return None
    except FileNotFoundError as e:
        logger.exception(
            f"File not found during BACI remapping (check input path '{input_path}'): {e}"
        )
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during BACI country code remapping: {e}")
        return None
