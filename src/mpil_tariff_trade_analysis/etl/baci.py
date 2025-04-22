import glob
import os
from pathlib import Path  # Add Path
from typing import Optional # Add Optional

import duckdb
import polars as pl      # Add polars
from tqdm.auto import tqdm

# Local imports
from mpil_tariff_trade_analysis.utils.iso_remapping import (
    apply_country_code_mapping,
    create_country_code_mapping_df,
    DEFAULT_BACI_COUNTRY_CODES_PATH,
    DEFAULT_WITS_COUNTRY_CODES_PATH,
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
        os.remove(output_file)

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
    Loads processed BACI data (e.g., partitioned parquet), remaps country codes ('i', 'j')
    to ISO numeric using the vectorized approach, and saves the result.

    Args:
        input_path: Path to the directory containing the input BACI parquet files
                    (e.g., output of baci_to_parquet_incremental or aggregate_baci).
                    Assumes a structure readable by pl.scan_parquet (e.g., single file,
                    directory of files, or hive-partitioned directory).
        output_path: Path to save the remapped data (e.g., a new directory or single file).
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
        # Ensure output directory exists if saving to a directory
        if output_path.suffix == "" or "/" in str(output_path.name): # Heuristic for directory path
             output_path.mkdir(parents=True, exist_ok=True)
             logger.debug(f"Ensured output directory exists: {output_path}")
        else:
             output_path.parent.mkdir(parents=True, exist_ok=True)
             logger.debug(f"Ensured parent directory exists for output file: {output_path.parent}")


        # Load the dataset as a LazyFrame
        # Use scan_parquet which handles directories, single files, and globs
        logger.info(f"Scanning input BACI dataset at: {input_path}...")
        lf = pl.scan_parquet(input_path) # Polars handles partitioning automatically if present

        # Define country code columns
        country_cols = ["i", "j"]
        logger.debug(f"Columns to remap: {country_cols}")

        # Generate the mapping DataFrame
        logger.info("Generating country code mapping...")
        # Pass reference paths explicitly
        mapping_df = create_country_code_mapping_df(
            lf,
            country_cols,
            baci_codes_path=baci_codes_path,
            wits_codes_path=wits_codes_path
        )

        if mapping_df.height == 0 and mapping_df.select(pl.col("original_code")).is_empty().all():
             logger.error("Country mapping DataFrame is empty, likely due to reference file loading errors. Skipping remapping.")
             return None

        # Apply the mapping to 'i' and 'j' columns
        logger.info("Applying mapping to 'i' (exporter) column...")
        lf = apply_country_code_mapping(
            lf=lf,
            mapping_df=mapping_df,
            original_col_name="i",
            new_col_name="i", # Overwrite original column
            drop_original=True,
        )

        logger.info("Applying mapping to 'j' (importer) column...")
        lf = apply_country_code_mapping(
            lf=lf,
            mapping_df=mapping_df,
            original_col_name="j",
            new_col_name="j", # Overwrite original column
            drop_original=True,
        )

        # Save the result
        # Decide whether to save as a single file or partitioned based on output_path
        logger.info(f"Saving remapped data to: {output_path}...")
        # Use sink_parquet for lazy operations, especially for large data
        # This requires deciding on partitioning strategy if output is a directory
        if output_path.is_dir():
             logger.warning(f"Output path {output_path} is a directory. Saving potentially large dataset without partitioning. Consider using sink_parquet with partitioning if needed.")
             # For simplicity, collect and write if it's a directory, but warn.
             # For very large data, sink_parquet with partition_by is better.
             lf.collect().write_parquet(output_path / "remapped_baci_data.parquet") # Example filename
        else:
             # If output_path is a file, collect and write.
             lf.collect().write_parquet(output_path)

        logger.info("BACI country code remapping completed successfully.")
        return output_path

    except pl.exceptions.ComputeError as e:
         logger.exception(f"Polars ComputeError during BACI remapping (check paths/data): {e}")
         return None
    except Exception as e:
        logger.exception(f"Unexpected error during BACI country code remapping: {e}")
        return None
