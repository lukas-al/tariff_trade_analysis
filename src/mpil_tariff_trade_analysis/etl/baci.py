import glob
import os

import duckdb
from tqdm.auto import tqdm

from mpil_tariff_trade_analysis.utils.logging_config import get_logger

# Set up logger for this module
LOGGER = get_logger(__name__)


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
            LOGGER.info(f"Created output directory: {output_folder}")
    else:
        output_file = f"{baci_folder}"

    LOGGER.info(f"Converting BACI files from {input_path} to {output_file}")

    # Compile all BACI tables into one table
    duckdb.sql(f"COPY( SELECT * FROM read_csv_auto('{input_path}/BACI*.csv') ) TO '{output_file}'")

    # Report result
    output_location = output_folder if output_folder else "project root"
    LOGGER.info(f"'{baci_folder}' successfully saved in '{output_location}'.")


def baci_to_parquet_incremental(hs, release, input_folder="raw", output_folder="final"):
    baci_folder = f"BACI_{hs}_V{release}"
    input_path = os.path.join(input_folder, baci_folder)
    output_file = os.path.join(output_folder, f"{baci_folder}")

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        LOGGER.info(f"Created output directory: {output_folder}")

    # Remove existing Parquet file if present.
    if os.path.exists(output_file):
        LOGGER.warning(f"Removing existing output file: {output_file}")
        os.remove(output_file)

    LOGGER.info("Running incremental conversion...")

    # Get all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(input_path, "BACI*.csv"))
    LOGGER.info(f"Found {len(csv_files)} CSV files to process")

    for i, csv_file in tqdm(enumerate(csv_files), desc="Processing CSV files"):
        file_basename = os.path.basename(csv_file)
        LOGGER.debug(f"Processing file {i + 1}/{len(csv_files)}: {file_basename}")
        sql_query = f"""
            COPY (
                SELECT *, '{i}' AS partition_col
                FROM read_csv_auto('{csv_file}')
            )
            TO '{output_file}'
            (FORMAT 'parquet', COMPRESSION 'SNAPPY', PARTITION_BY (partition_col), APPEND);
        """
        duckdb.sql(sql_query)

    LOGGER.info(f"'{baci_folder}.parquet' successfully saved in '{output_folder}'.")


def aggregate_baci(input, output, aggregation="country"):
    LOGGER.info(f"Aggregating BACI data with aggregation level: {aggregation}")

    if aggregation == "2digit":
        LOGGER.debug("Using 2-digit HS code aggregation")
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
        LOGGER.debug("Using 4-digit HS code aggregation")
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
        LOGGER.debug("Using country-level aggregation")
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

    LOGGER.info(f"Aggregation complete. Results saved to {output}")
