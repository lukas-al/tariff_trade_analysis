import gc  # Garbage collector
import logging
from pathlib import Path

import polars as pl

# Assuming your logging setup is accessible
from ..utils.logging_config import get_logger, setup_logging

# Import the core logic functions and constants from matching.py
from .matching import (  # DEFAULT_OUTPUT_PATH # We'll define a new output dir
    DEFAULT_BACI_PATH,
    DEFAULT_PREF_GROUPS_PATH,
    DEFAULT_WITS_MFN_PATH,
    DEFAULT_WITS_PREF_PATH,
    create_final_table,
    expand_preferential_tariffs,
    join_datasets,
    load_pref_group_mapping,
    rename_wits_mfn,
    rename_wits_pref,
)

logger = get_logger(__name__)

# --- Configuration for Chunking ---
# Define the column name used for chunking IN THE SOURCE PARQUET FILES
SOURCE_CHUNK_COLUMN = "year"
# Define the column name for partitioning IN THE FINAL OUTPUT
PARTITION_COLUMN = "Year"

# Define a new default output DIRECTORY for the partitioned dataset
DEFAULT_CHUNKED_OUTPUT_DIR = "data/final/unified_trade_tariff_partitioned"


def get_unique_years(paths: list[str | Path], column_name: str) -> list:
    """
    Scans multiple Parquet files or directories to find unique values
    in the specified column without loading full data.

    Args:
        paths: A list of paths to Parquet files or directories.
        column_name: The name of the column to find unique values for (e.g., 'year').

    Returns:
        A sorted list of unique values found in the column across all files.
    """
    unique_values = set()
    logger.info(f"Scanning for unique values in column '{column_name}' across paths: {paths}")

    for data_path in paths:
        try:
            # Scan the dataset, select only the unique values of the chunk column
            # Use low_memory=True for potentially very wide files, though less critical here
            lf = pl.scan_parquet(data_path)  # Scans all files if it's a directory
            if column_name in lf.columns:
                values = (
                    lf.select(pl.col(column_name).unique())
                    .collect(engine="streaming")
                    .get_column(column_name)
                    .to_list()
                )
                unique_values.update(values)
                logger.debug(f"Found {len(values)} unique '{column_name}' values in {data_path}.")
            else:
                logger.warning(
                    f"Chunk column '{column_name}' not found in schema for path: {data_path}"
                )
        except Exception as e:
            logger.warning(
                f"Could not read column '{column_name}' from {data_path}: {e}. Skipping this source for year detection."
            )

    if not unique_values:
        raise ValueError(
            f"No unique values found for column '{column_name}' in any of the provided paths. Cannot proceed."
        )

    # Convert to appropriate type if needed (e.g., int) and sort
    # Assuming years are integers here
    try:
        sorted_values = sorted([int(v) for v in unique_values if v is not None])
    except ValueError:
        logger.warning(
            f"Could not convert all values in '{column_name}' to integers. Sorting as strings."
        )
        sorted_values = sorted([str(v) for v in unique_values if v is not None])

    logger.info(
        f"Found {len(sorted_values)} unique values for chunking: {sorted_values[:5]}...{sorted_values[-5:]}"
    )
    return sorted_values


def run_chunked_matching_pipeline(
    baci_path: str | Path = DEFAULT_BACI_PATH,
    wits_mfn_path: str | Path = DEFAULT_WITS_MFN_PATH,
    wits_pref_path: str | Path = DEFAULT_WITS_PREF_PATH,
    pref_groups_path: str | Path = DEFAULT_PREF_GROUPS_PATH,
    output_dir: str | Path = DEFAULT_CHUNKED_OUTPUT_DIR,
    source_chunk_column: str = SOURCE_CHUNK_COLUMN,
    partition_column: str = PARTITION_COLUMN,
):
    """
    Processes the matching pipeline by chunking data based on a column (e.g., year)
    and writing to a Hive-partitioned Parquet dataset.

    Args:
        baci_path: Path to the BACI data (file or directory).
        wits_mfn_path: Path to the WITS MFN tariff data (file or directory).
        wits_pref_path: Path to the WITS Preferential tariff data (file or directory).
        pref_groups_path: Path to the WITS preferential groups mapping CSV.
        output_dir: The root directory to save the partitioned Parquet dataset.
        source_chunk_column: The column name in the source files used for filtering chunks.
        partition_column: The column name to use for partitioning the output dataset.
    """
    baci_path = Path(baci_path)
    wits_mfn_path = Path(wits_mfn_path)
    wits_pref_path = Path(wits_pref_path)
    pref_groups_path = Path(pref_groups_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting chunked matching pipeline.")
    logger.info(
        f"Source data paths: BACI='{baci_path}', MFN='{wits_mfn_path}', PREF='{wits_pref_path}'"
    )
    logger.info(f"Output directory (partitioned by '{partition_column}'): {output_dir}")
    logger.info(f"Chunking based on source column: '{source_chunk_column}'")

    # --- Load non-chunked data once ---
    # Pref group mapping is small, load it into memory for potentially faster access
    # during the expand step within the loop. Keep lazy if it's surprisingly large.
    logger.info(f"Loading preferential groups mapping from: {pref_groups_path}")
    try:
        # Using .collect() as it's likely small and used repeatedly
        pref_group_mapping = load_pref_group_mapping(pref_groups_path).collect()
        logger.info(
            f"Preferential group mapping loaded into memory (shape: {pref_group_mapping.shape})."
        )
    except Exception as e:
        logger.error(f"Failed to load preferential group mapping: {e}", exc_info=True)
        raise

    # --- Determine chunks (e.g., years) ---
    # Scan all relevant input datasets for the chunking column values
    chunk_values = get_unique_years([baci_path, wits_mfn_path, wits_pref_path], source_chunk_column)

    # --- Define Base LazyFrames (Scan Once) ---
    # These define HOW to read the data, filtering will be applied later per chunk
    logger.info("Defining base LazyFrames for input datasets...")
    baci_lazy_base = pl.scan_parquet(baci_path)
    wits_mfn_lazy_base = pl.scan_parquet(wits_mfn_path)
    wits_pref_lazy_base = pl.scan_parquet(wits_pref_path)

    # Check if the chunk column exists in the base lazy frames
    if source_chunk_column not in baci_lazy_base.columns:
        logger.warning(f"Chunk column '{source_chunk_column}' missing in BACI base scan.")
    if source_chunk_column not in wits_mfn_lazy_base.columns:
        logger.warning(f"Chunk column '{source_chunk_column}' missing in WITS MFN base scan.")
    if source_chunk_column not in wits_pref_lazy_base.columns:
        logger.warning(f"Chunk column '{source_chunk_column}' missing in WITS Pref base scan.")

    # --- Process each chunk ---
    for i, chunk_value in enumerate(chunk_values):
        logger.info(
            f"--- Processing Chunk {i + 1}/{len(chunk_values)}: {source_chunk_column} = {chunk_value} ---"
        )

        try:
            # Filter each base LazyFrame for the current chunk value
            # This operation is lazy and cheap.
            logger.debug(f"Applying filter: {source_chunk_column} == {chunk_value}")
            baci_chunk_lazy = baci_lazy_base.filter(pl.col(source_chunk_column) == chunk_value)
            wits_mfn_chunk_lazy = wits_mfn_lazy_base.filter(
                pl.col(source_chunk_column) == chunk_value
            )
            wits_pref_chunk_lazy = wits_pref_lazy_base.filter(
                pl.col(source_chunk_column) == chunk_value
            )

            # --- Apply core logic from matching.py to the filtered LazyFrames ---
            # Note: Pass the *filtered* lazy frames. Renaming happens within these steps.
            # The column 't' will be created here if source_chunk_column is 'year'.
            logger.debug("Renaming WITS MFN columns for chunk...")
            renamed_mfn_lazy = rename_wits_mfn(wits_mfn_chunk_lazy)

            logger.debug("Renaming WITS Pref columns for chunk...")
            renamed_pref_lazy = rename_wits_pref(wits_pref_chunk_lazy)

            logger.debug("Expanding preferential tariffs for chunk...")
            # Pass the collected pref_group_mapping DataFrame
            # Convert it back to LazyFrame just for this function call if the function expects Lazy
            # Or modify expand_preferential_tariffs to accept DataFrame or LazyFrame
            # Assuming expand_preferential_tariffs expects LazyFrame:
            expanded_pref_lazy = expand_preferential_tariffs(
                renamed_pref_lazy, pref_group_mapping.lazy()
            )

            logger.debug("Joining datasets for the chunk...")
            # This is the critical step where memory/disk usage is reduced
            joined_lazy = join_datasets(baci_chunk_lazy, renamed_mfn_lazy, expanded_pref_lazy)

            logger.debug("Creating final table structure for the chunk...")
            # This step renames 't' to 'Year' (our partition_column)
            final_table_lazy = create_final_table(joined_lazy)

            # --- Write the processed chunk to the partitioned dataset ---
            logger.info(
                f"Executing plan and writing chunk for {partition_column}={chunk_value} to {output_dir}"
            )

            # Check if the partition column exists before writing
            if partition_column not in final_table_lazy.columns:
                logger.error(
                    f"Partition column '{partition_column}' not found in the final schema for chunk {chunk_value}. Available columns: {final_table_lazy.columns}. Skipping write."
                )
                continue  # Skip to the next chunk

            # Use sink_parquet for efficient streaming write with partitioning
            final_table_lazy.sink_parquet(
                path=output_dir,
                partition_by=partition_column,
                # Ensure the partition value matches the data being written
                # Polars handles this automatically when partition_by is set
                # compression="zstd", # Optional: Choose compression
                # row_group_size=134217728, # Optional: Tune row group size (e.g., 128MB)
            )

            logger.info(f"Successfully wrote chunk for {partition_column}={chunk_value}.")

        except pl.exceptions.ComputeError as e:
            # Catch Polars computation errors specifically
            logger.error(f"Polars compute error processing chunk {chunk_value}: {e}", exc_info=True)
            logger.error(
                f"Query plan that failed:\n{final_table_lazy.explain(streaming=True)}"
            )  # Use streaming=True for explain
            # Decide whether to continue or stop
            # raise # Uncomment to stop on first compute error
        except Exception as e:
            logger.error(
                f"Failed to process or write chunk for {source_chunk_column}={chunk_value}: {e}",
                exc_info=True,
            )
            # Decide if you want to stop or continue with other chunks
            # raise # Uncomment to stop execution on first error
        finally:
            # Explicitly trigger garbage collection after each chunk
            # May help release memory, especially if complex objects were created
            gc.collect()
            logger.debug("Garbage collection triggered after chunk processing.")

    logger.info(
        f"--- Chunked matching pipeline finished. Partitioned data written to: {output_dir} ---"
    )


# --- Main Execution Example ---
if __name__ == "__main__":
    # Setup logging (adjust level as needed)
    # Make sure this is called *before* get_logger
    setup_logging(log_level=logging.INFO)
    # Re-get logger instance after setup, just in case
    logger = get_logger(__name__)

    logger.info("Running matching_chunked.py script...")

    # Define paths (use defaults or override here)
    # output_directory = "data/test_partitioned_output"
    output_directory = DEFAULT_CHUNKED_OUTPUT_DIR

    try:
        run_chunked_matching_pipeline(
            # You can override paths here if needed, e.g.:
            # baci_path="path/to/your/baci/data",
            output_dir=output_directory,
        )
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.critical(f"Chunked pipeline execution failed in __main__: {e}", exc_info=True)
        import sys

        sys.exit(1)
