import gc
import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

# Import the core logic functions and constants from matching.py
from mpil_tariff_trade_analysis.etl.matching_logic import (  # DEFAULT_OUTPUT_PATH # We'll define a new output dir
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

# Assuming your logging setup is accessible
from mpil_tariff_trade_analysis.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# --- Configuration for Chunking ---
# Define the column name used for chunking AFTER initial renaming
# This will be the consistent name across all base frames.
CHUNK_COLUMN_NAME = "t"

# Define the column name for partitioning IN THE FINAL OUTPUT
PARTITION_COLUMN = "Year"  # This comes from create_final_table

# Define a new default output DIRECTORY for the partitioned dataset
DEFAULT_CHUNKED_OUTPUT_DIR = "data/final/unified_trade_tariff_partitioned"


def get_unique_chunk_values(paths: list[str | Path], column_name: str) -> list:
    """
    Scans multiple Parquet files or directories to find unique values
    in the specified column without loading full data.

    Args:
        paths: A list of paths to Parquet files or directories containing the data
               *before* any renaming relevant to this function.
        column_name: The name of the column to find unique values for (e.g., 't' AFTER renaming).
                     This function assumes the column exists with this name in the scanned files,
                     or that the scan operation itself includes the necessary renaming if done lazily.
                     *** IMPORTANT: For this strategy, we assume the renaming to 't' happens *before*
                     this function is effectively called on the data source. ***

    Returns:
        A sorted list of unique values found in the column across all files.
    """
    unique_values = set()
    logger.info(f"Scanning for unique values in column '{column_name}' across paths: {paths}")

    # This function now expects the column 't' to exist in the sources
    # because we perform the renaming on the base LazyFrames beforehand.
    for data_path in paths:
        try:
            lf = pl.scan_parquet(data_path)
            # Check if the *original* year column likely exists before attempting rename/scan
            # This is a bit indirect, we rely on the calling context having done the rename
            # A more robust check might involve reading schema first if paths are complex.
            if "year" in lf.columns:  # Check if original 'year' exists for potential rename
                # Assume the rename to 'column_name' (e.g., 't') happens before collect
                lf_renamed_scan = lf.rename({"year": column_name})  # Apply rename for this scan
            else:
                # If 'year' isn't there, assume 'column_name' already exists
                lf_renamed_scan = lf

            if column_name in lf_renamed_scan.columns:
                values = (
                    lf_renamed_scan.select(pl.col(column_name).unique())
                    .collect(engine="streaming")  # Use streaming engine
                    .get_column(column_name)
                    .to_list()
                )
                unique_values.update(values)
                logger.debug(f"Found {len(values)} unique '{column_name}' values in {data_path}.")
            else:
                logger.warning(
                    f"Chunk column '{column_name}' (or original 'year') not found in schema for path: {data_path}"
                )
        except Exception as e:
            logger.warning(
                f"Could not read column '{column_name}' from {data_path}: {e}. Skipping this source for chunk value detection."
            )

    if not unique_values:
        raise ValueError(
            f"No unique values found for column '{column_name}' in any of the provided paths. Cannot proceed."
        )

    # Convert to appropriate type if needed (e.g., int) and sort
    try:
        sorted_values = sorted([int(v) for v in unique_values if v is not None])
    except ValueError:
        logger.warning(
            f"Could not convert all values in '{column_name}' to integers. Sorting as strings."
        )
        sorted_values = sorted([str(v) for v in unique_values if v is not None])

    logger.info(
        f"Found {len(sorted_values)} unique chunk values: {sorted_values[:5]}...{sorted_values[-5:]}"
    )
    return sorted_values


def run_chunked_matching_pipeline(
    baci_path: str | Path = DEFAULT_BACI_PATH,
    wits_mfn_path: str | Path = DEFAULT_WITS_MFN_PATH,
    wits_pref_path: str | Path = DEFAULT_WITS_PREF_PATH,
    pref_groups_path: str | Path = DEFAULT_PREF_GROUPS_PATH,
    output_dir: str | Path = DEFAULT_CHUNKED_OUTPUT_DIR,
    chunk_column: str = CHUNK_COLUMN_NAME,  # Use 't'
    partition_column: str = PARTITION_COLUMN,  # Use 'Year'
):
    """
    Processes the matching pipeline by chunking data based on a column (e.g., 't')
    and writing to a Hive-partitioned Parquet dataset using PyArrow.
    Initial renaming of year columns to 't' is done upfront.

    Args:
        baci_path: Path to the BACI data (file or directory).
        wits_mfn_path: Path to the WITS MFN tariff data (file or directory).
        wits_pref_path: Path to the WITS Preferential tariff data (file or directory).
        pref_groups_path: Path to the WITS preferential groups mapping CSV.
        output_dir: The root directory to save the partitioned Parquet dataset.
        chunk_column: The unified column name ('t') used for filtering chunks after initial rename.
        partition_column: The column name ('Year') to use for partitioning the output dataset.
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
    logger.info(f"Chunking based on unified column: '{chunk_column}'")  # Should log 't'

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

    # --- Define Base LazyFrames AND Perform Initial Renaming ---
    logger.info("Defining base LazyFrames and applying initial renames...")
    try:
        # Scan and immediately rename 'year' (or equivalent) to 't'
        baci_lazy_base = pl.scan_parquet(baci_path)  # .rename({"t": chunk_column})
        wits_mfn_lazy_base_renamed = rename_wits_mfn(pl.scan_parquet(wits_mfn_path))
        wits_pref_lazy_base_renamed = rename_wits_pref(pl.scan_parquet(wits_pref_path))

        # Verify 't' exists in all base frames after renaming
        if chunk_column not in baci_lazy_base.columns:
            raise ValueError(f"Chunk column '{chunk_column}' missing in BACI after rename attempt.")
        if chunk_column not in wits_mfn_lazy_base_renamed.columns:
            raise ValueError(f"Chunk column '{chunk_column}' missing in WITS MFN after rename.")
        if chunk_column not in wits_pref_lazy_base_renamed.columns:
            raise ValueError(f"Chunk column '{chunk_column}' missing in WITS Pref after rename.")

        logger.info("Base LazyFrames defined with consistent chunk column name 't'.")
        logger.debug(f"BACI base schema (post-rename): {baci_lazy_base.collect_schema()}")
        logger.debug(
            f"MFN base schema (post-rename): {wits_mfn_lazy_base_renamed.collect_schema()}"
        )
        logger.debug(
            f"Pref base schema (post-rename): {wits_pref_lazy_base_renamed.collect_schema()}"
        )

    except Exception as e:
        logger.error(
            f"Failed during base LazyFrame definition or initial renaming: {e}", exc_info=True
        )
        raise

    # --- Determine chunks (e.g., years using 't') ---
    # Now call get_unique_chunk_values looking for 't' in the original paths.
    # The function internally handles the logic assuming 't' should exist post-rename.
    chunk_values = get_unique_chunk_values(
        [baci_path, wits_mfn_path, wits_pref_path],
        chunk_column,  # Pass 't' as the column name
    )

    # --- Process each chunk ---
    output_schema = None  # Initialize schema variable outside the loop

    for i, chunk_value in enumerate(chunk_values):
        logger.info(
            f"--- Processing Chunk {i + 1}/{len(chunk_values)}: {chunk_column} = {chunk_value} ---"
        )

        try:
            # Filter each *already renamed* base LazyFrame for the current chunk value
            logger.debug(f"Applying filter: {chunk_column} == {chunk_value}")
            baci_chunk_lazy = baci_lazy_base.filter(pl.col(chunk_column) == chunk_value)
            wits_mfn_chunk_lazy = wits_mfn_lazy_base_renamed.filter(
                pl.col(chunk_column) == chunk_value
            )
            wits_pref_chunk_lazy = wits_pref_lazy_base_renamed.filter(
                pl.col(chunk_column) == chunk_value
            )

            # --- Apply core logic (NO renaming needed here anymore) ---
            logger.debug("Expanding preferential tariffs for chunk...")
            # Pass the filtered, already-renamed pref frame
            expanded_pref_lazy = expand_preferential_tariffs(
                wits_pref_chunk_lazy, pref_group_mapping.lazy()
            )

            logger.debug("Joining datasets for the chunk...")
            # Pass the filtered, already-renamed frames
            joined_lazy = join_datasets(baci_chunk_lazy, wits_mfn_chunk_lazy, expanded_pref_lazy)

            logger.debug("Creating final table structure for the chunk...")
            # This step renames 't' to 'Year' (our partition_column)
            final_table_lazy = create_final_table(joined_lazy)

            # --- Execute Plan, Convert to Arrow, and Write using PyArrow ---
            # (This part remains the same as the previous PyArrow implementation)
            logger.info(f"Executing Polars plan for chunk {chunk_value}...")
            chunk_df = final_table_lazy.collect(engine="streaming")
            logger.info(f"Collected chunk {chunk_value}. Shape: {chunk_df.shape}")

            if chunk_df.is_empty():
                logger.warning(
                    f"No data found for chunk {chunk_column}={chunk_value}. Skipping write."
                )
                continue

            logger.debug(f"Converting chunk {chunk_value} to PyArrow Table...")
            try:
                table = chunk_df.to_arrow()
            except Exception as e:
                logger.error(
                    f"Failed to convert Polars DataFrame to Arrow Table for chunk {chunk_value}: {e}",
                    exc_info=True,
                )
                continue

            if output_schema is None:
                output_schema = table.schema
                logger.info(
                    f"Established dataset schema from first non-empty chunk ({chunk_value}):\n{output_schema}"
                )
            else:
                try:
                    if table.schema != output_schema:
                        logger.warning(f"Schema for chunk {chunk_value} differs. Casting...")
                        table = table.cast(output_schema, safe=False)
                    else:
                        logger.debug(f"Schema for chunk {chunk_value} matches.")
                except Exception as e:
                    logger.error(
                        f"Failed to cast chunk {chunk_value} to established schema: {e}",
                        exc_info=True,
                    )
                    continue

            logger.info(f"Writing chunk {chunk_value} to partitioned dataset at: {output_dir}")
            try:
                pq.write_to_dataset(
                    table=table,
                    root_path=output_dir,
                    partition_cols=[partition_column],  # Use 'Year'
                    schema=output_schema,
                    existing_data_behavior="overwrite_or_ignore",
                    # write_statistics=True, # Default usually True
                    # compression='zstd',
                )
                logger.info(
                    f"Successfully wrote chunk for {partition_column}={chunk_value} using PyArrow."
                )
            except Exception as e:
                logger.error(
                    f"PyArrow failed to write partition for {partition_column}={chunk_value}: {e}",
                    exc_info=True,
                )

        # ... (rest of the try/except/finally block for the loop iteration) ...
        except pl.exceptions.ComputeError as e:
            logger.error(f"Polars compute error processing chunk {chunk_value}: {e}", exc_info=True)
            # Explain might fail if the lazy frame itself is problematic after error
            try:
                logger.error(f"Query plan that failed:\n{final_table_lazy.explain(streaming=True)}")
            except Exception as explain_e:
                logger.error(f"Could not explain query plan after error: {explain_e}")
        except Exception as e:
            logger.error(
                f"Failed to process or write chunk for {chunk_column}={chunk_value}: {e}",
                exc_info=True,
            )
        finally:
            # Explicitly trigger garbage collection after each chunk
            # May help release memory, especially if complex objects were created
            try:
                del chunk_df  # Explicit deletion hint
            except NameError:
                pass  # Ignore if chunk_df wasn't assigned due to error
            try:
                del table  # Explicit deletion hint
            except NameError:
                pass  # Ignore if table wasn't assigned due to error
            gc.collect()
            logger.debug("Garbage collection triggered after chunk processing.")

    logger.info(
        f"--- Chunked matching pipeline finished. Partitioned data written to: {output_dir} ---"
    )


# --- Main Execution Example ---
if __name__ == "__main__":
    setup_logging(log_level=logging.DEBUG)
    logger = get_logger(__name__)  # Re-get logger after setup

    logger.info("Running matching_chunked.py script...")
    output_directory = DEFAULT_CHUNKED_OUTPUT_DIR

    try:
        run_chunked_matching_pipeline(
            output_dir=output_directory,
        )
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.critical(f"Chunked pipeline execution failed in __main__: {e}", exc_info=True)
        import sys

        sys.exit(1)
