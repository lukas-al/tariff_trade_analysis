"""
Pipeline for merging cleaned BACI, WITS MFN, and WITS Preferential data.
Processes data in chunks based on a specified column (e.g., year) and
saves the output as a partitioned Parquet dataset.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional  # Added Optional and List

import polars as pl
import pyarrow.parquet as pq

# Import the core logic functions from matching_logic.py
# These functions now expect cleaned, renamed data as input.
# expand_preferential_tariffs is NO LONGER needed here.
from mpil_tariff_trade_analysis.etl.matching_logic import (  # load_pref_group_mapping, # No longer needed here
    create_final_table,
    join_datasets,
)
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration for Chunking ---
# Define the column name used for chunking in the *cleaned* input data.
# This should be the consistent name after cleaning/renaming (e.g., 't').
DEFAULT_CHUNK_COLUMN_NAME = "t"

# Define the column name for partitioning IN THE FINAL OUTPUT
DEFAULT_PARTITION_COLUMN = "Year"  # This comes from create_final_table

# Define a new default output DIRECTORY for the partitioned dataset
DEFAULT_MERGED_OUTPUT_DIR = "data/final/unified_trade_tariff_partitioned"


def get_unique_chunk_values(paths: List[str | Path], column_name: str) -> List[str]:
    """
    Scans multiple Parquet files or directories (containing *cleaned* data)
    to find unique values in the specified chunking column without loading full data.

    Args:
        paths: A list of paths to the cleaned Parquet files or directories.
        column_name: The name of the column to find unique values for (e.g., 't').

    Returns:
        A sorted list of unique string values found in the column across all files.
    """
    unique_values = set()
    logger.info(f"Scanning for unique values in chunk column '{column_name}' across paths: {paths}")

    for data_path in paths:
        try:
            # Scan the cleaned data source
            lf = pl.scan_parquet(data_path)

            if column_name in lf.columns:
                # Use streaming engine for potentially large datasets
                values = (
                    lf.select(pl.col(column_name).unique())
                    .collect(engine="streaming")  # Use streaming=True
                    .get_column(column_name)
                    # Ensure values are strings for consistent filtering later
                    .cast(pl.Utf8, strict=False)
                    .to_list()
                )
                unique_values.update(
                    val for val in values if val is not None
                )  # Exclude None values
                logger.debug(f"Found {len(values)} unique '{column_name}' values in {data_path}.")
            else:
                logger.warning(
                    f"Chunk column '{column_name}' not found in schema for path: {data_path}. Columns: {lf.columns}"
                )
        except Exception as e:
            logger.warning(
                f"Could not read column '{column_name}' from {data_path}: {e}. Skipping this source for chunk value detection."
            )

    if not unique_values:
        # If only one path was provided and failed, this might be critical
        if len(paths) == 1:
            raise ValueError(
                f"No unique values found for chunk column '{column_name}' in the primary path {paths[0]}. Cannot proceed."
            )
        # If multiple paths, maybe it's okay if one is missing, but log a warning
        logger.warning(
            f"No unique values found for chunk column '{column_name}' in any provided path."
        )
        return []  # Return empty list, the main loop should handle this

    # Sort values as strings to ensure consistent processing order
    sorted_values = sorted(list(unique_values))
    logger.info(
        f"Found {len(sorted_values)} unique string chunk values: {sorted_values[:5]}...{sorted_values[-5:]}"
    )
    return sorted_values


def validate_merged_chunk(df: pl.DataFrame, chunk_value: str) -> bool:
    """
    Performs basic validation on the merged DataFrame for a specific chunk.
    """
    # Columns expected after create_final_table
    expected_cols_subset = {
        "Year",
        "Source",
        "Target",
        "HS_Code",
        "effective_tariff_rate",
        # Add others like Quantity, Value, mfn_rate, pref_rate if selected
    }
    if "Quantity" in df.columns:
        expected_cols_subset.add("Quantity")
    if "Value" in df.columns:
        expected_cols_subset.add("Value")
    if "mfn_tariff_rate" in df.columns:
        expected_cols_subset.add("mfn_tariff_rate")
    if "pref_tariff_rate" in df.columns:
        expected_cols_subset.add("pref_tariff_rate")

    actual_cols = set(df.columns)

    if not expected_cols_subset.issubset(actual_cols):
        missing = expected_cols_subset - actual_cols
        logger.error(
            f"Validation Error (Chunk {chunk_value}): Missing expected columns in merged data. "
            f"Missing: {missing}, Found: {actual_cols}"
        )
        return False

    # Check for nulls in key identifiers
    key_cols = ["Year", "Source", "Target", "HS_Code"]
    null_counts = df.select([pl.col(c).is_null().sum() for c in key_cols]).row(0, named=True)
    for col, count in null_counts.items():
        if count > 0:
            logger.warning(
                f"Validation Warning (Chunk {chunk_value}): Found {count} null values in key column '{col}'."
            )
            # Decide if this should be an error or just a warning

    # Check data types in final output
    schema = df.schema
    if schema.get("Year") != pl.Utf8:  # create_final_table aliases 't' (Utf8) to 'Year'
        logger.error(
            f"Validation Error (Chunk {chunk_value}): Final 'Year' column type is {schema.get('Year')}, expected Utf8."
        )
        return False
    if schema.get("Source") != pl.Utf8:
        logger.error(
            f"Validation Error (Chunk {chunk_value}): Final 'Source' column type is {schema.get('Source')}, expected Utf8."
        )
        return False
    # Target might be list if BACI remapping produced lists and wasn't exploded? Check BACI output.
    # Assuming Target is Utf8 for now.
    if schema.get("Target") != pl.Utf8:
        logger.error(
            f"Validation Error (Chunk {chunk_value}): Final 'Target' column type is {schema.get('Target')}, expected Utf8."
        )
        return False
    if schema.get("HS_Code") != pl.Utf8:
        logger.error(
            f"Validation Error (Chunk {chunk_value}): Final 'HS_Code' column type is {schema.get('HS_Code')}, expected Utf8."
        )
        return False
    # Check tariff rates - they are likely Utf8 still, need casting for numeric checks
    if "effective_tariff_rate" in schema and schema["effective_tariff_rate"] != pl.Utf8:
        logger.warning(
            f"Validation Warning (Chunk {chunk_value}): Final 'effective_tariff_rate' column type is {schema['effective_tariff_rate']}, expected Utf8."
        )

    logger.info(f"Merged data chunk {chunk_value} basic validation passed.")
    return True


def run_merge_pipeline(
    config: Dict[str, Any],
    cleaned_baci_path: Path,
    cleaned_mfn_path: Path,
    cleaned_pref_path: Path,  # This is now the *expanded* pref data
) -> bool:
    """
    Processes the merging pipeline by chunking cleaned data based on a column
    and writing to a Hive-partitioned Parquet dataset.

    Args:
        config: The pipeline configuration dictionary. Expected keys:
            - MERGED_OUTPUT_DIR (Path for the final partitioned output)
            - CHUNK_COLUMN_NAME (Optional, defaults to 't')
            - PARTITION_COLUMN (Optional, defaults to 'Year')
        cleaned_baci_path: Path to the cleaned BACI data file/directory.
        cleaned_mfn_path: Path to the cleaned WITS MFN data file.
        cleaned_pref_path: Path to the cleaned and *expanded* WITS PREF data file.

    Returns:
        True if the pipeline completed successfully, False otherwise.
    """
    # pref_groups_path is no longer needed here
    output_dir = config.get("MERGED_OUTPUT_DIR", Path(DEFAULT_MERGED_OUTPUT_DIR))
    chunk_column = config.get("CHUNK_COLUMN_NAME", DEFAULT_CHUNK_COLUMN_NAME)
    partition_column = config.get("PARTITION_COLUMN", DEFAULT_PARTITION_COLUMN)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("--- Starting Data Merging Pipeline (Chunked) ---")
    logger.info(f"Cleaned BACI input: {cleaned_baci_path}")
    logger.info(f"Cleaned MFN input: {cleaned_mfn_path}")
    logger.info(f"Cleaned & Expanded PREF input: {cleaned_pref_path}")
    # logger.info(f"Preference Groups input: {pref_groups_path}") # Removed
    logger.info(f"Final output directory (partitioned by '{partition_column}'): {output_dir}")
    logger.info(f"Chunking based on column: '{chunk_column}'")

    # --- Load non-chunked data once ---
    # No non-chunked data needed here anymore (pref group mapping moved)

    # --- Define Base LazyFrames for Cleaned Data ---
    logger.info("Defining base LazyFrames for cleaned input data...")
    try:
        # Scan the cleaned data sources
        baci_lazy_base = pl.scan_parquet(cleaned_baci_path)
        wits_mfn_lazy_base = pl.scan_parquet(cleaned_mfn_path)
        # Scan the cleaned AND EXPANDED pref data
        wits_pref_lazy_base = pl.scan_parquet(cleaned_pref_path)

        # Verify chunk column exists in base frames
        if chunk_column not in baci_lazy_base.columns:
            raise ValueError(f"Chunk column '{chunk_column}' missing in cleaned BACI data.")
        if chunk_column not in wits_mfn_lazy_base.columns:
            raise ValueError(f"Chunk column '{chunk_column}' missing in cleaned WITS MFN data.")
        if chunk_column not in wits_pref_lazy_base.columns:
            raise ValueError(f"Chunk column '{chunk_column}' missing in cleaned WITS Pref data.")

        # Ensure chunk column type consistency (e.g., cast all to Utf8 for filtering)
        logger.debug(f"Casting chunk column '{chunk_column}' to Utf8 in base frames.")
        baci_lazy_base = baci_lazy_base.with_columns(pl.col(chunk_column).cast(pl.Utf8))
        wits_mfn_lazy_base = wits_mfn_lazy_base.with_columns(pl.col(chunk_column).cast(pl.Utf8))
        wits_pref_lazy_base = wits_pref_lazy_base.with_columns(pl.col(chunk_column).cast(pl.Utf8))

        logger.info("Base LazyFrames for cleaned data defined.")
        logger.debug(f"Cleaned BACI base schema: {baci_lazy_base.collect_schema()}")
        logger.debug(f"Cleaned MFN base schema: {wits_mfn_lazy_base.collect_schema()}")
        logger.debug(f"Cleaned Pref base schema: {wits_pref_lazy_base.collect_schema()}")

    except Exception as e:
        logger.error(
            f"Failed during base LazyFrame definition for cleaned data: {e}", exc_info=True
        )
        return False

    # --- Determine chunks (e.g., years using 't') ---
    try:
        chunk_values = get_unique_chunk_values(
            [cleaned_baci_path],  # Primarily use BACI for chunk values
            chunk_column,
        )
        if not chunk_values:
            logger.warning(
                f"No unique chunk values found in BACI data ({cleaned_baci_path}). Trying MFN/Pref..."
            )
            chunk_values = get_unique_chunk_values(
                [cleaned_mfn_path, cleaned_pref_path], chunk_column
            )
            if not chunk_values:
                logger.error("Could not determine chunk values from any source. Aborting.")
                return False

    except Exception as e:
        logger.error(f"Failed to determine chunk values: {e}", exc_info=True)
        return False

    # --- Process each chunk ---
    output_schema = None  # Initialize schema variable outside the loop

    for i, chunk_value in enumerate(chunk_values):
        logger.info(
            f"--- Processing Chunk {i + 1}/{len(chunk_values)}: {chunk_column} = {chunk_value} ---"
        )
        chunk_df: Optional[pl.DataFrame] = None  # Ensure chunk_df is defined for finally block
        table: Optional[Any] = None  # Ensure table is defined for finally block

        try:
            # Filter each base LazyFrame for the current chunk value
            logger.debug(f"Applying filter: {chunk_column} == '{chunk_value}'")
            baci_chunk_lazy = baci_lazy_base.filter(pl.col(chunk_column) == chunk_value)
            wits_mfn_chunk_lazy = wits_mfn_lazy_base.filter(pl.col(chunk_column) == chunk_value)
            # Filter the already expanded pref data
            wits_pref_chunk_lazy = wits_pref_lazy_base.filter(pl.col(chunk_column) == chunk_value)

            # --- Apply core merging logic (using functions from matching_logic) ---

            logger.debug("Joining datasets for the chunk...")
            # Pass the filtered BACI, filtered MFN, and filtered *expanded* Pref frames
            # Ensure column names match expectations of join_datasets (t, i, j, k etc.)
            # join_datasets expects 'baci', 'renamed_avemfn', 'expanded_pref'
            joined_lazy = join_datasets(
                baci=baci_chunk_lazy,
                renamed_avemfn=wits_mfn_chunk_lazy,
                cleaned_expanded_pref=wits_pref_chunk_lazy,  # Pass the filtered expanded pref data
            )

            logger.debug("Creating final table structure for the chunk...")
            # This step renames 't' to 'Year' (our partition_column)
            final_table_lazy = create_final_table(joined_lazy)

            # --- Execute Plan, Validate, Convert to Arrow, and Write ---
            logger.info(f"Executing Polars plan for chunk {chunk_value}...")
            # Use streaming execution if possible
            chunk_df = final_table_lazy.collect(engine="streaming")
            logger.info(f"Collected chunk {chunk_value}. Shape: {chunk_df.shape}")

            if chunk_df.is_empty():
                logger.warning(
                    f"No data found after merging for chunk {chunk_column}={chunk_value}. Skipping write."
                )
                continue

            # --- Validate Merged Chunk ---
            if not validate_merged_chunk(chunk_df, chunk_value):
                logger.error(f"Validation failed for merged chunk {chunk_value}. Skipping write.")
                # Consider whether to continue or abort pipeline on validation failure
                continue  # Skip writing this chunk

            # --- Write using PyArrow ---
            logger.debug(f"Converting chunk {chunk_value} to PyArrow Table...")
            try:
                table = chunk_df.to_arrow()
            except Exception as e:
                logger.error(
                    f"Failed to convert Polars DataFrame to Arrow Table for chunk {chunk_value}: {e}",
                    exc_info=True,
                )
                continue  # Skip writing this chunk

            if output_schema is None:
                output_schema = table.schema  # type: ignore
                logger.info(
                    f"Established dataset schema from first non-empty chunk ({chunk_value}):\n{output_schema}"
                )
            else:
                # Ensure schema consistency
                if table.schema != output_schema:  # type: ignore
                    logger.warning(
                        f"Schema for chunk {chunk_value} differs from established schema. Casting..."
                    )
                    try:
                        # Cast to the schema of the first chunk
                        table = table.cast(output_schema, safe=False)  # type: ignore
                        logger.debug(f"Successfully cast chunk {chunk_value} schema.")
                    except Exception as e:
                        logger.error(
                            f"Failed to cast chunk {chunk_value} to established schema: {e}. Skipping write.",
                            exc_info=True,
                        )
                        continue  # Skip writing this chunk

            logger.info(f"Writing chunk {chunk_value} to partitioned dataset at: {output_dir}")
            try:
                pq.write_to_dataset(
                    table=table,
                    root_path=str(output_dir),  # PyArrow might prefer string paths
                    partition_cols=[partition_column],  # Use 'Year' or configured column
                    schema=output_schema,
                    existing_data_behavior="overwrite_or_ignore",  # Safer than delete_matching if run concurrently
                    # compression='zstd', # Optional compression
                )
                logger.info(
                    f"Successfully wrote chunk for {partition_column}={chunk_value} using PyArrow."
                )
            except Exception as e:
                logger.error(
                    f"PyArrow failed to write partition for {partition_column}={chunk_value}: {e}",
                    exc_info=True,
                )
                # Decide if this error should stop the whole pipeline

        except pl.exceptions.ComputeError as e:
            logger.error(f"Polars compute error processing chunk {chunk_value}: {e}", exc_info=True)
            # Attempt to explain the failing plan
            try:
                logger.error(f"Query plan that failed:\n{final_table_lazy.explain(streaming=True)}")  # type: ignore
            except Exception as explain_e:
                logger.error(f"Could not explain query plan after error: {explain_e}")
        except Exception as e:
            logger.error(
                f"Failed to process or write chunk for {chunk_column}={chunk_value}: {e}",
                exc_info=True,
            )
            # Decide if this error should stop the whole pipeline (e.g., return False)
        finally:
            # Explicitly trigger garbage collection after each chunk
            del chunk_df
            del table
            gc.collect()
            logger.debug(f"Garbage collection triggered after chunk {chunk_value}.")

    logger.info(
        f"--- Chunked Merging Pipeline Finished. Partitioned data written to: {output_dir} ---"
    )
    # Basic check: did the output directory get created and has content?
    if not output_dir.exists() or not any(output_dir.iterdir()):
        logger.warning(
            f"Merge pipeline finished, but output directory {output_dir} is empty or does not exist."
        )
        # Return False if this indicates failure
        # return False
    return True  # Assume success if loop completes without returning False


# Example of how to run (optional, for direct execution)
if __name__ == "__main__":
    import sys

    from mpil_tariff_trade_analysis.utils.logging_config import setup_logging

    setup_logging(log_level="DEBUG")

    # --- Dummy Configuration for Testing ---
    # These paths should point to the *output* of the cleaning pipelines
    TEST_BASE_DATA_DIR = Path("data").resolve()
    TEST_INTERMEDIATE_DATA_DIR = TEST_BASE_DATA_DIR / "intermediate"
    TEST_FINAL_DATA_DIR = TEST_BASE_DATA_DIR / "final"

    # Assume cleaned data exists at these locations from previous steps
    cleaned_baci_path = TEST_INTERMEDIATE_DATA_DIR / "BACI_HS92_V202501_cleaned_remapped.parquet"
    cleaned_mfn_path = (
        TEST_INTERMEDIATE_DATA_DIR / "cleaned_wits_mfn" / "WITS_AVEMFN_cleaned.parquet"
    )
    # Point to the *expanded* pref data
    cleaned_pref_path = (
        TEST_INTERMEDIATE_DATA_DIR / "cleaned_wits_pref" / "WITS_AVEPref_cleaned_expanded.parquet"
    )

    test_config = {
        # PREF_GROUPS_PATH is no longer needed here
        "MERGED_OUTPUT_DIR": TEST_FINAL_DATA_DIR / "unified_trade_tariff_partitioned_refactored_v2",
        "CHUNK_COLUMN_NAME": "t",
        "PARTITION_COLUMN": "Year",
    }

    # Basic check if input data exists
    if not cleaned_baci_path.exists():
        logger.error(f"Cleaned BACI input not found: {cleaned_baci_path}")
        sys.exit(1)
    if not cleaned_mfn_path.exists():
        logger.error(f"Cleaned MFN input not found: {cleaned_mfn_path}")
        sys.exit(1)
    if not cleaned_pref_path.exists():
        logger.error(f"Cleaned & Expanded Pref input not found: {cleaned_pref_path}")
        sys.exit(1)

    logger.info(f"Running merge pipeline with test config: {test_config}")

    success = run_merge_pipeline(
        config=test_config,
        cleaned_baci_path=cleaned_baci_path,
        cleaned_mfn_path=cleaned_mfn_path,
        cleaned_pref_path=cleaned_pref_path,
    )

    if success:
        logger.info(
            f"Merge pipeline test completed successfully. Output: {test_config['MERGED_OUTPUT_DIR']}"
        )
        sys.exit(0)
    else:
        logger.error("Merge pipeline test failed.")
        sys.exit(1)
