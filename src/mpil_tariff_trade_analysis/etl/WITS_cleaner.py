"""
Clean the WITS tariff dataset & harmonise it with BACI, and across time.
"""

# Standard library imports
import glob
import os
import re
from pathlib import Path

import pandas as pd

# Third-party imports
import polars as pl

# Local/application imports
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)


OUTPUT_DIR = "data/intermediate"
DEFAULT_WITS_BASE_DIR = "data/raw/WITS_tariff"  # Add a default


def consolidate_wits_tariff_data(tariff_type="AVEMFN", base_dir=DEFAULT_WITS_BASE_DIR) -> pl.LazyFrame:
    """
    Consolidate all WITS tariff data for a specific tariff type into a single Polars LazyFrame.

    Args:
        tariff_type (str): Type of tariff to load (AVEMFN or AVEPref)
        base_dir (str): Base directory for the WITS tariff data

    Returns:
        polars.LazyFrame: A LazyFrame containing the consolidated data.
    """
    logger.info(f"Starting to consolidate {tariff_type} tariff data from {base_dir}")

    # Path to the specific tariff type directory
    tariff_dir = os.path.join(base_dir, tariff_type)
    logger.debug(f"Tariff directory: {tariff_dir}")

    # Pattern to extract metadata from directory names
    pattern = rf"{tariff_type}_(H\d+)_(\w+)_(\d+)_U2"

    # Find all CSV files for this tariff type
    all_csv_files = []

    try:
        subdirs = os.listdir(tariff_dir)
        logger.debug(f"Found {len(subdirs)} subdirectories in {tariff_dir}")
    except FileNotFoundError:
        logger.error(f"Directory not found: {tariff_dir}")
        raise

    for subdir in subdirs:
        match = re.match(pattern, subdir)
        if match:
            hs_revision, country_iso, year = match.groups()
            csv_path = os.path.join(tariff_dir, subdir)
            # Note: case-sensitive .CSV extension
            csv_files = glob.glob(os.path.join(csv_path, "*.CSV"))

            # logger.debug(f"Found {len(csv_files)} CSV files in {subdir}")

            for file in csv_files:
                if "JobID" in file:  # Only include the actual data files
                    all_csv_files.append(
                        {
                            "file_path": file,
                            "hs_revision": hs_revision,
                            "reporter_country": country_iso,
                            "year": int(year),
                        }
                    )

    logger.info(f"Found {len(all_csv_files)} CSV files for {tariff_type}")

    # Load and combine all CSV files
    dfs = []

    # Define schema based on observed CSV structure - use Utf8 for rate columns
    # to handle non-numeric values
    if tariff_type == "AVEMFN":
        schema = {
            "NomenCode": pl.Utf8,
            "Reporter_ISO_N": pl.Utf8,
            "Year": pl.Int32,
            "ProductCode": pl.Utf8,
            "Sum_Of_Rates": pl.Utf8,
            "Min_Rate": pl.Utf8,
            "Max_Rate": pl.Utf8,
            "SimpleAverage": pl.Utf8,
            "Nbr_NA_Lines": pl.Int32,
            "Nbr_Free_Lines": pl.Int32,
            "Nbr_AVE_Lines": pl.Int32,
            "Nbr_Dutiable_Lines": pl.Int32,
            "TotalNoOfValidLines": pl.Int32,
            "TotalNoOfLines": pl.Int32,
            "EstCode": pl.Utf8,
        }

    elif tariff_type == "AVEPref":
        schema = {
            "NomenCode": pl.Utf8,
            "Reporter_ISO_N": pl.Utf8,
            "Year": pl.Int32,
            "ProductCode": pl.Utf8,
            "Partner": pl.Utf8,
            "Sum_Of_Rates": pl.Utf8,
            "Min_Rate": pl.Utf8,
            "Max_Rate": pl.Utf8,
            "SimpleAverage": pl.Utf8,
            "Nbr_NA_Lines": pl.Int32,
            "Nbr_Free_Lines": pl.Int32,
            "Nbr_AVE_Lines": pl.Int32,
            "Nbr_Dutiable_Lines": pl.Int32,
            "TotalNoOfValidLines": pl.Int32,
            "TotalNoOfLines": pl.Int32,
            "EstCode": pl.Utf8,
        }

    else:
        raise KeyError("Wrong tariff type")

    successful_files = 0
    failed_files = 0

    for file_info in all_csv_files:
        try:
            # Load the CSV file lazily
            logger.debug(f"Scanning file: {file_info['file_path']}")
            df = pl.scan_csv(
                file_info["file_path"],
                schema=schema,
                infer_schema_length=0,  # Use our predefined schema
                try_parse_dates=False,  # No date columns
                separator=",",
                skip_rows_after_header=0,
                null_values=["", "NA"],
                truncate_ragged_lines=True,  # Handle any inconsistent lines
            )

            dfs.append(df)
            successful_files += 1
        except Exception as e:
            logger.error(f"Error loading file {file_info['file_path']}: {e}")
            failed_files += 1

    logger.info(
        f"Successfully scanned {successful_files} files, failed to scan {failed_files} files"
    )

    if not dfs:
        logger.warning("No valid CSV files found to scan.")
        # Depending on desired behavior, you might return an empty LazyFrame
        # or raise an error. Let's raise for now.
        raise FileNotFoundError("No WITS CSV files found or scanned successfully.")

    # Combine all dataframes
    logger.info("Combining all scanned dataframes")
    combined_df = pl.concat(dfs, how="vertical_relaxed") # Use relaxed to handle potential schema variations if any

    # Add a column for tariff type
    combined_df = combined_df.with_columns(pl.lit(tariff_type).alias("tariff_type"))

    if tariff_type == "AVEMFN":
        combined_df = combined_df.select(
            [
                pl.col("Year").cast(pl.Utf8, strict=False).alias("year"),
                pl.col("Reporter_ISO_N").alias("reporter_country"),
                pl.col("ProductCode").alias("product_code"),
                pl.col("NomenCode").alias("hs_revision"),
                pl.col("SimpleAverage").alias("tariff_rate"),
                pl.col("Min_Rate").alias("min_rate"),
                pl.col("Max_Rate").alias("max_rate"),
                pl.col("tariff_type"),
            ]
        )

    elif tariff_type == "AVEPref":
        combined_df = combined_df.select(
            [
                pl.col("Year").cast(pl.Utf8, strict=False).alias("year"),
                pl.col("Reporter_ISO_N").alias("reporter_country"),
                pl.col("Partner").alias("partner_country"),
                pl.col("ProductCode").alias("product_code"),
                pl.col("NomenCode").alias("hs_revision"),
                pl.col("SimpleAverage").alias("tariff_rate"),
                pl.col("Min_Rate").alias("min_rate"),
                pl.col("Max_Rate").alias("max_rate"),
                pl.col("tariff_type"),
            ]
        )

    # Remove the file writing part
    # final_path = Path(f"{tariff_dir}/WITS_{tariff_type}.parquet")
    # try:
    #     combined_df.sink_parquet(
    #         final_path,
    #     )
    #     logger.info(f"Sucesffuly sank WITS MFN post-consolidation to {final_path}")
    #     return final_path
    # except Exception as e:
    #     logger.error(f"Failed to write WITS {tariff_type} data:\n{e}")
    #     raise

    logger.info(f"Successfully consolidated WITS {tariff_type} data into a LazyFrame.")
    return combined_df


def vectorized_hs_translation(
    input_lf: pl.LazyFrame, mapping_dir: str = "data/raw/hs_reference"
) -> pl.LazyFrame:
    """
    Translates Harmonized System (HS) codes within a Polars LazyFrame to HS92 (H0).

    Args:
        input_lf (pl.LazyFrame): The input LazyFrame containing WITS data with a
                                 'product_code' and 'hs_revision' column.
        mapping_dir (str): Directory containing HS mapping CSV files (e.g., H1_to_H0.CSV).

    Returns:
        pl.LazyFrame: A LazyFrame with 'product_code' translated to HS92 where possible.
    """
    logger.info("Starting HS code translation to H0 (HS92).")
    # Define which HS revisions require mapping (all except H0)
    hs_versions = [f"H{i}" for i in range(1, 7)]  # H1, H2, H3, H4, H5, H6

    mapping_dfs = []
    for hs_version in hs_versions:
        # Build mapping file path (assumes file naming convention like H1_to_H0.CSV)
        path = Path(mapping_dir) / f"{hs_version}_to_H0.CSV"
        try:
            # Load mapping as pandas and convert it to a Polars DataFrame
            # Use scan_csv for lazy loading if files are large
            mapping_pl = pl.scan_csv(
                path,
                dtypes={"source_code": pl.Utf8, "target_code": pl.Utf8}, # Specify dtypes
                has_header=True, # Assuming header exists
                use_pyarrow=True, # Often faster
                encoding="iso-8859-1" # Keep encoding
            ).select(pl.col(pl.first()).alias("source_code"), pl.col(pl.last()).alias("target_code")) # Select first and last columns

            # mapping_pd = pd.read_csv(path, dtype=str, usecols=[0, 2], encoding="ISO-8859-1")
            # mapping_pd.columns = ["source_code", "target_code"]
            # mapping_pl = pl.from_pandas(mapping_pd) # Eager load

            mapping_pl = mapping_pl.with_columns(pl.lit(hs_version).alias("hs_revision"))
            mapping_pl = mapping_pl.with_columns(pl.col("source_code").str.zfill(6))
            mapping_dfs.append(mapping_pl)

        except Exception as e:
            logger.error(f"Error loading mapping file for {hs_version}: {path}. Error: {e}")
            # Decide if you want to continue without this mapping or raise
            # raise ValueError(f"Error loading mapping file for {hs_version}: \n {e}") from e
            continue # Skip this mapping if file is problematic

    if not mapping_dfs:
        logger.warning("No HS mapping files loaded. HS translation step will be skipped.")
        return input_lf # Return original if no mappings found

    # Combine all mappings into one Polars LazyFrame.
    mapping_all = pl.concat(mapping_dfs).lazy() # Ensure it's lazy

    # Use the input LazyFrame directly
    df = input_lf

    # Pre-process the main LazyFrame:
    # Ensure product codes are padded to 6 digits.
    df = df.with_columns(pl.col("product_code").str.zfill(6))

    # Split rows where translation is not needed (H0) from those that need translation.
    df_h0 = df.filter(pl.col("hs_revision") == "H0")
    df_non_h0 = df.filter(pl.col("hs_revision") != "H0")

    # Perform a vectorized join between df_non_h0 and the mapping dataframe.
    # The join is done on the hs_revision and the padded product_code versus mapping's source_code.
    df_non_h0 = df_non_h0.join(
        mapping_all,
        left_on=["hs_revision", "product_code"],
        right_on=["hs_revision", "source_code"],
        how="left",
    )

    # Create the translated HS code column: use target_code if available, otherwise fallback to the
    # original code. Ensure the final column name is 'product_code'.
    df_non_h0 = df_non_h0.with_columns(
        pl.when(pl.col("target_code").is_not_null()) # Check if target_code exists from join
        .then(pl.col("target_code"))
        .otherwise(pl.col("product_code")) # Keep original if no match
        .alias("product_code_translated") # Use a temporary name
    ).drop("product_code").rename({"product_code_translated": "product_code"}) # Rename back

    # Optionally drop unnecessary columns from join
    df_non_h0 = df_non_h0.drop(["target_code"]) # Drop the mapping target code column

    # Combine the rows which were already in H0 with the ones translated.
    # Ensure schemas match before concat
    # Select the same columns in the same order from both frames
    common_cols = df_h0.columns
    df_final = pl.concat([df_h0.select(common_cols), df_non_h0.select(common_cols)], how="vertical_relaxed")

    # Remove the file writing part
    # vectorised_output_path = Path(intermediate_file_path.stem + "_vectorised.parquet")
    # try:
    #     df_final.sink_parquet(vectorised_output_path)
    # except Exception as e:
    #     logger.error(f"Failed to sink hs_translated WITS file:\n{e}")
    #     raise
    # return vectorised_output_path

    logger.info("✅ HS code translation completed.")
    return df_final

# Remove the __main__ block as this file is now a library
# if __name__ == "__main__":
#    ...
