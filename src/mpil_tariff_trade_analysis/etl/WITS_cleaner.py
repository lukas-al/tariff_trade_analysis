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
DEFAULT_WITS_BASE_DIR = "data/raw/WITS_tariff" # Add a default


def load_wits_tariff_data(tariff_type="AVEMFN", base_dir=DEFAULT_WITS_BASE_DIR): # Use default here too
    """
    Load all WITS tariff data for a specific tariff type into a single Polars DataFrame.

    Args:
        tariff_type (str): Type of tariff to load (AVEMFN or AVEPref)
        base_dir (str): Base directory for the WITS tariff data

    Returns:
        pl.LazyFrame: Combined lazy dataframe with all tariff data
    """
    logger.info(f"Starting to load {tariff_type} tariff data from {base_dir}")

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
        return None

    for subdir in subdirs:
        match = re.match(pattern, subdir)
        if match:
            hs_revision, country_iso, year = match.groups()
            csv_path = os.path.join(tariff_dir, subdir)
            # Note: case-sensitive .CSV extension
            csv_files = glob.glob(os.path.join(csv_path, "*.CSV"))

            logger.debug(f"Found {len(csv_files)} CSV files in {subdir}")

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
            # Load the CSV file
            logger.debug(f"Loading file: {file_info['file_path']}")
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
        f"Successfully loaded {successful_files} files, failed to load {failed_files} files"
    )

    if not dfs:
        logger.warning("No valid CSV files found.")
        return None

    # Combine all dataframes
    logger.info("Combining all dataframes")
    combined_df = pl.concat(dfs)

    # Add a column for tariff type
    combined_df = combined_df.with_columns(pl.lit(tariff_type).alias("tariff_type"))

    if tariff_type == "AVEMFN":
        combined_df = combined_df.select(
            [
                pl.col("Year").alias("year"),
                pl.col("Reporter_ISO_N").alias("reporter_country"),
                pl.col("ProductCode").alias("product_code"),
                pl.col("NomenCode").alias("hs_revision"),
                pl.col("SimpleAverage").alias("tariff_rate"),
                pl.col("Min_Rate").alias("min_rate"),
                pl.col("Max_Rate").alias("max_rate"),
                pl.col("tariff_type"),
                # pl.col("SimpleAverage").cast(pl.Float64, strict=False).alias("tariff_rate"),
                # pl.col("Min_Rate").cast(pl.Float64, strict=False).alias("min_rate"),
                # pl.col("Max_Rate").cast(pl.Float64, strict=False).alias("max_rate"),
            ]
        )
        # .with_columns(
        #     pl.struct(["product_code", "hs_revision"]).map_elements(
        #         lambda row: translator.translate(row["product_code"], row["hs_revision"]),
        #         return_dtype=pl.Utf8
        #     ).alias("product_code_h0")
        # )

        logger.info("Translating HS codes...")
        combined_df = vectorized_hs_translation(combined_df)
    elif tariff_type == "AVEPref":
        combined_df = combined_df.select(
            [
                pl.col("Year").alias("year"),
                pl.col("Reporter_ISO_N").alias("reporter_country"),
                pl.col("Partner").alias("partner_country"),
                pl.col("ProductCode").alias("product_code"),
                pl.col("NomenCode").alias("hs_revision"),
                # pl.col("SimpleAverage").cast(pl.Float64, strict=False).alias("tariff_rate"),
                pl.col("SimpleAverage").alias("tariff_rate"),
                pl.col("tariff_type"),
                # pl.col("Min_Rate").cast(pl.Float64, strict=False).alias("min_rate"),
                pl.col("Min_Rate").alias("min_rate"),
                # pl.col("Max_Rate").cast(pl.Float64, strict=False).alias("max_rate"),
                pl.col("Max_Rate").alias("max_rate"),
            ]
        )

        logger.info("Translating HS codes...")
        combined_df = vectorized_hs_translation(combined_df)

    logger.info("Finished loading and processing WITS tariff data")
    return combined_df


def process_and_save_wits_data(
    tariff_type="AVEMFN",
    output_dir=OUTPUT_DIR,
    base_dir=DEFAULT_WITS_BASE_DIR # Add base_dir argument here
):
    """
    Process all WITS tariff data for a specific tariff type and save as a parquet file.

    Args:
        tariff_type (str): Type of tariff to process (AVEMFN or AVEPref)
        output_dir (str): Directory to save the output parquet file
        base_dir (str): Base directory containing the raw WITS tariff data folders

    Returns:
        str: Path to the saved parquet file or None on failure
    """
    logger.info(f"Starting processing of {tariff_type} tariff data")
    logger.info(f"Using base directory for raw data: {base_dir}") # Log the base dir being used

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created or confirmed output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return None

    # Load the tariff data, passing the base_dir
    logger.info(f"Loading {tariff_type} tariff data...")
    # Pass base_dir to the loading function
    df = load_wits_tariff_data(tariff_type=tariff_type, base_dir=base_dir)

    if df is None:
        logger.warning("No data loaded, cannot save.")
        return None

    # Save as parquet
    output_path = os.path.join(output_dir, f"WITS_{tariff_type}.parquet")
    logger.info(f"Processing and saving data to {output_path}...")

    try:
        # Materialize the lazy frame and write to parquet
        logger.info("   Materializing and writing to parquet...")
        df.collect().write_parquet(output_path)
        logger.info(f"Data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: \n {e}")
        return None

    return output_path


def vectorized_hs_translation(
    df: pl.LazyFrame, mapping_dir: str = "data/raw/hs_reference"
) -> pl.LazyFrame:
    """
    This function translates between harmonised system codes. The data in WITS is coded based on
    the HS at the time of the series date. This means we need to map from HS 1-6 to HS92,
    the HS our BACI data is in with the longest available series.

    !WARNING: Given inaccuracies which result from this process, it may be desirable in the future
    !    to use a different, shorter but more accurate BACI dataset coded in a more modern HS, and
    !    then to update this function to account for that.

    The HS in use is referred to as follows in the dataset:
    H0: HS 1988/92
    H1: HS 1996
    H2: HS 2002
    H3: HS 2007
    H4: HS 2012
    H5: HS 2017
    H6: HS 2022

    HS mappings are stored in the data/raw/hs_reference folder, with each file mapping from one
    HS to another. The files stored are H1_to_H0, H2_to_H0, H3_to_H0, H4_to_H0, H5_to_H0, H6_to_H0.

    !WARNING: I haven't fully validated that this is working, will assume so for now.

    """
    # Define which HS revisions require mapping (all except H0)
    hs_versions = [f"H{i}" for i in range(1, 7)]  # H1, H2, H3, H4, H5, H6

    mapping_dfs = []
    for hs_version in hs_versions:
        # Build mapping file path (assumes file naming convention like H1_to_H0.CSV)
        path = Path(mapping_dir) / f"{hs_version}_to_H0.CSV"
        try:
            # Load mapping as pandas and convert it to a Polars DataFrame
            mapping_pd = pd.read_csv(path, dtype=str, usecols=[0, 2], encoding="ISO-8859-1")
            mapping_pd.columns = ["source_code", "target_code"]
        except Exception as e:
            raise ValueError(f"Error loading mapping file for {hs_version}: \n {e}") from e

        mapping_pl = pl.from_pandas(mapping_pd)
        # Add the hs_revision column to indicate which mapping this is for.
        mapping_pl = mapping_pl.with_columns(pl.lit(hs_version).alias("hs_revision"))
        # Ensure source_code is consistently formatted (6-digit padding)
        mapping_pl = mapping_pl.with_columns(pl.col("source_code").str.zfill(6))
        mapping_dfs.append(mapping_pl)

    # Combine all mappings into one Polars LazyFrame.
    mapping_all = pl.concat(mapping_dfs).lazy()

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
    # original code.
    df_non_h0 = df_non_h0.with_columns(
        pl.when(pl.col("target_code").is_null())
        .then(pl.col("product_code"))
        .otherwise(pl.col("target_code"))
        .alias("product_code")
    )

    # Optionally drop unnecessary columns from join (if desired)
    df_non_h0 = df_non_h0.drop(["target_code"])

    # Combine the rows which were already in H0 with the ones translated.
    df_final = pl.concat([df_h0, df_non_h0])

    return df_final


# Modify the __main__ block to pass the base_dir if needed, or rely on default
if __name__ == "__main__":
    # Process and save AVEMFN tariff data
    logger.info("Starting WITS tariff data processing script")
    # Example: Explicitly pass base_dir if not using default, otherwise it uses the default
    # base_directory = "path/to/your/raw/data"
    # result = process_and_save_wits_data(tariff_type="AVEMFN", base_dir=base_directory)
    result = process_and_save_wits_data(tariff_type="AVEMFN") # Uses default base_dir

    if result:
        logger.info(f"Successfully processed and saved WITS AVEMFN tariff data to {result}")
    else:
        logger.error("Failed to process and save WITS AVEMFN tariff data")

    # Optionally process AVEPref data too
    logger.info("Starting processing of AVEPref tariff data")
    # result = process_and_save_wits_data(tariff_type="AVEPref", base_dir=base_directory)
    result = process_and_save_wits_data(tariff_type="AVEPref") # Uses default base_dir

    if result:
        logger.info(f"Successfully processed and saved WITS AVEPref tariff data to {result}")
    else:
        logger.error("Failed to process and save WITS AVEPref tariff data")
