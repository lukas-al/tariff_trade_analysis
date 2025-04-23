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
from mpil_tariff_trade_analysis.utils.iso_remapping import create_country_code_mapping_df
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)


OUTPUT_DIR = "data/intermediate"
DEFAULT_WITS_BASE_DIR = "data/raw/WITS_tariff"  # Add a default


def load_wits_tariff_data(
    tariff_type="AVEMFN", base_dir=DEFAULT_WITS_BASE_DIR
):  # Use default here too
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
                pl.col("Year").cast(pl.Utf8, strict=False).alias("year"),
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

        logger.info("Translating HS codes...")
        combined_df = vectorized_hs_translation(combined_df)
    elif tariff_type == "AVEPref":
        combined_df = combined_df.select(
            [
                pl.col("Year").cast(pl.Utf8, strict=False).alias("year"),
                pl.col("Reporter_ISO_N").alias("reporter_country"),
                pl.col("Partner").alias("partner_country"),
                pl.col("ProductCode").alias("product_code"),
                pl.col("NomenCode").alias("hs_revision"),
                # pl.col("SimpleAverage").cast(pl.Float64, strict=False).alias("tariff_rate"),
                pl.col("SimpleAverage").alias("tariff_rate"),
                # pl.col("Min_Rate").cast(pl.Float64, strict=False).alias("min_rate"),
                pl.col("Min_Rate").alias("min_rate"),
                # pl.col("Max_Rate").cast(pl.Float64, strict=False).alias("max_rate"),
                pl.col("Max_Rate").alias("max_rate"),
                pl.col("tariff_type"),
            ]
        )

        logger.info("Translating HS codes...")
        combined_df = vectorized_hs_translation(combined_df)

    logger.info("Finished initial loading and HS translation for WITS tariff data.")

    # --- Apply Vectorized Country Code Remapping ---
    country_cols_to_map = ["reporter_country"]
    if tariff_type == "AVEPref":
        country_cols_to_map.append("partner_country")

    logger.info(f"Starting country code remapping for columns: {country_cols_to_map}")

    # Apply the unified mapping and explosion logic directly
    # This function now handles the mapping, potential explosion, and column renaming/dropping
    combined_df = create_country_code_mapping_df(
        lf=combined_df,
        code_columns=country_cols_to_map,
        # Pass reference paths if they differ from defaults in iso_remapping.py
        # baci_codes_path=DEFAULT_BACI_COUNTRY_CODES_PATH,
        # wits_codes_path=DEFAULT_WITS_COUNTRY_CODES_PATH,
        # Assuming default column names in reference files are correct
        # baci_code_col="country_code", baci_name_col="country_name",
        # wits_code_col="ISO3", wits_name_col="Country Name",
        drop_original=True,  # This is the default, but explicit is fine
    )

    # Check if the remapping actually produced results (optional, based on function behavior)
    # The function logs internally if columns are skipped or mapping fails.
    # We might just check if the expected new columns exist.
    expected_new_cols = [f"{col}_iso_numeric" for col in country_cols_to_map]
    missing_cols = [col for col in expected_new_cols if col not in combined_df.columns]
    if missing_cols:
        logger.warning(
            f"Expected remapped columns missing after remapping: {missing_cols}. Check logs."
        )
    else:
        logger.info("Finished country code remapping for WITS data.")

    return combined_df


# process_and_save_wits_data is removed, saving is handled by specific pipelines


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
        mapping_pl = mapping_pl.with_columns(pl.lit(hs_version).alias("hs_revision"))
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


# Remove the __main__ block as this file is now a library
# if __name__ == "__main__":
#    ...
