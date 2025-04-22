from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import polars as pl
import pycountry
import pycountry.db

from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Default paths for reference data ---
# Adjust these paths as necessary
DEFAULT_BACI_COUNTRY_CODES_PATH = Path("data/raw/BACI_HS92_V202501/country_codes_V202501.csv")
DEFAULT_WITS_COUNTRY_CODES_PATH = Path("data/raw/WITS_country_codes.csv")


def load_reference_map(file_path: str | Path, code_col: str, name_col: str) -> Dict[str, str]:
    """Loads a CSV file into a dictionary mapping codes to names."""
    file_path = Path(file_path)  # Ensure it's a Path object
    logger.info(f"Attempting to load reference map from: {file_path}")
    try:
        # Ensure file exists before attempting to read
        if not file_path.is_file():
            logger.error(f"Reference map file not found: {file_path}")
            return {}

        df = pl.read_csv(file_path, infer_schema=False)  # Read all as strings
        # Ensure codes are treated as strings for consistent dictionary keys
        # Drop nulls to avoid issues with dict creation
        df_clean = df.select([code_col, name_col]).drop_nulls()
        mapping = dict(zip(df_clean[code_col].cast(pl.Utf8), df_clean[name_col], strict=False))
        logger.info(f"Successfully loaded {len(mapping)} entries from {file_path}")
        return mapping
    except pl.exceptions.ComputeError as e:
        # Catch Polars-specific errors like file not found during read_csv
        logger.error(f"Polars ComputeError loading reference map from {file_path}: {e}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error loading reference map from {file_path}: {e}")
        return {}


def remap_country_code_improved(
    code: Any, baci_map: Dict[str, str], wits_map: Dict[str, str]
) -> Optional[str]:
    """
    Remaps a given country code to the standard ISO 3166-1 numeric code.

    Uses pycountry for direct lookup first (after normalizing the code to a
    3-digit string). If not found, it attempts to find the code in the
    provided BACI and WITS mapping dictionaries, retrieves the country name,
    and uses pycountry's lookup or fuzzy search to find a match.

    Args:
        code: The input country code (expected to be numeric-like).
        baci_map: Dictionary mapping BACI numeric codes (str) to country names (str).
        wits_map: Dictionary mapping WITS numeric codes (str) to country names (str).

    Returns:
        The 3-digit ISO 3166-1 numeric code as a string (e.g., '840'),
        or None if no mapping could be confidently established.
    """
    if code is None:
        return None

    code_str_orig = str(code)
    if not code_str_orig:
        return None

    # 1. Normalize input code to 3-digit string
    cc_str = None
    is_numeric_like = False
    try:
        # Handle potential float inputs like '840.0' by converting to int first
        if isinstance(code, float):
            cc_str = str(int(code)).zfill(3)
            is_numeric_like = True
        else:
            # Handle potential non-numeric strings if possible, or just use original
            try:
                # Try converting to int first to handle "1", "01", etc. consistently
                cc_str = str(int(code_str_orig)).zfill(3)
                is_numeric_like = True
            except ValueError:
                # If it's not directly convertible to int (e.g., 'USA'), keep original for name lookup
                # We won't zfill non-numeric codes here as it might obscure them
                cc_str = code_str_orig
                logger.debug(
                    f"Code '{code_str_orig}' not directly numeric, attempting name lookup."
                )

    except ValueError:
        # This might catch errors if float conversion to int fails unexpectedly
        logger.warning(f"Input code '{code}' could not be normalized. Skipping.")
        return code

    # If normalization failed somehow, exit
    if cc_str is None:
        logger.warning(f"Normalization resulted in None for code '{code}'. Skipping.")
        return None

    # 2. Direct pycountry numeric lookup (Only if normalized code is 3-digit numeric)
    if is_numeric_like and len(cc_str) == 3:
        try:
            detected_country = pycountry.countries.get(numeric=cc_str)
            # Ensure the result has a numeric attribute before returning
            if (
                detected_country
                and hasattr(detected_country, "numeric")
                and detected_country.numeric
            ):
                logger.debug(f"Direct numeric match for '{code}' -> '{detected_country.numeric}'")
                return detected_country.numeric
        except LookupError:
            logger.debug(f"No direct pycountry numeric match for code '{cc_str}'.")
            pass
        except Exception as e:
            logger.error(f"Unexpected error during pycountry numeric lookup for '{cc_str}': {e}")
            pass
    elif is_numeric_like:
        logger.debug(
            f"Normalized code '{cc_str}' is numeric-like but not 3 digits, skipping direct numeric lookup."
        )
    # else: # Non-numeric code, handled below by name lookup

    # 3. Fallback Function (Helper for DRY principle)
    def _find_via_name(name: Optional[str]) -> Optional[str]:
        """Tries direct lookup then fuzzy search on a name."""
        if not name:
            return None
        try:
            # 3a. Try direct lookup first (faster, more accurate)
            direct_match = pycountry.countries.lookup(name)
            if direct_match and hasattr(direct_match, "numeric") and direct_match.numeric:
                logger.debug(f"Direct name lookup match for '{name}' -> '{direct_match.numeric}'")
                return direct_match.numeric
        except LookupError:
            # 3b. Try fuzzy search if direct lookup fails
            try:
                fuzzy_matches: List[pycountry.db.Country] = pycountry.countries.search_fuzzy(name)
                # Filter out results without a numeric code before accessing index 0
                valid_matches = [m for m in fuzzy_matches if hasattr(m, "numeric") and m.numeric]
                if valid_matches:
                    best_match = valid_matches[0]
                    logger.debug(
                        f"Used fuzzy match on name '{name}'. Best guess: {best_match.name} -> '{best_match.numeric}'"
                    )
                    return best_match.numeric
                else:
                    logger.debug(
                        f"Fuzzy search for '{name}' yielded no matches with numeric codes."
                    )
            except Exception as e:
                logger.warning(f"Error during pycountry fuzzy search for name '{name}': {e}")
        return None

    # 4. Attempt fallback using BACI name -> lookup/fuzzy search
    # Use the normalized cc_str as the key for the map
    baci_name = baci_map.get(cc_str)
    if baci_name:
        logger.debug(f"Found BACI name '{baci_name}' for code '{cc_str}'. Trying lookup.")
        baci_result = _find_via_name(baci_name)
        if baci_result:
            return baci_result

    # 5. Attempt fallback using WITS name -> lookup/fuzzy search
    # Use the normalized cc_str as the key for the map
    wits_name = wits_map.get(cc_str)
    if wits_name:
        logger.debug(f"Found WITS name '{wits_name}' for code '{cc_str}'. Trying lookup.")
        wits_result = _find_via_name(wits_name)
        if wits_result:
            return wits_result

    # 6. No match found
    logger.warning(
        f"Could not map country code '{code}' (normalized: '{cc_str}') using available methods."
    )
    return cc_str


def create_country_code_mapping_df(
    lf: pl.LazyFrame,
    code_columns: List[str],
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
    baci_code_col: str = "country_code",  # Column name in BACI ref file
    baci_name_col: str = "country_name",  # Column name in BACI ref file
    wits_code_col: str = "ISO3",  # Assumed column name in WITS ref file !! VERIFY !!
    wits_name_col: str = "Country Name",  # Assumed column name in WITS ref file !! VERIFY !!
) -> pl.DataFrame:
    """
    Generates a Polars DataFrame mapping original country codes found in a
    LazyFrame to standardized ISO 3166-1 numeric codes.

    Args:
        lf: The input Polars LazyFrame.
        code_columns: A list of column names in 'lf' containing country codes to map.
        baci_codes_path: Path to the BACI country codes reference CSV.
        wits_codes_path: Path to the WITS country codes reference CSV.
        baci_code_col: Column name for codes in the BACI reference file.
        baci_name_col: Column name for names in the BACI reference file.
        wits_code_col: Column name for codes in the WITS reference file.
        wits_name_col: Column name for names in the WITS reference file.

    Returns:
        A Polars DataFrame with columns 'original_code' and 'iso_numeric_code'.
    """
    logger.info(f"Generating country code mapping for columns: {code_columns}")

    # Load reference maps
    baci_map = load_reference_map(baci_codes_path, baci_code_col, baci_name_col)
    wits_map = load_reference_map(wits_codes_path, wits_code_col, wits_name_col)

    if not baci_map and not wits_map:
        logger.error("Failed to load both BACI and WITS reference maps. Cannot create mapping.")
        # Return an empty mapping DataFrame with correct schema
        return pl.DataFrame(
            {"original_code": [], "iso_numeric_code": []},
            schema={"original_code": pl.Utf8, "iso_numeric_code": pl.Utf8},
        )

    # Collect unique codes from all specified columns
    logger.info("Collecting unique country codes from the LazyFrame...")
    unique_codes_set: Set[Any] = set()
    try:
        # Select only the necessary columns before collecting unique values
        unique_lf = lf.select(code_columns).unique()
        # It's often more memory efficient to collect column by column if the unique set is large
        # but for country codes (~200-300), collecting together should be fine.
        unique_df = unique_lf.collect()
        for col in code_columns:
            # Add codes, ensuring None is handled gracefully and converting to string
            unique_codes_set.update(str(c) for c in unique_df[col].drop_nulls().to_list())
        logger.info(f"Found {len(unique_codes_set)} unique non-null codes to map.")
    except Exception as e:
        logger.exception(f"Error collecting unique codes: {e}")
        # Return an empty mapping DataFrame
        return pl.DataFrame(
            {"original_code": [], "iso_numeric_code": []},
            schema={"original_code": pl.Utf8, "iso_numeric_code": pl.Utf8},
        )

    # Remap each unique code
    original_codes_list = []
    mapped_codes_list = []
    logger.info("Remapping unique codes...")
    processed_count = 0
    for code in unique_codes_set:
        # Pass the string representation of the code
        mapped_code = remap_country_code_improved(code, baci_map, wits_map)
        original_codes_list.append(code)  # Store original (already string)
        mapped_codes_list.append(mapped_code)  # Can be None if mapping failed
        processed_count += 1
        if processed_count % 50 == 0:  # Log progress periodically
            logger.debug(f"Remapped {processed_count}/{len(unique_codes_set)} unique codes...")

    # Create the mapping DataFrame
    mapping_df = pl.DataFrame(
        {
            "original_code": original_codes_list,
            "iso_numeric_code": mapped_codes_list,
        },
        schema={"original_code": pl.Utf8, "iso_numeric_code": pl.Utf8},
    ).drop_nulls(subset=["original_code"])  # Ensure original_code is not null for joins

    # Log summary of mapping results
    null_mapped_count = mapping_df["iso_numeric_code"].is_null().sum()
    total_unique = len(unique_codes_set)
    # Ensure total_unique is not zero before division
    mapped_percentage = (
        ((total_unique - null_mapped_count) / total_unique * 100) if total_unique > 0 else 0
    )
    logger.info(
        f"Country code mapping DataFrame created. "
        f"Mapped {total_unique - null_mapped_count}/{total_unique} unique codes ({mapped_percentage:.2f}% success)."
    )
    if null_mapped_count > 0:
        logger.warning(
            f"{null_mapped_count} unique codes could not be mapped to an ISO numeric code."
        )
        # Optionally log the first few failed codes for debugging:
        failed_codes_sample = (
            mapping_df.filter(pl.col("iso_numeric_code").is_null())["original_code"]
            .head(10)
            .to_list()
        )
        logger.warning(f"Sample of unmapped codes: {failed_codes_sample}")

    return mapping_df


def apply_country_code_mapping(
    lf: pl.LazyFrame,
    mapping_df: pl.DataFrame,
    original_col_name: str,
    new_col_name: Optional[str] = None,
    drop_original: bool = True,
) -> pl.LazyFrame:
    """
    Applies the country code mapping to a specific column in a LazyFrame using a left join.

    Args:
        lf: The input Polars LazyFrame.
        mapping_df: The DataFrame containing 'original_code' and 'iso_numeric_code' columns.
        original_col_name: The name of the column in 'lf' to map.
        new_col_name: The desired name for the new column with mapped codes.
                      Defaults to original_col_name if None.
        drop_original: Whether to drop the original column after mapping.

    Returns:
        The modified Polars LazyFrame with the mapped country codes.
    """
    target_col_name = new_col_name if new_col_name is not None else original_col_name

    logger.info(
        f"Applying country code mapping to column '{original_col_name}' -> '{target_col_name}'"
    )

    # Ensure the original column is string type for joining with mapping_df
    # Also handle potential integers or other types that need casting
    lf = lf.with_columns(pl.col(original_col_name).cast(pl.Utf8))

    # Define a temporary unique name for the joined column to avoid conflicts
    temp_mapped_col = f"__{original_col_name}_iso_numeric_temp__"

    # Perform the join
    # Use lazy() on the mapping_df for potentially better optimization by Polars
    lf = lf.join(
        mapping_df.lazy().rename({"iso_numeric_code": temp_mapped_col}),  # Rename before join
        left_on=original_col_name,
        right_on="original_code",
        how="left",
    )

    logger.debug(f"Schema after join: {lf.collect_schema()}")

    # Use coalesce to fill nulls from the join with the original code
    # Create the final target column using coalesce
    lf = lf.with_columns(
        pl.coalesce(
            pl.col(temp_mapped_col),  # Use the mapped code if available (not null)
            pl.col(original_col_name),  # Otherwise, use the original code
        ).alias(target_col_name)  # Name the resulting column
    )

    # Drop the temporary mapped column as it's now incorporated into target_col_name
    lf = lf.drop(temp_mapped_col)
    logger.debug(f"Schema after coalesce and drop temp: {lf.collect_schema()}")

    # Drop the original column if requested AND if the target name is different
    if drop_original and original_col_name != target_col_name:
        # Check if original column still exists before dropping
        if original_col_name in lf.columns:
            logger.debug(f"Dropping original column '{original_col_name}'")
            lf = lf.drop(original_col_name)
        else:
            logger.warning(
                f"Requested to drop original column '{original_col_name}', but it was not found (possibly same as target)."
            )

    logger.info(
        f"Finished applying mapping for column '{original_col_name}'. Final schema: {lf.collect_schema()}"
    )
    return lf
