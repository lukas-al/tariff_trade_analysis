from pathlib import Path
from typing import Any, Dict, List, Optional  # Added List, Set

import polars as pl
import pycountry
import pycountry.db

from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Default paths for reference data ---
# Adjust these paths as necessary
DEFAULT_BACI_COUNTRY_CODES_PATH = Path("data/raw/BACI_HS92_V202501/country_codes_V202501.csv")
DEFAULT_WITS_COUNTRY_CODES_PATH = Path("data/raw/WITS_country_codes.csv")

# --- Hardcoded Mappings ---
# Map specific input codes directly to lists of ISO numeric codes
HARDCODED_CODE_MAP = {
    "697": [
        "352",
        "438",
        "578",
        "756",
    ],  # Europe EFTA, nes -> Iceland, Liechtenstein, Norway, Switzerland
    "490": ["158"],  # Other Asia, nes -> Taiwan (ISO 3166-1 numeric is 158)
}


def load_reference_map(file_path: str | Path, code_col: str, name_col: str) -> Dict[str, str]:
    """Loads a CSV file into a dictionary mapping codes to names."""
    file_path = Path(file_path)  # Ensure it's a Path object
    logger.info(f"Attempting to load reference map from: {file_path}")
    try:
        if not file_path.is_file():
            logger.error(f"Reference map file not found: {file_path}")
            return {}

        # Read all as strings initially to avoid type inference issues
        df = pl.read_csv(file_path, infer_schema=False)
        df_clean = df.select([code_col, name_col]).drop_nulls()
        # Ensure keys are strings after cleaning potential numeric interpretations
        mapping = dict(zip(df_clean[code_col], df_clean[name_col], strict=False))
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
) -> Optional[List[str]]:  # <--- Changed return type annotation
    """
    Remaps a given country code to a list of standard ISO 3166-1 numeric codes.

    Checks a hardcoded map first. If not found, uses pycountry for direct
    lookup (after normalizing the code). If still not found, it attempts
    to find the code in the provided BACI and WITS mapping dictionaries,
    retrieves the country name, and uses pycountry's lookup or fuzzy search.

    Args:
        code: The input country code (expected to be numeric-like or string).
        baci_map: Dictionary mapping BACI codes (str) to country names (str).
        wits_map: Dictionary mapping WITS codes (str) to country names (str).

    Returns:
        A list of 3-digit ISO 3166-1 numeric code strings (e.g., ['840']),
        or None if no mapping could be confidently established.
        The list will contain multiple codes for specific group mappings (e.g., EFTA).
    """
    if code is None:
        return None

    # --- 1. Normalize Input Code ---
    normalized_code_str = None
    try:
        # Attempt conversion for numeric-like codes first (e.g., 840.0, 490)
        # Convert to float then int handles cases like '840.0' -> 840
        normalized_code_str = str(int(float(str(code))))
    except (ValueError, TypeError):
        # Handle non-numeric codes or codes that fail conversion (e.g., 'USA', '697')
        normalized_code_str = str(code).strip()

    if not normalized_code_str:
        logger.debug(f"Skipping empty or None code after normalization: original='{code}'")
        return None  # Cannot process empty string

    # --- 2. Check Hardcoded Map ---
    if normalized_code_str in HARDCODED_CODE_MAP:
        iso_codes = HARDCODED_CODE_MAP[normalized_code_str]
        logger.debug(
            f"Code '{code}' (normalized: '{normalized_code_str}') mapped via hardcoded rule to {iso_codes}"
        )
        # Ensure codes are 3 digits if they came from the hardcoded map
        return [c.zfill(3) for c in iso_codes]

    # --- 3. Direct Pycountry Lookup (on normalized code) ---
    try:
        country = None
        # Try common lookups based on typical code formats
        if len(normalized_code_str) == 2 and normalized_code_str.isalpha():
            country = pycountry.countries.get(alpha_2=normalized_code_str.upper())
        elif len(normalized_code_str) == 3 and normalized_code_str.isalpha():
            country = pycountry.countries.get(alpha_3=normalized_code_str.upper())
        elif normalized_code_str.isdigit():
            # Pad with leading zeros if needed for numeric lookup
            padded_code = normalized_code_str.zfill(3)
            country = pycountry.countries.get(numeric=padded_code)
        # Add other direct lookups if needed (e.g., official_name)

        if country and hasattr(country, "numeric") and country.numeric:
            iso_numeric = country.numeric.zfill(3)
            logger.debug(
                f"Code '{code}' (normalized: '{normalized_code_str}') mapped via direct pycountry lookup to {iso_numeric}"
            )
            return [iso_numeric]  # Return as list
    except LookupError:
        logger.debug(
            f"Direct pycountry lookup for normalized code '{normalized_code_str}' failed. Trying name lookup."
        )
    except Exception as e:
        logger.warning(
            f"Unexpected error during direct pycountry lookup for code '{code}' (normalized: '{normalized_code_str}'): {e}"
        )

    # --- 4. Find Name using Reference Maps ---
    # Use the *normalized* code string as the key for the maps
    country_name = baci_map.get(normalized_code_str) or wits_map.get(normalized_code_str)

    if not country_name:
        logger.warning(
            f"Code '{code}' (normalized: '{normalized_code_str}') not found in BACI or WITS reference maps. Cannot map via name."
        )
        # Return the original normalized code as a list if no mapping found? Or None?
        # Returning None indicates failure to map to ISO standard.
        return None  # Indicate failure

    # --- 5. Pycountry Lookup/Search via Name ---
    def _find_via_name(name: Optional[str]) -> Optional[str]:
        """Tries direct lookup then fuzzy search on a name. Returns 3-digit numeric string."""
        if not name:
            return None
        try:
            # 5a. Try direct lookup by name (more reliable)
            direct_match = pycountry.countries.lookup(name)
            if direct_match and hasattr(direct_match, "numeric") and direct_match.numeric:
                logger.debug(
                    f"Direct name lookup match for '{name}' -> '{direct_match.numeric.zfill(3)}'"
                )
                return direct_match.numeric.zfill(3)
        except LookupError:
            # 5b. Try fuzzy search if direct lookup fails
            try:
                fuzzy_matches: List[pycountry.db.Country] = pycountry.countries.search_fuzzy(name)  # type: ignore
                valid_matches = [m for m in fuzzy_matches if hasattr(m, "numeric") and m.numeric]
                if valid_matches:
                    best_match = valid_matches[0]
                    # Maybe add logging here if multiple fuzzy results? Could indicate ambiguity.
                    logger.debug(
                        f"Used fuzzy match on name '{name}'. Best guess: {best_match.name} -> '{best_match.numeric.zfill(3)}'"
                    )
                    return best_match.numeric.zfill(3)
                else:
                    logger.debug(
                        f"Fuzzy search for '{name}' yielded no matches with numeric codes."
                    )
            except Exception as e:  # Catch potential errors during fuzzy search itself
                logger.warning(f"Error during pycountry fuzzy search for name '{name}': {e}")
        return None

    iso_code = _find_via_name(country_name)
    if iso_code:
        logger.debug(
            f"Code '{code}' (Name: '{country_name}') mapped via pycountry name lookup/fuzzy search to {iso_code}"
        )
        return [iso_code]  # Return as list
    else:
        logger.warning(
            f"Could not remap code '{code}' (Name: '{country_name}') to ISO numeric via pycountry name lookup."
        )
        # Return the original normalized code as a list if no mapping found? Or None?
        # Returning None indicates failure to map to ISO standard.
        return None  # Indicate failure


# --- New Vectorized Approach ---

def create_iso_mapping_table(
    unique_codes: pl.Series,
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
    baci_code_col: str = "country_code",
    baci_name_col: str = "country_name",
    wits_code_col: str = "ISO3",
    wits_name_col: str = "Country Name",
) -> pl.DataFrame:
    """
    Creates a mapping table from original codes to lists of ISO numeric codes.

    Args:
        unique_codes: A Polars Series containing unique original country codes.
        baci_codes_path: Path to the BACI country codes reference CSV.
        wits_codes_path: Path to the WITS country codes reference CSV.
        baci_codes_path: Path to the BACI country codes reference CSV.
        wits_codes_path: Path to the WITS country codes reference CSV.
        baci_code_col: Column name for codes in the BACI reference file.
        baci_name_col: Column name for names in the BACI reference file.
        wits_code_col: Column name for codes in the WITS reference file.
        wits_name_col: Column name for names in the WITS reference file.

    Returns:
        A Polars DataFrame with columns 'original_code' and 'iso_numeric_list'.
        'iso_numeric_list' contains lists of ISO 3166-1 numeric code strings.
        Returns an empty DataFrame if reference maps cannot be loaded.
    """
    logger.info(f"Creating ISO mapping table for {len(unique_codes)} unique codes.")

    # --- 1. Load Reference Maps ---
    try:
        baci_map = load_reference_map(baci_codes_path, baci_code_col, baci_name_col)
        wits_map = load_reference_map(wits_codes_path, wits_code_col, wits_name_col)
    except Exception as e:
        logger.exception(f"Failed to load reference maps: {e}")
        return pl.DataFrame(
            {"original_code": [], "iso_numeric_list": []},
            schema={"original_code": pl.Utf8, "iso_numeric_list": pl.List(pl.Utf8)},
        )

    if not baci_map and not wits_map:
        logger.error(
            "Both BACI and WITS reference maps failed to load or are empty. Cannot create mapping."
        )
        return pl.DataFrame(
            {"original_code": [], "iso_numeric_list": []},
            schema={"original_code": pl.Utf8, "iso_numeric_list": pl.List(pl.Utf8)},
        )
    elif not baci_map:
        logger.warning("BACI reference map failed to load or is empty.")
    elif not wits_map:
        logger.warning("WITS reference map failed to load or is empty.")

    # --- 2. Map Unique Codes ---
    mapping_data = []
    processed_count = 0
    failed_count = 0
    # Ensure unique codes are strings for consistent lookup
    unique_codes_str = unique_codes.cast(pl.Utf8).drop_nulls()

    for code in unique_codes_str:
        iso_list = remap_country_code_improved(code, baci_map, wits_map)
        if iso_list:  # Only add if mapping was successful
            mapping_data.append({"original_code": code, "iso_numeric_list": iso_list})
            processed_count += 1
        else:
            # Log the failure for the specific code
            logger.debug(f"Failed to map original code: '{code}'")
            failed_count += 1

    logger.info(
        f"Finished mapping unique codes. Success: {processed_count}, Failures: {failed_count}"
    )

    if not mapping_data:
        logger.warning("No successful mappings found for any unique codes.")
        return pl.DataFrame(
            {"original_code": [], "iso_numeric_list": []},
            schema={"original_code": pl.Utf8, "iso_numeric_list": pl.List(pl.Utf8)},
        )

    # --- 3. Create Mapping DataFrame ---
    mapping_df = pl.DataFrame(
        mapping_data,
        schema={"original_code": pl.Utf8, "iso_numeric_list": pl.List(pl.Utf8)},
    )

    logger.info(f"Created mapping table with {len(mapping_df)} entries.")
    return mapping_df


def apply_iso_mapping(
    lf: pl.LazyFrame,
    code_column_name: str,
    mapping_df: pl.DataFrame,
    output_list_col_name: Optional[str] = None,
    drop_original: bool = True,
) -> pl.LazyFrame:
    """
    Applies a pre-computed ISO code mapping table to a column in a LazyFrame using a join.

    Args:
        lf: The input Polars LazyFrame.
        code_column_name: The name of the column in 'lf' containing the original codes.
        mapping_df: The mapping DataFrame created by `create_iso_mapping_table`.
                    Must contain 'original_code' and 'iso_numeric_list' columns.
        output_list_col_name: The desired name for the new column containing the list
                              of ISO codes. Defaults to f"{code_column_name}_iso_list".
        drop_original: If True, the original code column is dropped after mapping.

    Returns:
        A Polars LazyFrame with the mapped ISO code list column added/replaced.
    """
    if code_column_name not in lf.columns:
        logger.error(f"Column '{code_column_name}' not found in LazyFrame. Cannot apply mapping.")
        return lf

    if not isinstance(mapping_df, pl.DataFrame) or mapping_df.is_empty():
        logger.warning(
            f"Mapping table is empty or invalid for column '{code_column_name}'. Skipping mapping."
        )
        return lf

    if "original_code" not in mapping_df.columns or "iso_numeric_list" not in mapping_df.columns:
        logger.error(
            "Mapping table is missing required columns ('original_code', 'iso_numeric_list')."
        )
        return lf

    if output_list_col_name is None:
        output_list_col_name = f"{code_column_name}_iso_list"

    logger.info(
        f"Applying ISO mapping to column '{code_column_name}', creating '{output_list_col_name}'."
    )

    # Ensure the join keys have compatible types (cast original code column to Utf8)
    lf_casted = lf.with_columns(pl.col(code_column_name).cast(pl.Utf8))

    # Perform the left join
    lf_joined = lf_casted.join(
        mapping_df.lazy(),
        left_on=code_column_name,
        right_on="original_code",
        how="left",
    )

    # Rename the joined list column
    # Note: The column from the right side of the join might be named "iso_numeric_list_right"
    # if "iso_numeric_list" already existed. Polars handles this, but let's be explicit.
    # We assume the new column is named "iso_numeric_list" after the join if it wasn't present,
    # or "iso_numeric_list_right" if it was. Let's check the schema.
    joined_schema = lf_joined.collect_schema()
    iso_list_col_actual = "iso_numeric_list"
    if iso_list_col_actual not in joined_schema:
        iso_list_col_actual = "iso_numeric_list_right"
        if iso_list_col_actual not in joined_schema:
            logger.error(
                f"Could not find the expected list column ('iso_numeric_list' or 'iso_numeric_list_right') after join for '{code_column_name}'. Aborting rename."
            )
            # Return the joined frame without rename/drop if the column isn't found
            return lf_joined

    lf_renamed = lf_joined.rename({iso_list_col_actual: output_list_col_name})

    # Drop the original column if requested
    if drop_original:
        logger.debug(f"Dropping original column: '{code_column_name}'")
        lf_final = lf_renamed.drop(code_column_name)
    else:
        lf_final = lf_renamed

    # Log rows that didn't get mapped (where the new list column is null)
    # This check is better done after collect, but we can add a note here.
    logger.info(
        f"Mapping applied for '{code_column_name}'. Check for nulls in '{output_list_col_name}' for codes that failed to map."
    )

    return lf_final
