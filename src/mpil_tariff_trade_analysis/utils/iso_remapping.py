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


def create_country_code_mapping_df(
    lf: pl.LazyFrame,
    code_columns: List[str],
    baci_codes_path: str | Path = DEFAULT_BACI_COUNTRY_CODES_PATH,
    wits_codes_path: str | Path = DEFAULT_WITS_COUNTRY_CODES_PATH,
    baci_code_col: str = "country_code",
    baci_name_col: str = "country_name",
    wits_code_col: str = "ISO3",  # Assumed column name in WITS ref file !! VERIFY !!
    wits_name_col: str = "Country Name",  # Assumed column name in WITS ref file !! VERIFY !!
    drop_original: bool = True,  # Option to keep original columns
) -> pl.LazyFrame:
    """
    Applies ISO 3166-1 numeric country code remapping to specified columns
    in a LazyFrame, handling hardcoded group expansions (like EFTA).

    It uses the `remap_country_code_improved` function, explodes the results
    for multi-country mappings, and replaces or adds the mapped columns.

    Args:
        lf: The input Polars LazyFrame.
        code_columns: A list of column names in 'lf' containing country codes to map.
        baci_codes_path: Path to the BACI country codes reference CSV.
        wits_codes_path: Path to the WITS country codes reference CSV.
        baci_code_col: Column name for codes in the BACI reference file.
        baci_name_col: Column name for names in the BACI reference file.
        wits_code_col: Column name for codes in the WITS reference file.
        wits_name_col: Column name for names in the WITS reference file.
        drop_original: If True, the original code columns are dropped.

    Returns:
        A Polars LazyFrame with the specified columns remapped to ISO numeric codes.
        Rows might be duplicated if a code maps to multiple ISO codes (e.g., EFTA).
    """
    logger.info(f"Starting country code remapping for columns: {code_columns}")

    # Load reference maps once
    baci_map = load_reference_map(baci_codes_path, baci_code_col, baci_name_col)
    wits_map = load_reference_map(wits_codes_path, wits_code_col, wits_name_col)

    if not baci_map or not wits_map:
        logger.error("Either BACI and WITS reference maps failed to load.")
        raise

    original_cols = lf.columns
    lf_processed = lf
    new_col_names_map = {}  # Map original col name to new col name

    for col_name in code_columns:
        if col_name not in lf.columns:
            logger.warning(f"Column '{col_name}' not found in LazyFrame schema. Skipping.")
            continue

        logger.info(f"Processing country codes in column: '{col_name}'")
        temp_list_col = f"__iso_list_{col_name}__"  # Temporary column for the list
        final_col_name = f"{col_name}_iso_numeric"  # Final column name after explode
        new_col_names_map[col_name] = final_col_name

        # Apply the updated mapping function which returns a list or None
        lf_processed = lf_processed.with_columns(
            pl.col(col_name)
            .map_elements(
                lambda code: remap_country_code_improved(code, baci_map, wits_map),
                return_dtype=pl.List(pl.Utf8),  # Expecting a list of strings
                skip_nulls=False,  # Let the function handle None input if necessary
                strategy="thread_local",  # Potentially faster for complex functions
            )
            .alias(temp_list_col)
        )

        # Filter out rows where mapping failed (returned None) before exploding
        lf_processed = lf_processed.filter(pl.col(temp_list_col).is_not_null())

        # Explode the list column to handle multi-country groups (like EFTA)
        lf_processed = lf_processed.explode(temp_list_col)

        # Rename the exploded column (which now contains single ISO codes)
        lf_processed = lf_processed.rename({temp_list_col: final_col_name})

        logger.debug(
            f"Remapped column '{col_name}' to '{final_col_name}'. Schema: {lf_processed.collect_schema()}"
        )

    # Determine final column set
    cols_to_keep = list(original_cols)  # Start with all original columns
    if drop_original:
        # Remove the original columns that were successfully processed
        for original_name in new_col_names_map.keys():
            if original_name in cols_to_keep:
                cols_to_keep.remove(original_name)

    # Add the new ISO numeric columns
    final_column_selection = cols_to_keep + list(new_col_names_map.values())

    # Select the final set of columns
    # Ensure the selection doesn't fail if a column wasn't processed
    valid_final_columns = [col for col in final_column_selection if col in lf_processed.columns]
    final_lf = lf_processed.select(valid_final_columns)

    logger.info(f"Finished country code remapping. Final schema: {final_lf.collect_schema()}")
    return final_lf
