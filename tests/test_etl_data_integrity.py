import math  # For checking approximate equality
from pathlib import Path

import polars as pl
import pytest

# --- Configuration ---
# Adjust base path if your test execution context is different
# Assumes tests run from the project root directory
TEST_DATA_DIR = Path("data")
INTERMEDIATE_DIR = TEST_DATA_DIR / "intermediate"
FINAL_DIR = TEST_DATA_DIR / "final"

# Paths to CLEANED intermediate data (outputs of cleaning pipelines)
BACI_CLEANED_PATH = INTERMEDIATE_DIR / "BACI_HS92_V202501_cleaned_remapped.parquet" # Example filename
WITS_MFN_CLEANED_PATH = INTERMEDIATE_DIR / "cleaned_wits_mfn" / "WITS_AVEMFN_cleaned.parquet"
WITS_PREF_CLEANED_PATH = INTERMEDIATE_DIR / "cleaned_wits_pref" / "WITS_AVEPref_cleaned_expanded.parquet"

# Path to final merged output
FINAL_OUTPUT_PATH_PARTITIONED = FINAL_DIR / "unified_trade_tariff_partitioned" # Keep consistent name


# --- Fixtures ---
@pytest.fixture(scope="module")
def cleaned_baci_df() -> pl.LazyFrame:
    """Loads the cleaned BACI data."""
    if not BACI_CLEANED_PATH.exists():
        pytest.skip(f"Cleaned BACI data not found at {BACI_CLEANED_PATH}")
    return pl.scan_parquet(BACI_CLEANED_PATH)


@pytest.fixture(scope="module")
def cleaned_wits_mfn_df() -> pl.LazyFrame:
    """Loads the cleaned WITS MFN data."""
    if not WITS_MFN_CLEANED_PATH.exists():
        pytest.skip(f"Cleaned WITS MFN data not found at {WITS_MFN_CLEANED_PATH}")
    return pl.scan_parquet(WITS_MFN_CLEANED_PATH)


@pytest.fixture(scope="module")
def cleaned_wits_pref_df() -> pl.LazyFrame:
    """Loads the cleaned and expanded WITS Preferential data."""
    if not WITS_PREF_CLEANED_PATH.exists():
        pytest.skip(f"Cleaned WITS Preferential data not found at {WITS_PREF_CLEANED_PATH}")
    return pl.scan_parquet(WITS_PREF_CLEANED_PATH)


# Optional BACI fixture if needed for specific checks

# Fixture for the final merged dataset
@pytest.fixture(scope="module")
def final_df() -> pl.LazyFrame:
    """Loads the final merged dataset (partitioned)."""
    # Check for partitioned data
    partitioned_glob_path = str(FINAL_OUTPUT_PATH_PARTITIONED / "**/*.parquet")
    if FINAL_OUTPUT_PATH_PARTITIONED.is_dir() and list(
        FINAL_OUTPUT_PATH_PARTITIONED.glob("**/*.parquet")
    ):
        print(f"Loading final partitioned data using glob: {partitioned_glob_path}")
        # scan_parquet handles glob patterns for partitioned datasets
        return pl.scan_parquet(partitioned_glob_path)
    else:
        pytest.skip(
            f"Final partitioned output data not found at {FINAL_OUTPUT_PATH_PARTITIONED}"
        )


# --- Helper Validation Function ---
def assert_row_exists(
    lf: pl.LazyFrame,
    filter_criteria: dict,
    expected_values: dict,
    tolerance: float = 0.01,  # Allow 1% difference for numeric values
    check_for_unique: bool = True,
):
    """
    Filters a LazyFrame based on criteria, collects the result,
    and asserts that the specified values match expected values within a tolerance.

    Args:
        lf: The Polars LazyFrame to query.
        filter_criteria: A dictionary where keys are column names and values are
                         the values to filter by (e.g., {"col_a": 1, "col_b": "x"}).
        expected_values: A dictionary where keys are column names and values are
                         the expected values for the *single* matching row.
        tolerance: The relative tolerance allowed for numeric comparisons.
                   Use None for exact comparison.
        check_for_unique: If True, assert that exactly one row matches the criteria.
    """
    query = lf
    filter_expressions = []
    for col, val in filter_criteria.items():
        # Ensure the column exists before filtering
        if col not in lf.columns:
            pytest.fail(
                f"Column '{col}' used in filter criteria not found in DataFrame schema: {lf.columns}"
            )
        # Adapt filter for potential list columns in BACI before explode in merge
        # This helper might need adjustment depending on when/where it's used.
        # For final_df, columns should be simple types after merge.
        filter_expressions.append(pl.col(col) == val)

    if filter_expressions:
        query = query.filter(pl.all_horizontal(filter_expressions))

    # Use streaming=True for potentially large final dataset
    result_df = query.collect(streaming=True)
    num_rows = len(result_df)

    if check_for_unique:
        assert num_rows == 1, (
            f"Expected exactly 1 row for criteria {filter_criteria}, but found {num_rows}."
        )
    else:
        assert num_rows > 0, (
            f"Expected at least 1 row for criteria {filter_criteria}, but found none."
        )

    # Take the first row if multiple found and check_for_unique is False
    result_row = result_df.row(0, named=True)

    for col, expected_val in expected_values.items():
        assert col in result_row, (
            f"Expected column '{col}' not found in result row. Available columns: {list(result_row.keys())}"
        )
        actual_val = result_row[col]

        # Attempt conversion to float for numeric comparison if types might be mixed (e.g., tariff as string)
        is_expected_numeric = isinstance(expected_val, (int, float))
        actual_numeric = None

        if is_expected_numeric:
            if actual_val is None:
                assert False, (
                    f"Column '{col}' is None, expected numeric value {expected_val} for criteria {filter_criteria}"
                )
            try:
                # Handle potential string representations of numbers
                actual_numeric = float(str(actual_val).strip())
            except (ValueError, TypeError):
                assert False, (
                    f"Column '{col}' could not be converted to float for comparison. Value: '{actual_val}' (Type: {type(actual_val)}), Criteria: {filter_criteria}"
                )

        if is_expected_numeric and tolerance is not None and actual_numeric is not None:
            # Use math.isclose for robust floating-point comparison
            assert math.isclose(actual_numeric, expected_val, rel_tol=tolerance), (
                f"Column '{col}' mismatch: Expected approx {expected_val} (tol={tolerance}), got {actual_numeric} for criteria {filter_criteria}"
            )
        elif is_expected_numeric and tolerance is not None and actual_numeric is None:
            # This case is handled by the initial None check above, but kept for clarity
            assert False, (
                f"Column '{col}' is None after conversion attempt, expected numeric value {expected_val} for criteria {filter_criteria}"
            )
        else:
            # Exact comparison for strings, integers (if tolerance is None), etc.
            # Handle comparison with None explicitly
            if expected_val is None:
                assert actual_val is None, (
                    f"Column '{col}' mismatch: Expected None, got {actual_val} (Type: {type(actual_val)}) for criteria {filter_criteria}"
                )
            elif actual_val is None:
                 assert False, (
                    f"Column '{col}' mismatch: Expected {expected_val}, got None for criteria {filter_criteria}"
                )
            # Explicitly cast expected value to string if comparing with string actual_val
            elif isinstance(actual_val, str) and not isinstance(expected_val, str):
                assert actual_val == str(expected_val), (
                    f"Column '{col}' mismatch (string vs non-string): Expected '{expected_val}', got '{actual_val}' for criteria {filter_criteria}"
                )
            else:
                assert actual_val == expected_val, (
                    f"Column '{col}' mismatch: Expected {expected_val} (Type: {type(expected_val)}), got {actual_val} (Type: {type(actual_val)}) for criteria {filter_criteria}"
                )


# --- Test Cases ---

# --- Cleaned WITS MFN Data Validation ---
@pytest.mark.parametrize(
    "reporter_iso, product_hs92, year_str, expected_tariff_str",
    [
        # Format: ("REPORTER_ISO_NUMERIC_STR", "HS92_CODE_STR", "YEAR_STR", "EXPECTED_MFN_TARIFF_STR_OR_NONE"),
        # Using string types as loaded by WITS_cleaner/pipeline
        # Sri Lanka (144), HS 310520, 2008 -> Expected MFN Tariff 2.5
        ("144", "310520", "2008", "2.5"),
        # Georgia (268), HS 300450, 2011 -> 0.0
        ("268", "300450", "2011", "0"), # WITS might store as "0"
        # Costa Rica (188), HS 321490, 2008 -> 5.0
        ("188", "321490", "2008", "5"),
        # Netherlands (528), HS 901839, 2007 -> 0.0
        ("528", "901839", "2007", "0"),
        # Panama (591), HS 290950, 2011 -> 0.0
        ("591", "290950", "2011", "0"),
        # USA (840), HS 845011, 2019 -> 1.4
        ("840", "845011", "2019", "1.4"),
        # USA (840), HS 845019, 2019 -> 1.8
        ("840", "845019", "2019", "1.8"),
        # Turkmenistan (795), HS 845019, 2019 -> No data (expect row might be missing or tariff is null)
        # Test for missing row or null value separately if needed. This checks for existing row with None.
        # ("795", "845019", "2019", None), # How WITS represents missing data needs verification ("NA", "", null?)
    ],
)
def test_cleaned_wits_mfn_values(cleaned_wits_mfn_df, reporter_iso, product_hs92, year_str, expected_tariff_str):
    """Verify specific MFN tariff values exist in the cleaned WITS MFN dataset."""
    # Use column names AFTER cleaning/renaming ('t', 'i', 'k', 'mfn_tariff_rate')
    criteria = {"i": reporter_iso, "k": product_hs92, "t": year_str}
    expected = {"mfn_tariff_rate": expected_tariff_str} # Compare as strings initially
    # Use exact comparison (tolerance=None) as we expect specific string representations from source
    assert_row_exists(cleaned_wits_mfn_df, criteria, expected, tolerance=None)


# --- Cleaned WITS Preferential Data Validation ---
@pytest.mark.parametrize(
    "reporter_iso, partner_iso, product_hs92, year_str, expected_tariff_str",
    [
        # --- Add validated cases AFTER expansion ---
        # Format: ("REP_ISO_STR", "PARTNER_ISO_STR", "HS92_CODE_STR", "YEAR_STR", "EXP_PREF_TARIFF_STR_OR_NONE"),
        # Example: If EU (e.g., 97) gives preference to CH (756) for product X in year Y
        # Find a specific case from the raw data and trace its expansion.
        # E.g., DE (276) reporting pref for CH (756) on HS 870321 in 2010 might be 0.0
        ("276", "756", "870321", "2010", "0"), # Hypothetical - VALIDATE THIS!
        # ("...", "...", "...", "...", "..."),  # Placeholder - REMOVE OR REPLACE
    ],
)
def test_cleaned_wits_pref_values(cleaned_wits_pref_df, reporter_iso, partner_iso, product_hs92, year_str, expected_tariff_str):
    """Verify specific Preferential tariff values exist in the cleaned & expanded WITS Pref dataset."""
    # Use column names AFTER cleaning/expansion/renaming ('t', 'i', 'j', 'k', 'pref_tariff_rate')
    criteria = {
        "i": reporter_iso,
        "j": partner_iso, # Partner is now individual ISO code
        "k": product_hs92,
        "t": year_str,
    }
    expected = {"pref_tariff_rate": expected_tariff_str} # Compare as strings
    assert_row_exists(cleaned_wits_pref_df, criteria, expected, tolerance=None)


# --- Final Merged Output Validation ---
@pytest.mark.parametrize(
    "source_iso, target_iso, hs_code_hs92, year_str, expected_value, expected_qty, expected_effective_tariff",
    [
        # Format: ("SOURCE_ISO_STR", "TARGET_ISO_STR", "HS92_CODE_STR", "YEAR_STR", EXP_VALUE_FLOAT, EXP_QTY_FLOAT, EXP_EFFECTIVE_TARIFF_FLOAT_OR_NONE),
        # --- Example Case from NETWORK.py (using ISO codes) ---
        # Germany ("276") -> Sri Lanka ("144"), HS "310520", 2008
        # Value: 11700, Qty: 30000. MFN was 2.5. Assume no pref. Effective = 2.5
        ("276", "144", "310520", "2008", 11700.0, 30000.0, 2.5),
        # --- Romania ("642") -> Georgia ("268"), HS "040690", 2000 ---
        # Value: 7829, Qty: 250. MFN was 0.0. Assume no pref. Effective = 0.0
        ("642", "268", "040690", "2000", 7829.0, 250.0, 0.0),
        # --- Add more validated cases, tracing from BACI + Cleaned Tariffs ---
        # USA (840) -> Mexico (484), HS 845011, 2019. BACI V/Q? MFN=1.4. Pref (NAFTA/USMCA)? Assume 0.0. Effective=0.0
        # Need actual BACI V/Q for this row. Let's assume V=100k, Q=500 for example.
        ("840", "484", "845011", "2019", 100000.0, 500.0, 0.0), # Hypothetical V/Q - VALIDATE! Tariff should be 0 due to NAFTA/USMCA.
        # ("...", "...", "...", "...", ..., ..., ...), # Placeholder - REMOVE OR REPLACE
    ],
)
def test_final_output_values(
    final_df, source_iso, target_iso, hs_code_hs92, year_str, expected_value, expected_qty, expected_effective_tariff
):
    """Verify specific trade flows have the correct value, quantity, and effective tariff in the final output."""
    # Use column names from create_final_table ('Year', 'Source', 'Target', 'HS_Code', 'Value', 'Quantity', 'effective_tariff_rate')
    criteria = {"Source": source_iso, "Target": target_iso, "HS_Code": hs_code_hs92, "Year": year_str}
    expected = {
        "Value": expected_value,
        "Quantity": expected_qty,
        "effective_tariff_rate": expected_effective_tariff, # This is the numeric coalesced rate
    }
    # Use tolerance for numeric comparisons (Value, Quantity, Tariff)
    assert_row_exists(final_df, criteria, expected, tolerance=0.01) # Adjust tolerance if needed
