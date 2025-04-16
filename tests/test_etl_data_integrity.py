
import pytest
import polars as pl
from pathlib import Path
import math # For checking approximate equality

# --- Configuration ---
# Adjust base path if your test execution context is different
# Assumes tests run from the project root directory
TEST_DATA_DIR = Path("data")
BACI_PATH = TEST_DATA_DIR / "intermediate" / "BACI_HS92_V202501" # Assuming dir of files
WITS_MFN_PATH = TEST_DATA_DIR / "intermediate" / "WITS_AVEMFN.parquet"
WITS_PREF_PATH = TEST_DATA_DIR / "intermediate" / "WITS_AVEPref.parquet"
FINAL_OUTPUT_PATH_SINGLE = TEST_DATA_DIR / "final" / "unified_trade_tariff.parquet"
FINAL_OUTPUT_PATH_PARTITIONED = TEST_DATA_DIR / "final" / "unified_trade_tariff_partitioned"

# --- Fixtures ---
@pytest.fixture(scope="module")
def wits_mfn_df() -> pl.LazyFrame:
    """Loads the cleaned WITS MFN data."""
    if not WITS_MFN_PATH.exists():
        pytest.skip(f"WITS MFN data not found at {WITS_MFN_PATH}")
    return pl.scan_parquet(WITS_MFN_PATH)

@pytest.fixture(scope="module")
def wits_pref_df() -> pl.LazyFrame:
    """Loads the cleaned WITS Preferential data."""
    if not WITS_PREF_PATH.exists():
        pytest.skip(f"WITS Preferential data not found at {WITS_PREF_PATH}")
    return pl.scan_parquet(WITS_PREF_PATH)

# Optional BACI fixture if needed for specific checks
# @pytest.fixture(scope="module")
# def baci_df() -> pl.LazyFrame:
#     """Loads the cleaned BACI data."""
#     baci_glob_path = str(BACI_PATH / "*.parquet") # Example if multiple files
#     if not list(BACI_PATH.glob("*.parquet")): # Check if any files match
#         pytest.skip(f"BACI data not found at {BACI_PATH}")
#     return pl.scan_parquet(baci_glob_path)

@pytest.fixture(scope="module")
def final_df() -> pl.LazyFrame:
    """Loads the final merged dataset."""
    final_path = None
    # Check for partitioned data first
    partitioned_glob_path = str(FINAL_OUTPUT_PATH_PARTITIONED / "**/*.parquet")
    if FINAL_OUTPUT_PATH_PARTITIONED.is_dir() and list(FINAL_OUTPUT_PATH_PARTITIONED.glob("**/*.parquet")):
        final_path = partitioned_glob_path # Use glob pattern for partitioned data
        print(f"Loading partitioned data using glob: {final_path}")
    elif FINAL_OUTPUT_PATH_SINGLE.exists():
        final_path = FINAL_OUTPUT_PATH_SINGLE
        print(f"Loading single file data from: {final_path}")
    else:
         pytest.skip(f"Final output data not found at {FINAL_OUTPUT_PATH_SINGLE} or {FINAL_OUTPUT_PATH_PARTITIONED}")

    # scan_parquet can handle single files and glob patterns
    return pl.scan_parquet(final_path)


# --- Helper Validation Function ---
def assert_row_exists(
    lf: pl.LazyFrame,
    filter_criteria: dict,
    expected_values: dict,
    tolerance: float = 0.01, # Allow 1% difference for numeric values
    check_for_unique: bool = True
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
             pytest.fail(f"Column '{col}' used in filter criteria not found in DataFrame schema: {lf.columns}")
        filter_expressions.append(pl.col(col) == val)

    if filter_expressions:
        query = query.filter(pl.all_horizontal(filter_expressions))

    result_df = query.collect()
    num_rows = len(result_df)

    if check_for_unique:
        assert num_rows == 1, \
            f"Expected exactly 1 row for criteria {filter_criteria}, but found {num_rows}."
    else:
        assert num_rows > 0, \
            f"Expected at least 1 row for criteria {filter_criteria}, but found none."

    # Take the first row if multiple found and check_for_unique is False
    result_row = result_df.row(0, named=True)

    for col, expected_val in expected_values.items():
        assert col in result_row, f"Expected column '{col}' not found in result row. Available columns: {list(result_row.keys())}"
        actual_val = result_row[col]

        # Attempt conversion to float for numeric comparison if types might be mixed (e.g., tariff as string)
        is_expected_numeric = isinstance(expected_val, (int, float))
        actual_numeric = None

        if is_expected_numeric:
            if actual_val is None:
                 assert False, f"Column '{col}' is None, expected numeric value {expected_val} for criteria {filter_criteria}"
            try:
                # Handle potential string representations of numbers
                actual_numeric = float(str(actual_val).strip())
            except (ValueError, TypeError):
                 assert False, f"Column '{col}' could not be converted to float for comparison. Value: '{actual_val}' (Type: {type(actual_val)}), Criteria: {filter_criteria}"

        if is_expected_numeric and tolerance is not None and actual_numeric is not None:
             # Use math.isclose for robust floating-point comparison
            assert math.isclose(actual_numeric, expected_val, rel_tol=tolerance), \
                f"Column '{col}' mismatch: Expected approx {expected_val} (tol={tolerance}), got {actual_numeric} for criteria {filter_criteria}"
        elif is_expected_numeric and tolerance is not None and actual_numeric is None:
             # This case is handled by the initial None check above, but kept for clarity
             assert False, f"Column '{col}' is None after conversion attempt, expected numeric value {expected_val} for criteria {filter_criteria}"
        else:
            # Exact comparison for strings, integers (if tolerance is None), etc.
            # Explicitly cast expected value to string if comparing with string actual_val
            if isinstance(actual_val, str) and not isinstance(expected_val, str):
                 assert actual_val == str(expected_val), \
                    f"Column '{col}' mismatch (string vs non-string): Expected '{expected_val}', got '{actual_val}' for criteria {filter_criteria}"
            else:
                assert actual_val == expected_val, \
                    f"Column '{col}' mismatch: Expected {expected_val} (Type: {type(expected_val)}), got {actual_val} (Type: {type(actual_val)}) for criteria {filter_criteria}"


# --- Test Cases ---

# --- WITS MFN Data Validation ---
@pytest.mark.parametrize(
    "reporter, product, year, expected_tariff",
    [
        # --- Example Case from NETWORK.py ---
        # Sri Lanka (144), HS 310520, 2008 -> Expected MFN Tariff (e.g., 2.5)
        # ** Use string codes consistent with your data **
        # !!! IMPORTANT: Replace 2.5 with the actual validated tariff rate !!!
        ("144", "310520", 2008, 2.5),

        # --- Add more validated cases ---
        # Format: ("COUNTRY_CODE_STR", "HS_CODE_STR", YEAR_INT, EXPECTED_MFN_TARIFF_FLOAT),
        # ("...", "...", ..., ...),
    ]
)
def test_wits_mfn_values(wits_mfn_df, reporter, product, year, expected_tariff):
    """Verify specific MFN tariff values exist in the WITS MFN dataset."""
    # *** Use column names from WITS_cleaner.py output ***
    criteria = {
        "reporter_country": reporter,
        "product_code": product,
        "year": year
    }
    # *** Target the correct tariff column name ***
    expected = {"tariff_rate": expected_tariff}
    assert_row_exists(wits_mfn_df, criteria, expected, tolerance=0.01)


# --- WITS Preferential Data Validation ---
@pytest.mark.parametrize(
    "reporter, partner, product, year, expected_tariff",
    [
        # --- Add validated cases ---
        # Format: ("REP_CODE_STR", "PARTNER_CODE_STR", "HS_CODE_STR", YEAR_INT, EXP_PREF_TARIFF_FLOAT),
         ("...", "...", "...", ..., ...), # Placeholder - REMOVE OR REPLACE
    ]
)
def test_wits_pref_values(wits_pref_df, reporter, partner, product, year, expected_tariff):
    """Verify specific Preferential tariff values exist in the WITS Pref dataset."""
    # *** Use column names from WITS_cleaner.py output ***
    criteria = {
        "reporter_country": reporter,
        "partner_country": partner,
        "product_code": product,
        "year": year
    }
    # *** Target the correct tariff column name ***
    expected = {"tariff_rate": expected_tariff}
    assert_row_exists(wits_pref_df, criteria, expected, tolerance=0.01)


# --- Final Merged Output Validation ---
@pytest.mark.parametrize(
    "source, target, hs_code, year, expected_value, expected_qty, expected_tariff",
    [
        # --- Example Case from NETWORK.py ---
        # Germany ("276") -> Sri Lanka ("144"), HS "310520", 2008
        # !!! IMPORTANT: Replace example values with actual validated data !!!
        ("276", "144", "310520", 2008, 11700.0, 30000.0, 2.5),

        # --- Romania ("642") -> Georgia ("268"), HS "040690", 2000 ---
        # !!! IMPORTANT: Replace example values with actual validated data !!!
        ("642", "268", "040690", 2000, 7829.0, 250.0, 0.0),

        # --- Add more validated cases ---
        # Format: ("SOURCE_CODE_STR", "TARGET_CODE_STR", "HS_CODE_STR", YEAR_INT, EXP_VALUE_FLOAT, EXP_QTY_FLOAT, EXP_TARIFF_FLOAT),
        # ("...", "...", "...", ..., ..., ..., ...), # Placeholder - REMOVE OR REPLACE
    ]
)
def test_final_output_values(final_df, source, target, hs_code, year, expected_value, expected_qty, expected_tariff):
    """Verify specific trade flows have the correct value, quantity, and effective tariff in the final output."""
    # *** Use column names from matching_logic.py's create_final_table ***
    criteria = {
        "Source": source,
        "Target": target,
        "HS_Code": hs_code,
        "Year": year
    }
    expected = {
        "Value": expected_value,
        "Quantity": expected_qty,
        "effective_tariff_rate": expected_tariff # This is the key calculated column
    }
    # Use a potentially larger tolerance for final values due to aggregation/matching nuances
    assert_row_exists(final_df, criteria, expected, tolerance=0.05) # Adjust tolerance if needed
