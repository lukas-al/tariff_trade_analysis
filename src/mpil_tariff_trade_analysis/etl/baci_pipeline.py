import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, glob, mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # IMPLEMENT THE BACI PIPELINE

    Start to finish - implement the BACI cleaning pipeline

    ## Structure:
    1. Load incrememntally and convert from CSV into a Parquet
    2. Explode any country codes which refer to regions.
    3. Remap the country codes.
    """
    )
    return


@app.cell
def _(Path):
    # Core Parameters
    hs_code = "HS92"
    baci_release = "202501"
    base_data_dir = Path("data").resolve()  # Use absolute paths for robustness

    # Define Directory Structure
    raw_data_dir = base_data_dir / "raw"
    intermediate_data_dir = base_data_dir / "intermediate"
    final_data_dir = base_data_dir / "final"

    baci_input_folder = raw_data_dir  # Parent dir of BACI_HSXX_VYYYYYY CSVs

    # --- Intermediate & Output Paths ---
    # BACI Paths
    baci_intermediate_parquet_name = f"BACI_{hs_code}_V{baci_release}_CLEAN.parquet"
    baci_intermediate_parquet_path = intermediate_data_dir / baci_intermediate_parquet_name
    return baci_intermediate_parquet_path, baci_release, hs_code, raw_data_dir


@app.cell
def _(Path, baci_release, glob, hs_code, pl, raw_data_dir):
    def scan_baci_csv_files(hs, release, input_folder_path, Path_module, glob_module, polars_module):
        baci_folder_name = f"BACI_{hs}_V{release}"
        # input_path is the directory containing the actual CSVs, e.g., data/raw/BACI_HS92_V202501/
        input_path = Path_module(input_folder_path) / baci_folder_name
        csv_search_pattern = str(input_path / "BACI*.csv")

        csv_files = glob_module.glob(csv_search_pattern)

        print(f"Found {len(csv_files)} CSV files to process in *{input_path}*")
        print(f"Search pattern: {csv_search_pattern}")

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching pattern: {csv_search_pattern}")

        # Scan all found CSV files into a single LazyFrame
        lf = polars_module.scan_csv(csv_files)

        print("Successfully scanned CSV files into a Polars LazyFrame.")
        return lf

    raw_lf = scan_baci_csv_files(
        hs=hs_code,
        release=baci_release,
        input_folder_path=raw_data_dir,
        Path_module=Path,
        glob_module=glob,
        polars_module=pl,
    )
    return (raw_lf,)


@app.cell
def _(raw_lf):
    # Inspect what we've created
    print(f"Raw LF head:\n{raw_lf.head().collect()}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Remap country codes

    1. Minor recast and padding
    2. Create a mapping table of these codes to ISO
    """
    )
    return


@app.cell
def _(pl, raw_lf):
    # 1: Minor recast and padding
    recast_lf = raw_lf.with_columns(
        pl.col("i").cast(pl.Utf8).str.zfill(3),
        pl.col("j").cast(pl.Utf8).str.zfill(3),
    )
    return (recast_lf,)


@app.cell
def _():
    from typing import Dict, List

    import pycountry

    def identify_iso_code(
        cc: str,
        baci_map_names: Dict[str, str],
        baci_map_iso3: Dict[str, str],
        wits_map_names: Dict[str, str],
        wits_map_iso3: Dict[str, str],
    ) -> List[str]:
        # Try find the matching code in our ISO set
        hardcoded_code_map = {
            "697": [
                "352",
                "438",
                "578",
                "756",
            ],  # Europe EFTA, nes -> Iceland, Liechtenstein, Norway, Switzerland
            "490": ["158"],  # Other Asia, nes -> Taiwan (ISO 3166-1 numeric is 158)
            "918": [
                "040",
                "056",
                "100",
                "191",
                "196",
                "203",
                "208",
                "233",
                "246",
                "250",
                "276",
                "300",
                "348",
                "372",
                "380",
                "428",
                "440",
                "442",
                "470",
                "528",
                "616",
                "620",
                "642",
                "703",
                "705",
                "724",
                "752",
                "492",  # Monaco
            ],  # European Customs Union incl. Monaco
        }

        # Check our hardcoded ccs first
        if cc in hardcoded_code_map:
            iso_nums = hardcoded_code_map[cc]
            print(f"Mapped {cc} to {iso_nums}")
            return iso_nums

        # Use a direct pycountry lookup
        try:
            direct_cc_match = pycountry.countries.lookup(cc)
            iso_nums = [direct_cc_match.numeric]
            print(f"Mapped {cc} to {iso_nums}")
            return iso_nums
        except LookupError:
            print(f"Direct pycountry lookup for code '{cc}' failed. Trying name and iso3 alpha lookup.")

        # If that fails, try get the name of the country from the provided mappings and use that for pycountry
        country_name = baci_map_names.get(cc) or wits_map_names.get(cc)
        iso3_code = baci_map_iso3.get(cc) or wits_map_iso3.get(cc)

        if country_name:
            try:
                direct_country_match = pycountry.countries.lookup(country_name)
                iso_nums = [direct_country_match.numeric]
                print(f"Mapped {cc} to {iso_nums} using country name")
                return iso_nums

            # Otherwise try a fuzzy match
            except LookupError:
                try:
                    fuzzy_matches = pycountry.countries.search_fuzzy(country_name)

                    if fuzzy_matches:
                        iso_nums = [fuzzy_matches[0].numeric]  # type: ignore
                        print(f"Mapped {cc} to {iso_nums} using country name *fuzzy*")
                        return iso_nums

                    else:
                        pass

                except LookupError:
                    print(f"Unable to fuzzy find a match for {country_name}. Code {cc}.")

        if iso3_code:
            print(f"Trying ISO3 alpha code for country {cc}.")

            try:
                direct_country_match = pycountry.countries.lookup(iso3_code)
                iso_nums = [direct_country_match.numeric]
                print(f"Mapped {cc} to {iso_nums} using iso3 alpha lookup")
                return iso_nums
            except LookupError:
                print(f"Unable to find any ISO alpha 3 code match for {cc}")

        print("Failed to match entirely. Returning original code.")
        return [cc]

    return (identify_iso_code,)


@app.cell
def _(Path, identify_iso_code, pl, recast_lf):
    # 2: Create a mapping table of these codes to ISO
    def create_mapping_df(lf: pl.LazyFrame, col_name: str) -> pl.DataFrame:
        # Get the unique codes from the series
        unique_codes = lf.select(pl.col(col_name).unique()).collect().to_series().to_list()

        #! FOR TESTING ONLY!
        # unique_codes.append("697")
        # print(unique_codes)

        # Load our reference data - this is marginally inefficient but feels cleaner
        baci_reference_path = Path("data/raw/BACI_HS92_V202501/country_codes_V202501.csv")
        baci_map = pl.read_csv(baci_reference_path, infer_schema=False, encoding="utf8")
        baci_map = baci_map.with_columns(pl.col("country_code").str.zfill(3))
        baci_map_names = dict(zip(baci_map["country_code"], baci_map["country_name"], strict=False))
        baci_map_iso3 = dict(zip(baci_map["country_code"], baci_map["country_iso3"], strict=False))
        # print(f"BACI Mapping: {baci_map_names}")
        # print(f"BACI Mapping: {baci_map_iso3}")

        wits_reference_path = Path("data/raw/WITS_country_codes.csv")
        wits_map = pl.read_csv(wits_reference_path, infer_schema=False, encoding="utf8")
        wits_map = wits_map.with_columns(pl.col("Numeric Code").str.zfill(3))
        wits_map_names = dict(zip(wits_map["Numeric Code"], wits_map["Country Name"], strict=False))
        wits_map_iso3 = dict(zip(wits_map["Numeric Code"], wits_map["ISO3"], strict=False))
        # print(f"WITS Mapping: {wits_map_names}")
        # print(f"WITS Mapping: {wits_map_iso3}")

        # Map the codes
        mapping_data = []
        for cc in unique_codes:
            iso_num_codes = identify_iso_code(cc, baci_map_names, baci_map_iso3, wits_map_names, wits_map_iso3)
            mapping_data.append({"original_code": cc, "iso_num_list": iso_num_codes})

        # Create a df
        mapping_df = pl.DataFrame(mapping_data).with_columns(pl.col("iso_num_list").cast(pl.List(pl.Utf8)))

        return mapping_df

    mapping_df_i = create_mapping_df(recast_lf, "i")
    mapping_df_j = create_mapping_df(recast_lf, "j")
    return mapping_df_i, mapping_df_j


@app.cell
def _(mapping_df_i):
    print(f"Mapping df for col I head:\n{mapping_df_i.head()}")
    return


@app.cell
def _():
    # # Create a test dataset
    # height = recast_lf.select(pl.len()).collect().item()
    # sample_sz = 100000
    # test_df = recast_lf.gather_every(height // sample_sz).collect()
    # test_lf = test_df.lazy()
    return


@app.cell
def _(mapping_df_i, mapping_df_j, pl, recast_lf):
    # Apply the iso mapping to the table to correct for the incorrect codes efficiently

    # --- i
    joined_lf = recast_lf.join(
        mapping_df_i.lazy(),
        left_on="i",
        right_on="original_code",
        how="left",
    )

    joined_lf = joined_lf.explode("iso_num_list").with_columns(pl.col("iso_num_list").alias("i")).drop("iso_num_list")

    # --- j
    joined_lf = joined_lf.join(
        mapping_df_j.lazy(),
        left_on="j",
        right_on="original_code",
        how="left",
    )

    joined_lf = joined_lf.explode("iso_num_list").with_columns(pl.col("iso_num_list").alias("j")).drop("iso_num_list")
    return (joined_lf,)


@app.cell
def _(joined_lf):
    print(f"Final joined_lf.head:\n{joined_lf.head().collect(engine='streaming')}")
    return


@app.cell
def _():
    # joined_lf.select(pl.len()).collect().item() # 258605611 -> Takes 28 mins to calculate...
    # Cast product codes to string...

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Write output
    This crashes when run as a marimo notebook - probably a thread / MP worker timeout
    """
    )
    return


@app.cell
def _(baci_intermediate_parquet_path, joined_lf):
    print(f"Sinking Parquet output to {baci_intermediate_parquet_path}")
    joined_lf.sink_parquet(baci_intermediate_parquet_path)
    return


if __name__ == "__main__":
    app.run()
