import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import os
    import re
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import polars as pl

    from typing import Dict, List
    import pycountry

    from mpil_tariff_trade_analysis.utils.pipeline_funcs import (
        vectorized_hs_translation,
    )

    return (
        Dict,
        List,
        Path,
        glob,
        mo,
        os,
        pd,
        pl,
        pycountry,
        re,
        vectorized_hs_translation,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # WITS PIPELINE
    Implement the WITS MFN & PREF pipeline from start to finish

    1. Load and consolidate
    2. Vectorised H0 translation
    3. Explode partner codes
    4. Remap country codes
    """
    )
    return


@app.cell
def _(glob, os, pl, re):
    # Load and consolidate all the WITS data
    def consolidate_wits_tariff_data(tariff_type="AVEMFN", base_dir="data/raw/WITS_tariff/") -> pl.LazyFrame:
        tariff_dir = os.path.join(base_dir, tariff_type)

        # 1: find all CSV files for this tariff type
        pattern = rf"{tariff_type}_(H\d+)_(\w+)_(\d+)_U2"
        all_csv_files = []
        try:
            subdirs = os.listdir(tariff_dir)
            # # logger.debug(f"Found {len(subdirs)} subdirectories in {WITS_BASE_DIR}")
            print(f"Found {len(subdirs)} subdirectories in {tariff_dir}")
        except FileNotFoundError:
            # # logger.error(f"Directory not found: {WITS_BASE_DIR}")
            raise

        for subdir in subdirs:
            match = re.match(pattern, subdir)
            if match:
                hs_revision, country_iso, year = match.groups()
                csv_path = os.path.join(tariff_dir, subdir)
                # Note: case-sensitive .CSV extension
                csv_files = glob.glob(os.path.join(csv_path, "*.CSV"))

                # # logger.debug(f"Found {len(csv_files)} CSV files in {subdir}")

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

        # # logger.info(f"Found {len(all_csv_files)} CSV files for {tariff_type}")
        print(f"Found {len(all_csv_files)} CSV files for {tariff_type}")
        # print(all_csv_files)

        # 2: Create a schema for the files
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

        # 3: Load the files and concat them
        successful_files = 0
        failed_files = 0
        dfs = []

        for file_info in all_csv_files:
            try:
                # Load the CSV file lazily
                # # logger.debug(f"Scanning file: {file_info['file_path']}")
                df = pl.scan_csv(
                    file_info["file_path"],
                    schema=schema,
                    # infer_schema_length=0,
                    try_parse_dates=False,
                    separator=",",
                    skip_rows_after_header=0,
                    null_values=["", "NA"],
                    truncate_ragged_lines=True,
                    ignore_errors=True,
                )

                dfs.append(df)
                successful_files += 1
            except Exception:
                # # logger.error(f"Error loading file {file_info['file_path']}: {e}")
                failed_files += 1

        # # logger.info(
        #     f"Successfully scanned {successful_files} files, failed to scan {failed_files} files"
        # )
        print(f"Successfully scanned {successful_files} files, failed to scan {failed_files} files")

        if not dfs:
            # # logger.warning("No valid CSV files found to scan.")
            raise FileNotFoundError("No WITS CSV files found or scanned successfully.")

        # Combine all dataframes
        # # logger.info("Combining all scanned dataframes")
        print("Combining all scanned dfs")
        combined_df = pl.concat(dfs, how="vertical_relaxed")  # Use relaxed to handle potential schema variations if any

        # Add a column for tariff type
        combined_df = combined_df.with_columns(pl.lit(tariff_type).alias("tariff_type"))

        combined_df.head().collect()

        if tariff_type == "AVEMFN":
            combined_df = combined_df.select(
                [
                    pl.col("Year").cast(pl.Utf8, strict=False).alias("year"),
                    pl.col("Reporter_ISO_N").alias("reporter_country"),
                    pl.col("ProductCode").alias("product_code"),
                    pl.col("NomenCode").alias("hs_revision"),
                    pl.col("SimpleAverage").alias("tariff_rate").str.strip_chars().cast(pl.Float32),
                    pl.col("Min_Rate").alias("min_rate").str.strip_chars().cast(pl.Float32),
                    pl.col("Max_Rate").alias("max_rate").str.strip_chars().cast(pl.Float32),
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
                    pl.col("SimpleAverage").alias("tariff_rate").str.strip_chars().cast(pl.Float32),
                    pl.col("Min_Rate").alias("min_rate").str.strip_chars().cast(pl.Float32),
                    pl.col("Max_Rate").alias("max_rate").str.strip_chars().cast(pl.Float32),
                    pl.col("tariff_type"),
                ]
            )

        return combined_df

    consolidated_lf_AVEMFN = consolidate_wits_tariff_data("AVEMFN", "data/raw/WITS_tariff/")
    consolidated_lf_AVEPref = consolidate_wits_tariff_data("AVEPref", "data/raw/WITS_tariff/")
    return consolidated_lf_AVEMFN, consolidated_lf_AVEPref


@app.cell
def _(consolidated_lf_AVEMFN, consolidated_lf_AVEPref):
    print(f"Consolidated lf avemfn:\n{consolidated_lf_AVEMFN.head().collect()}")
    print(f"Consolidated lf avepref:\n{consolidated_lf_AVEPref.head().collect()}")
    return


@app.cell
def _(mo):
    mo.md(r"""# Translate HS codes""")
    return


@app.cell
def _(
    consolidated_lf_AVEMFN,
    consolidated_lf_AVEPref,
    vectorized_hs_translation,
):
    print("--- TRANSLATING AVEMFN ---")
    translated_lf_AVEMFN = vectorized_hs_translation(consolidated_lf_AVEMFN)
    print("--- TRANSLATING AVEPref ---")
    translated_lf_AVEPref = vectorized_hs_translation(consolidated_lf_AVEPref)
    return translated_lf_AVEMFN, translated_lf_AVEPref


@app.cell
def _(translated_lf_AVEMFN):
    translated_lf_AVEMFN.head().collect()
    return


@app.cell
def _(pl, translated_lf_AVEMFN):
    # Now we've created duplicates, we need to join these duplicate product codes together
    aggregated_lf_AVEMFN = translated_lf_AVEMFN.group_by(["reporter_country", "year", "product_code", "tariff_type"]).agg(
        pl.mean("tariff_rate"),
        pl.mean("min_rate"),
        pl.mean("max_rate"),
    )

    print(f"Translated and regaggregated lf AVEMFN head:\n{aggregated_lf_AVEMFN.head().collect()}")
    return (aggregated_lf_AVEMFN,)


@app.cell
def _(pl, translated_lf_AVEPref):
    aggregated_lf_AVEPref = translated_lf_AVEPref.group_by(
        [
            "reporter_country",
            "partner_country",
            "year",
            "product_code",
            "tariff_type",
        ]
    ).agg(
        pl.mean("tariff_rate"),
        pl.mean("min_rate"),
        pl.mean("max_rate"),
    )

    print(f"Translated and regaggregated lf AVEPref head:\n{aggregated_lf_AVEPref.head().collect()}")
    return (aggregated_lf_AVEPref,)


@app.cell
def _(mo):
    mo.md(r"""# Remap partner codes for AVEPref""")
    return


@app.cell
def _(aggregated_lf_AVEPref, pd, pl):
    # Load the mapping
    pref_group_mapping = pd.read_csv(
        "data/raw/WITS_pref_groups/WITS_pref_groups.csv",
        encoding="iso-8859-1",
        usecols=[0, 2],
    )

    # Minor cleans and convert to polars
    pref_group_mapping.columns = ["pref_group_code", "country_iso_num"]
    pref_group_mapping["country_iso_num"] = pref_group_mapping["country_iso_num"].astype(str).str.zfill(3)
    pref_group_mapping = pref_group_mapping.groupby("pref_group_code").agg(list)
    pref_group_mapping_lf = pl.from_pandas(pref_group_mapping, include_index=True).lazy()

    # Join on the AVEpref dataset
    joined_pref_lf_AVEPref = aggregated_lf_AVEPref.join(
        pref_group_mapping_lf,
        left_on="partner_country",
        right_on="pref_group_code",
    )

    # Explode out to create entries for each individual country (memory intensive!)
    joined_pref_lf_AVEPref = joined_pref_lf_AVEPref.explode("country_iso_num")
    joined_pref_lf_AVEPref = joined_pref_lf_AVEPref.with_columns(pl.col("country_iso_num").alias("partner_country")).drop("country_iso_num")

    print(f"Joined Pref LF head, post explode:\n{joined_pref_lf_AVEPref.head().collect(engine='streaming')}")
    return (joined_pref_lf_AVEPref,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Translate country codes
    Now we have everything in ISO Numeric 3 digit space, we need to remap the WITS codes to our ISO ones.
    """
    )
    return


@app.cell
def _(Dict, List, pycountry):
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
            "230": ["231"],  # Ethiopia
            "200": ["203"],  # Czech Republic
            "729": ["736"],  # Sudan 1
            "728": ["736"],  # Sudan 2
            "688": ["891"],  # Serbia
            "499": ["890"],  # Montenegro
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
                        iso_nums = [fuzzy_matches[0].numeric]
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
def _(
    Path,
    aggregated_lf_AVEMFN,
    identify_iso_code,
    joined_pref_lf_AVEPref,
    pl,
):
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

    mapping_df_AVEPref_reporter = create_mapping_df(joined_pref_lf_AVEPref, "reporter_country")
    mapping_df_AVEPref_partner = create_mapping_df(joined_pref_lf_AVEPref, "partner_country")
    mapping_df_AVEMFN_reporter = create_mapping_df(aggregated_lf_AVEMFN, "reporter_country")
    return (
        mapping_df_AVEMFN_reporter,
        mapping_df_AVEPref_partner,
        mapping_df_AVEPref_reporter,
    )


@app.cell
def _(mapping_df_AVEPref_reporter):
    print(f"Mapping df for AVEPRef Reporter col:\n{mapping_df_AVEPref_reporter}")
    return


@app.cell
def _(
    aggregated_lf_AVEMFN,
    joined_pref_lf_AVEPref,
    mapping_df_AVEMFN_reporter,
    mapping_df_AVEPref_partner,
    mapping_df_AVEPref_reporter,
    pl,
):
    # Apply these mappings
    def apply_mapping(lf: pl.LazyFrame, mapping_df: pl.DataFrame, target_col_name: str) -> pl.LazyFrame:
        lf = lf.join(
            mapping_df.lazy(),
            left_on=target_col_name,
            right_on="original_code",
            how="left",
        )

        lf = lf.explode("iso_num_list").with_columns(pl.col("iso_num_list").alias(target_col_name)).drop("iso_num_list")

        return lf

    AVEMFN_lf_clean = apply_mapping(aggregated_lf_AVEMFN, mapping_df_AVEMFN_reporter, "reporter_country")

    # Apply mapping twice, once for reporter and partner country for the AVEPref
    AVEPref_lf_clean = apply_mapping(joined_pref_lf_AVEPref, mapping_df_AVEPref_reporter, "reporter_country")
    AVEPref_lf_clean = apply_mapping(AVEPref_lf_clean, mapping_df_AVEPref_partner, "partner_country")
    return AVEMFN_lf_clean, AVEPref_lf_clean


@app.cell
def _(AVEMFN_lf_clean, AVEPref_lf_clean):
    print(f"AVEMFN lf head, collected post retranslation, explosion, aggregation:\n{AVEMFN_lf_clean.head().collect(engine='streaming')}")

    print(f"AVEPref lf head, collected post retranslation, explosion, aggregation:\n{AVEPref_lf_clean.head().collect(engine='streaming')}")
    return


@app.cell
def _(mo):
    mo.md(r"""# Sink the results""")
    return


@app.cell
def _(AVEMFN_lf_clean):
    print("Sinking AVEMFN")
    AVEMFN_lf_clean.sink_parquet("data/intermediate/WITS_AVEMFN_CLEAN.parquet", compression="zstd")
    return


@app.cell
def _(AVEPref_lf_clean):
    print("Sinking AVEPREF")
    AVEPref_lf_clean.sink_parquet("data/intermediate/WITS_AVEPref_CLEAN.parquet", compression="zstd")
    return


if __name__ == "__main__":
    app.run()
