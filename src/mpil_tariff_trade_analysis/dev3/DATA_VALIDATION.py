import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import pycountry
    import pandas as pd
    import numpy as np

    import re
    import pdfplumber
    from pathlib import Path
    return Path, mo, np, pd, pdfplumber, pl, px, pycountry, re


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data validation
    The sole purpose of this notebook is to validate the tariff data I'm using. Whether that's WITS or something else.

    ## Approach
    A) Validating that the tariffs in WITS reported over the US-China trade war are as reported in US Census / official statistics.

    The PIIE provides [this page tracking tariffs](https://www.piie.com/research/piie-charts/2019/us-china-trade-war-tariffs-date-chart). We need to match this.

    B) Validating that intra-EU tariffs are as expected.

    C) Finding a non-WITS source to compare tariff levels against.

    # US-China trade war validation
    Test over the 2017 US-China trade war
    """
    )
    return


@app.cell(hide_code=True)
def _(pl):
    unified_lf = pl.scan_parquet("data/final/unified_trade_tariff_partitioned/")

    unified_lf.head().collect()
    return (unified_lf,)


@app.cell(hide_code=True)
def _(mo):
    PRODUCT_CODE = mo.ui.text(placeholder="Input HS code")
    EXPORTER_CODE = mo.ui.text(placeholder="China")
    IMPORTER_CODE = mo.ui.text(placeholder="USA")

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("Select your product code to check:"),
                    PRODUCT_CODE,
                ],
                justify="center",
            ),
            mo.hstack(
                [
                    mo.md("Select your exporter country code to check:"),
                    EXPORTER_CODE,
                ],
                justify="center",
            ),
            mo.hstack(
                [
                    mo.md("Select your importer country code to check:"),
                    IMPORTER_CODE,
                ],
                justify="center",
            ),
        ]
    )
    return EXPORTER_CODE, IMPORTER_CODE, PRODUCT_CODE


@app.cell(hide_code=True)
def _(
    EXPORTER_CODE,
    IMPORTER_CODE,
    PRODUCT_CODE,
    mo,
    pl,
    px,
    pycountry,
    unified_lf,
):
    reporter_cc = pycountry.countries.search_fuzzy(EXPORTER_CODE.value)[0].numeric
    partner_cc = pycountry.countries.search_fuzzy(IMPORTER_CODE.value)[0].numeric
    product_ts = (
        unified_lf.filter(
            pl.col("product_code") == PRODUCT_CODE.value,
            pl.col("reporter_country") == reporter_cc,
            pl.col("partner_country") == partner_cc,
        )
        .sort("year")
        .collect()
    )
    mo.vstack(
        [
            mo.md(
                f"## Effective Tariff Chart \n HS code: {PRODUCT_CODE.value}, exporter {reporter_cc}, importer {partner_cc}"
            ),
            px.line(product_ts, x="year", y="effective_tariff", markers=True),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Specifics

    From the Office of the USTS website, or relevant 'Federal Report' documents, we can obtain

    A) The specific chapter 99 heading

    B) The effective date that the tariff was implemented

    C) The tariff rate applied to that notice

    D) Any exclusions

    ## How
    To do this, we must extract the HS codes applied to each heading (available on the UCITS page), and map the UCITS heading to the applied tariff. Then we must check it against the WITS data rate, and if it's missing add it on. Kapow.
    """
    )
    return


@app.cell
def _(np, pd):
    data_records = [
        {
            "Effective Date": "2018-03-23",
            "Action Entity": "USA",
            "Tariff Target": "Steel imports from various countries",
            "Legal Basis": "Section 232",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-03-23",
            "Action Entity": "USA",
            "Tariff Target": "Aluminum imports from various countries",
            "Legal Basis": "Section 232",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.10,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-04-02",
            "Action Entity": "China",
            "Tariff Target": "128 U.S. products (Retaliation to Sec 232)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": 3_000_000_000,
            "Applied Rate": 0.15,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-04-02",
            "Action Entity": "China",
            "Tariff Target": "128 U.S. products (Retaliation to Sec 232)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": 3_000_000_000,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-07-06",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 1)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 34_000_000_000,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-07-06",
            "Action Entity": "China",
            "Tariff Target": "545 U.S. products (Retaliation to List 1)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": 34_000_000_000,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-08-23",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 2)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 16_000_000_000,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-08-23",
            "Action Entity": "China",
            "Tariff Target": "333 U.S. products (Retaliation to List 2)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": 16_000_000_000,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-09-24",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 3)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 200_000_000_000,
            "Applied Rate": 0.10,
            "Status": "Active",
        },
        {
            "Effective Date": "2019-05-10",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 3)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 200_000_000_000,
            "Applied Rate": 0.25,
            "Status": "Increased",
        },
        {
            "Effective Date": "2018-09-24",
            "Action Entity": "China",
            "Tariff Target": "U.S. products (Retaliation to List 3)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": 60_000_000_000,
            "Applied Rate": 0.05,
            "Status": "Active",
        },
        {
            "Effective Date": "2018-09-24",
            "Action Entity": "China",
            "Tariff Target": "U.S. products (Retaliation to List 3)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": 60_000_000_000,
            "Applied Rate": 0.10,
            "Status": "Active",
        },
        {
            "Effective Date": "2019-09-01",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 4A)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 120_000_000_000,
            "Applied Rate": 0.15,
            "Status": "Active",
        },
        {
            "Effective Date": "2019-09-01",
            "Action Entity": "China",
            "Tariff Target": "U.S. products (Retaliation to List 4A)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.05,
            "Status": "Active",
        },
        {
            "Effective Date": "2019-09-01",
            "Action Entity": "China",
            "Tariff Target": "U.S. products (Retaliation to List 4A)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.10,
            "Status": "Active",
        },
        {
            "Effective Date": "2019-12-15",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 4B)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 180_000_000_000,
            "Applied Rate": 0.15,
            "Status": "Suspended",
        },
        {
            "Effective Date": "2019-12-15",
            "Action Entity": "China",
            "Tariff Target": "U.S. products (Retaliation to List 4B)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.05,
            "Status": "Suspended",
        },
        {
            "Effective Date": "2019-12-15",
            "Action Entity": "China",
            "Tariff Target": "U.S. products (Retaliation to List 4B)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.10,
            "Status": "Suspended",
        },
        {
            "Effective Date": "2020-02-14",
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (List 4A)",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": 120_000_000_000,
            "Applied Rate": 0.075,
            "Status": "Reduced",
        },
        {
            "Effective Date": "2020-02-14",
            "Action Entity": "China",
            "Tariff Target": "Some U.S. goods (Reductions/Exemptions)",
            "Legal Basis": "Retaliation",
            "Trade Value (USD)": np.nan,
            "Applied Rate": np.nan,
            "Status": "Reduced/Exempted",
        },
    ]
    df_tariffs_comprehensive = pd.DataFrame(data_records)
    df_tariffs_comprehensive["Effective Date"] = pd.to_datetime(
        df_tariffs_comprehensive["Effective Date"]
    )

    # Add HTS Heading column, initialized with a suitable placeholder (e.g., None or pd.NA)
    df_tariffs_comprehensive["HTS Heading"] = pd.NA

    # Update existing rows with HTS codes from the new list where they match
    # For List 1 (9903.88.01)
    idx_list1 = df_tariffs_comprehensive[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 1)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2018-07-06")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.25)
    ].index
    if not idx_list1.empty:
        df_tariffs_comprehensive.loc[idx_list1, "HTS Heading"] = "9903.88.01"

    # For List 2 (9903.88.02)
    idx_list2 = df_tariffs_comprehensive[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 2)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2018-08-23")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.25)
    ].index
    if not idx_list2.empty:
        df_tariffs_comprehensive.loc[idx_list2, "HTS Heading"] = "9903.88.02"

    # For List 3 (9903.88.03) - initial 10% rate
    idx_list3_10 = df_tariffs_comprehensive[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 3)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2018-09-24")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.10)
    ].index
    if not idx_list3_10.empty:
        df_tariffs_comprehensive.loc[idx_list3_10, "HTS Heading"] = "9903.88.03"

    # For List 3 (9903.88.03) - increased 25% rate
    idx_list3_25 = df_tariffs_comprehensive[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 3)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2019-05-10")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.25)
    ].index
    if not idx_list3_25.empty:
        df_tariffs_comprehensive.loc[idx_list3_25, "HTS Heading"] = "9903.88.03"

    # For List 4A (9903.88.15) - initial 15% rate
    idx_list4a_15 = df_tariffs_comprehensive[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 4A)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2019-09-01")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.15)
    ].index
    if not idx_list4a_15.empty:
        df_tariffs_comprehensive.loc[idx_list4a_15, "HTS Heading"] = "9903.88.15"

    # For List 4A (9903.88.15) - reduced 7.5% rate
    idx_list4a_7_5 = df_tariffs_comprehensive[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 4A)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2020-02-14")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.075)
    ].index
    if not idx_list4a_7_5.empty:
        df_tariffs_comprehensive.loc[idx_list4a_7_5, "HTS Heading"] = "9903.88.15"


    # Prepare new records from the HTS list provided by the user
    new_hts_records = [
        # HTS 9903.88.04 - as per user's new list (20% rate)
        # This is treated as a distinct entry because the rate differs from List 3's general 10%/25%
        {
            "Effective Date": pd.to_datetime("2018-09-24"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.88.04)",
            "HTS Heading": "9903.88.04",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.20,
            "Status": "Active",
        },
        # HTS 9903.88.15 from new list: "September 1, 2019 ... + 7.5%".
        # This conflicts with List 4A history (15% from Sep 1, 2019, then 7.5% from Feb 14, 2020).
        # The reconciliation above correctly assigns HTS 9903.88.15 to the two phases of List 4A.
        # Adding this specific line from the new list as is would create a duplicate/conflict.
        # Thus, I will rely on the reconciled List 4A entries already updated with this HTS.
        # If the user *insisted* on adding this exact line even if conflicting, the code would differ.
        # For now, I assume the goal is a consistent, reconciled dataset.
        # Newer HTS codes
        {
            "Effective Date": pd.to_datetime("2024-09-27"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.91.01)",
            "HTS Heading": "9903.91.01",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": pd.to_datetime("2024-09-27"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.91.02)",
            "HTS Heading": "9903.91.02",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": pd.to_datetime("2024-09-27"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.91.03)",
            "HTS Heading": "9903.91.03",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": pd.to_datetime("2024-09-27"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.92.10)",
            "HTS Heading": "9903.92.10",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": pd.to_datetime("2025-01-01"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.91.04)",
            "HTS Heading": "9903.91.04",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
        {
            "Effective Date": pd.to_datetime("2025-01-01"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.91.05)",
            "HTS Heading": "9903.91.05",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },  # Note on value calculation for this HTS is omitted in cell per user.
        {
            "Effective Date": pd.to_datetime("2025-01-01"),
            "Action Entity": "USA",
            "Tariff Target": "Chinese products (under HTS 9903.91.11)",
            "HTS Heading": "9903.91.11",
            "Legal Basis": "Section 301",
            "Trade Value (USD)": np.nan,
            "Applied Rate": 0.25,
            "Status": "Active",
        },
    ]

    df_new_hts_records = pd.DataFrame(new_hts_records)

    # Concatenate the original (now updated) DataFrame with the truly new HTS records
    us_tariff_df = pd.concat(
        [df_tariffs_comprehensive, df_new_hts_records], ignore_index=True
    )

    new_tariff_data = [
        {
            "Effective Date": pd.to_datetime(
                "2025-02-05T00:00:00.000"
            ),  # Entered on or after February 5, 2025
            "Action Entity": "USA",
            "Tariff Target": "Products of China and Hong Kong (IEEPA)",
            "Legal Basis": "IEEPA",  # International Emergency Economic Powers Act
            "Trade Value (USD)": None,
            "Applied Rate": 0.10,  # 10 percent ad valorem
            "Status": "Active",  # Assumed active as it's an upcoming tariff
            "HTS Heading": "9903.01.20",
            # Note: The text states "prior to March 4, 2025".
            # This implies an end date. Representing tariff duration ranges
            # might require an additional 'End Date' column or a different data structure
            # if precise deactivation is needed. For simplicity, we're adding it as a point-in-time effective tariff.
        },
        {
            "Effective Date": pd.to_datetime(
                "2025-03-04T00:00:00.000"
            ),  # Entered on or after March 4, 2025
            "Action Entity": "USA",
            "Tariff Target": "Products of China and Hong Kong (IEEPA)",
            "Legal Basis": "IEEPA",  # International Emergency Economic Powers Act
            "Trade Value (USD)": None,
            "Applied Rate": 0.20,  # 20 percent ad valorem
            "Status": "Active",  # Assumed active
            "HTS Heading": "9903.01.24",
        },
    ]

    # Convert the list of new data into a DataFrame
    new_data_df = pd.DataFrame(new_tariff_data)

    # Append the new data to the original DataFrame
    # Using pd.concat is generally preferred over df.append for future pandas versions
    updated_df = pd.concat([us_tariff_df, new_data_df], ignore_index=True)

    # Sort by Effective Date and HTS Heading for better readability
    us_tariff_df = updated_df.sort_values(
        by=["Effective Date", "Action Entity", "HTS Heading", "Tariff Target"]
    ).reset_index(drop=True)

    us_tariff_df
    return (us_tariff_df,)


@app.cell
def _(us_tariff_df):
    us_tariff_df[us_tariff_df["Action Entity"] == "USA"]
    return


@app.cell
def _():
    STEEL_HS_CODE = "72"  # i.e. 78XXXX
    ALUMINIUM_HS_CODE = "76"  # i.e. 76XXXX
    return


@app.cell
def _(pd, pdfplumber, re):
    def extract_codes_from_pdf(pdf_path: str) -> pd.DataFrame:
        """
        Loads a PDF from the given path, extracts HS codes and Chapter 99 headings.
        Returns a DataFrame with "Product HS Code" and "Chapter 99 Heading".
        """
        document_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if page_text:
                        document_text += page_text + "\n"
        except FileNotFoundError:
            print(f"Error: PDF file not found at {pdf_path}")
            return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])
        except Exception as e:
            print(f"Error reading or processing PDF {pdf_path}: {e}")
            return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])

        if not document_text:
            print(f"No text could be extracted from PDF: {pdf_path}")
            return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])

        # Regex to find lines containing an HS code followed by a Chapter 99 code.
        # HS Code: XXXX.XX.XX or XXXX.XX.XXXX
        # Chapter 99 Code: XXXX.XX.XX
        # Assumes they are on the same line, separated by whitespace.
        # Uses ^ and $ with re.MULTILINE to match lines accurately.
        pattern = re.compile(
            r"^([0-9]{4}\.[0-9]{2}\.[0-9]{2}(?:[0-9]{2})?)\s+([0-9]{4}\.[0-9]{2}\.[0-9]{2})$",
            re.MULTILINE,
        )

        extracted_pairs = []
        matches = pattern.findall(document_text)

        for hs_code_str, ch99_code_str in matches:
            extracted_pairs.append(
                {
                    "Product HS Code": hs_code_str,
                    "Chapter 99 Heading": ch99_code_str,
                }
            )

        if not extracted_pairs:
            print(
                f"No HS code / Chapter 99 pairs found in PDF with the pattern: {pattern.pattern}"
            )
            print(
                "Please check the PDF structure or the regex if an empty DataFrame is unexpected."
            )
            return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])

        df = pd.DataFrame(
            extracted_pairs, columns=["Product HS Code", "Chapter 99 Heading"]
        )
        if not df.empty:
            df["Product HS Code"] = df["Product HS Code"].astype(str)
            df["Chapter 99 Heading"] = df["Chapter 99 Heading"].astype(str)
        return df


    def map_tariff_rates(
        extracted_codes_df: pd.DataFrame, tariff_rates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Maps tariff rates from tariff_rates_df onto the extracted codes DataFrame.
        """
        final_columns = [
            "Product HS Code",
            "Tariff Rate Applied",
            "Effective Date",
            "Chapter 99 Heading",
        ]
        if extracted_codes_df.empty:
            return pd.DataFrame(columns=final_columns)

        tariff_lookup = {}
        if tariff_rates_df is not None and not tariff_rates_df.empty:
            required_columns = ["HTS Heading", "Effective Date", "Applied Rate"]
            if not all(col in tariff_rates_df.columns for col in required_columns):
                print(
                    f"Error: tariff_rates_df missing one or more required columns: {required_columns}"
                )
                # Populate with default "not specified" values if lookup fails due to missing columns
                data_for_final_df = []
                for _, row in extracted_codes_df.iterrows():
                    data_for_final_df.append(
                        {
                            "Product HS Code": row["Product HS Code"],
                            "Tariff Rate Applied": "Rate not in provided DF",
                            "Effective Date": "Date not in provided DF",
                            "Chapter 99 Heading": row["Chapter 99 Heading"],
                        }
                    )
                final_df = pd.DataFrame(data_for_final_df, columns=final_columns)
            else:
                for _, row in tariff_rates_df.iterrows():
                    hts_heading = str(row["HTS Heading"]).strip()
                    try:
                        effective_date = pd.to_datetime(
                            row["Effective Date"]
                        ).strftime("%Y-%m-%d")
                    except ValueError:
                        effective_date = str(row["Effective Date"])

                    try:
                        applied_rate_numeric = float(row["Applied Rate"])
                    except ValueError:
                        applied_rate_numeric = 0.0

                    tariff_lookup[hts_heading] = {
                        "date": effective_date,
                        "rate": applied_rate_numeric,
                    }
        else:
            print(
                "Warning: tariff_rates_df is empty or None. Rates/dates will be 'not specified'."
            )

        data_for_final_df = []
        default_tariff_info = {
            "date": "Date not in provided DF",
            "rate": "Rate not in provided DF",
        }

        for _, row in extracted_codes_df.iterrows():
            hs_code = row["Product HS Code"]
            ch99_heading = row["Chapter 99 Heading"]

            tariff_info = tariff_lookup.get(ch99_heading, default_tariff_info)

            data_for_final_df.append(
                {
                    "Product HS Code": hs_code,
                    "Tariff Rate Applied": tariff_info["rate"],
                    "Effective Date": tariff_info["date"],
                    "Chapter 99 Heading": ch99_heading,
                }
            )

        if (
            not data_for_final_df
        ):  # Should not happen if extracted_codes_df is not empty
            return pd.DataFrame(columns=final_columns)

        final_df = pd.DataFrame(data_for_final_df, columns=final_columns)
        if not final_df.empty:
            final_df["Product HS Code"] = final_df["Product HS Code"].astype(str)
            final_df.drop_duplicates(
                subset=["Product HS Code", "Chapter 99 Heading"],
                inplace=True,
                keep="first",
            )
        return final_df
    return extract_codes_from_pdf, map_tariff_rates


@app.cell
def _(extract_codes_from_pdf):
    pdf_file_path = "data/raw/us_raw_tariff_data/HTS_HEADING_TO_HS_CODE.pdf"
    code_df = extract_codes_from_pdf(pdf_file_path)
    code_df
    return (code_df,)


@app.cell
def _(us_tariff_df):
    us_tariff_df
    return


@app.cell
def _(code_df, map_tariff_rates, us_tariff_df):
    merged_tariff_df = map_tariff_rates(code_df, us_tariff_df)
    merged_tariff_df["Product HS6"] = merged_tariff_df["Product HS Code"].apply(
        lambda x: x.replace(".", "")[:6]
    )

    print(merged_tariff_df)
    return (merged_tariff_df,)


@app.cell
def _(merged_tariff_df, pd):
    official_us_hs6_tariffs = (
        merged_tariff_df.groupby(["Product HS6", "Effective Date"])[
            "Tariff Rate Applied"
        ]
        .mean()
        .reset_index()
    )

    official_us_hs6_tariffs["Effective Date"] = pd.to_datetime(
        official_us_hs6_tariffs["Effective Date"]
    )

    official_us_hs6_tariffs["hs_revision"] = "HS6"
    official_us_hs6_tariffs.rename(
        columns={"Product HS6": "product_code"}, inplace=True
    )

    official_us_hs6_tariffs.head()
    return (official_us_hs6_tariffs,)


@app.cell
def _():
    # Remap it
    return


@app.cell
def _(Path, pd, pl):
    # Need to remap these to the HS Nomen 0


    def vectorized_hs_translation(
        input_lf: pl.LazyFrame, mapping_dir: str = "data/raw/hs_reference"
    ) -> pl.LazyFrame:
        print("Starting HS code translation to H0 (HS92).")

        # Define which HS revisions require mapping (all except H0)
        hs_versions = [
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
        ]  # H1, H2, H3, H4, H5, H6

        mapping_dfs_pd = []  # List to hold pandas DataFrames
        for hs_version in hs_versions:
            # Build mapping file path (assumes file naming convention like H1_to_H0.CSV)
            path = Path(mapping_dir) / f"{hs_version}_to_H0.CSV"

            try:
                mapping_pd = pd.read_csv(
                    path,
                    dtype=str,
                    usecols=[0, 2],
                    encoding="iso-8859-1",
                )

                mapping_pd.columns = ["source_code", "target_code"]
                mapping_pd["hs_revision"] = hs_version
                mapping_pd["source_code"] = mapping_pd["source_code"].astype(str)

                mapping_dfs_pd.append(mapping_pd)

            except FileNotFoundError as e:
                # # logger.warning(f"Mapping file not found for {hs_version}: {path}. Skipping.")
                raise ValueError(
                    f"Error loading mapping file for {hs_version}: \n {e}"
                ) from e

            except Exception as e:
                # # logger.error(f"Error loading mapping file for {hs_version}: {path}. Error: {e}")
                raise ValueError(
                    f"Error loading mapping file for {hs_version}: \n {e}"
                ) from e

        if not mapping_dfs_pd:
            # # logger.warning("No HS mapping files loaded.")
            raise

        # Combine all pandas mapping DataFrames into one.
        mapping_all_pd = pd.concat(mapping_dfs_pd, ignore_index=True)

        # Convert the combined pandas DataFrame to a Polars LazyFrame.
        schema = {
            "source_code": pl.Utf8,
            "target_code": pl.Utf8,
            "hs_revision": pl.Utf8,
        }
        mapping_all = pl.from_pandas(
            mapping_all_pd, schema_overrides=schema
        ).lazy()

        df = input_lf
        # df = df.with_columns(
        #     # pl.col("source_code").str.pad_end(6, fill_char="0"),
        #     pl.col("product_code").str.pad_end(6, fill_char="0")
        # )

        # Split rows where translation is not needed (H0) from those that need translation.
        df_h0 = df.filter(pl.col("hs_revision") == "H0")
        df_non_h0 = df.filter(pl.col("hs_revision") != "H0")

        # Perform a vectorized join between df_non_h0 and the mapping dataframe.
        df_non_h0 = df_non_h0.join(
            mapping_all,
            left_on=["hs_revision", "product_code"],
            right_on=["hs_revision", "source_code"],
            how="left",
        )

        # Create the translated HS code column: use target_code if available, otherwise fallback to the
        # original code.
        print(f"DF non H0 after join: {df_non_h0.head().collect()}")

        df_non_h0 = (
            df_non_h0.with_columns(
                pl.when(
                    pl.col("target_code").is_not_null()
                )  # Check if target_code exists from join
                .then(pl.col("target_code"))
                .otherwise(pl.col("product_code"))  # Keep original if no match
                .alias("product_code_translated")  # Use a temporary name
            )
            .drop("product_code")
            .rename({"product_code_translated": "product_code"})
        )

        # Optionally drop unnecessary columns from join
        df_non_h0 = df_non_h0.drop(
            ["target_code"]
        )  # Drop the mapping target code column

        print(f"DF non H0 after join and merge: {df_non_h0.head().collect()}")

        # Combine the rows which were already in H0 with the ones translated.
        common_cols = df_h0.collect_schema().names()
        df_final = pl.concat(
            [df_h0.select(common_cols), df_non_h0.select(common_cols)],
            how="vertical_relaxed",
        )

        # logger.info("✅ HS code translation completed.")
        return df_final
    return (vectorized_hs_translation,)


@app.cell
def _(official_us_hs6_tariffs, pl, vectorized_hs_translation):
    remapped_lf = pl.LazyFrame(official_us_hs6_tariffs)

    remapped_lf = vectorized_hs_translation(remapped_lf)

    remapped_df = remapped_lf.collect().to_pandas()
    remapped_df
    return (remapped_df,)


@app.cell
def _(mo):
    mo.md(r"""Compare the official_us_hs6_tariffs which I've extracted from the US data, with the WITS dataset.""")
    return


@app.cell
def _(unified_lf):
    unified_lf.collect_schema()
    return


@app.cell
def _(pl, unified_lf):
    filtered_lf = unified_lf.filter(
        pl.col("reporter_country") == "156", pl.col("partner_country") == "840"
    )
    print(filtered_lf.head().collect())

    filtered_df = filtered_lf.collect()
    return (filtered_df,)


@app.cell
def _():
    # # For each item the officuial_us_hs6, find the corresponding entry in our WITS data between US China.
    # # Then see how different they are.


    # for row in official_us_hs6_tariffs.iterrows():
    #     hs6_c = row[1]["Product HS6"]
    #     year_applied = row[1]["Effective Date"].year
    #     official_tariff = row[1]["Tariff Rate Applied"]

    #     wits_data = filtered_df.filter(
    #         pl.col("product_code") == hs6_c, pl.col("year") == str(year_applied)
    #     )

    #     print(wits_data)
    #     print(hs6_c, year_applied, official_tariff)
    #     break
    return


@app.cell
def _():
    return


@app.cell
def _(filtered_df, mo, pd, pl, remapped_df):
    results_list = []
    processed_rows_count = 0
    matched_official_rows_count = 0
    no_match_official_rows_count = 0
    total_wits_matches_collected = 0

    for index, row in mo.status.progress_bar(
        remapped_df.iterrows(), title="Mapping values", total=len(remapped_df)
    ):
        processed_rows_count += 1
        # Product code and original tariff details from remapped_df (official data)
        official_hs6_c_from_remapped = str(row["product_code"])
        original_year_applied_from_remapped = pd.to_datetime(
            row["Effective Date"]
        ).year
        original_tariff_from_remapped = row["Tariff Rate Applied"]

        # Filter WITS data based on product_code and year from remapped_df
        # This assumes WITS product_code should exactly match remapped_df product_code for a valid join
        wits_data = filtered_df.filter(
            (pl.col("product_code") == official_hs6_c_from_remapped)
            & (pl.col("year") == str(original_year_applied_from_remapped))
        )

        if wits_data.height > 0:
            matched_official_rows_count += (
                1  # Official row found at least one WITS match
            )
            for wits_row_dict in wits_data.to_dicts():
                total_wits_matches_collected += 1

                # These will be the final "official" tariff and year for the combined row
                final_official_tariff = original_tariff_from_remapped
                final_official_year = original_year_applied_from_remapped
                tariff_rule_source = None

                # Get the product code from the matched WITS record.
                # Due to the filter condition (pl.col("product_code") == official_hs6_c_from_remapped),
                # wits_product_code will be the same as official_hs6_c_from_remapped.
                # The check is essentially on this common product code.
                wits_product_code = str(wits_row_dict.get("product_code", ""))

                if wits_product_code.startswith("72"):
                    final_official_tariff = 0.25  # 25%
                    final_official_year = 2018
                    tariff_rule_source = (
                        "Section 232 Steel (WITS Product Code Trigger)"
                    )
                elif wits_product_code.startswith("76"):
                    final_official_tariff = 0.10  # 10%
                    final_official_year = 2018
                    tariff_rule_source = (
                        "Section 232 Aluminium (WITS Product Code Trigger)"
                    )

                combined_data = {
                    "official_hs6_code": official_hs6_c_from_remapped,  # From remapped_df
                    "official_year_applied": final_official_year,  # Potentially overridden by S232 rule
                    "official_tariff_rate": final_official_tariff,  # Potentially overridden by S232 rule
                    **wits_row_dict,  # All columns from WITS data
                }
                if tariff_rule_source:
                    combined_data["tariff_rule_source"] = tariff_rule_source
                results_list.append(combined_data)
        else:
            # No WITS data found for the product_code and year from remapped_df
            no_match_official_rows_count += 1
            # print(
            #     f"No matching WITS data found for Official HS6: {official_hs6_c_from_remapped}, Year: {original_year_applied_from_remapped}"
            # )

    results_df = None
    if results_list:
        results_df = pl.DataFrame(results_list)
        results_df = results_df.with_columns(
            (pl.col("official_tariff_rate") * 100).alias(
                "official_tariff_rate_percentage"
            )
        )  # Consider if you want to replace official_tariff_rate or add a new column
        print("\nFinal Combined DataFrame:")
        print(results_df)
    else:
        print("\nNo results were collected.")

    # Print processing statistics
    print("\n--- Processing Statistics ---")
    print(
        f"Total rows processed from official tariffs data: {processed_rows_count}"
    )
    print(
        f"Number of official tariff rows with matching WITS data: {matched_official_rows_count}"
    )
    print(
        f"Number of official tariff rows with NO matching WITS data: {no_match_official_rows_count}"
    )

    if results_df is not None:
        print(f"Total rows in the final combined DataFrame: {results_df.height}")
        print(
            f"Total individual WITS records matched and included: {total_wits_matches_collected}"
        )
    else:
        print("Total rows in the final combined DataFrame: 0")
        print("Total individual WITS records matched and included: 0")

    # Optional: Sanity check for row counts
    if processed_rows_count != (
        matched_official_rows_count + no_match_official_rows_count
    ):
        print(
            "Warning: Discrepancy in processed row counts. This indicates an issue in the counting logic."
        )
    return (results_df,)


@app.cell
def _(results_df):
    print(results_df)
    return


@app.cell
def _(results_df):
    results_df[
        "official_hs6_code",
        "official_tariff_rate_percentage",
        "effective_tariff",
        "year",
    ]
    return


@app.cell
def _(px, results_df):
    px.histogram(results_df["effective_tariff", "official_tariff_rate_percentage"])
    return


@app.cell
def _():
    return


@app.cell
def _(px, results_df):
    px.scatter(
        results_df,
        x="official_tariff_rate_percentage",
        y="effective_tariff",
        color="year",
        hover_data=["product_code"],
    )
    return


@app.cell
def _(filtered_df, pl, px, results_df):
    # Plot the tariff pre-application and post application
    prod_cc = "511219"
    product_df = filtered_df.filter(pl.col("product_code") == prod_cc)

    product_df = product_df.sort("year")

    matched_official_tariff = results_df.filter(pl.col("product_code") == prod_cc)

    product_df = product_df.join(
        matched_official_tariff["year", "official_tariff_rate_percentage"],
        on="year",
        how="left",
    )

    product_df = product_df.with_columns(
        pl.col("official_tariff_rate_percentage").forward_fill()
    )

    product_df = product_df.with_columns(
        (
            pl.col("official_tariff_rate_percentage") - pl.col("effective_tariff")
        ).alias("difference_between_tariffs")
    )

    px.line(
        product_df,
        x="year",
        y=["effective_tariff", "official_tariff_rate_percentage"],
    )
    return


@app.cell
def _(filtered_df, pl, px, results_df):
    official_tariffs_subset = results_df.select(
        ["product_code", "year", "official_tariff_rate_percentage"]
    )

    # Join the effective tariffs (from filtered_df) with the official tariffs
    # This brings official_tariff_rate_percentage into the filtered_df context
    # for all product-year combinations present in filtered_df.
    data_with_both_tariffs = filtered_df.join(
        official_tariffs_subset, on=["product_code", "year"], how="left"
    )

    # Handle missing official_tariff_rate_percentage for each product:
    # 1. Sort by product and year to ensure forward fill works correctly chronologically for each product.
    #    This sorting is crucial for the .over("product_code") to process rows in the correct order for ffill.
    data_with_both_tariffs = data_with_both_tariffs.sort(["product_code", "year"])

    # 2. Forward fill official_tariff_rate_percentage within each product's timeline.
    # 3. Fill any remaining NaNs with 0 (as per "If the official_tariff rate is null, then leave it at 0").
    #    This handles cases where a product has no initial official tariff or no official tariff data at all.
    data_with_both_tariffs = data_with_both_tariffs.with_columns(
        pl.col("official_tariff_rate_percentage")
        .forward_fill()
        .over("product_code")  # Apply forward fill partitioned by product_code
        .fill_null(
            0
        )  # If official_tariff rate is null (even after ffill for a product), set to 0
    )

    # Calculate the absolute discrepancy between the two tariff rates
    data_with_both_tariffs = data_with_both_tariffs.with_columns(
        (pl.col("official_tariff_rate_percentage") - pl.col("effective_tariff"))
        .abs()
        .alias("absolute_discrepancy")
    )

    # Calculate the average absolute discrepancy for each year across all products
    yearly_average_discrepancy = (
        data_with_both_tariffs.group_by("year")
        .agg(pl.mean("absolute_discrepancy").alias("average_abs_discrepancy"))
        .sort("year")
    )

    # Plot the average absolute discrepancy over the years
    fig_discrepancy_line = px.line(
        yearly_average_discrepancy,
        x="year",
        y="average_abs_discrepancy",
        title="Average Absolute Tariff Discrepancy Over Years",
        labels={
            "year": "Year",
            "average_abs_discrepancy": "Average Absolute Discrepancy",
        },
    )

    fig_discrepancy_line.show()
    return


@app.cell
def _(results_df):
    results_df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Trade-weighted average tariff between the US and China

    1. Create a unified df
    2. Calculate the trade weighted average with and without the new tariffs
    """
    )
    return


@app.cell
def _(filtered_df):
    filtered_df  # <- Reporter China, partner USA (US Tariffs on China)
    return


@app.cell
def _(filtered_df):
    selected_df = filtered_df.select(
        [
            "year",
            "product_code",
            "value",
            "quantity",
            "effective_tariff",
            "unit_value_detrended",
        ]
    )

    selected_df.head()
    return (selected_df,)


@app.cell
def _(pl, px, results_df, selected_df):
    official_and_wits_df = selected_df.join(
        other=results_df.select(
            ["product_code", "year", "official_tariff_rate_percentage"]
        ),
        on=["product_code", "year"],
        how="left",
    )

    official_and_wits_df = official_and_wits_df.with_columns(
        pl.col("official_tariff_rate_percentage")
        .forward_fill()
        .over("product_code")  # Apply forward fill partitioned by product_code
        .fill_null(0)
    )

    official_and_wits_df = official_and_wits_df.with_columns(
        (pl.col("effective_tariff") + pl.col("official_tariff_rate_percentage"))
        .alias("full_tariff")
        .cast(pl.Float32)
    )

    # Calculate total value per year for weighting
    total_value_per_year = official_and_wits_df.group_by("year").agg(
        pl.sum("value").alias("total_yearly_value")
    )

    # Calculate weighted tariffs
    weighted_tariffs_df = (
        official_and_wits_df.with_columns(
            (pl.col("effective_tariff") * pl.col("value")).alias(
                "weighted_effective_tariff_value"
            ),
            (pl.col("full_tariff") * pl.col("value")).alias(
                "weighted_full_tariff_value"
            ),
        )
        .group_by("year")
        .agg(
            pl.sum("weighted_effective_tariff_value").alias(
                "sum_weighted_effective"
            ),
            pl.sum("weighted_full_tariff_value").alias("sum_weighted_full"),
        )
    )

    # Join with total yearly value and calculate the final trade-weighted tariffs
    final_weighted_tariffs_df = (
        weighted_tariffs_df.join(total_value_per_year, on="year", how="left")
        .with_columns(
            (
                pl.col("sum_weighted_effective") / pl.col("total_yearly_value")
            ).alias("trade_weighted_effective_tariff"),
            (pl.col("sum_weighted_full") / pl.col("total_yearly_value")).alias(
                "trade_weighted_full_tariff"
            ),
        )
        .select(
            [
                "year",
                "trade_weighted_effective_tariff",
                "trade_weighted_full_tariff",
            ]
        )
    )

    # Ensure 'year' is sorted for the line chart
    final_weighted_tariffs_df = final_weighted_tariffs_df.sort("year")

    plot_df = final_weighted_tariffs_df.unpivot(
        index="year",
        on=["trade_weighted_effective_tariff", "trade_weighted_full_tariff"],
        variable_name="tariff_type",
        value_name="weighted_tariff_rate",
    )

    # Create the line chart
    fig = px.line(
        plot_df.to_pandas(),
        x="year",
        y="weighted_tariff_rate",
        color="tariff_type",
        title="Trade-Weighted Effective and Full Tariffs Over Time",
        labels={
            "year": "Year",
            "weighted_tariff_rate": "Trade-Weighted Tariff Rate (%)",
            "tariff_type": "Tariff Type",
        },
    )
    fig.show()

    # print(final_weighted_tariffs_df)
    return


if __name__ == "__main__":
    app.run()
