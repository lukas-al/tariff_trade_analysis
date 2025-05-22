import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from typing import Optional

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl

    from mpil_tariff_trade_analysis.utils.pipeline_funcs import (
        vectorized_hs_translation,
    )
    return Optional, Path, mo, np, pd, pl, vectorized_hs_translation


@app.cell
def _():
    print("--- Adding manually identified US-China tariffs --- ")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Add USA exceptional tariffs to the data

    1. Hard coded mapping of Chapter 99 codes to tariff rate and applied date. These come from detailed reading of the announcement documents.
    2. Extract hscode to ch99 mapping from UCITS PDF documents.
    3. Aggregate the 10 digit codes and calculate an average tariff for them using the HS-CH99 mapping.
    4. Join these onto the unified dataset, specifically for US exports to China.

    ## Option 1: Manually extract from CH99 filings.
    ## Option 2: Use Carter-Mix for replication purposes

    Going with option 2 for now...
    """
    )
    return


@app.cell
def _(np, pd):
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
            "date": np.nan,
            "rate": np.nan,
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
    return


@app.cell
def _(mo):
    mo.md(r"""# V1 - Scraping UCITS documents""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Manual Records
    Data which was manually collected on Trade War 1 tariffs.
    """
    )
    return


@app.cell(hide_code=True)
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
    df_tariffs_comprehensive["HTS Heading"] = pd.NA

    # Update HTS Headings for specific records
    df_tariffs_comprehensive.loc[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 1)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2018-07-06")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.25),
        "HTS Heading",
    ] = "9903.88.01"

    df_tariffs_comprehensive.loc[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 2)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2018-08-23")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.25),
        "HTS Heading",
    ] = "9903.88.02"

    df_tariffs_comprehensive.loc[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 3)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2018-09-24")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.10),
        "HTS Heading",
    ] = "9903.88.03"

    df_tariffs_comprehensive.loc[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 3)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2019-05-10")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.25),
        "HTS Heading",
    ] = "9903.88.03"

    df_tariffs_comprehensive.loc[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 4A)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2019-09-01")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.15),
        "HTS Heading",
    ] = "9903.88.15"

    df_tariffs_comprehensive.loc[
        (df_tariffs_comprehensive["Tariff Target"] == "Chinese products (List 4A)")
        & (
            df_tariffs_comprehensive["Effective Date"]
            == pd.to_datetime("2020-02-14")
        )
        & (df_tariffs_comprehensive["Applied Rate"] == 0.075),
        "HTS Heading",
    ] = "9903.88.15"

    new_hts_records_list = [
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
        },
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
    df_new_hts_records = pd.DataFrame(new_hts_records_list)

    new_tariff_data_list = [
        {
            "Effective Date": pd.to_datetime("2025-02-05T00:00:00.000"),
            "Action Entity": "USA",
            "Tariff Target": "Products of China and Hong Kong (IEEPA)",
            "Legal Basis": "IEEPA",
            "Trade Value (USD)": None,
            "Applied Rate": 0.10,
            "Status": "Active",
            "HTS Heading": "9903.01.20",
        },
        {
            "Effective Date": pd.to_datetime("2025-03-04T00:00:00.000"),
            "Action Entity": "USA",
            "Tariff Target": "Products of China and Hong Kong (IEEPA)",
            "Legal Basis": "IEEPA",
            "Trade Value (USD)": None,
            "Applied Rate": 0.20,
            "Status": "Active",
            "HTS Heading": "9903.01.24",
        },
    ]
    df_new_tariff_data = pd.DataFrame(new_tariff_data_list)

    us_tariff_df = pd.concat(
        [df_tariffs_comprehensive, df_new_hts_records, df_new_tariff_data],
        ignore_index=True,
    )

    us_tariff_df = us_tariff_df.sort_values(
        by=["Effective Date", "Action Entity", "HTS Heading", "Tariff Target"]
    ).reset_index(drop=True)

    us_tariff_df
    return


@app.cell(hide_code=True)
def _(pd):
    # Exemptions
    exclusion_data_1_6 = {
        "8412.21.0075": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8418.69.0120": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8480.71.8045": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8482.10.5044": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8482.10.5048": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8482.10.5052": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8525.60.1010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.05",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/2018-28277.pdf",
            }
        ],
        "8412.21.0045": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.06",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_11152.pdf",
            }
        ],
        "8430.31.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.06",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_11152.pdf",
            }
        ],
        "8607.21.1000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.06",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_11152.pdf",
            }
        ],
        "8479.90.9496": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            }
        ],
        "8481.10.0090": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            },
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.08",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_21389.pdf",
            },
        ],
        "8481.90.9040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            }
        ],
        "8482.10.5032": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            }
        ],
        "8504.90.9690": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            }
        ],
        "8515.90.4000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            }
        ],
        "8536.50.9065": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.07",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_16310.pdf",
            },
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            },
        ],
        "8407.21.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.08",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_21389.pdf",
            }
        ],
        "8427.10.4000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.08",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_21389.pdf",
            }
        ],
        "8473.40.1000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.08",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_21389.pdf",
            }
        ],
        "8483.50.9040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.08",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_21389.pdf",
            }
        ],
        "8537.10.8000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8411.99.9085": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8413.50.0010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8413.50.0070": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8413.70.2004": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8413.70.2005": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8413.70.2090": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8413.91.9010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            },
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            },
        ],
        "8413.91.9060": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8414.90.4190": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8418.69.0110": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8418.69.0180": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8419.19.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8419.40.0080": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8419.90.3000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8421.99.0080": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8427.10.8090": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8428.90.0290": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8429.40.0020": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8429.52.1010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8430.10.0000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8431.39.0010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8431.49.9010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8455.22.0000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8482.10.5060": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8482.40.0000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8482.99.0500": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8482.99.6595": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8483.90.8010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8501.53.8040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8504.40.9550": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8504.40.9580": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8505.90.8000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8515.80.0000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8531.10.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8531.10.0050": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8531.80.0070": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8531.80.0085": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8535.30.0070": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8535.30.0080": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8535.40.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8535.40.0050": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8535.90.0080": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8536.50.9035": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8536.90.4000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8543.30.9040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "9018.19.9560": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "9024.10.0000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.10",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_25895.pdf",
            }
        ],
        "8402.90.0010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8402.90.0090": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8408.90.9010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8408.90.9020": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8411.99.9090": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8412.39.0080": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8412.80.1000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8413.60.0030": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8413.70.2015": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8413.91.9095": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8429.40.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8429.51.1015": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8429.51.1040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8429.51.1045": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8429.51.1050": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8429.51.5010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8431.10.0010": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8431.31.0040": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8431.31.0060": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8431.39.0070": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8431.49.9044": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8432.90.0060": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8501.10.4060": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8427.10.8070": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8427.10.8095": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
        "8504.40.4000": [
            {
                "date_of_exclusion_start": "2018-07-06",
                "chapter_99_code": "9903.88.11",
                "source_document_url": "https://ustr.gov/sites/default/files/enforcement/301Investigations/Notice_of_Product_Exclusions.pdf",
            }
        ],
    }

    # Initialize lists for each column
    hts_codes = []
    dates_of_exclusion_start = []
    chapter_99_codes = []
    source_document_urls = []

    # Iterate through the original dictionary
    for hts_code, exclusion_details_list in exclusion_data_1_6.items():
        for exclusion_detail in exclusion_details_list:
            hts_codes.append(hts_code)
            dates_of_exclusion_start.append(
                exclusion_detail["date_of_exclusion_start"]
            )
            chapter_99_codes.append(exclusion_detail["chapter_99_code"])
            source_document_urls.append(exclusion_detail["source_document_url"])

    # Create the restructured dictionary
    restructured_exclusion_data = pd.DataFrame(
        {
            "hts_code": hts_codes,
            "date_of_exclusion_start": dates_of_exclusion_start,
            "chapter_99_code": chapter_99_codes,
            "source_document_url": source_document_urls,
        }
    )

    restructured_exclusion_data
    return


@app.cell
def _(mo):
    mo.md(r"""## Extract data from PDF""")
    return


@app.cell
def _():
    # def extract_codes_from_pdf(pdf_path: str) -> pd.DataFrame:
    #     """
    #     Loads a PDF from the given path, extracts HS codes and Chapter 99 headings.
    #     Returns a DataFrame with "Product HS Code" and "Chapter 99 Heading".
    #     """
    #     document_text = ""
    #     try:
    #         with pdfplumber.open(pdf_path) as pdf:
    #             for page in pdf.pages:
    #                 page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
    #                 if page_text:
    #                     document_text += page_text + "\n"
    #     except FileNotFoundError:
    #         print(f"Error: PDF file not found at {pdf_path}")
    #         return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])
    #     except Exception as e:
    #         print(f"Error reading or processing PDF {pdf_path}: {e}")
    #         return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])

    #     if not document_text:
    #         print(f"No text could be extracted from PDF: {pdf_path}")
    #         return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])

    #     # Regex to find lines containing an HS code followed by a Chapter 99 code.
    #     # HS Code: XXXX.XX.XX or XXXX.XX.XXXX
    #     # Chapter 99 Code: XXXX.XX.XX
    #     # Assumes they are on the same line, separated by whitespace.
    #     pattern = re.compile(
    #         r"^([0-9]{4}\.[0-9]{2}\.[0-9]{2}(?:[0-9]{2})?)\s+([0-9]{4}\.[0-9]{2}\.[0-9]{2})$",
    #         re.MULTILINE,
    #     )

    #     extracted_pairs = []
    #     matches = pattern.findall(document_text)

    #     for hs_code_str, ch99_code_str in matches:
    #         extracted_pairs.append(
    #             {
    #                 "Product HS Code": hs_code_str,
    #                 "Chapter 99 Heading": ch99_code_str,
    #             }
    #         )

    #     if not extracted_pairs:
    #         print(
    #             f"No HS code / Chapter 99 pairs found in PDF with the pattern: {pattern.pattern}"
    #         )
    #         print(
    #             "Please check the PDF structure or the regex if an empty DataFrame is unexpected."
    #         )
    #         return pd.DataFrame(columns=["Product HS Code", "Chapter 99 Heading"])

    #     df = pd.DataFrame(
    #         extracted_pairs, columns=["Product HS Code", "Chapter 99 Heading"]
    #     )
    #     if not df.empty:
    #         df["Product HS Code"] = df["Product HS Code"].astype(str)
    #         df["Chapter 99 Heading"] = df["Chapter 99 Heading"].astype(str)
    #     return df

    # pdf_file_path = "data/raw/us_raw_tariff_data/HTS_HEADING_TO_HS_CODE_M25.pdf"
    # hs_to_ch99_mapping = extract_codes_from_pdf(pdf_file_path)

    # hs_to_ch99_mapping
    return


@app.cell
def _():
    # hs_to_ch99_mapping.to_csv("test_hstoch99")
    return


@app.cell
def _(mo):
    mo.md(
        r"""## Map tariff rates to HS codes using CH99 mapping and manual data"""
    )
    return


@app.cell
def _():
    # us_tariff_df_mapped = map_tariff_rates(hs_to_ch99_mapping, us_tariff_df)
    # us_tariff_df_mapped["Tariff Rate Applied"] = us_tariff_df_mapped[
    #     "Tariff Rate Applied"
    # ].astype(float)

    # us_tariff_df_mapped
    return


@app.cell
def _():
    # us_tariff_df_mapped["Product HS6"] = us_tariff_df_mapped[
    #     "Product HS Code"
    # ].apply(lambda x: x.replace(".", "")[:6])

    # official_us_hs6_tariffs = (
    #     us_tariff_df_mapped.groupby(["Product HS6", "Effective Date"])[
    #         "Tariff Rate Applied"
    #     ]
    #     .mean()
    #     .reset_index()
    # )

    # official_us_hs6_tariffs["Effective Date"] = pd.to_datetime(
    #     official_us_hs6_tariffs["Effective Date"]
    # )

    # official_us_hs6_tariffs["hs_revision"] = "HS6"
    # official_us_hs6_tariffs.rename(
    #     columns={"Product HS6": "product_code"}, inplace=True
    # )

    # official_us_hs6_tariffs.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## Remap the HS6 codes to HS Nom 0""")
    return


@app.cell
def _():
    # official_us_hs6_tariffs_remapped = vectorized_hs_translation(
    #     pl.LazyFrame(official_us_hs6_tariffs)
    # )
    return


@app.cell
def _():
    # official_us_tariffs_df = official_us_hs6_tariffs_remapped.collect().to_pandas()
    # official_us_tariffs_df
    return


@app.cell
def _(mo):
    mo.md(r"""# V2 - Use Carter Mix data""")
    return


@app.cell
def _(Path, pd, pl):
    base_dir = Path("data/raw/CarterMix/")
    list1 = pl.read_csv(base_dir / "Tariffs" / "part1.csv")
    list2 = pl.read_csv(base_dir / "Tariffs" / "part2.csv")
    list3 = pl.read_csv(base_dir / "Tariffs" / "part3.csv")
    list4 = pl.read_csv(base_dir / "Tariffs" / "part4a.csv")

    cm_us_tariffs = pl.concat([list1, list2, list3, list4]).to_pandas()
    cm_us_tariffs["time"] = pd.to_datetime(cm_us_tariffs["time"], format="%Ym%m")
    cm_us_tariffs["hs_code"] = cm_us_tariffs["hs_code"].astype(str)

    # Shorten the codes
    cm_us_tariffs["hs_code"] = cm_us_tariffs["hs_code"].apply(lambda x: x[:-2])
    cm_us_tariffs = cm_us_tariffs.rename(
        columns={
            "hs_code": "product_code",
            "time": "Effective Date",
            "tariff_add": "Tariff Rate Applied",
        }
    )
    cm_us_tariffs = cm_us_tariffs.groupby(
        "product_code"
    ).mean()  # Take the average when grouping across the new shortened codes
    cm_us_tariffs["hs_revision"] = "HS6"
    cm_us_tariffs = cm_us_tariffs.reset_index()
    cm_us_tariffs["Tariff Rate Applied"] = (
        cm_us_tariffs["Tariff Rate Applied"] * 100
    )

    cm_us_tariffs.head(1)
    return (cm_us_tariffs,)


@app.cell
def _(cm_us_tariffs, pl, vectorized_hs_translation):
    # Remap the cartermix to use HS0
    cm_us_tariffs_remapped = vectorized_hs_translation(pl.LazyFrame(cm_us_tariffs))
    return


@app.cell
def _(mo):
    mo.md(r"""# Join selected tariff dataset with the existing dataset""")
    return


@app.cell
def _(pl):
    # Previous 'unified_lf'
    unified_path = "/Users/lukasalemu/Documents/00. Bank of England/03. MPIL/tariff_trade_analysis/data/final/unified_trade_tariff_partitioned"

    unified_lf = pl.scan_parquet(unified_path)
    unified_lf.head().collect()
    return unified_lf, unified_path


@app.cell
def _(Optional, pd, pl):
    def combine_us_official_tariffs_with_unified_optimized(
        unified_lf: pl.LazyFrame,
        official_df: pd.DataFrame,
        max_year_for_ffill: Optional[int] = None,
    ) -> pl.LazyFrame:
        # Part 1: Prepare initial official tariff data from official_df
        # This calculates the mean tariff per product_code and year.
        official_base_tariffs_lf = (
            pl.from_pandas(official_df)
            .lazy()
            .with_columns(
                pl.col("product_code").cast(pl.String),
                pl.col("Effective Date").dt.year().alias("year"),
                pl.col("Tariff Rate Applied")
                .cast(pl.Float64)
                .alias("base_official_tariff_join"),
            )
            .select(["product_code", "year", "base_official_tariff_join"])
            # Drop rows where essential data for aggregation or join is missing
            .drop_nulls(
                subset=["product_code", "year", "base_official_tariff_join"]
            )
            .group_by(["product_code", "year"])
            .agg(pl.col("base_official_tariff_join").mean())  # Taking mean tariff
        )

        # Part 2: Determine the year range for scaffolding and forward-filling
        # Use provided max_year_for_ffill or determine from unified_lf
        actual_max_year_for_ffill = max_year_for_ffill
        if actual_max_year_for_ffill is None:
            max_year_collected = (
                unified_lf.select(pl.col("year").cast(pl.Int32).max())
                .collect()
                .item()
            )
            if max_year_collected is not None:
                actual_max_year_for_ffill = max_year_collected
            # If still None, it means unified_lf is empty or has no valid years; densification might be limited.

        min_year_collect = official_base_tariffs_lf.select(
            pl.col("year").min().alias("min_year")
        ).collect()
        min_year_in_official_lf = None
        if not min_year_collect.is_empty():  # Should be a 1-row DF
            min_year_in_official_lf = min_year_collect.item(0, "min_year")

        # Create a LazyFrame of all years to be covered by the scaffold
        all_years_lf = pl.LazyFrame(
            {"year": []}, schema={"year": pl.Int32}
        )  # Default to empty
        if (
            min_year_in_official_lf is not None
            and actual_max_year_for_ffill is not None
            and min_year_in_official_lf <= actual_max_year_for_ffill
        ):
            all_years_lf = pl.LazyFrame(
                {
                    "year": range(
                        min_year_in_official_lf, actual_max_year_for_ffill + 1
                    )
                }
            )

        # Part 3: Densify official_data_lf
        # Create a scaffold of unique product_codes from official_df crossed with all relevant years
        unique_products_lf = official_base_tariffs_lf.select(
            "product_code"
        ).unique()

        # product_year_scaffold_lf will be empty if unique_products_lf or all_years_lf is empty
        product_year_scaffold_lf = unique_products_lf.join(
            all_years_lf, how="cross"
        )

        # Join the scaffold with the actual tariff data, then sort and forward-fill
        densified_official_tariffs_lf = (
            product_year_scaffold_lf.join(
                official_base_tariffs_lf, on=["product_code", "year"], how="left"
            )
            .sort("product_code", "year")
            .with_columns(
                pl.col("base_official_tariff_join")
                .forward_fill()
                .over("product_code")
            )
        )

        # Prepare the final official_data_lf to be joined with unified_lf:
        # Add dummy reporter/partner countries and filter out rows where no tariff exists (even after ffill)
        official_data_to_join_lf = (
            densified_official_tariffs_lf.with_columns(
                pl.lit("156").cast(pl.String).alias("reporter_country"),  # China
                pl.lit("840").cast(pl.String).alias("partner_country"),  # USA
            )
            .filter(
                pl.col(
                    "base_official_tariff_join"
                ).is_not_null()  # Only keep product-years that have an actual tariff
            )
            .select(
                [
                    "product_code",
                    "year",
                    "base_official_tariff_join",
                    "reporter_country",
                    "partner_country",
                ]
            )
        )

        # Part 4: Join with unified_lf and apply logic
        # Ensure join key types in unified_lf are compatible
        unified_lf = unified_lf.with_columns(
            pl.col("product_code").cast(pl.String),
            pl.col("year").cast(pl.Int32),
            pl.col("reporter_country").cast(pl.String),  # Ensure type consistency
            pl.col("partner_country").cast(pl.String),  # Ensure type consistency
        )

        # Left join unified_lf with the densified and forward-filled official tariff data
        unified_lf = unified_lf.join(
            official_data_to_join_lf,
            on=["product_code", "year", "reporter_country", "partner_country"],
            how="left",
        )

        # Conditions for applying tariffs (S232 or general)
        # apply_tariff_condition relies on base_official_tariff_join being non-null,
        # which means a match occurred on product, year, and the specific reporter/partner pair.
        apply_tariff_condition = pl.col("partner_country") == "840"

        s232_steel_cond = apply_tariff_condition & pl.col(
            "product_code"
        ).str.starts_with("72")
        s232_aluminium_cond = apply_tariff_condition & pl.col(
            "product_code"
        ).str.starts_with("76")

        general_tariff_cond = apply_tariff_condition & ~(
            s232_steel_cond | s232_aluminium_cond
        )

        # Calculate the single 'official_effective_tariff' column
        unified_lf = unified_lf.with_columns(
            pl.when(s232_steel_cond)
            .then(pl.lit(0.25))
            .when(s232_aluminium_cond)
            .then(pl.lit(0.10))
            .when(general_tariff_cond)
            .then(pl.col("base_official_tariff_join"))
            .otherwise(None)  # Null if no conditions met for an official tariff
            .cast(pl.Float32)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("official_tariff")
        )

        # # Multiply by 100 to bring into same space as the average_tariff_official previously
        # unified_lf = unified_lf.with_columns(pl.col("official_tariff"))

        unified_lf = unified_lf.with_columns(
            (pl.col("average_tariff") + pl.col("official_tariff")).alias(
                "average_tariff_official"
            )
        )
        # Drop the intermediate helper column
        unified_lf = unified_lf.drop("base_official_tariff_join")

        return unified_lf
    return (combine_us_official_tariffs_with_unified_optimized,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Perform the join

    Testing the use of the data extracted from their website, vs tariffs in Carter Mix (which are date appropriate)
    """
    )
    return


@app.cell
def _(
    cm_us_tariffs,
    combine_us_official_tariffs_with_unified_optimized,
    pl,
    unified_lf,
):
    enhanced_unified_lf = combine_us_official_tariffs_with_unified_optimized(
        unified_lf,
        cm_us_tariffs,  # official_us_tariffs_df
        max_year_for_ffill=2023,
    )

    enhanced_unified_lf = enhanced_unified_lf.with_columns(
        pl.col("year").cast(pl.String)
    )

    print("New schema for unified_lf: ", enhanced_unified_lf.collect_schema())
    return (enhanced_unified_lf,)


@app.cell
def _(enhanced_unified_lf):
    print(enhanced_unified_lf.head().collect(engine="streaming"))
    return


@app.cell
def _(enhanced_unified_lf, pl, unified_path):
    # Sink the output
    print("Sinking back to unified_lf")
    enhanced_unified_lf.sink_parquet(
        pl.PartitionByKey(
            base_path=unified_path,
            by=pl.col("year"),
        ),
        mkdir=True,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""<!-- # Compare with data in Carter-Mix replication materials -->""")
    return


@app.cell
def _():
    # base_dir = Path("data/raw/CarterMix/")
    # list1 = pl.read_csv(base_dir / "Tariffs" / "part1.csv")
    # list2 = pl.read_csv(base_dir / "Tariffs" / "part2.csv")
    # list3 = pl.read_csv(base_dir / "Tariffs" / "part3.csv")
    # list4 = pl.read_csv(base_dir / "Tariffs" / "part4a.csv")

    # cm_us_tariffs = pl.concat([list1, list2, list3, list4]).to_pandas()
    # cm_us_tariffs["time"] = pd.to_datetime(cm_us_tariffs["time"], format="%Ym%m")
    # cm_us_tariffs["hs_code"] = cm_us_tariffs["hs_code"].astype(str)

    # # Shorten the codes
    # cm_us_tariffs["hs_code"] = cm_us_tariffs["hs_code"].apply(lambda x: x[:-2])
    # cm_us_tariffs = cm_us_tariffs.rename(columns={"hs_code": "product_code"})
    # cm_us_tariffs = cm_us_tariffs.groupby("product_code").mean()
    # cm_us_tariffs["hs_revision"] = "HS6"
    # cm_us_tariffs = cm_us_tariffs.reset_index()
    # cm_us_tariffs["tariff_add"] = cm_us_tariffs["tariff_add"] * 100
    # #
    # cm_us_tariffs
    # # cm_us_tariffs = vectorized_hs_translation(cm_us_tariffs)
    return


@app.cell
def _():
    # official_us_tariffs_df.sort_values("product_code")
    return


@app.cell
def _():
    # cm_us_tariffs["month_year"] = cm_us_tariffs["time"].dt.to_period("M")
    # official_us_tariffs_df["month_year"] = official_us_tariffs_df[
    #     "Effective Date"
    # ].dt.to_period("M")

    # # Perform the merge
    # comparison_df = pd.merge(
    #     cm_us_tariffs,
    #     official_us_tariffs_df,
    #     left_on=["product_code", "month_year", "tariff_add"],
    #     right_on=["product_code", "month_year", "Tariff Rate Applied"],
    #     how="inner",
    # )

    # # Display rows where all three conditions match
    # comparison_df

    # # cm_us_tariffs["month_year"] = cm_us_tariffs["time"].dt.to_period("M")
    # # official_us_tariffs_df["month_year"] = official_us_tariffs_df[
    # #     "Effective Date"
    # # ].dt.to_period("M")

    # # # Perform an outer merge with an indicator
    # # comparison_df_outer = pd.merge(
    # #     cm_us_tariffs,
    # #     official_us_tariffs_df,
    # #     left_on=["product_code", "month_year", "tariff_add"],
    # #     right_on=["product_code", "month_year", "Tariff Rate Applied"],
    # #     how="outer",
    # #     indicator=True,
    # # )

    # # # Filter for rows that are not in both DataFrames
    # # non_matching_rows = comparison_df_outer[
    # #     comparison_df_outer["_merge"] != "both"
    # # ]

    # # # Display non-matching rows
    # # print("Rows in cm_us_tariffs but not in official_us_tariffs_df:")
    # # print(non_matching_rows[non_matching_rows["_merge"] == "left_only"])

    # # print("\nRows in official_us_tariffs_df but not in cm_us_tariffs:")
    # # print(non_matching_rows[non_matching_rows["_merge"] == "right_only"])
    return


@app.cell
def _():
    # # Try load the hs6 tariff data from CM
    # cm_hs6_tariffs = pd.read_stata(base_dir / "PanelElasticity" / "HS6Tariffs.dta")
    # cm_hs6_tariffs = pd.read_stata(base_dir / "PanelElasticity" / "HS6TariffsNonChina.dta")
    # cm_hs6_tariffs
    return


if __name__ == "__main__":
    app.run()
