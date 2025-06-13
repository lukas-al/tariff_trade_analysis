import os
import pickle
import re
from typing import Dict, List, Optional

import polars as pl
import pycountry
import pyfixest
from tqdm import tqdm

# ---- PARAMETERS ----
# List of independent variables (column names in the data)
INDEPENDENT_VARS = [
    "quantity",
    "value",
    "unit_value",
    # Add more as needed
]

# Data path
UNIFIED_LF_PATH = "/Users/lukasalemu/Documents/00. Bank of England/03. MPIL/tariff_trade_analysis/data/final/unified_trade_tariff_partitioned"

# Output directory for pickles
OUTPUT_DIR = "/Users/lukasalemu/Documents/00. Bank of England/03. MPIL/tariff_trade_analysis/notebooks/dev5/regression_pickles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Country Codes
USA_CC = pycountry.countries.search_fuzzy("USA")[0].numeric  # type: ignore
CHINA_CC = pycountry.countries.search_fuzzy("China")[0].numeric  # type: ignore
BRAZIL_CC = pycountry.countries.search_fuzzy("Brazil")[0].numeric  # type: ignore
IRELAND_CC = pycountry.countries.search_fuzzy("Ireland")[0].numeric  # type: ignore
JAPAN_CC = pycountry.countries.search_fuzzy("Japan")[0].numeric  # type: ignore
ITALY_CC = pycountry.countries.search_fuzzy("Italy")[0].numeric  # type: ignore
SOUTHAFRICA_CC = pycountry.countries.search_fuzzy("South Africa")[0].numeric  # type: ignore
UK_CC = pycountry.countries.search_fuzzy("United Kingdom")[0].numeric  # type: ignore
GERMANY_CC = pycountry.countries.search_fuzzy("Germany")[0].numeric  # type: ignore
FRANCE_CC = pycountry.countries.search_fuzzy("France")[0].numeric  # type: ignore
KOREA_CC = pycountry.countries.search_fuzzy("Korea")[0].numeric  # type: ignore
TURKEY_CC = pycountry.countries.search_fuzzy("Turkiye")[0].numeric  # type: ignore
AUSTRALIA_CC = pycountry.countries.search_fuzzy("Australia")[0].numeric  # type: ignore
SAUDI_CC = pycountry.countries.search_fuzzy("Saudi Arabia")[0].numeric  # type: ignore
MEXICO_CC = pycountry.countries.search_fuzzy("Mexico")[0].numeric  # type: ignore
CANADA_CC = pycountry.countries.search_fuzzy("Canada")[0].numeric  # type: ignore
INDONESIA_CC = pycountry.countries.search_fuzzy("Indonesia")[0].numeric  # type: ignore
INDIA_CC = pycountry.countries.search_fuzzy("India")[0].numeric  # type: ignore
VIETNAM_CC = pycountry.countries.search_fuzzy("Vietnam")[0].numeric  # type: ignore
RUSSIA_CC = pycountry.countries.search_fuzzy("Russia")[0].numeric  # type: ignore
HONGKONG_CC = pycountry.countries.search_fuzzy("Hong Kong")[0].numeric  # type: ignore


# ---- UTILITY FUNCTIONS ----
def get_oil_exporting_countries(lzdf: pl.LazyFrame, oil_export_percentage_threshold: float) -> List[str]:
    total_exports = lzdf.group_by("reporter_country").agg(pl.sum("value").alias("total_value"))
    oil_exports = lzdf.filter(pl.col("product_code").str.starts_with("27")).group_by("reporter_country").agg(pl.sum("value").alias("oil_value"))
    summary = total_exports.join(oil_exports, on="reporter_country", how="left").with_columns(pl.col("oil_value").fill_null(0.0))
    summary = summary.with_columns(((pl.col("oil_value") / pl.col("total_value")) * 100).alias("oil_export_percentage"))
    filtered_countries = summary.filter(pl.col("oil_export_percentage") > oil_export_percentage_threshold)
    return filtered_countries.collect()["reporter_country"].to_list()


def prepare_analysis_data(
    source_lf: pl.LazyFrame,
    top_n: Optional[int] = None,
    selection_year: Optional[str] = None,
    year_range_to_keep: Optional[List[str]] = None,
    selection_method: str = "total_trade",
    oil_export_threshold: Optional[float] = 50.0,
    countries_to_exclude: Optional[List[str]] = None,
    countries_to_include: Optional[List[str]] = None,
    product_codes_to_exclude: Optional[List[str]] = None,
) -> pl.LazyFrame:
    if countries_to_include and (top_n or selection_year):
        raise ValueError("'countries_to_include' cannot be used with 'top_n' or 'selection_year'.")
    if not countries_to_include and not (top_n and selection_year):
        raise ValueError("Either 'countries_to_include' or both 'top_n' and 'selection_year' must be provided.")
    lf = source_lf
    if product_codes_to_exclude:
        exclusion_expr = pl.any_horizontal(pl.col("product_code").str.starts_with(code) for code in product_codes_to_exclude)
        lf = lf.filter(~exclusion_expr)
    if oil_export_threshold is not None:
        oil_countries = get_oil_exporting_countries(lf, oil_export_threshold)
        lf = lf.filter(~pl.col("reporter_country").is_in(oil_countries))
    if countries_to_include:
        top_countries_list = countries_to_include
    else:
        trade_in_year_lf = lf.filter(pl.col("year") == selection_year)
        if selection_method == "importers":
            grouped = trade_in_year_lf.group_by("partner_country").agg(pl.sum("value").alias("import_value")).sort("import_value", descending=True)
            if top_n is not None:
                grouped = grouped.head(top_n)
            top_countries_df = grouped.collect()
            top_countries_list = top_countries_df["partner_country"].to_list()
        elif selection_method == "total_trade":
            exports_lf = trade_in_year_lf.select(pl.col("reporter_country").alias("country"), "value")
            imports_lf = trade_in_year_lf.select(pl.col("partner_country").alias("country"), "value")
            grouped = (
                pl.concat([exports_lf, imports_lf]).group_by("country").agg(pl.sum("value").alias("total_trade")).sort("total_trade", descending=True)
            )
            if top_n is not None:
                grouped = grouped.head(top_n)
            top_countries_df = grouped.collect()
            top_countries_list = top_countries_df["country"].to_list()
        else:
            raise ValueError("selection_method must be 'importers' or 'total_trade'")
    if countries_to_exclude:
        top_countries_list = [str(c) for c in top_countries_list if c not in countries_to_exclude]
    analysis_lf = lf.filter(pl.col("reporter_country").is_in(top_countries_list) & pl.col("partner_country").is_in(top_countries_list))
    if year_range_to_keep:
        analysis_lf = analysis_lf.filter(pl.col("year").is_in(year_range_to_keep))
    return analysis_lf


def run_saturated_regression(
    data: pl.LazyFrame,
    formula: str,
    year_range: List[str],
    countries_to_exclude: Optional[List[str]] = None,
    vcov: Optional[str] = None,
    filter_expression: Optional[pl.Expr] = None,
):
    USA_CC = "840"
    CHINA_CC = "156"
    tariff_expr = (
        pl.col("average_tariff_official")
        .filter((pl.col("partner_country") == USA_CC) & (pl.col("reporter_country") == CHINA_CC))
        .mean()
        .over(["year", "product_code"])
        .alias("tariff_us_china")
    )
    base_lf = data.with_columns(
        pl.col("partner_country").alias("importer"),
        pl.col("reporter_country").alias("exporter"),
        tariff_expr,
    )
    if countries_to_exclude is None:
        countries_to_exclude = [CHINA_CC, USA_CC]
    importers = base_lf.select("importer").unique().filter(~pl.col("importer").is_in(countries_to_exclude)).collect().to_series().to_list()
    interaction_expressions = []
    formula_terms = []
    for country_code in importers:
        for year in year_range:
            term_name = f"gamma_{country_code}_{year}"
            formula_terms.append(term_name)  # type: ignore
            interaction_filter = (pl.col("importer") == country_code) & (pl.col("exporter") == CHINA_CC) & (pl.col("year") == year)
            expression = pl.when(interaction_filter).then(pl.col("tariff_us_china")).otherwise(0.0).alias(term_name)
            interaction_expressions.append(expression)
    final_lf = base_lf.with_columns(*interaction_expressions)
    final_lf = final_lf.drop(
        [
            "tariff_rate_pref",
            "min_rate_pref",
            "max_rate_pref",
            "tariff_rate_mfn",
            "min_rate_mfn",
            "max_rate_mfn",
            "average_tariff",
            "value_global_trend",
            "value_detrended",
            "quantity_global_trend",
            "quantity_detrended",
            "price_global_trend",
            "unit_value_detrended",
            "value_global_trend_right",
            "quantity_global_trend_right",
            "price_global_trend_right",
            "reporter_country",
            "partner_country",
        ]
    )
    saturated_effects = " + ".join(formula_terms)
    if "|" in formula:
        parts = formula.split("|", 1)
        final_formula = f"{parts[0].strip()} ~ {saturated_effects} | {parts[1].strip()}"
    else:
        final_formula = f"{formula.strip()} ~ {saturated_effects}"
    dependent_var_col = re.findall(r"\b\w+\b", formula.split("~")[0].strip())[-1]
    clean_df = final_lf.drop_nulls(subset=[dependent_var_col, "tariff_us_china"]).collect().to_pandas()
    # Always use a supported string for vcov
    if vcov is None:
        vcov = "hetero"
    model = pyfixest.feols(fml=final_formula, data=clean_df, vcov=vcov)  # type: ignore
    return model


# ---- MAIN SCRIPT ----
def main():
    # Steel/Aluminum product codes to exclude
    alu_steel_product_codes = [
        "720610",
        "720690",
        "720711",
        "720712",
        "720719",
        "720720",
        "720810",
        "720825",
        "720826",
        "720827",
        "720836",
        "720837",
        "720838",
        "720839",
        "720840",
        "720851",
        "720852",
        "720853",
        "720854",
        "720890",
        "720915",
        "720916",
        "720917",
        "720918",
        "720925",
        "720926",
        "720927",
        "720928",
        "720990",
        "721011",
        "721012",
        "721020",
        "721030",
        "721041",
        "721049",
        "721050",
        "721061",
        "721069",
        "721070",
        "721090",
        "721113",
        "721114",
        "721119",
        "721123",
        "721129",
        "721190",
        "721210",
        "721220",
        "721230",
        "721240",
        "721250",
        "721260",
        "721310",
        "721320",
        "721391",
        "721399",
        "721410",
        "721420",
        "721430",
        "721491",
        "721499",
        "721510",
        "721550",
        "721590",
        "721610",
        "721621",
        "721622",
        "721631",
        "721632",
        "721633",
        "721640",
        "721650",
        "721699",
        "721710",
        "721720",
        "721730",
        "721790",
        "721810",
        "721891",
        "721899",
        "721911",
        "721912",
        "721913",
        "721914",
        "721921",
        "721922",
        "721923",
        "721924",
        "721931",
        "721932",
        "721933",
        "721934",
        "721935",
        "721990",
        "722011",
        "722012",
        "722020",
        "722090",
        "722100",
        "722211",
        "722219",
        "722220",
        "722230",
        "722240",
        "722300",
        "722410",
        "722490",
        "722511",
        "722519",
        "722530",
        "722540",
        "722550",
        "722591",
        "722592",
        "722599",
        "722611",
        "722619",
        "722620",
        "722691",
        "722692",
        "722699",
        "722710",
        "722720",
        "722790",
        "722810",
        "722820",
        "722830",
        "722840",
        "722850",
        "722860",
        "722870",
        "722880",
        "722920",
        "722990",
        "730110",
        "730210",
        "730240",
        "730290",
        "730411",
        "730419",
        "730422",
        "730423",
        "730424",
        "730429",
        "730431",
        "730439",
        "730441",
        "730449",
        "730451",
        "730459",
        "730490",
        "730511",
        "730512",
        "730519",
        "730520",
        "730531",
        "730539",
        "730590",
        "730611",
        "730619",
        "730621",
        "730629",
        "730630",
        "730640",
        "730650",
        "730661",
        "730669",
        "730690",
        "760110",
        "760120",
        "760410",
        "760421",
        "760429",
        "760511",
        "760519",
        "760521",
        "760529",
        "760611",
        "760612",
        "760691",
        "760692",
        "760711",
        "760719",
        "760720",
        "760810",
        "760820",
        "760900",
    ]
    # Load data
    print("Loading and preparing data...")
    unified_lf = pl.scan_parquet(UNIFIED_LF_PATH)
    # Prepare data
    analysis_lf = prepare_analysis_data(
        source_lf=unified_lf,
        top_n=30,
        selection_year="2017",
        year_range_to_keep=[str(y) for y in range(2016, 2021)],
        selection_method="total_trade",
        oil_export_threshold=50.0,
        countries_to_exclude=[RUSSIA_CC, HONGKONG_CC, ITALY_CC, IRELAND_CC],
        product_codes_to_exclude=alu_steel_product_codes,
    )
    # Run regression for each independent variable
    results: Dict[str, object] = {}
    print(f"Starting regressions for {len(INDEPENDENT_VARS)} dependent variables...")
    for dep_var in tqdm(INDEPENDENT_VARS, desc="Regressions", unit="var"):
        formula = f"log({dep_var}) | importer^year^product_code + importer^exporter"
        print(f"\n[INFO] Running regression for: {dep_var}")
        model = run_saturated_regression(
            data=analysis_lf,
            formula=formula,
            year_range=[str(y) for y in range(2017, 2021)],
        )
        results[dep_var] = model
        # Save each model as a separate pickle as well
        model_path = os.path.join(OUTPUT_DIR, f"model_{dep_var}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[INFO] Saved model for {dep_var} to {model_path}")
    # Save all models in a single dictionary pickle
    all_models_path = os.path.join(OUTPUT_DIR, "all_models.pkl")
    with open(all_models_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] Saved all models to {all_models_path}")


if __name__ == "__main__":
    main()
