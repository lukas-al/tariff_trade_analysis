# src/mpil_tariff_trade_analysis/etl/matching_duckdb.py

import logging
from pathlib import Path

import duckdb

# Assuming your logging setup is accessible via this import path
from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration ---
# Keep the same configuration as the other versions
DEFAULT_BACI_PATH = "data/final/BACI_HS92_V202501"
DEFAULT_WITS_MFN_PATH = "data/final/WITS_AVEMFN.parquet"
DEFAULT_WITS_PREF_PATH = "data/final/WITS_AVEPref.parquet"
DEFAULT_PREF_GROUPS_PATH = "data/raw/WITS_pref_groups/WITS_pref_groups.csv"
DEFAULT_OUTPUT_PATH = "data/final/unified_trade_tariff_duckdb.parquet"  # Changed output name
DEFAULT_DUCKDB_TEMP_DIR = "data/intermediate/duckdb_temp"  # Directory for temp DB file


# --- Main Execution (DuckDB) ---


def run_matching_pipeline_duckdb(
    baci_path: str | Path = DEFAULT_BACI_PATH,
    wits_mfn_path: str | Path = DEFAULT_WITS_MFN_PATH,
    wits_pref_path: str | Path = DEFAULT_WITS_PREF_PATH,
    pref_groups_path: str | Path = DEFAULT_PREF_GROUPS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    duckdb_temp_dir: str | Path = DEFAULT_DUCKDB_TEMP_DIR,
):
    """
    Runs the full data matching pipeline using DuckDB.

    Args:
        baci_path: Path to the BACI data (Parquet).
        wits_mfn_path: Path to the WITS MFN tariff data (Parquet).
        wits_pref_path: Path to the WITS Preferential tariff data (Parquet).
        pref_groups_path: Path to the WITS preferential groups mapping CSV.
        output_path: Path to save the final unified Parquet file.
        duckdb_temp_dir: Directory to store the temporary DuckDB database file.
    """
    logger.info("Starting data matching pipeline (DuckDB)...")

    baci_path = Path(baci_path).resolve()
    wits_mfn_path = Path(wits_mfn_path).resolve()
    wits_pref_path = Path(wits_pref_path).resolve()
    pref_groups_path = Path(pref_groups_path).resolve()
    output_path = Path(output_path).resolve()
    duckdb_temp_dir = Path(duckdb_temp_dir).resolve()

    # Ensure paths exist for input files
    for p in [baci_path, wits_mfn_path, wits_pref_path, pref_groups_path]:
        if not p.exists():
            logger.error(f"Input file not found: {p}")
            raise FileNotFoundError(f"Input file not found: {p}")

    # Ensure directories exist for output and temp db
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_temp_dir.mkdir(parents=True, exist_ok=True)
    db_file = duckdb_temp_dir / "matching_pipeline.duckdb"

    # Delete existing temp DB file if it exists
    if db_file.exists():
        logger.warning(f"Deleting existing temporary DuckDB file: {db_file}")
        try:
            db_file.unlink()
        except OSError as e:
            logger.error(f"Error deleting existing DB file {db_file}: {e}")
            raise

    logger.info(f"Using temporary DuckDB database file: {db_file}")
    con = None  # Initialize connection variable
    try:
        # Connect to DuckDB, creating or opening the file
        con = duckdb.connect(database=str(db_file), read_only=False)
        logger.info("DuckDB connection established.")

        # 1. Load Preferential Group Mapping (CSV -> Aggregate -> View)
        logger.info(f"Loading and processing preferential group mapping: {pref_groups_path}")
        # Use read_csv_auto for simplicity, specify encoding.
        # Aggregate partners into a list for each region code.
        sql_load_groups = f"""
        CREATE OR REPLACE TEMP VIEW pref_group_mapping AS
        SELECT
            "RegionCode" AS region_code,
            list(CAST("Partner" AS VARCHAR)) FILTER (WHERE "Partner" IS NOT NULL) AS partner_list
        FROM read_csv_auto('{pref_groups_path}', ENCODING='utf-8', header=true, ignore_errors=true)
        GROUP BY region_code;
        """
        con.sql(sql_load_groups)
        logger.info("TEMP VIEW 'pref_group_mapping' created.")
        if logger.isEnabledFor(logging.DEBUG):
            count = con.sql("SELECT COUNT(*) FROM pref_group_mapping").fetchone()[0]
            logger.debug(f"Pref group mapping view count: {count}")
            logger.debug(
                f"Pref group mapping schema:\n{con.sql('DESCRIBE pref_group_mapping').df()}"
            )

        # 2. Load main data sources as views
        logger.info(f"Creating TEMP VIEW for BACI data: {baci_path}")
        con.sql(f"""
            CREATE OR REPLACE TEMP VIEW baci_raw AS 
            SELECT * FROM read_parquet('{baci_path}/*/*.parquet', hive_partitioning=true)
        """)
        logger.info("TEMP VIEW 'baci_raw' created.")
        if logger.isEnabledFor(logging.DEBUG):
            count = con.sql("SELECT COUNT(*) FROM baci_raw").fetchone()[0]
            logger.debug(f"BACI raw view count: {count}")

        logger.info(f"Creating TEMP VIEW for WITS MFN data: {wits_mfn_path}")
        con.sql(f"CREATE OR REPLACE TEMP VIEW avemfn_raw AS FROM read_parquet('{wits_mfn_path}');")
        logger.info("TEMP VIEW 'avemfn_raw' created.")

        logger.info(f"Creating TEMP VIEW for WITS Preferential data: {wits_pref_path}")
        con.sql(
            f"CREATE OR REPLACE TEMP VIEW avepref_raw AS FROM read_parquet('{wits_pref_path}');"
        )
        logger.info("TEMP VIEW 'avepref_raw' created.")

        # 3. Rename columns (create new views)
        logger.info("Creating TEMP VIEW 'renamed_avemfn'.")
        sql_rename_mfn = """
        CREATE OR REPLACE TEMP VIEW renamed_avemfn AS
        SELECT
            year AS t,
            reporter_country AS i,
            product_code AS k,
            tariff_rate AS mfn_tariff_rate,
            min_rate AS mfn_min_tariff_rate,
            max_rate AS mfn_max_tariff_rate,
            tariff_type -- Keep for potential debugging
        FROM avemfn_raw;
        """
        con.sql(sql_rename_mfn)
        logger.info("TEMP VIEW 'renamed_avemfn' created.")

        logger.info("Creating TEMP VIEW 'renamed_avepref'.")
        sql_rename_pref = """
        CREATE OR REPLACE TEMP VIEW renamed_avepref AS
        SELECT
            year AS t,
            reporter_country AS i,
            CAST(partner_country AS VARCHAR) AS j, -- Cast partner to VARCHAR early
            product_code AS k,
            tariff_rate AS pref_tariff_rate,
            min_rate AS pref_min_tariff_rate,
            max_rate AS pref_max_tariff_rate
        FROM avepref_raw;
        """
        con.sql(sql_rename_pref)
        logger.info("TEMP VIEW 'renamed_avepref' created.")

        # 4. Expand Preferential Tariffs
        logger.info("Expanding preferential tariffs...")

        # 4a. Join renamed preferential data with group mapping
        logger.debug("Creating TEMP VIEW 'joined_pref_mapping'.")
        sql_join_pref_map = """
        CREATE OR REPLACE TEMP VIEW joined_pref_mapping AS
        SELECT
            pref.*,
            map.partner_list
        FROM renamed_avepref pref
        LEFT JOIN pref_group_mapping map ON pref.j = map.region_code;
        """
        con.sql(sql_join_pref_map)
        logger.debug("TEMP VIEW 'joined_pref_mapping' created.")

        # 4b. Create the final partner list (use mapping list or original 'j')
        logger.debug("Creating TEMP VIEW 'pref_with_final_list'.")
        sql_final_list = """
        CREATE OR REPLACE TEMP VIEW pref_with_final_list AS
        SELECT
            *,
            CASE
                WHEN partner_list IS NOT NULL AND len(partner_list) > 0 THEN partner_list
                ELSE [j] -- Create a single-element list with the original partner code
            END AS final_partner_list
        FROM joined_pref_mapping;
        """
        con.sql(sql_final_list)
        logger.debug("TEMP VIEW 'pref_with_final_list' created.")

        # 4c. Explode (UNNEST) the final partner list
        logger.info("Creating TEMP VIEW 'expanded_pref' using UNNEST.")
        sql_explode = """
        CREATE OR REPLACE TEMP VIEW expanded_pref AS
        SELECT
            t,
            i,
            UNNEST(final_partner_list) AS j_individual, -- Explode the list
            k,
            pref_tariff_rate,
            pref_min_tariff_rate,
            pref_max_tariff_rate
        FROM pref_with_final_list;
        """
        # This is the potentially expensive step
        try:
            con.sql(sql_explode)
            logger.info("TEMP VIEW 'expanded_pref' created successfully.")
            if logger.isEnabledFor(logging.DEBUG):
                count = con.sql("SELECT COUNT(*) FROM expanded_pref").fetchone()[0]
                logger.debug(f"Expanded pref view count: {count}")  # This forces computation
        except Exception as e:
            logger.error(
                f"CRITICAL: Failed to create 'expanded_pref' view (UNNEST step). Error: {e}",
                exc_info=True,
            )
            raise

        # 5. Join Datasets
        logger.info("Joining BACI, MFN, and expanded Preferential datasets...")

        # 5a. Join BACI with MFN
        # Ensure join keys are compatible types (casting if needed)
        logger.debug("Creating TEMP VIEW 'joined_mfn'.")
        sql_join_baci_mfn = """
        CREATE OR REPLACE TEMP VIEW joined_mfn AS
        SELECT
            b.t,
            CAST(b.i AS VARCHAR) AS i, -- Cast BACI keys
            CAST(b.j AS VARCHAR) AS j,
            CAST(b.k AS VARCHAR) AS k,
            b.v, -- Assuming 'v' is value in BACI
            b.q, -- Assuming 'q' is quantity in BACI
            mfn.mfn_tariff_rate,
            mfn.mfn_min_tariff_rate,
            mfn.mfn_max_tariff_rate,
            mfn.tariff_type
        FROM baci_raw b
        LEFT JOIN renamed_avemfn mfn
            ON b.t = mfn.t
            AND CAST(b.i AS VARCHAR) = mfn.i -- Match MFN 'i' type
            AND CAST(b.k AS VARCHAR) = mfn.k; -- Match MFN 'k' type
        """
        con.sql(sql_join_baci_mfn)
        logger.debug("TEMP VIEW 'joined_mfn' created.")

        # 5b. Join result with Expanded Preferential
        logger.debug("Creating TEMP VIEW 'joined_all'.")
        sql_join_all = """
        CREATE OR REPLACE TEMP VIEW joined_all AS
        SELECT
            jm.*,
            ep.pref_tariff_rate,
            ep.pref_min_tariff_rate,
            ep.pref_max_tariff_rate
        FROM joined_mfn jm
        LEFT JOIN expanded_pref ep
            ON jm.t = ep.t
            AND jm.i = ep.i
            AND jm.j = ep.j_individual -- Match expanded pref partner code
            AND jm.k = ep.k;
        """
        con.sql(sql_join_all)
        logger.debug("TEMP VIEW 'joined_all' created.")

        # 6. Calculate Effective Tariff Rate
        logger.info("Calculating effective tariff rate.")
        sql_effective_tariff = """
        CREATE OR REPLACE TEMP VIEW effective_tariff_calc AS
        SELECT
            *,
            COALESCE(pref_tariff_rate, mfn_tariff_rate) AS effective_tariff_rate
        FROM joined_all;
        """
        con.sql(sql_effective_tariff)
        logger.info("TEMP VIEW 'effective_tariff_calc' created.")

        # 7. Create Final Table Structure (View)
        logger.info("Creating final table structure view 'final_unified_table'.")
        # Adjust v, q column names if they differ in your BACI data
        sql_final_select = """
        CREATE OR REPLACE TEMP VIEW final_unified_table AS
        SELECT
            t AS "Year",
            i AS "Source",
            j AS "Target",
            k AS "HS_Code",
            q AS "Quantity", -- Assuming 'q' exists
            v AS "Value",    -- Assuming 'v' exists
            mfn_tariff_rate,
            pref_tariff_rate,
            effective_tariff_rate,
            mfn_min_tariff_rate, -- Optional
            mfn_max_tariff_rate, -- Optional
            pref_min_tariff_rate, -- Optional
            pref_max_tariff_rate, -- Optional
            tariff_type          -- Optional
        FROM effective_tariff_calc;
        """
        con.sql(sql_final_select)
        logger.info("TEMP VIEW 'final_unified_table' created.")
        if logger.isEnabledFor(logging.DEBUG):
            count = con.sql("SELECT COUNT(*) FROM final_unified_table").fetchone()[0]
            logger.debug(f"Final unified table view count: {count}")  # Forces computation
            logger.debug(
                f"Final unified table schema:\n{con.sql('DESCRIBE final_unified_table').df()}"
            )

        # 8. Save Result to Parquet
        logger.info(f"Saving final result to: {output_path}")
        sql_save = f"""
        COPY (SELECT * FROM final_unified_table)
        TO '{output_path}' (FORMAT PARQUET, CODEC 'ZSTD');
        """
        try:
            con.sql(sql_save)
            logger.info(f"Successfully saved final data to {output_path}")
        except Exception as e:
            logger.critical(f"Failed to save final result to Parquet. Error: {e}", exc_info=True)
            raise

        logger.info("Data matching pipeline (DuckDB) finished successfully.")

    except Exception as e:
        logger.critical(f"An error occurred during the DuckDB pipeline: {e}", exc_info=True)
        # Re-raise the exception after logging
        raise
    finally:
        # Ensure the connection is closed even if errors occur
        if con:
            logger.info("Closing DuckDB connection.")
            con.close()
            logger.info("DuckDB connection closed.")
        # Optionally delete the temp file after closing connection
        # if db_file.exists():
        #     logger.info(f"Deleting temporary DuckDB file: {db_file}")
        #     db_file.unlink()


if __name__ == "__main__":
    # Setup logging
    import logging

    from mpil_tariff_trade_analysis.utils.logging_config import setup_logging

    # Set log level via setup_logging
    setup_logging(log_level=logging.DEBUG)  # Use DEBUG for detailed logs
    logger = get_logger(__name__)  # Re-get logger after setup

    logger.info("Running matching_duckdb.py script directly.")
    try:
        run_matching_pipeline_duckdb()
    except Exception:
        # Error should already be logged by the pipeline function
        logger.error("DuckDB pipeline execution failed in __main__.")
        import sys

        sys.exit(1)
