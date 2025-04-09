import os
import glob

from tqdm.auto import tqdm
import duckdb


def baci_to_parquet(hs, release, input_folder="raw", output_folder="final"):
    # Determine input path

    baci_folder = f"BACI_{hs}_V{release}"

    if input_folder is not None:
        input_path = os.path.join(input_folder, baci_folder)
    else:
        input_path = baci_folder

    # Determine output file

    if output_folder is not None:
        output_file = os.path.join(output_folder, f"{baci_folder}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    else:
        output_file = f"{baci_folder}"

    # Compile all BACI tables into one table

    duckdb.sql(f"COPY( SELECT * FROM read_csv_auto('{input_path}/BACI*.csv') ) TO '{output_file}'")

    # Report result
    if output_folder is not None:
        print(f"'{baci_folder}' successfully saved in '{output_folder}'.")
    else:
        print(f"'{baci_folder}' successfully saved in project root.")


def baci_to_parquet_incremental(hs, release, input_folder="raw", output_folder="final"):
    baci_folder = f"BACI_{hs}_V{release}"
    input_path = os.path.join(input_folder, baci_folder)
    output_file = os.path.join(output_folder, f"{baci_folder}")

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Remove existing Parquet file if present.
    if os.path.exists(output_file):
        os.remove(output_file)

    print("Running incremental conversion...")

    # Get all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(input_path, "BACI*.csv"))

    for i, csv_file in tqdm(enumerate(csv_files), desc="Processing CSV files"):
        sql_query = f"""
            COPY (
                SELECT *, '{i}' AS partition_col
                FROM read_csv_auto('{csv_file}')
            )
            TO '{output_file}'
            (FORMAT 'parquet', COMPRESSION 'SNAPPY', PARTITION_BY (partition_col), APPEND);
        """
        duckdb.sql(sql_query)

    print(f"'{baci_folder}.parquet' successfully saved in '{output_folder}'.")


def aggregate_baci(input, output, aggregation="country"):
    if aggregation == "2digit":
        duckdb.sql(
            f"""
            COPY (
                SELECT t, i, j, k2, SUM(v) AS v, SUM(q) AS q
                FROM SELECT t, i, j, SUBSTRING(k, -6, 2) AS k2, v, q
                FROM read_parquet('{input}/*/*.parquet', hive_partitioning=true))
                GROUP BY t, i, j, k2
                ORDER BY t
            ) TO '{output}'
            """
        )

    elif aggregation == "4digit":
        duckdb.sql(
            f"""
            COPY (
                SELECT t, i, j, k4, SUM(v) AS v, SUM(q) AS q
                FROM SELECT t, i, j, substring(k, -6, 4) AS k4, v, q 
                FROM read_parquet('{input}/*/*.parquet', hive_partitioning=true)
                GROUP BY t, i, j, k4
                ORDER BY t
            ) TO '{output}'
            """
        )

    else:
        duckdb.sql(
            f"""
            COPY (
                SELECT t, i, j, SUM(v) AS v, SUM(q) AS q
                FROM read_parquet('{input}/*/*.parquet', hive_partitioning=true)
                GROUP BY t, i, j
                ORDER BY t
            ) TO '{output}'
            """
        )
