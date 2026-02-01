#!/usr/bin/env python
# coding: utf-8

import click
import pandas as pd
from sqlalchemy import create_engine
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--pg-user", default="root")
@click.option("--pg-pass", default="root")
@click.option("--pg-host", default="localhost")
@click.option("--pg-port", default=5432, type=int)
@click.option("--pg-db", default="ny_taxi")
@click.option("--dataset", type=click.Choice(["yellow", "green", "zones"]), required=True)
@click.option("--year", default=2021, type=int)
@click.option("--month", default=1, type=int)
@click.option("--chunksize", default=100_000, type=int)
def run(pg_user, pg_pass, pg_host, pg_port, pg_db, dataset, year, month, chunksize):
    """
    Ingest NYC taxi or zones data into Postgres.
    """

    # 1️⃣ Validate inputs early (fail fast)
    validate_inputs(dataset, year, month)

    # 2️⃣ Create Postgres engine
    engine = create_engine(
        f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    )

    # 3️⃣ Get dataframe iterator and target table name
    df_iter, table_name = get_dataframe_iterator(
        dataset=dataset,
        year=year,
        month=month,
        chunksize=chunksize
    )

    # 4️⃣ Write to Postgres
    write_to_postgres(df_iter, table_name, engine, dataset)


# -----------------------------
# Validation
# -----------------------------
def validate_inputs(dataset, year, month):
    if dataset in {"yellow", "green"}:
        if year < 2009:
            raise click.BadParameter("year must be >= 2009")
        if not 1 <= month <= 12:
            raise click.BadParameter("month must be between 1 and 12")


# -----------------------------
# Dispatcher
# -----------------------------
def get_dataframe_iterator(dataset, year, month, chunksize):
    if dataset == "yellow":
        return ingest_yellow(year, month, chunksize), "yellow_taxi_data"
    if dataset == "green":
        return ingest_green(year, month), "green_taxi_data"
    if dataset == "zones":
        return ingest_zones(), "zones"


# -----------------------------
# Ingestion functions
# -----------------------------
def ingest_yellow(year, month, chunksize):
    """
    Reads yellow taxi CSV from GitHub in chunks.
    """
    url = (
        "https://github.com/DataTalksClub/nyc-tlc-data/"
        f"releases/download/yellow/yellow_tripdata_{year}-{month:02d}.csv.gz"
    )

    dtype = {
        "VendorID": "Int64",
        "passenger_count": "Int64",
        "trip_distance": "float64",
        "RatecodeID": "Int64",
        "store_and_fwd_flag": "string",
        "PULocationID": "Int64",
        "DOLocationID": "Int64",
        "payment_type": "Int64",
        "fare_amount": "float64",
        "extra": "float64",
        "mta_tax": "float64",
        "tip_amount": "float64",
        "tolls_amount": "float64",
        "improvement_surcharge": "float64",
        "total_amount": "float64",
        "congestion_surcharge": "float64",
    }

    parse_dates = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime"
    ]

    return pd.read_csv(
        url,
        dtype=dtype,
        parse_dates=parse_dates,
        iterator=True,
        chunksize=chunksize
    )


def ingest_green(year, month):
    """
    Reads green taxi Parquet from cloudfront (small enough for memory).
    """
    url = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"green_tripdata_{year}-{month:02d}.parquet"
    )
    df = pd.read_parquet(url)
    return [df]  # wrap in list to behave like iterator


def ingest_zones():
    """
    Reads taxi zone lookup CSV from GitHub.
    """
    url = (
        "https://github.com/DataTalksClub/nyc-tlc-data/"
        "releases/download/misc/taxi_zone_lookup.csv"
    )
    df = pd.read_csv(url)
    return [df]  # wrap in list to behave like iterator


# -----------------------------
# Postgres writing
# -----------------------------
def write_to_postgres(df_iter, table_name, engine, dataset):
    """
    Writes dataframe iterator to Postgres with atomic transactions and logging.
    """
    logging.info(f"Loading dataset={dataset} into table={table_name}")

    with engine.begin() as conn:
        first = True
        for df in tqdm(df_iter, desc=f"Ingesting {table_name}"):
            
            if first:
                df.columns = [c.lower() for c in df.columns]
                # create table schema (no data)
                df.head(0).to_sql(
                    name=table_name,
                    con=conn,
                    if_exists="replace",
                    index=False
                )
                first = False

            logging.info(f"Writing chunk with {len(df)} rows")
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists="append",
                index=False,
                method="multi"
            )


if __name__ == '__main__':
    run()
