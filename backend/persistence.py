"""MySQL storage for completed loan estimates."""
import os

import pymysql
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "loan_requests"


def _connect(database=True):
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    if not host or not user:
        raise RuntimeError("MYSQL_HOST and MYSQL_USER must be configured")
    options = {
        "host": host,
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": user,
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "charset": "utf8mb4",
        "connect_timeout": 5,
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": True,
    }
    if database:
        options["database"] = os.getenv("MYSQL_DATABASE", "Loan_Information")
    return pymysql.connect(**options)


def initialize_database():
    """Create the configured database and table if they do not exist."""
    database_name = os.getenv("MYSQL_DATABASE", "Loan_Information")
    if not database_name.replace("_", "").isalnum():
        raise RuntimeError("MYSQL_DATABASE contains unsupported characters")
    with _connect(database=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4")
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                    loan_request_id CHAR(8) PRIMARY KEY,
                    name VARCHAR(120) NOT NULL,
                    age TINYINT UNSIGNED NOT NULL,
                    annual_income DOUBLE NOT NULL,
                    credit_score SMALLINT UNSIGNED NOT NULL,
                    employment_status VARCHAR(32) NOT NULL,
                    loan_amount DOUBLE NOT NULL,
                    collateral DOUBLE NOT NULL,
                    monthly_debt DOUBLE NOT NULL,
                    probability DOUBLE NOT NULL,
                    decision VARCHAR(16) NOT NULL,
                    created_at DATETIME NOT NULL
                ) CHARACTER SET utf8mb4"""
            )


def save_loan_request(record):
    initialize_database()
    columns = (
        "loan_request_id", "name", "age", "annual_income", "credit_score",
        "employment_status", "loan_amount", "collateral", "monthly_debt",
        "probability", "decision", "created_at",
    )
    placeholders = ", ".join(["%s"] * len(columns))
    query = f"INSERT INTO `{TABLE_NAME}` ({', '.join(columns)}) VALUES ({placeholders})"
    values = [record[column] for column in columns]
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, values)


def find_loan_request(loan_request_id):
    initialize_database()
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM `{TABLE_NAME}` WHERE loan_request_id = %s",
                (loan_request_id,),
            )
            record = cursor.fetchone()
    if record and record.get("created_at"):
        record["created_at"] = record["created_at"].isoformat()
    return record
