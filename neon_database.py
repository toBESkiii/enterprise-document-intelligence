"""Reusable Neon reader connection; run this file to check access safely."""

import os
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row


def read_neon_settings():
    key = "NEON_READONLY_DATABASE_URL"
    if key in os.environ:
        url = os.environ[key].strip()
    else:
        path = Path(__file__).resolve().parent / ".env.app"
        if not path.is_file():
            raise ValueError("Create .env.app beside this script first.")
        values = dotenv_values(path, interpolate=False, encoding="utf-8-sig")
        url = (values.get(key) or "").strip()

    if not url.startswith(("postgresql://", "postgres://")):
        raise ValueError("Set NEON_READONLY_DATABASE_URL to a URL, not a psql command.")
    try:
        settings = conninfo_to_dict(url)
    except psycopg.Error:
        raise ValueError("The reader URL could not be read. Copy it again from Neon.") from None

    host = settings.get("host", "").lower()
    if not host.endswith(".neon.tech") or any(c in host for c in ", /\\\t\r\n"):
        raise ValueError("Expected one Neon hostname, not a local database.")
    if settings.get("hostaddr") or settings.get("service"):
        raise ValueError("Use the connection URL copied from Neon's Connect window.")
    if settings.get("user") != "edi_reader":
        raise ValueError("Select the edi_reader role in Neon, not the owner account.")
    if settings.get("dbname") != "neandb":
        raise ValueError("Select the existing neandb database in Neon.")
    if not settings.get("password"):
        raise ValueError("The reader URL must include its password. Do not share it.")
    if settings.get("sslmode") not in {"require", "verify-ca", "verify-full"}:
        raise ValueError("Keep sslmode=require or stronger in the reader URL.")
    return settings


def connect_neon_database():
    settings = read_neon_settings()
    settings.update(
        connect_timeout=15,
        application_name="edi_neon_reader",
        options="-c statement_timeout=15000",
    )
    connection = psycopg.connect(**settings, row_factory=dict_row)
    connection.read_only = True
    return connection


def check_reader_access():
    print("Checking the Neon application account; no models will be loaded.")
    with connect_neon_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT current_database() AS database_name,
                       current_user AS active_user, session_user AS login_user,
                       current_setting('transaction_read_only') AS read_only
            """)
            identity = cursor.fetchone()
            if identity["active_user"] != "edi_reader" or identity["login_user"] != "edi_reader":
                raise ValueError("The active database account is not edi_reader.")
            if identity["database_name"] != "neandb" or identity["read_only"] != "on":
                raise ValueError("Expected neandb with a read-only transaction.")
            print("Database: neandb | Account: edi_reader | Transaction: read-only")

            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_roles
                    WHERE pg_has_role(current_user, oid, 'MEMBER')
                      AND (rolsuper OR rolcreatedb OR rolcreaterole
                           OR rolreplication OR rolbypassrls
                           OR rolname = 'neon_superuser')
                ) AS elevated
            """)
            if cursor.fetchone()["elevated"]:
                raise ValueError("This reader has elevated role access. Do not deploy it yet.")
            print("Elevated role access: NO")

            for table in ("edi.documents", "edi.chunks"):
                cursor.execute("""
                    SELECT has_table_privilege(current_user, %s, 'SELECT') AS can_read,
                           (has_table_privilege(current_user, %s,
                                'INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
                            OR has_any_column_privilege(current_user, %s,
                                'INSERT, UPDATE, REFERENCES')) AS can_write,
                           pg_has_role(current_user, relowner, 'MEMBER') AS owner_access
                    FROM pg_class WHERE oid = %s::regclass
                """, (table, table, table, table))
                access = cursor.fetchone()
                if not access or not access["can_read"]:
                    raise ValueError(f"The reader cannot read all columns of {table}.")
                if access["can_write"] or access["owner_access"]:
                    raise ValueError(f"The reader has write or ownership access to {table}.")
                print(f"{table}: read YES | write NO | ownership NO")

            cursor.execute("""
                SELECT (SELECT COUNT(*) FROM edi.documents) AS documents,
                       (SELECT COUNT(*) FROM edi.chunks) AS chunks
            """)
            counts = cursor.fetchone()
            print(f"Documents visible: {counts['documents']} | Chunks visible: {counts['chunks']}")
            if (counts["documents"], counts["chunks"]) != (10, 2994):
                raise ValueError("Expected 10 documents and 2994 chunks. Check the Neon branch.")

    print("Read-only account check passed for the two application tables.")
    print("No database changes, model calls or application changes were made.")


def main():
    try:
        check_reader_access()
    except psycopg.Error as error:
        print(f"Database check failed. SQLSTATE: {error.sqlstate or 'unavailable'}")
        print("Check Neon connectivity, the reader URL, and its table permissions.")
        print("Keep the password and connection URL private.")
        raise SystemExit(1) from None
    except ValueError as error:
        print(f"Stopped: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("Could not read .env.app. Check its location and file permissions.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()