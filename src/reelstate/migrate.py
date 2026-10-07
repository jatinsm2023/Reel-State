"""Apply db/migrations/*.sql in filename order, once each."""
import psycopg

from .config import DATABASE_URL, MIGRATIONS_DIR


def main() -> None:
    # a plain connection: on a brand-new database the vector type does not exist until migration 001 runs
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            print(f"applying {path.name}")
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
        conn.commit()
    print("migrations up to date")


if __name__ == "__main__":
    main()
