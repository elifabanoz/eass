import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


class MigrationTests(unittest.TestCase):
    def test_initial_migration_creates_scans_table(self):
        with tempfile.TemporaryDirectory() as temp:
            url = "sqlite:///" + (Path(temp) / "eass.db").as_posix()
            config = Config("alembic.ini")
            with patch.dict(os.environ, {"DATABASE_URL": url}):
                command.upgrade(config, "head")
            engine = create_engine(url)
            self.assertIn("scans", inspect(engine).get_table_names())
            self.assertIn(
                "source", {column["name"] for column in inspect(engine).get_columns("scans")}
            )
            self.assertIn("alembic_version", inspect(engine).get_table_names())
            self.assertIn("scan_reports", inspect(engine).get_table_names())
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
