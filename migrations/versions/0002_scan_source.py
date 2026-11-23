import sqlalchemy as sa
from alembic import op

revision = "0002_scan_source"
down_revision = "0001_create_scans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scans",
        sa.Column("source", sa.String(length=16), nullable=False, server_default="rest"),
    )


def downgrade() -> None:
    op.drop_column("scans", "source")
