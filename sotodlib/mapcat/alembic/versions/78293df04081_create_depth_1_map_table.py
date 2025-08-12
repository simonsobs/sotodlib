"""create depth-1 map table
Add processing tracking table

Revision ID: 78293df04081
Revises:
Create Date: 2025-08-12 11:39:10.066048

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "78293df04081"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "depth_one_maps",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("map_path", sa.String, nullable=False),
        sa.Column("ot", sa.String, nullable=False),
        sa.Column("arrays", sa.Integer, nullable=False),
        sa.Column("frequency", sa.String, nullable=False),
        sa.Column(
            "ctime", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime?
    )

    op.create_table(
        "time_domain_processing",
        sa.Column(
            "name",
            sa.String,
            sa.ForeignKey("depth_one_maps.name", nullable=False, primary_key=True),
        ),
        sa.Column(
            "processing_start", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime
        sa.Column(
            "processing_end", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime
        sa.Column("processing_status", sa.String, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("depth_one_maps")
    op.drop_table("time_domain_processing")
