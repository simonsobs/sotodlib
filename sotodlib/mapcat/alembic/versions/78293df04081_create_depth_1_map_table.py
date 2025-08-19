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
        sa.Column("map_name", sa.String, unique=True, nullable=False),
        sa.Column("map_path", sa.String, nullable=False),
        sa.Column(
            "tube_slot", sa.String, nullable=False
        ),  # TODO: Maybe make this a literal?
        sa.Column(
            "wafers", sa.String, nullable=False
        ),  # TODO: I'm not sure this field makes sense as a map may use TODs with different wafer availability. I.E., wafers 0,1,2 may be available for one TOD but only 0,2 for another
        sa.Column("frequency", sa.String, nullable=False),
        sa.Column(
            "ctime", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime?
    )

    op.create_table(
        "time_domain_processing",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "map_name",
            sa.String,
            sa.ForeignKey("depth_one_maps.name"),
            nullable=False,
        ),
        sa.Column(
            "processing_start", sa.Float, nullable=True
        ),  # TODO: should this be sa.Datetime
        sa.Column(
            "processing_end", sa.Float, nullable=True
        ),  # TODO: should this be sa.Datetime
        sa.Column("processing_status", sa.String, nullable=False),
    )

    op.create_table(
        "depth_1_pointing_residuals",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "map_name",
            sa.String,
            sa.ForeignKey("depth_one_maps.map_name"),
            nullable=False,
        ),
        sa.Column("ra_offset", sa.Float),
        sa.Column("dec_offset", sa.Float),
    )

    op.create_table(
        "tod_depth_one",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "map_name",
            sa.String,
            sa.ForeignKey("depth_one_maps.map_name"),
        ),
        sa.Column("pwv", sa.Float),
        sa.Column(
            "ctime", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime?
        sa.Column(
            "start_time", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime?
        sa.Column(
            "stop_time", sa.Float, nullable=False
        ),  # TODO: should this be sa.Datetime?
        sa.Column("nsamples", sa.Integer),
        sa.Column("telescope", sa.String),
        sa.Column("telescope_flavor", sa.String),
        sa.Column("tube_slot", sa.String),
        sa.Column("tube_flavor", sa.String),
        sa.Column("frequency", sa.String),
        sa.Column("type", sa.String),
        sa.Column("subtype", sa.String),
        sa.Column("wafer_count", sa.Integer),
        sa.Column("duration", sa.Float),
        sa.Column("az_center", sa.Float),
        sa.Column("az_throw", sa.Float),
        sa.Column("el_center", sa.Float),
        sa.Column("el_throw", sa.Float),
        sa.Column("roll_center", sa.Float),
        sa.Column("roll_throw", sa.Float),
        sa.Column("wafer_slots_list", sa.String),
        sa.Column("stream_ids_list", sa.String),
    )

    op.create_table(
        "pipeline_information",
        sa.Column(
            "map_name",
            sa.String,
            sa.ForeignKey("depth_one_maps.map_name"),
            primary_key=True,
        ),
        sa.Column("sotodlib_version", sa.String, nullable=False),
        sa.Column("map_maker", sa.String, nullable=False),
        sa.Column("preprocess_info", sa.JSON),
    )

    op.create_table(
        "depth_one_sky_coverage",
        sa.Column(
            "map_name",
            sa.String,
            sa.ForeignKey("depth_one_maps.map_name"),
            primary_key=True,
        ),
        sa.Column("patch_coverage", sa.String, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("depth_one_maps")
    op.drop_table("time_domain_processing")
    op.drop_table("depth_1_pointing_offsets")
    op.drop_table("tod_depth_one")
    op.drop_table("pipelin_information")
    op.drop_table("depth_one_sky_coverage")
