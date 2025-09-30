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
        sa.Column("map_id", sa.Integer, primary_key=True),
        sa.Column("map_name", sa.String, unique=True, nullable=False),
        sa.Column("map_path", sa.String, nullable=False),
        sa.Column(
            "tube_slot", sa.String, nullable=False
        ),  # TODO: Maybe make this a literal?
        sa.Column("frequency", sa.String, nullable=False),
        sa.Column("ctime", sa.Float, nullable=False),
        sa.Column("start_time", sa.Float, nullable=False),
        sa.Column("stop_time", sa.Float, nullable=False),
    )

    op.create_table(
        "time_domain_processing",
        sa.Column("processing_status_id", sa.Integer, primary_key=True),
        sa.Column(
            "map_id",
            sa.Integer,
            sa.ForeignKey(
                "depth_one_maps.map_id", ondelete="CASCADE", onupdate="CASCADE"
            ),
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
        "depth_one_pointing_residuals",
        sa.Column("pointing_residual_id", sa.Integer, primary_key=True),
        sa.Column(
            "map_id",
            sa.Integer,
            sa.ForeignKey(
                "depth_one_maps.map_id", ondelete="CASCADE", onupdate="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("ra_offset", sa.Float),
        sa.Column("dec_offset", sa.Float),
    )

    op.create_table(
        "tod_depth_one",
        sa.Column("tod_id", sa.Integer, primary_key=True),
        sa.Column("obs_id", sa.String, nullable=False),
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
        sa.Column("scan_type", sa.String),
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
        "link_tod_to_depth_one_map",
        sa.Column(
            "tod_id", sa.Integer, sa.ForeignKey("tod_depth_one.tod_id"), primary_key=True
        ),
        sa.Column(
            "map_id",
            sa.Integer,
            sa.ForeignKey("depth_one_maps.map_id"),
            primary_key=True,
        ),
    )

    op.create_table(
        "pipeline_information",
        sa.Column("pipeline_information_id", sa.Integer, primary_key=True),
        sa.Column(
            "map_id",
            sa.Integer,
            sa.ForeignKey(
                "depth_one_maps.map_id", ondelete="CASCADE", onupdate="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("sotodlib_version", sa.String, nullable=False),
        sa.Column("map_maker", sa.String, nullable=False),
        sa.Column("preprocess_info", sa.JSON),
    )

    op.create_table(
        "depth_one_sky_coverage",
        sa.Column("patch_id", sa.Integer, primary_key=True),
        sa.Column("x", sa.CHAR, nullable=False),
        sa.Column("y", sa.CHAR, nullable=False),
        sa.Column(
            "map_id",
            sa.Integer,
            sa.ForeignKey("depth_one_maps.map_id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    op.create_table(
        "depth_one_coadds",
        sa.Column("coadd_id", sa.Integer, primary_key=True),
        sa.Column("coadd_name", sa.String, nullable=False),
        sa.Column("coadd_type", sa.String, nullable=False),
        sa.Column("coadd_path", sa.String, nullable=False),
        sa.Column("frequency", sa.String, nullable=False),
        sa.Column("ctime", sa.Float, nullable=False),
        sa.Column("start_time", sa.Float, nullable=False),
        sa.Column("stop_time", sa.Float, nullable=False),
    )

    op.create_table(
        "link_depth_one_map_to_coadd",
        sa.Column(
            "map_id",
            sa.Integer,
            sa.ForeignKey("depth_one_maps.map_id"),
            primary_key=True,
        ),
        sa.Column(
            "coadd_id",
            sa.Integer,
            sa.ForeignKey("depth_one_coadds.coadd_id"),
            primary_key=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("depth_one_maps")
    op.drop_table("time_domain_processing")
    op.drop_table("depth_1_pointing_offsets")
    op.drop_table("tod_depth_one")
    op.drop_table("link_tod_to_depth_one_map")
    op.drop_table("pipelin_information")
    op.drop_table("depth_one_sky_coverage")
