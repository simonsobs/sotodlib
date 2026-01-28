from sotodlib.core import Context
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib.mapmaking.utils import downsample_obs
from sotodlib import coords
from pixell import enmap, enplot
from sotodlib.qa.quality_reports.report_data import ReportData, ReportDataConfig
from typing import Literal, Union, Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import datetime as dt
import os
import json
from pathlib import Path
from sotodlib.qa.quality_reports import plots
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
import sys
import yaml
import argparse
import traceback
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta
from importlib import reload
reload(plots)

from sotodlib.site_pipeline.utils.pipeline import main_launcher


def generate_coverage_map(ctx_path: str, obs_id: str):
    """
    Load an obs_id and generate a hits map with pixell.
    """
    ctx = Context(ctx_path)
    res = 10./60 * coords.DEG

    geom = enmap.fullsky_geometry(res=res, proj='car')
    aman = ctx.get_obs(obs_id, no_signal=True)
    aman.restrict("dets", np.isfinite(aman.focal_plane.gamma))
    aman = downsample_obs(aman, 100, skip_signal=True)
    aman.restrict("dets", aman.dets.vals[::10])
    p = coords.P.for_tod(aman, geom=geom, comps='T')
    w = p.to_weights(aman)

    return w


def parse_range(folder_name):
    try:
        start, end = folder_name.split("_")
        return dt.datetime.strptime(start, "%Y%m%d"), dt.datetime.strptime(end, "%Y%m%d")
    except Exception:
        return None, None


def create_manifest(base_dir: str, output_file: str):
    manifest = []

    for cadence in ["weekly", "monthly"]:
        parent = os.path.join(base_dir, cadence)
        if not os.path.exists(parent):
            continue

        entries = []
        candidates = []

        for folder_name in sorted(os.listdir(parent)):
            folder_path = os.path.join(parent, folder_name)
            if not os.path.isdir(folder_path):
                continue

            index_path = os.path.join(folder_path, "report.html")
            if not os.path.exists(index_path):
                continue

            rel_path = os.path.relpath(index_path, start=base_dir).replace(os.sep, "/")
            rel_path = f"../../{rel_path}"

            entries.append({
                "label": f"{cadence} / {folder_name}",
                "rel_path": rel_path
            })

            start, end = parse_range(folder_name)
            if end is not None:
                candidates.append((end, start, rel_path))

        if candidates:
            _, _, latest_path = max(candidates)
            entries.insert(0, {
                "label": f"{cadence} / latest",
                "rel_path": latest_path
            })

        manifest.extend(entries)

    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)


class GenerateReportConfig:
    def __init__(
        self,
        platform: Literal["satp1", "satp2", "satp3", "lat"],
        site_url: str,
        report_interval: Literal["weekly", "monthly"],
        start_time: Union[dt.datetime, float, str],
        output_root: str,
        data_config: Dict[str, Any],
        stop_time: Union[dt.datetime, float, str, None] = None,
        overwrite_html: bool = False,
        overwrite_data: bool = False,
        skip_html: bool = False,
        template_dir: Optional[str] = None,
    ) -> None:
        self.platform: Literal["satp1", "satp2", "satp3", "lat"] = platform
        self.site_url: str = site_url
        self.report_interval = report_interval

        def convert_to_datetime(
            time: Union[dt.datetime, float, str, None],
        ) -> dt.datetime:
            if isinstance(time, type(None)):
                return dt.datetime.now(tz=dt.timezone.utc)
            if isinstance(time, str):
                return dt.datetime.fromisoformat(time)
            elif isinstance(time, (int, float)):
                return dt.datetime.fromtimestamp(time)
            elif isinstance(time, dt.datetime):
                return time
            else:
                raise Exception(f"Could not convert type {type(time)} to datetime")

        self.start_time = convert_to_datetime(start_time)
        self.stop_time = convert_to_datetime(stop_time)
        self.output_root = output_root
        self.overwrite_html = overwrite_html
        self.overwrite_data = overwrite_data
        self.template_dir = template_dir
        self.data_config = data_config
        self.skip_html = skip_html

        self.time_intervals: List[Tuple[dt.datetime, dt.datetime]] = []
        if self.report_interval == "weekly":
            delta = dt.timedelta(weeks=1)
        elif self.report_interval == "monthly":
            delta = relativedelta(months=1)
        start: dt.datetime = self.start_time
        self.time_intervals = []
        now = dt.datetime.now(tz=dt.timezone.utc)
        while start < self.stop_time:
            stop: dt.datetime = start + delta
            if stop > now - dt.timedelta(hours=1):
                # Give a buffer of 1 hour to compile report for previous interval
                break
            self.time_intervals.append((start, stop))
            start += delta

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateReportConfig":
        return GenerateReportConfig(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "GenerateReportConfig":
        with open(path, "r") as f:
            return GenerateReportConfig(**yaml.safe_load(f))


def _main(
    cfg: str,
    executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
    as_completed_callable: Callable,
    start_time: Optional[Union[dt.datetime, float, str]] = None,
    stop_time: Optional[Union[dt.datetime, float, str]] = None,
    report_interval: Optional[str] = None,
    overwrite_html: Optional[bool] = None,
    overwrite_data: Optional[bool] = None,
    load_source_footprints: Optional[bool] = None,
    make_cov_map: Optional[bool] = None,
) -> None:

    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if start_time is not None:
        cfg["start_time"] = start_time

    if stop_time is not None:
        cfg["stop_time"] = stop_time

    if report_interval is not None and report_interval in ["weekly", "monthly"]:
        cfg["report_interval"] = report_interval

    if overwrite_html is not None:
        cfg["overwrite_html"] = overwrite_html

    if overwrite_data is not None:
        cfg["overwrite_data"] = overwrite_data

    if load_source_footprints is not None:
        cfg["data_config"]["load_source_footprints"] = load_source_footprints

    if make_cov_map is not None:
        cfg["data_config"]["make_cov_map"] = make_cov_map

    cfg = GenerateReportConfig.from_dict(cfg)

    base_path = os.path.join(cfg.output_root, cfg.report_interval)

    n_failed = 0

    for start_time, stop_time in tqdm(cfg.time_intervals, total=len(cfg.time_intervals)):
        time_str = f"{start_time:%Y%m%d}_{stop_time:%Y%m%d}"
        subdir = os.path.join(base_path, time_str)
        report_file = os.path.join(subdir, "report.html")
        data_file = os.path.join(subdir, "data.h5")
        map_fits_file = os.path.join(subdir, "cov.fits")
        map_png_file = os.path.join(subdir, "cov.png")

        data_cfg = ReportDataConfig(
            start_time=start_time,
            stop_time=stop_time,
            platform=cfg.platform,
            site_url=cfg.site_url,
            **cfg.data_config,
        )

        longterm_path = None
        longterm_data = None

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        try:
            if not os.path.exists(data_file) or cfg.overwrite_data:
                data: ReportData = ReportData.build(data_cfg)
                data.save(data_file)
            else:
                data = ReportData.load(data_file)

            if data_cfg.make_cov_map:
                if not os.path.exists(map_fits_file) or cfg.overwrite_data:
                    cmb_obs_list = [o for o in data.obs_list if o.obs_subtype == "cmb"]

                    futures = []
                    for o in cmb_obs_list:
                        futures.append(executor.submit(
                                generate_coverage_map,
                                data.cfg.ctx_path,
                                o.obs_id))

                    total = len(futures)
                    for future in tqdm(as_completed_callable(futures), total=total, desc="generate_cov_map"):
                        if data.w is None:
                            data.w = future.result()
                        else:
                            data.w += future.result()
                        futures.remove(future)

                    enmap.write_map(map_fits_file, data.w)
                    f = enplot.plot(data.w, grid=True, downgrade=1, mask=0, ticks=10)
                    enplot.write(map_png_file, f[0])
                else:
                    data.w = enmap.read_map(map_fits_file)
                    if not os.path.exists(map_png_file):
                        f = enplot.plot(data.w, grid=True, downgrade=1, mask=0, ticks=10)
                        enplot.write(map_png_file, f[0])

            if not os.path.exists(report_file) or cfg.overwrite_html and not cfg.skip_html:
                if data.cfg.longterm_obs_file != longterm_path:
                    longterm_data = ReportData.load(data.cfg.longterm_obs_file)
                    longterm_path = data.cfg.longterm_obs_file
                render_report(
                    cfg,
                    data,
                    report_file,
                    template_dir=cfg.template_dir,
                    longterm_data=longterm_data,
                    map_png_file=map_png_file,
                )
                # update longterm obs file
                if longterm_path is not None and cfg.report_interval == "monthly":
                    data.save(longterm_path, map_path=None, overwrite=False, update_footprints=False)
        except Exception as e:
            tb = ''.join(traceback.format_tb(e.__traceback__))
            print(f"Failed to generate report for {time_str}: {tb} {e}")
            n_failed += 1

    create_manifest(cfg.output_root, os.path.join(cfg.output_root, "manifest.json"))

    if n_failed > 0:
        raise RuntimeError(f"{n_failed} reports failed to generate")


def render_report(
    cfg: GenerateReportConfig,
    data: ReportData,
    output_path: str,
    template_dir=None,
    longterm_data: Optional[ReportData] = None,
    map_png_file: Optional[str] = None,
):
    if template_dir is None:
        template_dir = os.path.join(os.path.dirname(__file__), "templates")

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html")

    obs_efficiency_plots = plots.wafer_obs_efficiency(data)
    source_footprint_plots = plots.source_footprints(data)
    figures: Dict[str, go.Figure] = {
        "obs_efficiency_heatmap": obs_efficiency_plots.heatmap,
        "obs_efficiency_pie": obs_efficiency_plots.pie,
        "boresight_vs_time": plots.boresight_vs_time(data),
        "hwp_freq_vs_time": plots.hwp_freq_vs_time(data),
        "yield_vs_pwv": plots.yield_vs_pwv(data, longterm_data=longterm_data),
        "pwv_yield_vs_time": plots.pwv_and_yield_vs_time(data),
        "pwv_and_array_nep_vs_time": plots.pwv_and_nep_vs_time(data, field_name="array"),
        "array_nep_vs_pwv": plots.nep_vs_pwv(data, longterm_data=longterm_data, field_name="array"),
        "pwv_and_det_nep_vs_time": plots.pwv_and_nep_vs_time(data, field_name="det"),
        "det_nep_vs_pwv": plots.nep_vs_pwv(data, longterm_data=longterm_data, field_name="det"),
        "source_focalplane": source_footprint_plots.focalplane,
        "source_table": source_footprint_plots.table,
        "map_png": plots.cov_map_plot(map_png_file),
    }

    html_kw = dict(full_html=False, include_plotlyjs=False)
    if isinstance(data.cfg.start_time, str):
        start_time_str = data.cfg.start_time
    else:
        start_time_str = data.cfg.start_time.isoformat()
    if isinstance(data.cfg.stop_time, str):
        stop_time_str = data.cfg.stop_time
    else:
        stop_time_str = data.cfg.stop_time.isoformat()
    jinja_data = {
        "data": data,
        "report_interval": cfg.report_interval.capitalize(),
        "plots": {
            k: v if isinstance(v, str) else v.to_html(**html_kw)
            for k, v in figures.items()
        },
        "general_stats": {
            "Platform": cfg.platform,
            "Start time": dt.datetime.fromisoformat(start_time_str).strftime("%A %m/%d/%Y  %H:%M (UTC)"),
            "Stop time": dt.datetime.fromisoformat(stop_time_str).strftime("%A %m/%d/%Y  %H:%M (UTC)"),
            "Number of Observations": len([o for o in data.obs_list if o.obs_type == "obs"]),
            "Number of CMB Observations": len([o for o in data.obs_list if o.obs_subtype == "cmb"]),
            "Number of Cal Observations": len([o for o in data.obs_list if o.obs_subtype == "cal" and o.obs_type == "obs"]),
            "Time Spent on CMB Observations (hrs)": np.round(np.sum(np.array([o.duration for o in data.obs_list if o.obs_subtype == "cmb"])) / 3600, 1),
            "Time Spent on Cal Observations (hrs)": np.round(np.sum(np.array([o.duration for o in data.obs_list if o.obs_subtype == "cal" and o.obs_type == "obs"])) / 3600, 1),
            "Average Duration of CMB Observations (hrs)": np.round(np.nanmean(np.array([o.duration for o in data.obs_list if o.obs_subtype == "cmb"])) / 3600, 2),
            "Average Duration of Cal Observations (hrs)": np.round(np.nanmean(np.array([o.duration for o in data.obs_list if o.obs_subtype == "cal" and o.obs_type == "obs"])) / 3600, 2),
            "Average PWV (mm)": np.round(np.nanmean([o.pwv for o in data.obs_list]), 3),
        }
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(template.render(jinja_data))


def get_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        p = argparse.ArgumentParser()
    else:
        p = parser
    p.add_argument(
        "cfg", type=str, help="yaml configuration file."
    )
    return p


def main(
    cfg: str,
    start_time: Optional[Union[dt.datetime, float, str]] = None,
    stop_time: Optional[Union[dt.datetime, float, str]] = None,
    report_interval: Optional[str] = None,
    overwrite_html: Optional[bool] = None,
    overwrite_data: Optional[bool] = None,
    load_source_footprints: Optional[bool] = None,
    make_cov_map: Optional[bool] = None,
    nproc: int = 1
) -> None:

    rank, executor, as_completed_callable = get_exec_env(nproc)

    if rank == 0:
        _main(
            cfg=cfg,
            executor=executor,
            as_completed_callable=as_completed_callable,
            start_time=start_time,
            stop_time=stop_time,
            report_interval=report_interval,
            overwrite_html=overwrite_html,
            overwrite_data=overwrite_data,
            load_source_footprints=load_source_footprints,
            make_cov_map=make_cov_map,
        )

if __name__ == '__main__':
    main_launcher(main, get_parser)
