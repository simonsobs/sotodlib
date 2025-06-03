from sotodlib.qa.quality_reports.report_data import ReportData, ReportDataConfig
from typing import Literal, Union, Dict, Any, Optional, Tuple, List
import datetime as dt
import os
from sotodlib.qa.quality_reports import plots
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
import sys
import yaml
import traceback
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta
from importlib import reload
reload(plots)


class GenerateReportConfig:
    def __init__(
        self,
        platform: Literal["satp1", "satp2", "satp3", "lat"],
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
            if stop > now - dt.timedelta(days=1):
                # Give a buffer of 1 day to compile report for previous interval
                break
            self.time_intervals.append((start, stop))
            start += delta

    @classmethod
    def from_yaml(cls, path: str) -> "GenerateReportConfig":
        with open(path, "r") as f:
            return GenerateReportConfig(**yaml.safe_load(f))


def main(cfg: GenerateReportConfig) -> None:
    base_path = os.path.join(cfg.output_root, cfg.platform, cfg.report_interval)

    for start_time, stop_time in tqdm(cfg.time_intervals):
        time_str = f"{start_time:%Y%m%d}_{stop_time:%Y%m%d}"
        subdir = os.path.join(base_path, time_str)
        report_file = os.path.join(subdir, "report.html")
        data_file = os.path.join(subdir, "data.h5")

        data_cfg = ReportDataConfig(
            start_time=start_time,
            stop_time=stop_time,
            platform=cfg.platform,
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

            if not os.path.exists(report_file) or cfg.overwrite_html and not cfg.skip_html:
                if data.cfg.longterm_obs_file != longterm_path:
                    longterm_data = ReportData.load(data.cfg.longterm_obs_file)
                    longterm_path = data.cfg.longterm_obs_file
                render_report(
                    data,
                    report_file,
                    template_dir=cfg.template_dir,
                    longterm_data=longterm_data
                )
        except Exception as e:
            tb = ''.join(traceback.format_tb(e.__traceback__))
            print(f"Failed to generate report for {time_str}: {tb} {e}")


def render_report(
    data: ReportData,
    output_path: str,
    template_dir=None,
    longterm_data: Optional[ReportData] = None,
):
    if template_dir is None:
        template_dir = os.path.join(os.path.dirname(__file__), "templates")

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html")

    obs_efficiency_plots = plots.wafer_obs_efficiency(data)
    figures: Dict[str, go.Figure] = {
        "obs_efficiency_heatmap": obs_efficiency_plots.heatmap,
        "obs_efficiency_pie": obs_efficiency_plots.pie,
        "yield_vs_pwv": plots.yield_vs_pwv(data, longterm_data=longterm_data),
        "pwv_yield_vs_time": plots.pwv_and_yield_vs_time(data),
        "cal_footprints": plots.cal_footprints(data),
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
        "plots": {k: v.to_html(**html_kw) for k, v in figures.items()},
        "general_stats": {
            "Start time": dt.datetime.fromisoformat(start_time_str).strftime("%A %m/%d/%Y  %H:%M (UTC)"),
            "Stop time": dt.datetime.fromisoformat(stop_time_str).strftime("%A %m/%d/%Y  %H:%M (UTC)"),
            "Number of Observations": len([o for o in data.obs_list if o.obs_type == "obs"]),
        }
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(template.render(jinja_data))


if __name__ == '__main__':
    cfg_file = sys.argv[1]
    cfg = GenerateReportConfig.from_yaml(cfg_file)
    main(cfg)
