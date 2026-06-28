import plotly
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors
import os
import pandas as pd
from dataclasses import dataclass, field
import datetime as dt
from copy import deepcopy
import ast
from collections import defaultdict

from typing import List, Tuple, TYPE_CHECKING, Dict, Any, Optional

from .report_data import ReportData, Footprint


BASE_COLORS = [
    '#4477AA', '#EE6677', '#228833',
    '#CCBB44', '#66CCEE', '#AA3377',
    '#BBBBBB'
]


MARKER_COLORS = (
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00",
)


MARKER_SYMBOLS = (
    "circle", "square", "diamond",
    "cross", "x", "triangle-up",
)


LAT_TUBE_SLOTS = (
    "c1", "i1", "i2", "i3", "i4", "i5", "i6",
    "o1", "o2", "o3", "o4", "o5", "o6",
)


SAT_WAFER_SLOTS = tuple(f"ws{i}" for i in range(7))


def get_wafers(platform: str) -> list[str]:
    """Return wafer identifiers for a given platform."""

    platform = platform.lower().strip()

    if platform == "lat":
        return [
            f"{tube}_{wafer}"
            for tube in LAT_TUBE_SLOTS
            for wafer in ("ws0", "ws1", "ws2")
        ]

    if platform in {"satp1", "satp2", "satp3"}:
        return list(SAT_WAFER_SLOTS)

    raise ValueError(f"Unknown platform: {platform!r}")


def get_discrete_distinct_colors(n, reverse=False):
    """Get a list of colors from a list or interpolate between them.
    First and last color are fixed for idle and CMB obs
    """
    n_base = len(BASE_COLORS)

    if n <= n_base:
        # Use subset with fixed endpoints
        colors = [BASE_COLORS[0]] + BASE_COLORS[1:-1][:n - 2] + [BASE_COLORS[-1]]
    else:
        # Interpolate between base colors
        base_rgb = [mcolors.to_rgb(c) for c in BASE_COLORS]
        interpolated = np.linspace(0, n_base - 1, n)
        colors = []
        for i in interpolated:
            i_low = int(np.floor(i))
            i_high = min(i_low + 1, n_base - 1)
            t = i - i_low
            rgb = (1 - t) * np.array(base_rgb[i_low]) + t * np.array(base_rgb[i_high])
            colors.append(mcolors.to_hex(rgb))

    if reverse:
        colors = list(reversed(colors))

    return colors


@dataclass
class Colors:
    """
    Helper class for getting a list of colors.
    """
    names: List[str]
    reverse: bool = False
    colormap: Dict[str, str] = field(init=False)

    def __post_init__(self):
        n = len(self.names)
        discrete_colors = get_discrete_distinct_colors(n, self.reverse)
        self.colormap = {name: color for name, color in zip(self.names, discrete_colors)}

    def __getitem__(self, name: str) -> str:
        return self.colormap.get(name, "#CCCCCC")

    def __repr__(self):
        return f"Colors({self.colormap})"


def obs_hover(o, exclude=None):
    """
    Add obs info to hover text.
    """

    if exclude is None:
        exclude = set()

    def is_empty(v):
        if v is None:
            return True

        if isinstance(v, np.ndarray):
            if v.size == 0:
                return True

            if np.issubdtype(v.dtype, np.number):
                return np.all(~np.isfinite(v))

            return False

        if isinstance(v, (float, np.floating)):
            return not np.isfinite(v)

        if isinstance(v, (list, tuple, dict, set)):
            return len(v) == 0

        if isinstance(v, str):
            return v.strip() == ""

        return False

    def fmt(v):
        if isinstance(v, (float, np.floating)) and not isinstance(v, bool):
            return float(np.round(v, 2))
        return v

    meta = {}

    for k, v in vars(o).items():
        if (
            k.startswith("_")
            or callable(v)
            or k in exclude
            or is_empty(v)
        ):
            continue

        v = fmt(v)
        meta[k] = v

    lines = [f"{k}: {v}" for k, v in meta.items()]
    return "<br>".join(lines)


# ============================================================
# Observation efficiency plots.
# ============================================================


@dataclass
class ObsEfficiencyPlots:
    pie: go.Figure
    pie_good_pwv: go.Figure
    heatmap: go.Figure


def wafer_obs_efficiency(d: ReportData, nsegs: int=2000, good_pwv_lim: float=3) -> ObsEfficiencyPlots:
    """Plot heatmap of wafer vs time and pie chart showing observation efficiency"""

    def wafer_index(o, wafer, wafers, platform):
        """Get index in data vector for wafer"""
        if platform == "lat":
            return np.where(np.asarray(wafers) == f"{o.obs_tube_slot}_{wafer}")[0][0]
        return int(wafer.strip()[-1])

    def obs_value(o, obs_values, cal_targets):
        """Get value for data based on obs type and subtype"""
        if o.obs_type != "obs":
            return obs_values[o.obs_type]

        if o.obs_subtype == "cmb":
            return obs_values["cmb"]

        if o.obs_subtype == "cal":
            for tag in o.obs_tags.split(","):
                if tag in cal_targets:
                    return obs_values[tag]
            return obs_values["cal"]

        return obs_values["obs"]

    def fill_obs_array(arr, tstamps, d, wafers, obs_values):
        """Populate data array with values"""
        for o in d.obs_list:
            mask = (tstamps > o.start_time) & (tstamps < o.stop_time)
            value = obs_value(o, obs_values, d.cfg.cal_targets)

            for wafer in o.wafer_slots_list.split(","):
                idx = wafer_index(o, wafer, wafers, d.cfg.platform)
                arr[idx, mask] = value

    def make_pie(arr, obs_types):
        """Create pie chart from data array"""
        vals, counts = np.unique(arr, return_counts=True)
        reverse = True if (arr == (len(obs_types) - 1)).all() else False

        labels = [obs_types[v] for v in vals]
        colors = Colors(names=labels, reverse=reverse)

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=100 * counts / counts.sum(),
                textinfo="label+percent",
                marker=dict(colors=[colors[l] for l in labels]),
            )
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
        )

        return fig, colors

    wafers = get_wafers(d.cfg.platform)

    obs_types = ["cmb", "obs", "cal", "oper", *d.cfg.cal_targets, "idle"]
    obs_values = {k: i for i, k in enumerate(obs_types)}

    # All data pie chart
    times = pd.date_range(d.cfg.start_time, d.cfg.stop_time, nsegs)
    tstamps = times.astype(np.int64) / 1e9

    data = np.full((len(wafers), nsegs), obs_values["idle"], dtype=int)
    fill_obs_array(data, tstamps, d, wafers, obs_values)
    pie, colors = make_pie(data, obs_types)

    # Use 10 minute segments for PWV pie chart
    start = pd.Timestamp(d.cfg.start_time)
    stop = pd.Timestamp(d.cfg.stop_time)
    nsegs_pwv = int(np.ceil((stop - start).total_seconds() / 600))

    # Get segments with good pwv
    t_edges = pd.date_range(start, stop, periods=nsegs_pwv + 1)
    t_edges = np.array([t.timestamp() for t in t_edges])

    data_pwv = np.full(nsegs_pwv, np.nan)

    pwv_times = d.pwv[0]
    pwv_values = d.pwv[1]

    # Average pwv for each segment
    for i in range(nsegs_pwv):
        m = (pwv_times >= t_edges[i]) & (pwv_times < t_edges[i + 1])
        if np.any(m):
            data_pwv[i] = np.mean(pwv_values[m])

    good = np.isfinite(data_pwv) & (data_pwv < good_pwv_lim)

    pwv_times = pd.date_range(start, stop, nsegs_pwv)
    pwv_tstamps = pwv_times.astype(np.int64) / 1e9

    data_good_pwv = np.full((len(wafers), nsegs_pwv), obs_values["idle"], dtype=int)
    fill_obs_array(data_good_pwv, pwv_tstamps, d, wafers, obs_values)
    pie_good_pwv, _ = make_pie(data_good_pwv[:, good], obs_types)

    # Efficiency heatmap
    unique = np.unique(data)
    mapping = {v: i for i, v in enumerate(unique)}

    z = np.vectorize(mapping.get)(data)

    labels = [obs_types[v] for v in unique]

    colorscale = []
    for i, label in enumerate(labels):
        colorscale.extend([
            (i / len(labels), colors[label]),
            ((i + 1) / len(labels), colors[label]),
        ])

    text = np.vectorize(dict(enumerate(labels)).get)(z)
    hover_text = [
        [
            (
                f"Time: {time}<br>"
                f"Wafer: {wafer}<br>"
                f"Type: {text[i, j]}"
            )
            for j, time in enumerate(times)
        ]
        for i, wafer in enumerate(wafers)
    ]

    heatmap = go.Figure(
        data=go.Heatmap(
            z=z,
            x=times,
            y=wafers,
            text=hover_text,
            hoverinfo="text",
            zmin=0,
            zmax=len(labels) - 1,
            colorscale=colorscale,
            colorbar=dict(
                tickvals=list(range(len(labels))),
                ticktext=labels,
            ),
            ygap=1,
        )
    )

    height = {
        "lat": 700,
        "satp1": 300,
        "satp2": 300,
        "satp3": 300,
    }.get(d.cfg.platform, 400)

    heatmap.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
    )

    return ObsEfficiencyPlots(pie=pie, pie_good_pwv=pie_good_pwv, heatmap=heatmap)


# ============================================================
# Time series plots for HKDB fields and other metadata such as
# hwp dir, boresight, and so on.
# ============================================================


def obsdb_scatter_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    xlim,
    symbols=None,
    name=None,
    hovertext=None,
    fig=None,
):
    """Generic scatter plot for obsdb fields"""

    x = np.asarray(x)
    y = np.asarray(y)

    if symbols is None:
        symbols = np.full(len(x), "default")
    else:
        symbols = np.asarray(symbols)

    uniq_symbols = np.unique(symbols)

    symbol_map = {
        s: MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)]
        for i, s in enumerate(uniq_symbols)
    }

    if fig is None:
        fig = go.Figure()

    for s in uniq_symbols:
        m = symbols == s

        fig.add_trace(
            go.Scatter(
                x=x[m],
                y=y[m],
                mode="markers",
                marker=dict(
                    size=8,
                    symbol=symbol_map[s],
                ),
                hovertext=None if hovertext is None else np.asarray(hovertext)[m],
                name=f"{name}_{s}" if name else str(s),
                showlegend=True,
            )
        )

    fig.update_layout(
        margin=dict(l=40, r=40, t=140, b=40),
        height=550,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.92,
            yanchor="top",
            font=dict(size=18),
        ),
    )

    fig.update_xaxes(
        title_text=xlabel,
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        range=list(xlim),
    )

    fig.update_yaxes(
        title_text=ylabel,
        showgrid=True,
        gridcolor="lightgray",
    )

    return fig


def plot_obs_timeseries(
    d: ReportData,
    field: str,
    ylabel: str,
    title: str,
    transform=None,
) -> go.Figure:
    """
    Generic observation timeseries plot.
    """

    obs = [o for o in d.obs_list if o.obs_type == "obs"]
    if not obs:
        return go.Figure()

    t = np.array([o.start_time for o in obs])
    y = np.array([
        np.nan if getattr(o, field, None) is None else getattr(o, field)
        for o in obs
    ])
    sub_types = np.array([o.obs_subtype for o in obs])

    hovertext = np.array([obs_hover(o) for o in obs])

    if transform is not None:
        y = transform(y)

    order = np.argsort(t)

    t = t[order]
    y = y[order]
    sub_types = sub_types[order]
    hovertext = np.array(hovertext)[order]

    x = [dt.datetime.fromtimestamp(tt) for tt in t]

    return obsdb_scatter_plot(
        x,
        y,
        xlabel="Time (UTC)",
        ylabel=ylabel,
        title=title,
        xlim=[d.cfg.start_time, d.cfg.stop_time],
        symbols=sub_types,
        name=field,
        hovertext=hovertext,
    )


def boresight_vs_time(d: ReportData) -> go.Figure:
    """Boresight angle and corotator vs time."""

    obs = [o for o in d.obs_list if o.obs_type == "obs"]
    if not obs:
        return go.Figure()

    tstamps = np.array([o.start_time for o in obs])
    boresight = np.array([
        np.nan if o.boresight is None else o.boresight
        for o in obs
    ])
    sub_types = np.array([o.obs_subtype for o in obs])

    boresight = np.round(boresight, 1)

    if d.cfg.platform == "lat":
        el_center = np.array([
            np.nan if o.el_center is None else o.el_center
            for o in obs
        ])
    else:
        el_center = None

    hovertext = np.array([obs_hover(o) for o in obs])

    order = np.argsort(tstamps)

    tstamps = tstamps[order]
    boresight = boresight[order]
    sub_types = sub_types[order]
    hovertext = np.array(hovertext)[order]

    if el_center is not None:
        el_center = el_center[order]

    x = [dt.datetime.fromtimestamp(t) for t in tstamps]

    fig = obsdb_scatter_plot(
        x,
        boresight,
        xlabel="Time (UTC)",
        ylabel="Boresight [deg]",
        title="Boresight",
        xlim=[d.cfg.start_time, d.cfg.stop_time],
        symbols=sub_types,
        name="bore",
        hovertext=hovertext,
    )

    if d.cfg.platform == "lat":
        corot = boresight + el_center - 60

        for s in np.unique(sub_types):
            m = sub_types == s

            fig.add_trace(
                go.Scatter(
                    x=np.array(x)[m],
                    y=corot[m],
                    mode="markers",
                    name=f"corot_{s}",
                    marker=dict(size=8, symbol="circle"),
                    yaxis="y2",
                    hovertext=hovertext[m],
                )
            )

        fig.update_layout(
            yaxis2=dict(
                title="Corotator [deg]",
                overlaying="y",
                side="right",
            )
        )

    return fig


def scan_type_vs_time(d: ReportData) -> go.Figure:
    """Scan type vs time."""

    options = {"type1", "type2", "type3"}

    hovertext = []
    obs = []
    for o in d.obs_list:
        if o.obs_type != "obs":
            continue

        tags = set(o.obs_tags.split(",")) if o.obs_tags else set()
        match = next((t for t in tags if t in options), None)

        if match is not None:
            obs.append((o.start_time, match, o.obs_subtype))
            hovertext.append(obs_hover(o))

    if not obs:
        return go.Figure()

    tstamps, scan_types, sub_types = map(np.array, zip(*obs))

    order = np.argsort(tstamps)

    tstamps = tstamps[order]
    scan_types = scan_types[order]
    sub_types = sub_types[order]
    hovertext = np.array(hovertext)[order]

    x = [dt.datetime.fromtimestamp(t) for t in tstamps]

    return obsdb_scatter_plot(
        x,
        scan_types,
        xlabel="Time (UTC)",
        ylabel="Scan Type",
        title="Scan Type",
        xlim=[d.cfg.start_time, d.cfg.stop_time],
        symbols=sub_types,
        name="scan_type",
        hovertext=hovertext,
    )


def hwp_freq_vs_time(d: ReportData) -> go.Figure:
    """Mean HWP frequency vs time (SAT only)."""

    if d.cfg.platform not in {"satp1", "satp2", "satp3"}:
        return go.Figure()

    obs = [
        o for o in d.obs_list
        if o.obs_type == "obs"
    ]

    if not obs:
        return go.Figure()

    tstamps = np.array([o.start_time for o in obs])
    hwp = np.array([
        np.nan if o.hwp_freq_mean is None else o.hwp_freq_mean
        for o in obs
    ])
    sub_types = np.array([o.obs_subtype for o in obs])

    hwp = np.round(hwp, 2)

    hovertext = np.array([obs_hover(o) for o in obs])

    order = np.argsort(tstamps)

    tstamps = tstamps[order]
    hwp = hwp[order]
    sub_types = sub_types[order]
    hovertext = np.array(hovertext)[order]

    x = [dt.datetime.fromtimestamp(t) for t in tstamps]

    return obsdb_scatter_plot(
        x,
        hwp,
        xlabel="Time (UTC)",
        ylabel="Mean HWP Freq [Hz]",
        title="Mean HWP Frequency",
        xlim=[d.cfg.start_time, d.cfg.stop_time],
        symbols=sub_types,
        name="hwp_freq",
        hovertext=hovertext,
    )


# ============================================================
# PWV, Yield, and NEPs (array averaged and effective detector)
# related plots.
# ============================================================


def pwv_vs_time(d: ReportData, fig: go.Figure, ds_factor: int=5):
    """ Helper function to plot PWV vs time"""
    pwvs = np.array(deepcopy(d.pwv[1][::ds_factor]))
    ts = np.array([dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in d.pwv[0][::ds_factor]])

    mask = pwvs >= 3

    starts = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == 1)[0]
    ends = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == -1)[0]

    for i, (s, e) in enumerate(zip(starts, ends)):
        fig.add_trace(
            go.Scatter(
                x=ts[s:e],
                y=np.full(e-s, 4.0),
                fill="tozeroy",
                mode="none",
                fillcolor="rgba(128,128,128,0.1)",
                connectgaps=False,
                showlegend=(i==0),
                name="PWV ≥ 3 mm" if i==0 else None
            ),
            secondary_y=True
        )

    pwvs[(pwvs > 4) | (pwvs < .1)] = np.nan

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=pwvs,
            mode="markers",
            name="PWV [mm]",
            line=dict(color="#CC79A7", width=2, dash="solid"),
            opacity=0.6,
        ),
        secondary_y=True,
    )

    return fig


def pwv_and_timeseries_vs_time(
    d: "ReportData",
    mode: str,
    field_name: str | None = None,
) -> go.Figure:
    """PWV and time-series plot (yield or NEP)."""

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    times = defaultdict(list)
    values = defaultdict(lambda: defaultdict(list))
    hover = defaultdict(list)

    for obs in d.obs_list:
        if obs.obs_subtype != "cmb":
            continue

        t = dt.datetime.fromtimestamp(obs.start_time, tz=dt.timezone.utc)

        if mode == "yield":
            if obs.num_valid_dets.size == 0:
                continue

            for b in obs.num_valid_dets.dtype.names:
                times[b].append(t)
                values[b]["yield"].append(obs.num_valid_dets[b][0])
                hover[b].append(obs_hover(obs))

        elif mode == "nep":
            field = obs.array_nep if field_name == "array" else obs.det_nep
            if field.size == 0:
                continue

            for b in field.dtype.names:
                times[b].append(t)
                hover[b].append(obs_hover(obs))

                for i, pol in enumerate(field[b].dtype.names):
                    values[b][pol].append(field[b][0][i])

    i = 0

    for b in sorted(values):

        for k, arr in values[b].items():

            label = (
                f"{k} ({b})"
                if mode == "nep"
                else f"Valid Dets ({b})"
            )

            fig.add_trace(
                go.Scatter(
                    x=times[b],
                    y=arr,
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=MARKER_COLORS[i % len(MARKER_COLORS)],
                        symbol=MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)],
                        size=8,
                        line=dict(width=1, color="black"),
                    ),
                    hovertext=hover[b],
                ),
                secondary_y=False,
            )
            i += 1

    fig = pwv_vs_time(d, fig)

    if mode == "yield":
        title = "Valid Detectors and PWV"
        y_left = "Num Valid Dets"
        tpad = 80

    else:
        title = f"{field_name.capitalize()} NEP and PWV"
        y_left = r"$\rm{NEP\ [aW/\sqrt{Hz}]}$"
        tpad = 140

    fig.update_layout(
        margin=dict(l=40, r=40, t=tpad, b=40),
        height=550,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=18),
        ),
    )

    fig.update_xaxes(
        title_text="Time (UTC)",
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=y_left,
        secondary_y=False,
        showgrid=True,
        gridcolor="lightgray",
    )

    fig.update_yaxes(
        title_text="PWV [mm]",
        secondary_y=True,
        showgrid=True,
    )

    return fig


def pwv_and_yield_vs_time(d):
    return pwv_and_timeseries_vs_time(d, mode="yield")


def pwv_and_nep_vs_time(d, field_name="array"):
    return pwv_and_timeseries_vs_time(d, mode="nep", field_name=field_name)


def field_vs_pwv(
    d: "ReportData",
    mode: str,
    longterm_data: Optional["ReportData"] = None,
    field_name: str | None = None,
) -> go.Figure:
    """Generic PWV vs field plot (yield or NEP)."""

    fig = go.Figure()

    if longterm_data is not None:

        long_vals = defaultdict(lambda: defaultdict(list))
        long_pwvs = []

        for obs in longterm_data.obs_list:
            if np.isfinite(obs.pwv):

                if mode == "yield":
                    field = obs.num_valid_dets
                else:
                    field = obs.array_nep if field_name == "array" else obs.det_nep

                if field.size == 0:
                    continue

                long_pwvs.append(obs.pwv)

                for b in field.dtype.names:
                    if mode == "yield":
                        long_vals[b]["yield"].append(field[b][0])
                    else:
                        for i, pol in enumerate(field[b].dtype.names):
                            long_vals[b][pol].append(field[b][0][i])

        c = 0
        for b in sorted(long_vals):
            for k, vals in long_vals[b].items():

                fig.add_trace(
                    go.Histogram2dContour(
                        x=long_pwvs,
                        y=vals,
                        colorscale=[[0, MARKER_COLORS[c]], [1, MARKER_COLORS[c]]],
                        contours_coloring="lines",
                        showscale=False,
                        opacity=0.7,
                        name=f"longterm {b}_{k}",
                        showlegend=True,
                    )
                )
                c += 1

    vals = defaultdict(lambda: defaultdict(list))
    pwvs = []
    hover = defaultdict(list)

    for obs in d.obs_list:
        if (
            not np.isfinite(obs.pwv)
            or obs.obs_subtype != "cmb"
        ):
            continue

        pwvs.append(obs.pwv)

        if mode == "yield":
            field = obs.num_valid_dets
            if field.size == 0:
                continue

            for b in field.dtype.names:
                vals[b]["yield"].append(field[b][0])
                hover[b].append(obs_hover(obs))

        else:
            field = obs.array_nep if field_name == "array" else obs.det_nep
            if field.size == 0:
                continue

            for b in field.dtype.names:
                hover[b].append(obs_hover(obs))
                for i, pol in enumerate(field[b].dtype.names):
                    vals[b][pol].append(field[b][0][i])

    i = 0
    for b in sorted(vals):
        for k, arr in vals[b].items():

            label = (
                f"Valid Dets ({b})"
                if mode == "yield"
                else f"{k.split('_')[-1]} ({b})"
            )

            fig.add_trace(
                go.Scatter(
                    x=pwvs,
                    y=arr,
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=MARKER_COLORS[i % len(MARKER_COLORS)],
                        symbol=MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)],
                        size=8,
                        line=dict(width=1, color="black"),
                    ),
                    hovertext=hover[b],
                )
            )
            i += 1

    if mode == "yield":
        title = "Valid Detectors and PWV"
        ylab = "Num Valid Dets"
        tpad = 80
    else:
        title = f"{field_name.capitalize()} NEP and PWV"
        ylab = r"$\rm{NEP\ [aW/\sqrt{Hz}]}$"
        tpad = 140

    fig.update_layout(
        margin=dict(l=40, r=40, t=tpad, b=40),
        height=550,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=18),
        ),
    )

    fig.update_xaxes(
        title_text="PWV [mm]",
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=ylab,
        showgrid=True,
        gridcolor="lightgray",
    )

    return fig


def yield_vs_pwv(d, longterm_data=None):
    return field_vs_pwv(d, mode="yield", longterm_data=longterm_data)


def nep_vs_pwv(d, longterm_data=None, field_name=None):
    return field_vs_pwv(
        d,
        mode="nep",
        longterm_data=longterm_data,
        field_name=field_name,
    )


def cov_map_plot(map_png_file: str, embed:bool=True) -> str:
    import base64

    if map_png_file is None or not os.path.isfile(map_png_file):
        return "<p>Coverage map not found</p>"

    # Embed image as base64
    if embed:
        with open(map_png_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%; height:auto;" '
            f'alt="Coverage Map">'
        )

    # Otherwise use relative path
    return (
        f'<img src="./{os.path.basename(map_png_file)}" '
        f'style="max-width:100%; height:auto;" '
        f'alt="Coverage Map">'
    )


# ============================================================
# Source footprint plots.
# ============================================================


@dataclass
class SourceFootprintPlots:
    focalplane: go.Figure
    table: go.Figure


def source_footprints(d: "ReportData") -> go.Figure:

    if d.cfg.platform in ["satp1", "satp2", "satp3"]:
        wafer_rad = 0.10488
        wafer_sep = 0.219911
        pie_width = 0.25

        angs = np.arange(-np.pi / 2, 3 * np.pi / 2, np.pi / 3)
        x0s = [0] + [np.cos(a) * wafer_sep for a in angs]
        y0s = [0] + [np.sin(a) * wafer_sep for a in angs]

        wafers = ['ws0', 'ws1', 'ws6', 'ws5', 'ws4', 'ws3', 'ws2']

        wafer_centers = {}
        for i, wafer in enumerate(wafers):
            wafer_centers[wafer] = (x0s[i], y0s[i])

    elif d.cfg.platform == "lat":
        wafer_rad = 0.005236
        pie_width = 0.0625
        x0s = [
            -0.00619278,  0.00317458,  0.00316847,
            -0.03301884, -0.02378116, -0.02375480,
            -0.00610289,  0.00317318,  0.00323130,
             0.02089648,  0.03016575,  0.03018966,
             0.02082073,  0.03019804,  0.03015737,
            -0.00610289,  0.00323881,  0.00316568,
            -0.03301779, -0.02374625, -0.02378954,
            -0.02350644, -0.03283837, -0.02360279,
             0.03350055,  0.02415571,  0.02406826,
             0.05734820,  0.05733965,  0.04790702,
             0.02416426,  0.03350072,  0.02405971,
            -0.03283854, -0.02349789, -0.02361134,
            -0.05998592, -0.05038556, -0.05038556,
        ]

        y0s = [
             5.24e-06,  0.00548417, -0.00548732,
            -0.01565927, -0.01019534, -0.02093994,
            -0.03111852, -0.02575652, -0.03649641,
            -0.01562349, -0.01019778, -0.02094709,
             0.01557305,  0.02094255,  0.01019325,
             0.03112830,  0.03649204,  0.02575216,
             0.01557287,  0.02093540,  0.01019080,
            -0.05206474, -0.04667656, -0.04128960,
            -0.04669977, -0.05206684, -0.04129466,
             0.00539656, -0.00540162,  4.89e-06,
             0.05207190,  0.04668983,  0.04129973,
             0.04668634,  0.05205968,  0.04128472,
             3.50e-07,  0.00561629, -0.00561577,
        ]

        wafers = [
            'c1_ws0', 'c1_ws1', 'c1_ws2',
            'i1_ws0', 'i1_ws1', 'i1_ws2',
            'i2_ws0', 'i2_ws1', 'i2_ws2',
            'i3_ws0', 'i3_ws1', 'i3_ws2',
            'i4_ws0', 'i4_ws1', 'i4_ws2',
            'i5_ws0', 'i5_ws1', 'i5_ws2',
            'i6_ws0', 'i6_ws1', 'i6_ws2',
            'o1_ws0', 'o1_ws1', 'o1_ws2',
            'o2_ws0', 'o2_ws1', 'o2_ws2',
            'o3_ws0', 'o3_ws1', 'o3_ws2',
            'o4_ws0', 'o4_ws1', 'o4_ws2',
            'o5_ws0', 'o5_ws1', 'o5_ws2',
            'o6_ws0', 'o6_ws1', 'o6_ws2',
        ]

        wafer_centers = {}
        for i, wafer in enumerate(wafers):
            wafer_centers[wafer] = (x0s[i], y0s[i])

    def hex(x0, y0):
        angs = np.linspace(0, 2 * np.pi, 7)
        pts = np.array(
            [
                x0 + wafer_rad * np.cos(angs),
                y0 + wafer_rad * np.sin(angs),
            ]
        )
        return pts

    xs = [x for x, y in wafer_centers.values()]
    ys = [y for x, y in wafer_centers.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span > y_span:
        y_mid = (y_max + y_min) / 2
        y_min = y_mid - x_span / 2
        y_max = y_mid + x_span / 2
    else:
        x_mid = (x_max + x_min) / 2
        x_min = x_mid - y_span / 2
        x_max = x_mid + y_span / 2
    x_margin = x_span * 0.05
    y_margin = x_span * 0.05

    x_range = [x_min - 1.1*wafer_rad, x_max + 1.1*wafer_rad]
    y_range = [y_min - 1.1*wafer_rad, y_max + 1.1*wafer_rad]

    def get_normalized_positions(positions, x_range, y_range, fig_width, fig_height):
        x0, x1 = x_range
        y0, y1 = y_range
        x_span = x1 - x0
        y_span = y1 - y0

        data_aspect = x_span / y_span
        fig_aspect = fig_width / fig_height

        norm_positions = {}

        if data_aspect > fig_aspect:
            # Y is scaled (shorter)
            scale = fig_aspect / data_aspect
            y_offset = (1 - scale) / 2
            for key, (x, y) in positions.items():
                x_paper = (x - x0) / x_span
                y_paper = y_offset + scale * (y - y0) / y_span
                norm_positions[key] = (x_paper, y_paper)
        else:
            # X is scaled (shorter)
            scale = data_aspect / fig_aspect
            x_offset = (1 - scale) / 2
            for key, (x, y) in positions.items():
                x_paper = x_offset + scale * (x - x0) / x_span
                y_paper = (y - y0) / y_span
                norm_positions[key] = (x_paper, y_paper)

        return norm_positions

    norm_positions = get_normalized_positions(wafer_centers, x_range, y_range, 900, 900)

    def normalize_values(values, target_total=100):
        total = sum(values)
        if total == 0:
            return values  # leave empty pie alone
        return [v * target_total / total for v in values]

    fig = go.Figure()

    for i, (x0, y0) in enumerate(zip(x0s, y0s)):
        pts = hex(x0, y0)
        fig.add_trace(
            go.Scatter(
                x=pts[0],
                y=pts[1],
                fill="toself",
                fillcolor="grey",
                opacity=0.3,
                mode="lines",
                marker=dict(color="grey"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_annotation(
            x=x0,
            y=y0 + 0.9*wafer_rad,
            text=wafers[i],
            showarrow=False,
            font=dict(size=12),
            xanchor='center',
            yanchor='bottom',
            xref='x',
            yref='y'
        )

    ntargets = 0
    target_colors = {}
    if d.source_footprints is not None:
        wafer_map = defaultdict(list)
        for fp in d.source_footprints:
            wafer_map[fp.wafer].append(fp)

        if wafer_map:
            for wafer, group in wafer_map.items():
                x, y = norm_positions[wafer]
                labels = [fp.source for fp in group]
                values = [fp.count for fp in group]
                hovertext = [
                    f"Wafer: {fp.wafer}<br>Source: {fp.source}<br>Count: {fp.count}<br>ObsIDs:<br>" + "<br>".join(fp.obsids)
                    for fp in group
                ]

                x0 = x - pie_width / 2
                x1 = x + pie_width / 2
                y0 = y - pie_width / 2
                y1 = y + pie_width / 2

                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=normalize_values(values),
                        name=wafer,
                        domain=dict(x=[x0, x1], y=[y0, y1]),
                        scalegroup='wafer_pies',
                        hoverinfo='text',
                        hovertext=hovertext
                    )
                )

    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[y_min, y_max],
        mode='markers',
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        xaxis=dict(
            range=x_range,
            title='Xi (rad)',
            showgrid=True,
            zeroline=True,
            fixedrange=True
        ),
        yaxis=dict(
            range=y_range,
            title='Eta (rad)',
            showgrid=True,
            zeroline=True,
            fixedrange=True
        ),
        width=800,
        height=800,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    def make_obsid_links(obsids):
        if not obsids:
            return ""
        links = [
            f'<a href="{d.cfg.site_url}/site-pipeline/{d.cfg.platform}/preprocess/{obsid.split("_")[1][:5]}/{obsid}/" target="_blank">{obsid}</a>'
            for obsid in obsids
        ]
        return "<br>".join(links) + ("<br>" if len(links) == 1 else "")

    if d.source_footprints is not None:
        wafers = sorted(set(fp.wafer for fp in d.source_footprints))
        sources = sorted(set(fp.source for fp in d.source_footprints))

        lookup = {
            (fp.wafer, fp.source): fp.obsids
            for fp in d.source_footprints
        }

        wafer_col = wafers
        source_cols = []
        for source in sources:
            col = []
            for wafer in wafers:
                obsids = lookup.get((wafer, source), [])
                col.append(make_obsid_links(obsids))
            source_cols.append(col)
    else:
        wafer_col = []
        source_cols = []
        sources = []

    table_columns = [wafer_col] + source_cols
    header_labels = ["Wafers"] + sources

    table = go.Figure(data=[go.Table(
        header=dict(
            values=header_labels,
            fill_color='lightgrey',
            align='left',
            font=dict(size=12),
            line_color='black'
        ),
        cells=dict(
            values=table_columns,
            fill_color='white',
            align='left',
            line_color='black',
            height=30
        )
    )])

    table.update_layout(
        width=500,
        height=400 + 30 * len(wafers),
        margin=dict(l=20, r=20, t=20, b=20),
        dragmode=False
    )

    return SourceFootprintPlots(focalplane=fig, table=table)
