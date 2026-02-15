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


def get_wafers(platform: str):
    """Get wafer and tube slots"""
    if platform  == "lat":
        tube_slots = ["c1", "i1", "i3", "i4", "i5", "i6"]
        wafer_slots = ["ws0", "ws1", "ws2"]
        wafers = [f"{x}_{y}" for x in tube_slots for y in wafer_slots]
    elif platform in ["satp1", "satp2", "satp3"]:
        wafers = [f"ws{i} " for i in range(7)]
    else:
        raise ValueError(f"Uknown platform {platform}")

    return wafers


def get_discrete_distinct_colors(n, reverse=False):
    """Get a list of colors from a list or interpolate between them.
    First and last color are fixed for idle and CMB obs
    """
    base_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
    n_base = len(base_colors)

    if n <= n_base:
        # Use subset with fixed endpoints
        colors = [base_colors[0]] + base_colors[1:-1][:n - 2] + [base_colors[-1]]
    else:
        # Interpolate between base colors
        base_rgb = [mcolors.to_rgb(c) for c in base_colors]
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


@dataclass
class ObsEfficiencyPlots:
    pie: go.Figure
    heatmap: go.Figure


def obsdb_line_plot(x, y, xlabel, ylabel, title, xlim):
    """Generic function to make a Plotly scatter plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=ylabel,
            line=dict(color="#0072B2", width=2),
            marker=dict(size=8)
        ))

    # layout
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
        range=[xlim[0], xlim[1]]
    )
    fig.update_yaxes(
        title_text=ylabel,
        showgrid=True,
        gridcolor="lightgray",
    )

    return fig


def boresight_vs_time(d: ReportData) -> go.Figure:
    """Boresight angle and corotator in degrees vs time"""
    tstamps = [o.start_time for o in d.obs_list]
    boresight = [o.boresight if o.boresight is not None else np.nan for o in d.obs_list]
    boresight = np.round(boresight, 1)

    if not tstamps:
        return go.Figure()

    if d.cfg.platform == "lat":
        el_center = [o.el_center if o.el_center is not None else np.nan for o in d.obs_list]
        tstamps, boresight, el_center = zip(*sorted(zip(tstamps, boresight, el_center)))
        el_center = list(el_center)
    else:
        tstamps, boresight = zip(*sorted(zip(tstamps, boresight)))
    tstamps = list(tstamps)
    boresight = list(boresight)
    tstamps = [
        dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        for t in tstamps
    ]

    fig = obsdb_line_plot(tstamps, boresight, "Time (UTC)", "Boresight [deg]", "Boresight",
                         xlim=[d.cfg.start_time, d.cfg.stop_time])

    if d.cfg.platform == "lat":
        corot = np.array(boresight) + np.array(el_center) - 60

        fig.add_trace(
        go.Scatter(
            x=tstamps,
            y=corot,
            mode='markers',
            name="Corotator [deg]",
            line=dict(color="#E69F00", width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))

        fig.update_layout(
        yaxis2=dict(
            title="Corotator [deg]",
            overlaying="y",
            side="right"
        ))

    return fig


def hwp_freq_vs_time(d: ReportData) -> go.Figure:
    """Mean HWP frequency vs time"""
    if d.cfg.platform in ["satp1", "satp2", "satp3"]:
        tstamps = [o.start_time for o in d.obs_list]
        hwp_freq_mean = [o.hwp_freq_mean if o.hwp_freq_mean is not None else np.nan for o in d.obs_list]
        hwp_freq_mean = np.round(hwp_freq_mean, 2)

        if not tstamps:
            return go.Figure()

        tstamps, hwp_freq_mean = zip(*sorted(zip(tstamps, hwp_freq_mean)))
        tstamps = [
            dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            for t in tstamps
        ]

        return obsdb_line_plot(tstamps, hwp_freq_mean, "Time (UTC)", "Mean HWP Freq [Hz]", "Mean HWP Freq",
                              xlim=[d.cfg.start_time, d.cfg.stop_time])
    else:
        return go.Figure()


def wafer_obs_efficiency(d: ReportData, nsegs=2000) -> ObsEfficiencyPlots:
    """Plot heatmap of wafer vs time and pie chart showing observation efficiency"""

    wafers = get_wafers(d.cfg.platform)
    nwafers = len(wafers)

    times = pd.date_range(d.cfg.start_time, d.cfg.stop_time, nsegs).to_pydatetime()
    tstamps = np.array([t.timestamp() for t in times])

    obs_types = ["cmb", "obs", "cal", "oper"] + d.cfg.cal_targets + ["idle"]
    obs_values = {k: i for i, k in enumerate(obs_types)}

    data = np.ones((nwafers, nsegs), dtype=int) * (len(obs_types) - 1)

    for o in d.obs_list:
        m = np.logical_and.reduce([tstamps > o.start_time, tstamps < o.stop_time])
        wafer_slots_list = o.wafer_slots_list.split(",")
        for wafer_slot in wafer_slots_list:
            if d.cfg.platform == "lat":
                idx = np.where(np.array(wafers) == o.obs_tube_slot + "_" + wafer_slot)[0][0]
            else:
                idx = int(wafer_slot.strip()[-1])
            if o.obs_type == "obs":
                if o.obs_subtype == "cmb":
                    data[idx][m] = obs_values[o.obs_subtype]
                elif o.obs_subtype == "cal":
                    matches = [item for item in o.obs_tags.split(',') if item in d.cfg.cal_targets]
                    if matches:
                        data[idx][m] = obs_values[matches[0]]
                    else:
                        data[idx][m] = obs_values[o.obs_subtype]
                else:
                    data[idx][m] = obs_values[o.obs_type]
            else:
                data[idx][m] = obs_values[o.obs_type]

    reverse = False
    if (data == (len(obs_types) - 1)).all():
        reverse = True

    # Compile data for pie chart
    unique_vals, counts = np.unique(data, return_counts=True)
    percentages = counts / counts.sum() * 100
    labels = [obs_types[i] for i in unique_vals]
    COLORS = Colors(names=labels, reverse=reverse)
    colorscale: List[Tuple[float, str]] = []
    ntypes = len(labels)
    for i, t in enumerate(labels):
        colorscale.extend([(i / ntypes, COLORS[t]), ((i + 1) / ntypes, COLORS[t])])
    pie_colors = [COLORS[t] for t in labels]
    colorbar=dict(tickvals=list(range(len(labels))), ticktext=labels,)

    unique_sorted = np.sort(np.unique(data))
    mapping = {v: i for i, v in enumerate(unique_sorted)}
    vectorized_map = np.vectorize(mapping.get)
    data = vectorized_map(data)

    value_to_label = {i: lbl for i, lbl in enumerate(labels)}
    text = np.vectorize(value_to_label.get)(data)

    ys = wafers

    hover_text = []
    for yi, yval in enumerate(ys):
        row = []
        for xi, xval in enumerate(times):
            label = text[yi, xi]
            row.append(f"x: {xval}<br>y: {yval}<br>z: {label}")
        hover_text.append(row)

    heatmap = go.Figure(
        data=go.Heatmap(
            z=data,
            zmin=0,
            zmax=len(labels) - 1,
            x=times,
            y=ys,
            text=hover_text,
            hoverinfo="text",
            colorscale=colorscale,
            colorbar=dict(tickvals=list(range(len(labels))), ticktext=labels,),
            ygap=1,
        ),
    )
    heatmap.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)

    pie = go.Figure(
        data=go.Pie(
            labels=labels,
            values=percentages,
            textinfo="label+percent",
        ),
    )
    pie.update_traces(marker=dict(colors=pie_colors))
    pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)

    return ObsEfficiencyPlots(pie=pie, heatmap=heatmap)


colors = ["#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00"]
markers = ["circle", "square", "diamond", "cross", "x", "triangle-up"]


def pwv_and_yield_vs_time(d: "ReportData") -> go.Figure:
    """PWV and yield as a function of time"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    yields = defaultdict(list)
    times = defaultdict(list)
    hover = defaultdict(list)

    # build a dict keyed off of bandpass for data from all obs
    for obs in d.obs_list:
        obs_time = dt.datetime.fromtimestamp(obs.start_time, tz=dt.timezone.utc)
        if obs.num_valid_dets.size > 0:
            for b in obs.num_valid_dets.dtype.names:
                times[b].append(obs_time)
                yields[b].append(obs.num_valid_dets[b][0])
                hover[b].append(obs.obs_id)

    # yield trace
    for i, b in enumerate(sorted(yields)):
        fig.add_trace(
            go.Scatter(
                x=times[b],
                y=yields[b],
                mode="markers",
                name=f"Valid Dets ({b})",
                marker=dict(
                    color=colors[i % len(colors)],
                    symbol=markers[i % len(markers)],
                    size=8,
                    line=dict(width=1, color="black"),
                ),
                hovertext=hover[b],
            ),
            secondary_y=False,
        )

    if d.obs_list:
        # PWV trace
        ds_factor = 10
        #pwvs = deepcopy(d.pwv[1][::ds_factor])
        #pwvs[(pwvs > 4) | (pwvs < .1)] = np.nan
        #ts = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in d.pwv[0][::ds_factor]]

        ts, pwvs = zip(*sorted(
            ((dt.datetime.fromtimestamp(o.start_time, tz=dt.timezone.utc), o.pwv) for o in d.obs_list)
        ))
        ts = list(ts)
        pwvs = list(pwvs)

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

    # layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
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
            text="Valid Detectors and PWV",
            x=0.5,
            xanchor="center",
            y=0.97,
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
        title_text="Num Valid Dets",
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


def yield_vs_pwv(d: "ReportData", longterm_data: Optional["ReportData"] = None) -> go.Figure:
    """Number of valid detectors as a function of PWV. Current interval is plotted over
    longterm contours.
    """
    fig = go.Figure()

    if longterm_data is not None:
        long_yields = defaultdict(list)
        long_pwvs = []
        for obs in longterm_data.obs_list:
            if obs.num_valid_dets.size > 0:
                if np.isfinite(obs.pwv):
                    long_pwvs.append(obs.pwv)
                    for b in obs.num_valid_dets.dtype.names:
                        long_yields[b].append(obs.num_valid_dets[b][0])

        for i, b in enumerate(sorted(long_yields)):
            fig.add_trace(
                go.Histogram2dContour(
                    x=long_pwvs,
                    y=long_yields[b],
                    colorscale=[[0, colors[i]], [1, colors[i]]],
                    contours_coloring="lines",
                    showscale=False,
                    opacity=0.7,
                    name=f"longterm {b}",
                )
            )

    yields = defaultdict(list)
    pwvs = []
    hover = defaultdict(list)

    # build a dict keyed off of bandpass for data from all obs
    for obs in d.obs_list:
        if obs.num_valid_dets.size > 0:
             if np.isfinite(obs.pwv):
                pwvs.append(obs.pwv)
                for b in obs.num_valid_dets.dtype.names:
                    yields[b].append(obs.num_valid_dets[b][0])
                    hover[b].append(obs.obs_id)

    # yield and pwv trace
    for i, b in enumerate(sorted(yields)):
        fig.add_trace(
            go.Scatter(
                x=pwvs,
                y=yields[b],
                mode="markers",
                name=f"Valid Dets ({b})",
                marker=dict(
                    color=colors[i % len(colors)],
                    symbol=markers[i % len(markers)],
                    size=8,
                    line=dict(width=1, color="black"),
                ),
                hovertext=hover[b],
            )
        )

    # layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
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
            text="Valid Detectors and PWV",
            x=0.5,
            xanchor="center",
            y=0.97,
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
        title_text="Num Valid Dets",
        showgrid=True,
        gridcolor="lightgray",
    )

    return fig


def pwv_and_nep_vs_time(d: "ReportData", field_name: str = None) -> go.Figure:
    """PWV and NEPs as a function of time"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    neps = defaultdict(lambda: defaultdict(list))
    times = defaultdict(list)
    hover = defaultdict(list)

    # build a dict keyed off of bandpass for data from all obs
    for obs in d.obs_list:
        obs_time = dt.datetime.fromtimestamp(obs.start_time, tz=dt.timezone.utc)
        if field_name == "array":
            field = obs.array_nep
        elif field_name == "det":
            field = obs.det_nep
        if field.size > 0:
            for b in obs.array_nep.dtype.names:
                times[b].append(obs_time)
                hover[b].append(obs.obs_id)
                for i, pol in enumerate(field[b].dtype.names):
                    if pol not in neps[b].keys():
                        neps[b][pol] = []
                    neps[b][pol].append(field[b][0][i])

    # nep trace
    i = 0
    for b in sorted(neps):
        for pol, nep in neps[b].items():
            fig.add_trace(
                go.Scatter(
                    x=times[b],
                    y=nep,
                    mode="markers",
                    name=f"{pol.split('_')[-1]} ({b})",
                    marker=dict(
                        color=colors[i % len(colors)],
                        symbol=markers[i % len(markers)],
                        size=8,
                        line=dict(width=1, color="black"),
                    ),
                    hovertext=hover[b],
                ),
                secondary_y=False,
            )
            i += 1

    # PWV trace
    if d.obs_list:
        ds_factor = 10
        #pwvs = deepcopy(d.pwv[1][::ds_factor])
        #pwvs[(pwvs > 4) | (pwvs < .1)] = np.nan
        #ts = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in d.pwv[0][::ds_factor]]

        ts, pwvs = zip(*sorted(
            ((dt.datetime.fromtimestamp(o.start_time, tz=dt.timezone.utc), o.pwv) for o in d.obs_list)
        ))
        ts = list(ts)
        pwvs = list(pwvs)

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

    # layout
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
            text=f"{field_name.capitalize()} NEP and PWV",
            x=0.5,
            xanchor="center",
            y=0.92,
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
        title_text=r"$\rm{NEP\ [aW/\sqrt{Hz}]}$",
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


def nep_vs_pwv(d: "ReportData", longterm_data: Optional["ReportData"] = None,
               field_name: str = None) -> go.Figure:
    """NEP as a function of PWV. Current interval is plotted over longterm contours.
    """
    fig = go.Figure()

    if longterm_data is not None:
        long_neps = defaultdict(lambda: defaultdict(list))
        long_pwvs = []

        for obs in longterm_data.obs_list:
            if field_name == "array":
                field = obs.array_nep
            elif field_name == "det":
                field = obs.det_nep
            if field.size > 0:
                if np.isfinite(obs.pwv):
                    long_pwvs.append(obs.pwv)
                    for b in field.dtype.names:
                        for i, pol in enumerate(field[b].dtype.names):
                            if pol not in long_neps[b].keys():
                                long_neps[b][pol] = []
                            long_neps[b][pol].append(field[b][0][i])

        i = 0
        for b in sorted(long_neps):
            for pol, long_nep in long_neps[b].items():
                fig.add_trace(
                    go.Histogram2dContour(
                        x=long_pwvs,
                        y=long_nep,
                        colorscale=[[0, colors[i]], [1, colors[i]]],
                        contours_coloring="lines",
                        showscale=False,
                        opacity=0.7,
                        name=f"longterm {b}",
                    )
                )
                i += 1

    neps = defaultdict(lambda: defaultdict(list))
    pwvs = []
    hover = defaultdict(list)

    # build a dict keyed off of bandpass for data from all obs
    for obs in d.obs_list:
        if field_name == "array":
            field = obs.array_nep
        elif field_name == "det":
            field = obs.det_nep
        if field.size > 0:
            if np.isfinite(obs.pwv):
                pwvs.append(obs.pwv)
                for b in field.dtype.names:
                    hover[b].append(obs.obs_id)
                    for i, pol in enumerate(field[b].dtype.names):
                        if pol not in neps[b].keys():
                            neps[b][pol] = []
                        neps[b][pol].append(field[b][0][i])

    i = 0
    for b in sorted(neps):
        for pol, nep in neps[b].items():
            fig.add_trace(
                go.Scatter(
                    x=pwvs,
                    y=nep,
                    mode="markers",
                    name=f"{pol.split('_')[-1]} ({b})",
                    marker=dict(
                        color=colors[i % len(colors)],
                        symbol=markers[i % len(markers)],
                        size=8,
                        line=dict(width=1, color="black"),
                    ),
                    hovertext=hover[b],
                )
            )
            i += 1

    # layout
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
            text=f"{field_name.capitalize()} NEP and PWV",
            x=0.5,
            xanchor="center",
            y=0.92,
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
        title_text=r"$\rm{NEP\ [aW/\sqrt{Hz}]}$",
        showgrid=True,
        gridcolor="lightgray",
    )

    return fig

def cov_map_plot(map_png_file: str, embed:bool=True) -> str:
    import base64

    if map_png_file is None or not os.path.isfile(map_png_file):
        return "<p>Coverage map not found</p>"

    # Embed image as base64 (guaranteed to work)
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
        pie_width = 0.125
        x0s = [-0.00647517,  0.00316777,  0.00316777, -0.03335673, -0.02370855,
               -0.02371379,  0.02070833,  0.03023957,  0.03025179,  0.0204762 ,
                0.03025005,  0.03023957, -0.00637918,  0.00327947,  0.00325853,
               -0.03330437, -0.02369634, -0.02370855]

        y0s = [ 0.        ,  0.00560425, -0.00560425, -0.01579872, -0.00995536,
               -0.02117608, -0.01556659, -0.0099571 , -0.02117957,  0.01579872,
                0.02117957,  0.0099571 ,  0.03112446,  0.03673045,  0.02551671,
                0.01556834,  0.02117608,  0.01021716]

        wafers = ['c1_ws0', 'c1_ws1', 'c1_ws2',
                  'i1_ws0', 'i1_ws1', 'i1_ws2',
                  'i3_ws0', 'i3_ws1', 'i3_ws2',
                  'i4_ws0', 'i4_ws1', 'i4_ws2',
                  'i5_ws0', 'i5_ws1', 'i5_ws2',
                  'i6_ws0', 'i6_ws1', 'i6_ws2']

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

    norm_positions = get_normalized_positions(wafer_centers, x_range, y_range, 700, 700)

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
