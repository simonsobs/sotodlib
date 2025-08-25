import plotly
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import plotly.express as px
import datetime as dt
from copy import deepcopy
import ast
from collections import defaultdict

from typing import List, Tuple, TYPE_CHECKING, Dict, Any, Optional

from .report_data import ReportData, Footprint, obs_list_to_arr

band_colors = [
    "#D55E00",
    "#56B4E9",
]

def get_discrete_distinct_colors(n, reverse=False):
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


def wafer_obs_efficiency(d: ReportData, nsegs=2000) -> ObsEfficiencyPlots:
    if d.cfg.platform  == "lat":
        tube_slots = ["c1", "i1", "i3", "i4", "i5", "i6"]
        wafer_slots = ["ws0", "ws1", "ws2"]
        wafers = [f"{x}_{y}" for x in tube_slots for y in wafer_slots]
    elif d.cfg.platform in ["satp1", "satp2", "satp3"]:
        wafers = [f"ws{i} " for i in range(7)]
    else:
        raise ValueError(f"Uknown platform {d.cfg.platform}")

    nwafers = len(wafers)

    times = pd.date_range(d.cfg.start_time, d.cfg.stop_time, nsegs).to_pydatetime()
    tstamps= np.array([t.timestamp() for t in times])

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


def yield_vs_pwv(d: "ReportData", longterm_data: Optional["ReportData"] = None) -> go.Figure:
    fig = go.Figure()

    if longterm_data is not None:
        band_data = defaultdict(lambda: {"pwv": [], "yield": []})

        for obs in longterm_data.obs_list:
            if not obs.num_valid_dets:
                continue

            band_yields = ast.literal_eval(obs.num_valid_dets)

            for band, val in band_yields.items():
                if np.isfinite(val):
                    band_data[band]["pwv"].append(obs.pwv)
                    band_data[band]["yield"].append(val)

        band_index = 0
        for band, vals in band_data.items():
            fig.add_trace(
                go.Histogram2dContour(
                    x=vals["pwv"],
                    y=vals["yield"],
                    colorscale=[[0, band_colors[band_index]], [1, band_colors[band_index]]],
                    contours_coloring="lines",
                    showscale=False,
                    opacity=0.7,
                    name=f"longterm {band}"
                )
            )
            band_index +=1

    band_index = 0
    band_data = defaultdict(lambda: {'x': [], 'y': [], 'hover': []})

    for obs in d.obs_list:
        ts = dt.datetime.fromtimestamp(obs.start_time, tz=dt.timezone.utc).isoformat()
        if obs.num_valid_dets:
            for band, val in ast.literal_eval(obs.num_valid_dets).items():
                if not np.isfinite(val):
                    continue
                band_data[band]['x'].append(obs.pwv)
                band_data[band]['y'].append(val)
                band_data[band]['hover'].append(obs.obs_id)

    for band, data in band_data.items():
        color = band_colors[band_index % len(band_colors)]
        fig.add_trace(
            go.Scatter(
                x=data['x'],
                y=data['y'],
                mode="markers",
                name=f"{band}",
                marker=dict(color=color),
                hovertext=data['hover'],
            )
        )
        band_index += 1

    fig.update_layout(
        xaxis=dict(title="PWV"),
        yaxis=dict(title="Valid Dets"),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig



def pwv_and_yield_vs_time(d: "ReportData") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Collect yields per band
    yields_by_band = defaultdict(list)
    times_by_band = defaultdict(list)
    hover_by_band = defaultdict(list)

    for obs in d.obs_list:
        obs_time = dt.datetime.fromtimestamp(obs.start_time, tz=dt.timezone.utc)
        if obs.num_valid_dets:
            for band, val in ast.literal_eval(obs.num_valid_dets).items():
                times_by_band[band].append(obs_time)
                yields_by_band[band].append(val)
                hover_by_band[band].append(obs.obs_id)

    band_index = 0

    for band in sorted(yields_by_band):
        color = band_colors[band_index % len(band_colors)]
        fig.add_trace(
            go.Scatter(
                x=times_by_band[band],
                y=yields_by_band[band],
                mode='markers',
                name=f"Valid Dets ({band})",
                marker=dict(color=color),
                hovertext=hover_by_band[band],
            ),
            secondary_y=False,
        )
        band_index += 1

    # Add PWV trace
    ds_factor = 10
    pwvs = deepcopy(d.pwv[1][::ds_factor])
    pwvs[(pwvs > 4) | (pwvs < .1)] = np.nan
    ts = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in d.pwv[0][::ds_factor]]

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=pwvs,
            mode='lines',
            name="PWV",
            marker=dict(color="#E69F00"),
            opacity=0.5,
        ),
        secondary_y=True
    )

    # Axis labels and layout
    fig.update_yaxes(title_text='Num Valid Dets', secondary_y=False)
    fig.update_yaxes(title_text='PWV', secondary_y=True)
    fig.update_layout(
        margin={k: 0 for k in ['l', 'r', 't', 'b']},
        height=500
    )

    return fig


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
    
    wafers = sorted(set(fp.wafer for fp in d.source_footprints))
    sources = sorted(set(fp.source for fp in d.source_footprints))
    
    lookup = {
        (fp.wafer, fp.source): fp.obsids
        for fp in d.source_footprints
    }
    
    def make_obsid_links(obsids):
        if not obsids:
            return ""
        links = [
            f'<a href="{d.cfg.site_url}/site-pipeline/{d.cfg.platform}/preprocess/{obsid.split("_")[1][:5]}/{obsid}/" target="_blank">{obsid}</a>'
            for obsid in obsids
        ]
        return "<br>".join(links) + ("<br>" if len(links) == 1 else "")
    
    wafer_col = wafers
    source_cols = []
    for source in sources:
        col = []
        for wafer in wafers:
            obsids = lookup.get((wafer, source), [])
            col.append(make_obsid_links(obsids))
        source_cols.append(col)

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
