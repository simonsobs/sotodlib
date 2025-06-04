import plotly
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass
import plotly.express as px
import datetime as dt
from copy import deepcopy

from typing import List, Tuple, TYPE_CHECKING, Dict, Any, Optional

from .report_data import ReportData, Footprint, obs_list_to_arr


@dataclass
class Colors:
    oper: str = DEFAULT_PLOTLY_COLORS[0]
    obs: str = DEFAULT_PLOTLY_COLORS[1]
    idle: str = DEFAULT_PLOTLY_COLORS[2]
    cmb: str =  DEFAULT_PLOTLY_COLORS[3]
    cal: str =  DEFAULT_PLOTLY_COLORS[4]

    def __getitem__(self, name) -> str:
        return getattr(self, name)


COLORS = Colors()


@dataclass
class ObsEfficiencyPlots:
    pie: go.Figure
    heatmap: go.Figure


def wafer_obs_efficiency(d: ReportData, nsegs=2000) -> ObsEfficiencyPlots:
    if d.cfg.platform  == "lat":
        nwafers = 18
    else:
        nwafers = 7
    data = np.zeros((nwafers, nsegs), dtype=int)
    times = pd.date_range(d.cfg.start_time, d.cfg.stop_time, nsegs).to_pydatetime()
    tstamps= np.array([t.timestamp() for t in times])

    obs_types = ["idle", "obs", "cmb", "cal", "oper"]
    obs_values = {k: i for i, k in enumerate(obs_types)}
    colorscale: List[Tuple[float, str]] = []
    ntypes = len(obs_types)
    for i, t in enumerate(obs_types):
        colorscale.extend([(i / ntypes, COLORS[t]), ((i + 1) / ntypes, COLORS[t])])

    for o in d.obs_list:
        m = np.logical_and.reduce([tstamps > o.start_time, tstamps < o.stop_time])
        if d.cfg.platform == "lat":
            wafer_slots_list = o.stream_ids_list.split(",")
        else:
            wafer_slots_list = o.wafer_slots_list.split(",")
        for wafer_slot in wafer_slots_list:
            idx = int(wafer_slot.strip()[-1])
            if o.obs_type == "obs":
                if o.obs_subtype in obs_types:
                    data[idx][m] = obs_values[o.obs_subtype]
                else:
                    data[idx][m] = obs_values[o.obs_type]
            else:
                data[idx][m] = obs_values[o.obs_type]
    # Compile data for pie chart
    unique_vals, counts = np.unique(data, return_counts=True)
    percentages = counts / counts.sum() * 100
    labels = [obs_types[i] for i in unique_vals]
    pie_colors = [COLORS[t] for t in labels]

    ys = [f"ws{i} " for i in range(7)]
    heatmap = go.Figure(
        data=go.Heatmap(
            z=data,
            x=times,
            y=ys,
            colorscale=colorscale,
            colorbar=dict(tickvals=[0, 1, 2, 3, 4], ticktext=["None", "Obs", "CMB", "Cal", "Oper"]),
        ),
    )
    heatmap.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=150)

    pie = go.Figure(
        data=go.Pie(
            labels=labels,
            values=percentages,
            textinfo="label+percent",
        ),
    )
    pie.update_traces(marker=dict(colors=pie_colors))
    pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=150)

    return ObsEfficiencyPlots(pie=pie, heatmap=heatmap)


def yield_vs_pwv(d: "ReportData", longterm_data: Optional["ReportData"]=None) -> go.Figure():
    fig = go.Figure()
    hovertemplate = "<br>".join(
        [
            "<b>PWV: </b> %{x}",
            "<b>Biased Dets: </b> %{y}",
            "<b>ObsId: </b> %{customdata[0]}",
            "<b>Datetime: </b> %{customdata[1]}",
        ]
    )
    if longterm_data is not None:
        arr = obs_list_to_arr(longterm_data.obs_list)
        xs = arr["pwv"]
        ys = arr["num_valid_dets"]
        fig.add_trace(
            go.Histogram2dContour(
                x=xs,
                y=ys,
                colorscale="Blues",
                contours_coloring="heatmap",
            )
        )

    obs_arr = obs_list_to_arr(d.obs_list)
    fig.add_trace(
        go.Scatter(
            x=obs_arr["pwv"].tolist(),
            y=obs_arr["num_valid_dets"].tolist(),
            customdata=np.column_stack([obs_arr["obs_id"].astype(str), obs_arr["start_time"]]),
            mode="markers",
            marker=dict(
                opacity=1,
                color="red",
            ),
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        xaxis=dict(title="PWV"),
        yaxis=dict(title="Valid Dets"),
        # height=500,
        # width=500,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def pwv_and_yield_vs_time(d: "ReportData") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    times, yields, htext = [], [], []
    for obs in d.obs_list:
        times.append(dt.datetime.fromtimestamp(obs.start_time, tz=dt.timezone.utc))
        yields.append(obs.num_valid_dets)
        htext.append(obs.obs_id)

    ds_factor=10
    pwvs = deepcopy(d.pwv[1][::ds_factor])
    pwvs[(pwvs > 4) | (pwvs < .1)] = np.nan
    ts = [dt.datetime.fromtimestamp(t) for t in d.pwv[0][::ds_factor]]
    fig.add_trace(
        go.Scatter(
            x=ts, y=pwvs,
            mode='lines',
            name="PWV",
            opacity=0.5,
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=yields,
            mode='markers',
            name="Num Valid Dets",
            hovertext=htext,
        ),
        secondary_y=False,
    )
    margin = {k: 50 for k in ['l', 'r', 't', 'b']}
    fig.update_yaxes(title_text='Num Valid Dets', secondary_y=False)
    fig.update_yaxes(title_text='PWV', secondary_y=True)
    fig.update_layout(
        margin=margin
    )
    return fig


def cal_footprints(d: "ReportData") -> go.Figure:

    if d.cfg.platform in ["satp1", "satp2", "satp3"]:
        wafer_rad = 0.10488
        wafer_sep = 0.219911

        angs = np.arange(-np.pi / 2, 3 * np.pi / 2, np.pi / 3)
        x0s = [0] + [np.cos(a) * wafer_sep for a in angs]
        y0s = [0] + [np.sin(a) * wafer_sep for a in angs]

    elif d.cfg.platform == "lat":
        wafer_rad = 0.005236
        x0s = [-0.00647517,  0.00316777,  0.00316777, -0.03335673, -0.02370855,
               -0.02371379,  0.02070833,  0.03023957,  0.03025179,  0.0204762 ,
                0.03025005,  0.03023957, -0.00637918,  0.00327947,  0.00325853,
               -0.03330437, -0.02369634, -0.02370855]

        y0s = [ 0.        ,  0.00560425, -0.00560425, -0.01579872, -0.00995536,
               -0.02117608, -0.01556659, -0.0099571 , -0.02117957,  0.01579872,
                0.02117957,  0.0099571 ,  0.03112446,  0.03673045,  0.02551671,
                0.01556834,  0.02117608,  0.01021716]

    def hex(x0, y0):
        angs = np.linspace(0, 2 * np.pi, 7)
        pts = np.array(
            [
                x0 + wafer_rad * np.cos(angs),
                y0 + wafer_rad * np.sin(angs),
            ]
        )
        return pts

    fig = go.Figure()

    for x0, y0 in zip(x0s, y0s):
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

    ntargets = 0
    target_colors = {}
    if d.cal_footprints is not None:
        for fp in d.cal_footprints:
            if fp.bounds is None:
                continue
            if fp.target not in target_colors:
                target_colors[fp.target] = DEFAULT_PLOTLY_COLORS[ntargets]
                ntargets += 1
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        fill="toself",
                        name=fp.target,
                        mode="lines",
                        marker=dict(color=target_colors[fp.target]),
                        fillcolor=target_colors[fp.target],
                        opacity=0.5,
                        legendgroup=fp.target,
                        showlegend=True,
                    )
                )

        xs, ys = fp.bounds.T
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                name=fp.obs_id,
                mode="lines",
                marker=dict(color=target_colors[fp.target]),
                fillcolor=target_colors[fp.target],
                opacity=0.2,
                legendgroup=fp.target,
                showlegend=False,
            )
        )

    # margin = 50
    layout: Dict[str, Any] = dict(
        xaxis_title="Xi",
        yaxis_title="Eta",
    )
    margin=0
    if margin is not None:
        layout["margin"] = dict(t=margin, b=margin, l=margin, r=margin)
    layout['margin']['t'] = 50
    fig.update_layout(layout)
    return fig
