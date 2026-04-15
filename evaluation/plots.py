from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# Soft natural palette matching the UI theme
PALETTE = {
    'sage':    '#6e9f6e',
    'sage_lt': '#9dbf9d',
    'blue':    '#5a93cc',
    'blue_lt': '#88b4de',
    'amber':   '#c8922a',
    'amber_lt':'#e8b84a',
    'rose':    '#d06868',
    'rose_lt': '#e49898',
    'dust':    '#b5a08a',
    'dust_lt': '#cfc0ae',
    'teal':    '#5a9fa0',
    'lavender':'#9b8ec4',
    'bg':      '#fdfcfa',
    'surface': '#f6f3ee',
    'ink':     '#2e261c',
    'muted':   '#6b5a48',
    'line':    '#e4d9cc',
}

BAR_COLORS = [
    PALETTE['sage'], PALETTE['blue'], PALETTE['amber'], PALETTE['rose'],
    PALETTE['sage_lt'], PALETTE['blue_lt'], PALETTE['amber_lt'], PALETTE['rose_lt'],
]

FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'

BASE_LAYOUT = dict(
    font=dict(family=FONT_FAMILY, color=PALETTE['ink'], size=12),
    paper_bgcolor=PALETTE['bg'],
    plot_bgcolor=PALETTE['bg'],
    margin=dict(l=60, r=40, t=72, b=60),
    hoverlabel=dict(
        bgcolor=PALETTE['surface'],
        bordercolor=PALETTE['line'],
        font=dict(family=FONT_FAMILY, color=PALETTE['ink'], size=12),
    ),
    xaxis=dict(
        gridcolor=PALETTE['line'],
        linecolor=PALETTE['line'],
        zerolinecolor=PALETTE['line'],
        tickfont=dict(color=PALETTE['muted']),
        title_font=dict(color=PALETTE['muted']),
    ),
    yaxis=dict(
        gridcolor=PALETTE['line'],
        linecolor=PALETTE['line'],
        zerolinecolor=PALETTE['line'],
        tickfont=dict(color=PALETTE['muted']),
        title_font=dict(color=PALETTE['muted']),
    ),
)

PLOTLY_CONFIG = {'displayModeBar': True, 'displaylogo': False, 'responsive': True}

# Entrance animation transition applied to all figures
ENTRANCE_TRANSITION = dict(duration=700, easing='cubic-in-out')


def plot_feature_importance(feature_importances: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = list(feature_importances.keys())
    values = list(feature_importances.values())

    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(features))]

    fig = go.Figure(data=[go.Bar(
        x=features,
        y=values,
        marker=dict(color=colors, line=dict(color=PALETTE['bg'], width=1.5)),
        hovertemplate='<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>',
    )])

    layout = dict(**BASE_LAYOUT)
    layout['title'] = dict(
        text='Random Forest Feature Importance',
        font=dict(size=16, color=PALETTE['ink']),
        x=0.5, xanchor='center', pad=dict(b=12),
    )
    layout['xaxis_title'] = 'Feature'
    layout['yaxis_title'] = 'Importance'
    layout['height'] = 400
    layout['bargap'] = 0.35
    layout['transition'] = ENTRANCE_TRANSITION
    fig.update_layout(**layout)

    fig.write_html(output_path, config=PLOTLY_CONFIG)


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix)

    # Short flat abbreviations — no line breaks
    abbrev = {
        'dot_product': 'dot_prod',
        'linear_regression_inference': 'lin_reg',
        'logistic_regression_approx': 'log_reg',
        'mean': 'mean',
        'variance': 'variance',
    }
    short_labels = [abbrev.get(l, l[:10]) for l in labels]

    annotations = []
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            annotations.append(dict(
                x=col, y=row,
                text=str(array[row, col]),
                showarrow=False,
                font=dict(
                    color='white' if array[row, col] > array.max() / 2 else PALETTE['ink'],
                    size=13,
                )
            ))

    colorscale = [
        [0.0,  '#f0f5fb'],
        [0.25, '#b8d2ec'],
        [0.5,  '#6fa8d8'],
        [0.75, '#3a74b0'],
        [1.0,  '#1e4d80'],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=array,
        x=short_labels,
        y=short_labels,
        colorscale=colorscale,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        colorbar=dict(
            title=dict(text='Count', side='right'),
            thickness=14,
            len=0.85,
        ),
    ))

    layout = dict(**BASE_LAYOUT)
    layout['title'] = None
    layout['xaxis'] = dict(
        gridcolor=PALETTE['line'],
        linecolor=PALETTE['line'],
        zerolinecolor=PALETTE['line'],
        title=dict(text='Predicted', font=dict(size=12, color=PALETTE['muted'])),
        side='bottom',
        tickangle=0,
        tickfont=dict(size=11, color=PALETTE['muted']),
    )
    layout['yaxis'] = dict(
        gridcolor=PALETTE['line'],
        linecolor=PALETTE['line'],
        zerolinecolor=PALETTE['line'],
        title=dict(text='True', font=dict(size=12, color=PALETTE['muted'])),
        autorange='reversed',
        tickfont=dict(size=11, color=PALETTE['muted']),
    )
    layout['height'] = 380
    layout['margin'] = dict(l=80, r=60, t=24, b=80)
    layout['annotations'] = annotations
    layout['transition'] = ENTRANCE_TRANSITION
    fig.update_layout(**layout)

    fig.write_html(output_path, config=PLOTLY_CONFIG)


def plot_accuracy_comparison(results: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    accuracies = list(results.values())

    short_names = []
    for name in names:
        parts = name.split('_')
        if len(parts) == 2:
            short_names.append(f"{parts[0].capitalize()}<br>{parts[1].capitalize()}")
        else:
            short_names.append(name.replace('_', '<br>').title())

    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(short_names))]

    fig = go.Figure(data=[go.Bar(
        x=short_names,
        y=accuracies,
        marker=dict(color=colors, line=dict(color=PALETTE['bg'], width=1.5)),
        text=[f'{a:.2f}' for a in accuracies],
        textposition='outside',
        textfont=dict(size=11, color=PALETTE['muted']),
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.3f}<extra></extra>',
    )])

    layout = dict(**BASE_LAYOUT)
    layout['title'] = dict(
        text='Model Accuracy Comparison',
        font=dict(size=16, color=PALETTE['ink']),
        x=0.5, xanchor='center', pad=dict(b=12),
    )
    layout['xaxis_title'] = 'Model'
    layout['yaxis_title'] = 'Accuracy'
    layout['yaxis'] = dict(**BASE_LAYOUT['yaxis'], range=[0, 1.15])
    layout['height'] = 400
    layout['bargap'] = 0.3
    layout['transition'] = ENTRANCE_TRANSITION
    fig.update_layout(**layout)

    fig.write_html(output_path, config=PLOTLY_CONFIG)
