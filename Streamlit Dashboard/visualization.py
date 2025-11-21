# visualization.py
# Data visualization functions using Plotly, Matplotlib, and Seaborn

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import io
import base64

from config import *
from utils import is_categorical, get_feature_type, parse_csv_with_separator


def create_distribution_plot(data: pd.Series, feature_name: str, plot_type: str = 'auto'):
    """
    Create appropriate distribution plot based on data type.
    
    Args:
        data: Pandas Series with feature data
        feature_name: Name of the feature
        plot_type: 'auto', 'bar', 'pie', 'histogram', 'box'
        
    Returns:
        Plotly figure object
    """
    # Auto-detect plot type
    if plot_type == 'auto':
        feature_type = get_feature_type(data, feature_name)
        
        if feature_type == 'categorical':
            plot_type = 'bar'
        elif feature_type == 'discrete':
            plot_type = 'histogram'
        else:
            plot_type = 'histogram'
    
    # Create plot based on type
    if plot_type == 'bar':
        return create_bar_chart(data, feature_name)
    elif plot_type == 'pie':
        return create_pie_chart(data, feature_name)
    elif plot_type == 'histogram':
        return create_histogram(data, feature_name)
    elif plot_type == 'box':
        return create_box_plot(data, feature_name)
    else:
        return create_histogram(data, feature_name)


def create_bar_chart(data: pd.Series, feature_name: str):
    """Create bar chart for categorical data."""
    value_counts = data.value_counts().sort_index()
    
    # Check if we have labels for this feature
    labels_map = FEATURE_VALUE_LABELS.get(feature_name, {})
    
    # Map numeric values to text labels if available
    if labels_map:
        # Create list of labels in the same order as value_counts
        x_labels = [labels_map.get(int(val), str(val)) for val in value_counts.index]
    else:
        x_labels = value_counts.index
    
    fig = px.bar(
        x=x_labels,
        y=value_counts.values,
        labels={'x': feature_name, 'y': 'Count'},
        title=f'Distribution of {feature_name}',
        color=value_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=PLOT_HEIGHT,
        showlegend=False,
        xaxis_title=feature_name,
        yaxis_title='Count',
        xaxis={'tickangle': -45}  # Angle labels for better readability
    )
    
    return fig


def create_pie_chart(data: pd.Series, feature_name: str):
    """Create pie chart for categorical data."""
    value_counts = data.value_counts()
    
    # Check if we have labels for this feature
    labels_map = FEATURE_VALUE_LABELS.get(feature_name, {})
    
    # Map numeric values to text labels if available
    if labels_map:
        names_labels = [labels_map.get(int(val), str(val)) for val in value_counts.index]
    else:
        names_labels = value_counts.index
    
    fig = px.pie(
        values=value_counts.values,
        names=names_labels,
        title=f'Distribution of {feature_name}'
    )
    
    fig.update_layout(height=PLOT_HEIGHT)
    
    return fig


def create_histogram(data: pd.Series, feature_name: str):
    """Create histogram for numerical data."""
    fig = px.histogram(
        data,
        x=data.values,
        nbins=30,
        title=f'Distribution of {feature_name}',
        labels={'x': feature_name, 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        height=PLOT_HEIGHT,
        xaxis_title=feature_name,
        yaxis_title='Count',
        showlegend=False
    )
    
    # Add mean line
    mean_val = data.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top"
    )
    
    return fig


def create_box_plot(data: pd.Series, feature_name: str):
    """Create box plot for numerical data."""
    fig = px.box(
        y=data.values,
        title=f'Box Plot of {feature_name}',
        labels={'y': feature_name}
    )
    
    fig.update_layout(
        height=PLOT_HEIGHT,
        showlegend=False
    )
    
    return fig


def create_correlation_heatmap(training_data_path: str = TRAINING_DATA_PATH):
    """
    Create correlation heatmap for all features including Target.
    
    Args:
        training_data_path: Path to training data CSV
        
    Returns:
        Plotly figure object
    """
    try:
        # Load data
        df = parse_csv_with_separator(training_data_path, CSV_SEPARATOR)
        
        # Find target column
        target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
        
        if target_col is None:
            raise ValueError("Target column not found")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Ensure target is included
        if target_col not in numeric_df.columns:
            numeric_df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale=HEATMAP_COLORSCALE,
            aspect="auto",
            title="Feature Correlation Heatmap"
        )
        
        fig.update_layout(
            height=800,
            width=900,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {e}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig


def create_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 20):
    """
    Create bar chart for feature importance ranking.
    
    Args:
        importance_df: DataFrame with columns ['feature', 'importance']
        top_n: Number of top features to display
        
    Returns:
        Plotly figure object
    """
    # Get top N features
    top_features = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=max(500, top_n * 25),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_risk_distribution_pie(risk_counts: dict):
    """
    Create pie chart for risk level distribution (including None/not predicted).
    
    Args:
        risk_counts: Dictionary with risk levels and counts
        
    Returns:
        Plotly figure object
    """
    # Include all risk levels (including None), filter only zero counts
    filtered_counts = {k: v for k, v in risk_counts.items() if v > 0}
    
    if not filtered_counts:
        # No students at all
        fig = go.Figure()
        fig.add_annotation(
            text="No students found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Use all risk colors including 'None' (green)
    colors = {
        'None': RISK_COLORS['None'],
        'Mild': RISK_COLORS['Mild'],
        'Moderate': RISK_COLORS['Moderate'],
        'Severe': RISK_COLORS['Severe']
    }
    
    labels = list(filtered_counts.keys())
    values = list(filtered_counts.values())
    color_list = [colors.get(label, '#999999') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=color_list),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Student Distribution by Risk Level",
        height=PLOT_HEIGHT
    )
    
    return fig


def create_risk_bar_chart(risk_counts: dict):
    """
    Create bar chart for risk level counts (including None/not predicted).
    
    Args:
        risk_counts: Dictionary with risk levels and counts
        
    Returns:
        Plotly figure object
    """
    # Include all risk levels (including 'None')
    filtered_counts = {k: v for k, v in risk_counts.items() if v > 0}
    
    labels = list(filtered_counts.keys())
    values = list(filtered_counts.values())
    colors_list = [RISK_COLORS.get(label, '#999999') for label in labels]
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors_list),
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Student Count by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Number of Students",
        height=PLOT_HEIGHT,
        showlegend=False
    )
    
    return fig


def create_prediction_probability_histogram(students_df: pd.DataFrame):
    """
    Create histogram of prediction probabilities.
    
    Args:
        students_df: DataFrame with student data including prediction_probability
        
    Returns:
        Plotly figure object
    """
    # Filter students with predictions
    predicted_df = students_df[students_df['prediction_probability'].notna()]
    
    if len(predicted_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No predictions available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = px.histogram(
        predicted_df,
        x='prediction_probability',
        nbins=30,
        title='Distribution of Dropout Probabilities',
        labels={'prediction_probability': 'Dropout Probability', 'count': 'Number of Students'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add risk threshold lines
    fig.add_vline(x=0.5, line_dash="dash", line_color="yellow", annotation_text="Mild Threshold")
    fig.add_vline(x=0.65, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
    fig.add_vline(x=0.85, line_dash="dash", line_color="red", annotation_text="Severe Threshold")
    
    fig.update_layout(
        height=PLOT_HEIGHT,
        xaxis_title="Dropout Probability",
        yaxis_title="Number of Students",
        showlegend=False
    )
    
    return fig


def create_summary_metrics_cards(stats: dict):
    """
    Create metric cards for dashboard summary.
    
    Args:
        stats: Dictionary with various statistics
        
    Returns:
        HTML string for metric cards
    """
    html = '<div style="display: flex; gap: 20px; flex-wrap: wrap;">'
    
    metrics = [
        {
            'title': 'Total Students',
            'value': stats.get('total_students', 0),
            'icon': '👨‍🎓',
            'color': '#1f77b4'
        },
        {
            'title': 'Predicted Students',
            'value': stats.get('predicted_students', 0),
            'icon': '🎯',
            'color': '#2ca02c'
        },
        {
            'title': 'At-Risk (Mild)',
            'value': stats.get('risk_counts', {}).get('Mild', 0),
            'icon': '🟡',
            'color': RISK_COLORS['Mild']
        },
        {
            'title': 'At-Risk (Moderate)',
            'value': stats.get('risk_counts', {}).get('Moderate', 0),
            'icon': '🟠',
            'color': RISK_COLORS['Moderate']
        },
        {
            'title': 'At-Risk (Severe)',
            'value': stats.get('risk_counts', {}).get('Severe', 0),
            'icon': '🔴',
            'color': RISK_COLORS['Severe']
        },
        {
            'title': 'Total Educators',
            'value': stats.get('total_educators', 0),
            'icon': '👨‍🏫',
            'color': '#9467bd'
        }
    ]
    
    for metric in metrics:
        html += f'''
        <div style="
            background: linear-gradient(135deg, {metric['color']}22, {metric['color']}44);
            border-left: 4px solid {metric['color']};
            padding: 20px;
            border-radius: 8px;
            min-width: 200px;
            flex: 1;
        ">
            <div style="font-size: 36px; margin-bottom: 10px;">{metric['icon']}</div>
            <div style="font-size: 32px; font-weight: bold; color: {metric['color']};">{metric['value']}</div>
            <div style="font-size: 14px; color: #666; margin-top: 5px;">{metric['title']}</div>
        </div>
        '''
    
    html += '</div>'
    return html


def plot_training_vs_input_comparison(training_data: pd.Series, input_data: pd.Series, feature_name: str):
    """
    Create comparison plot between training and input data distributions.
    
    Args:
        training_data: Training data series
        input_data: Input data series
        feature_name: Name of the feature
        
    Returns:
        Plotly figure object
    """
    # Check if we have labels for this feature
    labels_map = FEATURE_VALUE_LABELS.get(feature_name, {})
    
    # If we have labels, map the data
    if labels_map:
        # Map training data
        training_data_mapped = training_data.map(lambda x: labels_map.get(int(x) if pd.notna(x) else x, str(x)))
        # Map input data
        input_data_mapped = input_data.map(lambda x: labels_map.get(int(x) if pd.notna(x) else x, str(x)))
    else:
        training_data_mapped = training_data
        input_data_mapped = input_data
    
    fig = go.Figure()
    
    # Training data histogram
    fig.add_trace(go.Histogram(
        x=training_data_mapped,
        name='Training Data',
        opacity=0.7,
        marker_color='#1f77b4'
    ))
    
    # Input data histogram
    fig.add_trace(go.Histogram(
        x=input_data_mapped,
        name='Input Data',
        opacity=0.7,
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title=f'Comparison: {feature_name}',
        xaxis_title=feature_name,
        yaxis_title='Count',
        barmode='overlay',
        height=PLOT_HEIGHT,
        legend=dict(x=0.7, y=0.95),
        xaxis={'tickangle': -45}  # Angle labels for better readability
    )
    
    return fig


if __name__ == '__main__':
    # Test mode
    print("=== Visualization Module Test ===")
    
    # Test with sample data
    sample_data = pd.Series(np.random.randint(0, 10, 100))
    
    try:
        fig = create_distribution_plot(sample_data, "Test Feature")
        print("✅ Distribution plot created")
    except Exception as e:
        print(f"❌ Error: {e}")
