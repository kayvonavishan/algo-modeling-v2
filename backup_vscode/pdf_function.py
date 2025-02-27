###############
# CREATE PDF
################

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import DMatrix
from matplotlib.backends.backend_pdf import PdfPages
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib
matplotlib.use("Agg")

def create_label_distribution_plots(df_train, df_test):
    """
    Creates pie charts showing the distribution of labels in train and test sets.
    
    Parameters:
    -----------
    df_train : pandas.DataFrame
        Training data with 'label' column
    df_test : pandas.DataFrame
        Test data with 'label' column
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing two pie charts side by side
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Get label distributions
    train_dist = df_train['label'].value_counts().sort_index()
    test_dist = df_test['label'].value_counts().sort_index()
    
    # Calculate percentages
    train_pct = (train_dist / len(df_train) * 100).round(1)
    test_pct = (test_dist / len(df_test) * 100).round(1)
    
    # Custom labels with percentages
    train_labels = [f'Label {label}: {pct}%' for label, pct in zip(train_dist.index, train_pct)]
    test_labels = [f'Label {label}: {pct}%' for label, pct in zip(test_dist.index, test_pct)]
    
    # Create color map
    num_classes = max(len(train_dist), len(test_dist))
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    # Plot train distribution
    ax1.pie(train_dist, labels=train_labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('Label Distribution in Training Set\n'
                 f'(Total Samples: {len(df_train):,})', pad=20)
    
    # Plot test distribution
    ax2.pie(test_dist, labels=test_labels, colors=colors, autopct='%1.1f%%')
    ax2.set_title('Label Distribution in Test Set\n'
                 f'(Total Samples: {len(df_test):,})', pad=20)
    
    # Add overall title
    plt.suptitle('Label Distribution Comparison', fontsize=14, y=1.05)
    
    plt.tight_layout()
    
    return fig


def create_labels_over_time_plot(df_features_master_for_model):
    """
    Creates a matplotlib figure showing the distribution of labels over time bins.
    
    Parameters:
    -----------
    df_features_master_for_model : pd.DataFrame
        DataFrame containing the features and labels with datetime index
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the labels over time visualization
    """
    # Ensure the index is of datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_features_master_for_model.index):
        df_features_master_for_model.index = pd.to_datetime(df_features_master_for_model.index)
    
    # Copy the dataframe for label analysis
    df_copy = df_features_master_for_model.copy()
    df_copy['date'] = df_copy.index
    
    # Split into 10 bins
    df_copy['bin'] = pd.qcut(df_copy.index, q=10, labels=False)
    
    # Calculate percentages
    bin_summary = df_copy.groupby('bin')['label'].value_counts(normalize=True).unstack().fillna(0) * 100
    bin_summary = bin_summary.apply(np.floor).astype(int)
    
    # Get date ranges
    bin_date_range = df_copy.groupby('bin').agg(
        min_date=('date', 'min'),
        max_date=('date', 'max')
    ).reset_index()
    
    bin_date_range['date_range'] = (
        bin_date_range['min_date'].dt.strftime('%b %Y') + 
        ' - ' + 
        bin_date_range['max_date'].dt.strftime('%b %Y')
    )
    
    # Prepare data for plotting
    bin_summary_df = bin_summary.reset_index().merge(
        bin_date_range[['bin', 'date_range']], 
        on='bin'
    )
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get unique labels and create color map
    labels = sorted(bin_summary.columns)
    num_labels = len(labels)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_labels))
    
    # Plot stacked bars
    bottom = np.zeros(len(bin_summary_df))
    
    for i, label in enumerate(labels):
        values = bin_summary_df[label]
        bars = ax.bar(
            bin_summary_df['date_range'], 
            values, 
            bottom=bottom,
            label=f'Label {label}',
            color=colors[i]
        )
        bottom += values
        
        # Add percentage labels on the bars
        for j, bar in enumerate(bars):
            if values[j] > 0:  # Only add label if percentage is greater than 0
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bottom[j] - height/2,
                    f'{values[j]}%',
                    ha='center',
                    va='center',
                    rotation=0 if height > 10 else 90,  # Rotate text if bar is thin
                    color='white' if height > 20 else 'black'  # Adjust text color based on bar height
                )
    
    # Customize the plot
    plt.title('Percentage of Each Label in Each Time Bin', pad=20)
    plt.xlabel('Date Range (Month Year)')
    plt.ylabel('Percentage (%)')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def create_debug_text_pages(debug_output, max_lines_per_page=50, max_pages=5):
    """
    Creates matplotlib figures for debug text, limited to max_pages.
    """
    figures = []
    debug_output.finalize()
    
    # Calculate total available lines
    total_available_lines = max_lines_per_page * max_pages
    
    # Process sections while respecting page limit
    current_total_lines = 0
    
    for section_title, section_content in debug_output.sections:
        text_content = ''.join(section_content)
        lines = text_content.split('\n')
        
        # Check if adding this section would exceed limit
        if current_total_lines + len(lines) > total_available_lines:
            # Truncate the section
            remaining_lines = total_available_lines - current_total_lines
            if remaining_lines > 0:
                lines = lines[:remaining_lines]
                lines.append("\n[Output truncated due to page limit...]")
            else:
                continue
        
        current_total_lines += len(lines)
        
        # Split into pages
        for i in range(0, len(lines), max_lines_per_page):
            if len(figures) >= max_pages:
                break
                
            page_lines = lines[i:min(i + max_lines_per_page, len(lines))]
            
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            
            page_text = f"{section_title}\n\n" + '\n'.join(page_lines)
            
            plt.text(0.1, 0.95, page_text,
                    fontsize=8,
                    fontfamily='monospace',
                    verticalalignment='top',
                    transform=plt.gca().transAxes)
            
            figures.append(fig)
            
            if len(figures) >= max_pages:
                break
    
    return figures

from typing import List
from matplotlib.figure import Figure
from typing import List
from matplotlib.figure import Figure
def create_probability_return_histogram(symbol_trades, symbol, highest_class, filtered_for_highest=False, exclude_highest=False):
    """Helper function to create histogram with consistent styling.
    
    Parameters:
        symbol_trades: DataFrame containing trade data
        symbol: Trading symbol identifier
        highest_class: The highest prediction class
        filtered_for_highest: Boolean to filter only for highest class predictions
        exclude_highest: Boolean to filter out highest class predictions
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Input validation - can't have both filters active
    if filtered_for_highest and exclude_highest:
        raise ValueError("Cannot both filter for and exclude highest class simultaneously")

    # Apply filtering based on parameters
    if filtered_for_highest:
        symbol_trades = symbol_trades[symbol_trades['predicted_class'] == highest_class]
        if len(symbol_trades) == 0:
            return None  # Return None if no trades match the filter
    elif exclude_highest:
        symbol_trades = symbol_trades[symbol_trades['predicted_class'] != highest_class]
        if len(symbol_trades) == 0:
            return None  # Return None if no trades match the filter

    # Get the actual range of probabilities
    min_prob = symbol_trades['highest_class_probability'].min()
    max_prob = symbol_trades['highest_class_probability'].max()

    # Create 5 bins with approximately equal counts using qcut
    n_bins = 5
    symbol_trades['prob_bin'] = pd.qcut(
        symbol_trades['highest_class_probability'], 
        q=n_bins,
        labels=[f'Quintile {i+1}' for i in range(n_bins)]
    )

    # Calculate statistics for each bin
    bin_stats = symbol_trades.groupby('prob_bin').agg({
        'return_percentage': ['mean', 'count', 'std'],
        'highest_class_probability': ['min', 'max']
    }).round(3)

    bin_stats.columns = ['mean_return', 'count', 'std', 'min_prob', 'max_prob']

    # Create more informative bin labels showing the probability ranges
    bin_stats['label'] = bin_stats.apply(
        lambda x: f'{x["min_prob"]:.3f}-{x["max_prob"]:.3f}', axis=1
    )

    # Plot histogram-style bars
    bars = ax.bar(range(len(bin_stats)), bin_stats['mean_return'])

    # Add value labels on top of bars
    for i, row in enumerate(bin_stats.itertuples()):
        label_text = f'Return: {row.mean_return:.2f}%\nCount: {row.count}'
        ax.text(i, row.mean_return, label_text, 
                ha='center', 
                va='bottom' if row.mean_return >= 0 else 'top')

    # Set x-axis labels with rotation for better readability
    plt.xticks(range(len(bin_stats)), bin_stats['label'], rotation=45, ha='right')

    # Set appropriate title based on filtering
    if filtered_for_highest:
        title_prefix = 'Highest Class Only: '
    elif exclude_highest:
        title_prefix = 'Excluding Highest Class: '
    else:
        title_prefix = 'All Classes: '

    ax.set_title(f'{title_prefix}Average Return (%) by Highest Class Probability - {symbol}\n' +
                 f'Range: [{min_prob:.3f}, {max_prob:.3f}]')
    ax.set_xlabel('Highest Class Probability Range')
    ax.set_ylabel('Average Return (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def generate_class_metric_plots(trades_df: pd.DataFrame, initial_capital: float = 10000) -> List[Figure]:
   figs = []
   
   for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
       
        metrics = {
            'Average Winning Trade Return (%)': lambda x: x[x['return'] > 0]['return_percentage'].mean(),
            'Average Losing Trade Return (%)': lambda x: x[x['return'] <= 0]['return_percentage'].mean(),
            'Average Return (%)': lambda x: x['return_percentage'].mean(),
            'Number of Trades': lambda x: x['return_percentage'].count(),
            'Win Rate (%)': lambda x: (x['return'] > 0).mean() * 100,
            'Std Deviation (%)': lambda x: x['return_percentage'].std(),
            'Average Hold Time (Hours)': lambda x: x['hold_time_hours'].mean(),
            'Average Position Size (%)': lambda x: (x['position_size'] / x['capital'] * 100).mean()  # Changed this line
        }
       
        for title, func in metrics.items():
           fig, ax = plt.subplots(figsize=(10, 6))
           class_metrics = symbol_trades.groupby('predicted_class').apply(func)
           bars = class_metrics.plot(kind='bar', ax=ax)
           
           for i, v in enumerate(class_metrics):
               ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
               
           ax.set_title(f'{title} by Predicted Class - {symbol} (Full Test Set)')
           ax.set_xlabel('Predicted Class')
           ax.set_ylabel(title)
           plt.grid(True, alpha=0.3)
           plt.tight_layout()
           figs.append(fig)


        # Get the highest possible class number
        highest_class = max(symbol_trades['predicted_class'])
        
        # Create histogram for all trades
        fig_all = create_probability_return_histogram(symbol_trades, symbol, highest_class, filtered_for_highest=False, exclude_highest=False)
        if fig_all is not None:
            figs.append(fig_all)

        # Create histogram for highest class trades only
        fig_highest = create_probability_return_histogram(symbol_trades, symbol, highest_class, filtered_for_highest=True, exclude_highest=False)
        if fig_highest is not None:
            figs.append(fig_highest)

        # Create histogram excluding highest class trades
        fig_exclude_highest = create_probability_return_histogram(symbol_trades, symbol, highest_class, filtered_for_highest=False, exclude_highest=True)
        if fig_exclude_highest is not None:
            figs.append(fig_exclude_highest)
   
   return figs

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import List 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List

def generate_time_series_metric_plot(trades_df: pd.DataFrame, initial_capital: float = 10000, window: int = 200) -> List[Figure]:
    figs = []
    
    limited_y_metrics = [
        'Average Winning Trade Return (%)',
        'Average Losing Trade Return (%)',
        'Average Return (%)'
    ]
    
    for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        
        # Create a complete date range from earliest to latest buy_timestamp
        date_range = pd.date_range(
            start=symbol_trades['buy_timestamp'].min(),
            end=symbol_trades['buy_timestamp'].max(),
            freq='D'
        )
        
        metrics = {
            'Average Winning Trade Return (%)': lambda x: x[x['return'] > 0]['return_percentage'].mean(),
            'Average Losing Trade Return (%)': lambda x: x[x['return'] <= 0]['return_percentage'].mean(),
            'Average Return (%)': lambda x: x['return_percentage'].mean(),
            'Average Position Size (%)': lambda x: (x['position_size'] / x['capital'] * 100).mean()
        }
        
        for title, metric_func in metrics.items():
            # Create figure with adjusted size to accommodate legend
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111)
            
            # Group by (buy_timestamp, predicted_class) then compute the metric
            grouped = symbol_trades.groupby(['buy_timestamp', 'predicted_class']).apply(metric_func).reset_index(name='metric')
            
            # Pivot so that each class is a column, then reindex by the daily date_range
            pivot_df = grouped.pivot(index='buy_timestamp', columns='predicted_class', values='metric')
            pivot_df = pivot_df.reindex(date_range)
            
            # 1) Forward-fill missing days so we don't have NaN
            pivot_df = pivot_df.ffill()
            
            # 2) Apply an EMA (exponential weighted average) over the chosen window
            ema_df = pivot_df.ewm(span=window, adjust=False).mean()
            
            # Plot lines with distinct colors
            num_classes = ema_df.shape[1]
            palette = sns.color_palette("husl", num_classes)
            
            for i, col in enumerate(ema_df.columns):
                ax.plot(
                    ema_df.index, 
                    ema_df[col], 
                    label=f'Class {col}', 
                    color=palette[i], 
                    linewidth=2
                )
            
            # Optionally limit y-axis for certain metrics
            if title in limited_y_metrics:
                ax.set_ylim(-3.0, 3.0)
            
            # Customize plot
            ax.set_title(f'{title} (EMA) by Predicted Class - {symbol}', fontsize=14, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Add legend off to the side
            ax.legend(
                title='Predicted Class', 
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                title_fontsize=12
            )
            
            # Adjust layout to prevent legend cutoff
            plt.subplots_adjust(right=0.85)
            
            figs.append(fig)
    
    return figs

def generate_time_series_metric_plot_long_short(trades_df: pd.DataFrame, initial_capital: float = 10000, window: int = 200) -> List[Figure]:
    figs = []
    
    limited_y_metrics = [
        'Average Winning Trade Return (%)',
        'Average Losing Trade Return (%)',
        'Average Return (%)'
    ]
    
    for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        
        # Create a complete date range from earliest to latest buy_timestamp
        date_range = pd.date_range(
            start=symbol_trades['buy_timestamp'].min(),
            end=symbol_trades['buy_timestamp'].max(),
            freq='D'
        )
        
        metrics = {
            'Average Winning Trade Return (%)': lambda x: x[x['return'] > 0]['return_percentage'].mean(),
            'Average Losing Trade Return (%)': lambda x: x[x['return'] <= 0]['return_percentage'].mean(),
            'Average Return (%)': lambda x: x['return_percentage'].mean(),
            'Average Position Size (%)': lambda x: (x['position_size'] / x['capital'] * 100).mean()
        }
        
        for title, metric_func in metrics.items():
            # Create figure with adjusted size to accommodate legend
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111)
            
            # Group by (buy_timestamp, is_short) then compute the metric
            grouped = symbol_trades.groupby(['buy_timestamp', 'is_short']).apply(metric_func).reset_index(name='metric')
            
            # Pivot so that long/short are columns
            pivot_df = grouped.pivot(index='buy_timestamp', columns='is_short', values='metric')
            pivot_df = pivot_df.reindex(date_range)
            
            # Rename columns for clarity
            pivot_df.columns = ['Long', 'Short']
            
            # Forward-fill missing days
            pivot_df = pivot_df.ffill()
            
            # Apply EMA
            ema_df = pivot_df.ewm(span=window, adjust=False).mean()
            
            # Plot lines with distinct colors
            ax.plot(ema_df.index, ema_df['Long'], label='Long', color='green', linewidth=2)
            ax.plot(ema_df.index, ema_df['Short'], label='Short', color='red', linewidth=2)
            
            # Optionally limit y-axis for certain metrics
            if title in limited_y_metrics:
                ax.set_ylim(-3.0, 3.0)
            
            # Customize plot
            ax.set_title(f'{title} (EMA) by Position Type - {symbol}', fontsize=14, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Add legend
            ax.legend(
                title='Position Type', 
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                title_fontsize=12
            )
            
            # Adjust layout to prevent legend cutoff
            plt.subplots_adjust(right=0.85)
            
            figs.append(fig)
    
    return figs



def create_price_plot(filtered_df, start_date, end_date):
    """
    Creates a matplotlib figure showing close price with labels and close raw,
    handling time series gaps by using categorical x-axis.
    Legend labels are sorted in ascending order.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create categorical x-axis values
    timestamp_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in filtered_df.index]
    x_categorical = np.arange(len(timestamp_list))
    
    # Plot close price and close raw using categorical x positions
    line1 = ax.plot(x_categorical, filtered_df['close'], color='black', label='Close Price', linewidth=1)[0]
    line2 = ax.plot(x_categorical, filtered_df['close_raw'], color='blue', linestyle=':', label='Close Raw', linewidth=1)[0]
    
    # Prepare label colors
    unique_labels = filtered_df['label'].unique()
    valid_labels = unique_labels[(unique_labels != -999) & (unique_labels != -100)]
    sorted_labels = np.sort(valid_labels)
    negative_labels = sorted_labels[sorted_labels < 0]
    positive_labels = sorted_labels[sorted_labels > 0]
    
    # Create color maps with more dramatic gradients
    label_color_map = {-100: 'lightgray'}
    
    # Create red color map for negative labels with more dramatic gradient
    if len(negative_labels) > 0:
        red_colors = [
            mcolors.to_rgba('lightpink'),
            mcolors.to_rgba('red'),
            mcolors.to_rgba('darkred')
        ]
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', red_colors)
        
        red_indices = np.linspace(0.2, 1.0, len(negative_labels))
        red_colors = [red_cmap(idx) for idx in red_indices]
        for label, color in zip(reversed(negative_labels), red_colors):
            label_color_map[label] = color
    
    # Create green color map for positive labels with more dramatic gradient
    if len(positive_labels) > 0:
        green_colors = [
            mcolors.to_rgba('lightgreen'),
            mcolors.to_rgba('green'),
            mcolors.to_rgba('darkgreen')
        ]
        green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greens', green_colors)
        
        green_indices = np.linspace(0.2, 1.0, len(positive_labels))
        green_colors = [green_cmap(idx) for idx in green_indices]
        for label, color in zip(positive_labels, green_colors):
            label_color_map[label] = color
    
    # Store scatter plot handles for legend ordering
    scatter_handles = []
    
    # Plot markers for each label using categorical x positions in sorted order
    for label in sorted_labels:  # Use sorted_labels instead of valid_labels
        mask = filtered_df['label'] == label
        masked_indices = np.where(mask)[0]
        scatter = ax.scatter(
            masked_indices,
            filtered_df[mask]['close'],
            color=label_color_map.get(label, 'gray'),
            s=50,
            alpha=0.8,
            label=f'Label {label}'
        )
        scatter_handles.append(scatter)
    
    # Set x-axis ticks and labels
    tick_spacing = max(len(timestamp_list) // 10, 1)
    tick_positions = x_categorical[::tick_spacing]
    tick_labels = [timestamp_list[i] for i in range(0, len(timestamp_list), tick_spacing)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Customize the plot
    ax.set_title(f"Close Price with Labels and Close Raw\n{start_date} to {end_date}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    
    # Add legend with ordered handles
    # Combine line and scatter handles in desired order
    all_handles = [line1, line2] + scatter_handles
    all_labels = ['Close Price', 'Close Raw'] + [f'Label {label}' for label in sorted_labels]
    
    ax.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_signals_plot(filtered_df, backtesting_df, symbol, start_date, end_date):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create categorical x-axis values
    timestamp_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in filtered_df.index]
    x_categorical = np.arange(len(timestamp_list))
    
    # Plot close_raw using categorical x positions
    close_prices = filtered_df['close_raw']
    ax.plot(x_categorical, close_prices, color='black', label='Close Price', linewidth=1)
    
    # Calculate offset for buy/sell signals (0.5% of price range)
    price_range = close_prices.max() - close_prices.min()
    offset = price_range * 0.04
    
    # Filter backtesting DataFrame
    symbol_df = backtesting_df[backtesting_df['symbol'] == symbol]
    
    # Create index mapping for signals
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(filtered_df.index)}
    
    # Get full range of possible classes
    min_possible_class = 0
    pred_columns = [col for col in symbol_df.columns if col.startswith('prediction_raw_class_')]
    max_possible_class = max(int(col.split('_')[-1]) for col in pred_columns)
    all_possible_labels = np.arange(min_possible_class, max_possible_class + 1)
    midpoint = len(all_possible_labels) // 2

    # Split into negative and positive groups
    negative_labels = all_possible_labels[:midpoint]
    positive_labels = all_possible_labels[midpoint:]
    
    # Create color maps
    label_color_map = {-100: 'lightgray'}
    
    # Red color map for negative labels (darker red for more negative)
    if len(negative_labels) > 0:
        red_colors = [
            mcolors.to_rgba('darkred'),
            mcolors.to_rgba('red'),
            mcolors.to_rgba('lightcoral'),
            mcolors.to_rgba('lightpink')
        ]
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', red_colors)
        
        # Lower labels → lower indices (darker reds)
        red_indices = np.linspace(0, 1.0, len(negative_labels))
        red_colors = [red_cmap(idx) for idx in red_indices]
        
        for label, color in zip(negative_labels, red_colors):
            label_color_map[label] = color
    
    # Green color map for positive labels (darker green for more positive)
    if len(positive_labels) > 0:
        green_colors = [
            mcolors.to_rgba('lightgreen'),
            mcolors.to_rgba('limegreen'),
            mcolors.to_rgba('green'),
            mcolors.to_rgba('darkgreen')
        ]
        green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greens', green_colors)
        
        green_indices = np.linspace(0, 1.0, len(positive_labels))
        green_colors = [green_cmap(idx) for idx in green_indices]
        
        for label, color in zip(positive_labels, green_colors):
            label_color_map[label] = color
    
    # Plot predicted labels using categorical x positions
    unique_labels = symbol_df['predicted_label'].unique()
    # Sort valid labels so they appear in ascending order in the legend
    valid_labels = sorted([lbl for lbl in unique_labels if lbl not in [-999, -100]])

    for label in valid_labels:
        mask = symbol_df['predicted_label'] == label
        mask_timestamps = symbol_df[mask].index
        mask_indices = [timestamp_to_idx[ts] for ts in mask_timestamps if ts in timestamp_to_idx]
        if mask_indices:
            mask_prices = [filtered_df['close_raw'].loc[ts] for ts in mask_timestamps if ts in timestamp_to_idx]
            ax.scatter(
                mask_indices,
                mask_prices,
                color=label_color_map.get(label, 'gray'),
                s=25,
                alpha=0.8,
                label=f'Predicted Label {label}'
            )
    
    # Plot buy signals
    buy_mask = symbol_df['buy_final'] == 1
    if buy_mask.any():
        buy_timestamps = symbol_df[buy_mask].index
        buy_indices = [timestamp_to_idx[ts] for ts in buy_timestamps if ts in timestamp_to_idx]
        buy_prices = [filtered_df['close_raw'].loc[ts] + offset for ts in buy_timestamps if ts in timestamp_to_idx]
        
        ax.scatter(
            buy_indices,
            buy_prices,
            color='green',
            s=100,
            alpha=0.6,
            label='Buy Signal',
            marker='^'
        )
    
    # Plot sell signals
    sell_mask = symbol_df['sell'] == 1
    if sell_mask.any():
        sell_timestamps = symbol_df[sell_mask].index
        sell_indices = [timestamp_to_idx[ts] for ts in sell_timestamps if ts in timestamp_to_idx]
        sell_prices = [filtered_df['close_raw'].loc[ts] - offset for ts in sell_timestamps if ts in timestamp_to_idx]
        
        ax.scatter(
            sell_indices,
            sell_prices,
            color='red',
            s=100,
            alpha=0.6,
            label='Sell Signal',
            marker='v'
        )
    
    # Set x-axis ticks and labels
    tick_spacing = max(len(timestamp_list) // 10, 1)
    tick_positions = x_categorical[::tick_spacing]
    tick_labels = [timestamp_list[i] for i in range(0, len(timestamp_list), tick_spacing)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Customize the plot
    ax.set_title(f"Close Price with Predicted Labels and Buy/Sell Signals - {symbol}\n{start_date} to {end_date}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_smoothed_signals_plot(filtered_df, backtesting_df, symbol, start_date, end_date):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create categorical x-axis values
    timestamp_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in filtered_df.index]
    x_categorical = np.arange(len(timestamp_list))
    
    # Plot close_raw using categorical x positions
    close_prices = filtered_df['close_raw']
    ax.plot(x_categorical, close_prices, color='black', label='Close Price', linewidth=1)
    
    # Calculate offset for buy/sell signals (0.5% of price range)
    price_range = close_prices.max() - close_prices.min()
    offset = price_range * 0.04
    
    # Filter backtesting DataFrame
    symbol_df = backtesting_df[backtesting_df['symbol'] == symbol]
    
    # Create index mapping for signals
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(filtered_df.index)}
    
    # Get full range of possible classes
    min_possible_class = 0
    pred_columns = [col for col in symbol_df.columns if col.startswith('prediction_raw_class_')]
    max_possible_class = max(int(col.split('_')[-1]) for col in pred_columns)
    all_possible_labels = np.arange(min_possible_class, max_possible_class + 1)
    midpoint = len(all_possible_labels) // 2

    # Split into negative and positive groups
    negative_labels = all_possible_labels[:midpoint]
    positive_labels = all_possible_labels[midpoint:]
    
    # Create color maps
    label_color_map = {-100: 'lightgray'}
    
    # Red color map for negative labels (darker red for more negative)
    if len(negative_labels) > 0:
        red_colors = [
            mcolors.to_rgba('darkred'),
            mcolors.to_rgba('red'),
            mcolors.to_rgba('lightcoral'),
            mcolors.to_rgba('lightpink')
        ]
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', red_colors)
        
        red_indices = np.linspace(0, 1.0, len(negative_labels))
        red_colors = [red_cmap(idx) for idx in red_indices]
        
        for label, color in zip(negative_labels, red_colors):
            label_color_map[label] = color
    
    # Green color map for positive labels (darker green for more positive)
    if len(positive_labels) > 0:
        green_colors = [
            mcolors.to_rgba('lightgreen'),
            mcolors.to_rgba('limegreen'),
            mcolors.to_rgba('green'),
            mcolors.to_rgba('darkgreen')
        ]
        green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greens', green_colors)
        
        green_indices = np.linspace(0, 1.0, len(positive_labels))
        green_colors = [green_cmap(idx) for idx in green_indices]
        
        for label, color in zip(positive_labels, green_colors):
            label_color_map[label] = color
    
    # Plot predicted_class_moving_avg using categorical x positions
    unique_labels = symbol_df['predicted_class_moving_avg'].dropna().unique()
    # Sort valid labels so they appear in ascending order in the legend
    valid_labels = sorted([lbl for lbl in unique_labels if lbl not in [-999, -100]])

    for label in valid_labels:
        mask = symbol_df['predicted_class_moving_avg'] == label
        mask_timestamps = symbol_df[mask].index
        mask_indices = [timestamp_to_idx[ts] for ts in mask_timestamps if ts in timestamp_to_idx]
        if mask_indices:
            mask_prices = [filtered_df['close_raw'].loc[ts] for ts in mask_timestamps if ts in timestamp_to_idx]
            ax.scatter(
                mask_indices,
                mask_prices,
                color=label_color_map.get(label, 'gray'),
                s=25,
                alpha=0.8,
                label=f'Smoothed Class {label}'
            )
    
    # Plot buy signals
    buy_mask = symbol_df['buy_final'] == 1
    if buy_mask.any():
        buy_timestamps = symbol_df[buy_mask].index
        buy_indices = [timestamp_to_idx[ts] for ts in buy_timestamps if ts in timestamp_to_idx]
        buy_prices = [filtered_df['close_raw'].loc[ts] + offset for ts in buy_timestamps if ts in timestamp_to_idx]
        
        ax.scatter(
            buy_indices,
            buy_prices,
            color='green',
            s=100,
            alpha=0.6,
            label='Buy Signal',
            marker='^'
        )
    
    # Plot sell signals
    sell_mask = symbol_df['sell'] == 1
    if sell_mask.any():
        sell_timestamps = symbol_df[sell_mask].index
        sell_indices = [timestamp_to_idx[ts] for ts in sell_timestamps if ts in timestamp_to_idx]
        sell_prices = [filtered_df['close_raw'].loc[ts] - offset for ts in sell_timestamps if ts in timestamp_to_idx]
        
        ax.scatter(
            sell_indices,
            sell_prices,
            color='red',
            s=100,
            alpha=0.6,
            label='Sell Signal',
            marker='v'
        )
    
    # Set x-axis ticks and labels
    tick_spacing = max(len(timestamp_list) // 10, 1)
    tick_positions = x_categorical[::tick_spacing]
    tick_labels = [timestamp_list[i] for i in range(0, len(timestamp_list), tick_spacing)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Customize the plot
    ax.set_title(f"Close Price with Smoothed Predictions and Buy/Sell Signals - {symbol}\n{start_date} to {end_date}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_prediction_probabilities_plot(filtered_df, backtesting_df, symbol, start_date, end_date):
    """
    Creates a plot showing probability time series for each class
    
    Parameters:
    -----------
    filtered_df : DataFrame
        Original DataFrame with close_raw prices (used for time alignment)
    backtesting_df : DataFrame
        DataFrame containing prediction_raw_class_X columns
    symbol : str
        Symbol to filter for in backtesting_df
    start_date, end_date : str
        Date range to filter for
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create categorical x-axis values
    timestamp_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in filtered_df.index]
    x_categorical = np.arange(len(timestamp_list))
    
    # Filter backtesting DataFrame for symbol
    symbol_df = backtesting_df[backtesting_df['symbol'] == symbol]
    
    # Get prediction probability columns and sort by class number
    prob_columns = [col for col in symbol_df.columns if col.startswith('prediction_raw_class_')]
    prob_columns.sort(key=lambda x: int(x.split('_')[-1]))
    
    num_classes = len(prob_columns)
    midpoint = num_classes // 2

    # Create color maps for the two halves
    red_colors = [
        mcolors.to_rgba('darkred'),
        mcolors.to_rgba('red'),
        mcolors.to_rgba('lightpink')
    ]
    green_colors = [
        mcolors.to_rgba('lightgreen'),
        mcolors.to_rgba('green'),
        mcolors.to_rgba('darkgreen')
    ]
    
    red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', red_colors)
    green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greens', green_colors)

    # Create index mapping
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(filtered_df.index)}
    
    # Plot each class probability with appropriate color
    for i, col in enumerate(prob_columns):
        class_num = int(col.split('_')[-1])
        timestamps = symbol_df.index
        indices = [timestamp_to_idx[ts] for ts in timestamps if ts in timestamp_to_idx]
        probabilities = [symbol_df.loc[ts, col] for ts in timestamps if ts in timestamp_to_idx]
        
        if indices:  # Only plot if we have points
            # Choose color based on whether class is in first or second half
            if i < midpoint:
                # Use red gradient for "negative" classes
                # Updated so that i=0 gets the darkest red.
                color_idx = i / max(1, midpoint - 1)
                color = red_cmap(color_idx)
            else:
                # Use green gradient for "positive" classes
                color_idx = (i - midpoint) / max(1, midpoint - 1)
                color = green_cmap(color_idx)
                
            ax.plot(indices, probabilities, 
                    label=f'Class {class_num}',
                    color=color,
                    linewidth=1,
                    alpha=0.8)
    
    # Set x-axis ticks and labels
    tick_spacing = max(len(timestamp_list) // 10, 1)
    tick_positions = x_categorical[::tick_spacing]
    tick_labels = [timestamp_list[i] for i in range(0, len(timestamp_list), tick_spacing)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Customize the plot
    ax.set_title(f"Class Probabilities Over Time - {symbol}\n{start_date} to {end_date}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_hma_probabilities_plot(filtered_df, backtesting_df, symbol, start_date, end_date):
    """
    Creates two subplots showing HMA of probability time series for each class,
    separated into normal and short signals.
    """
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Create categorical x-axis values
    timestamp_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in filtered_df.index]
    x_categorical = np.arange(len(timestamp_list))
    
    # Filter backtesting DataFrame for symbol
    symbol_df = backtesting_df[backtesting_df['symbol'] == symbol]
    
    def get_class_number(col_name):
        parts = col_name.split('_')
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
        return -1
    
    # Separate columns into normal and short
    hma_columns = [col for col in symbol_df.columns if col.startswith('HMA_prediction_raw_class_')]
    normal_columns = [col for col in hma_columns if not col.endswith('_short')]
    short_columns = [col for col in hma_columns if col.endswith('_short')]

    # Create color maps
    red_colors = [mcolors.to_rgba('darkred'), mcolors.to_rgba('red'), mcolors.to_rgba('lightpink')]
    green_colors = [mcolors.to_rgba('lightgreen'), mcolors.to_rgba('green'), mcolors.to_rgba('darkgreen')]
    red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', red_colors)
    green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greens', green_colors)

    # Create index mapping
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(filtered_df.index)}
    
    def plot_columns(columns, ax, title_suffix):
        # Group columns by class number
        columns_by_class = {}
        for col in columns:
            class_num = get_class_number(col)
            if class_num not in columns_by_class:
                columns_by_class[class_num] = []
            columns_by_class[class_num].append(col)
        
        unique_classes = sorted(columns_by_class.keys())
        midpoint = len(unique_classes) // 2
        
        for i, class_num in enumerate(unique_classes):
            cols = columns_by_class[class_num]
            for col in cols:
                timestamps = symbol_df.index
                indices = [timestamp_to_idx[ts] for ts in timestamps if ts in timestamp_to_idx]
                hma_values = [symbol_df.loc[ts, col] for ts in timestamps if ts in timestamp_to_idx]
                
                if indices:  # Only plot if we have points
                    if i < midpoint:
                        color_idx = i / max(1, midpoint - 1)
                        color = red_cmap(color_idx)
                    else:
                        color_idx = (i - midpoint) / max(1, midpoint - 1)
                        color = green_cmap(color_idx)
                        
                    ax.plot(indices, hma_values, 
                           label=f'Class {class_num}',
                           color=color,
                           linewidth=1,
                           alpha=0.8)
        
        # Set x-axis ticks and labels
        tick_spacing = max(len(timestamp_list) // 10, 1)
        tick_positions = x_categorical[::tick_spacing]
        tick_labels = [timestamp_list[i] for i in range(0, len(timestamp_list), tick_spacing)]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Customize the subplot
        ax.set_title(f"HMA of Class Probabilities - {title_suffix}")
        ax.set_xlabel('Time')
        ax.set_ylabel('HMA of Probability')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot normal signals on top subplot
    plot_columns(normal_columns, ax1, "Normal Signals")
    
    # Plot short signals on bottom subplot
    plot_columns(short_columns, ax2, "Short Signals")
    
    # Add overall title
    fig.suptitle(f"{symbol} - {start_date} to {end_date}", y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_hma_probabilities_minmax_plot(filtered_df, backtesting_df, symbol, start_date, end_date):
    """
    Creates two subplots showing HMA of probability time series for the MINIMUM and MAXIMUM classes,
    separated into normal and short signals.
    """
    plt.style.use('default')
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Create categorical x-axis values
    timestamp_list = [ts.strftime('%Y-%m-%d %H:%M') for ts in filtered_df.index]
    x_categorical = np.arange(len(timestamp_list))
    
    # Filter backtesting DataFrame for symbol
    symbol_df = backtesting_df[backtesting_df['symbol'] == symbol]
    
    def get_class_number(col_name):
        parts = col_name.split('_')
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
        return -1
    
    # Separate columns into normal and short
    hma_columns = [col for col in symbol_df.columns if col.startswith('HMA_prediction_raw_class_')]
    normal_columns = [col for col in hma_columns if not col.endswith('_short')]
    short_columns = [col for col in hma_columns if col.endswith('_short')]
    
    # Create color maps
    red_colors = [mcolors.to_rgba('darkred'), mcolors.to_rgba('red'), mcolors.to_rgba('lightpink')]
    green_colors = [mcolors.to_rgba('lightgreen'), mcolors.to_rgba('green'), mcolors.to_rgba('darkgreen')]
    red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', red_colors)
    green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_greens', green_colors)
    
    # Create index mapping
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(filtered_df.index)}
    
    # Function to plot on a specific axis
    def plot_columns(columns, ax, title_suffix):
        class_numbers = sorted(set(get_class_number(col) for col in columns))
        if len(class_numbers) < 2:
            return
        
        min_class = min(class_numbers)
        max_class = max(class_numbers)
        midpoint = len(class_numbers) // 2
        
        # Group columns by class number
        columns_by_class = {}
        for col in columns:
            class_num = get_class_number(col)
            if class_num not in columns_by_class:
                columns_by_class[class_num] = []
            columns_by_class[class_num].append(col)
        
        # Plot min and max classes
        for class_num in [min_class, max_class]:
            if class_num in columns_by_class:
                cols = columns_by_class[class_num]
                for col in cols:
                    timestamps = symbol_df.index
                    plot_indices = [timestamp_to_idx[ts] for ts in timestamps if ts in timestamp_to_idx]
                    hma_values = [symbol_df.loc[ts, col] for ts in timestamps if ts in timestamp_to_idx]
                    
                    if not plot_indices:
                        continue
                    
                    if class_num < midpoint:
                        color_idx = 1 - (class_num / max(1, midpoint - 1)) if midpoint > 1 else 1.0
                        color = red_cmap(color_idx)
                    else:
                        color_idx = (class_num - midpoint) / max(1, len(class_numbers) - 1 - midpoint)
                        color = green_cmap(color_idx)
                    
                    ax.plot(plot_indices, hma_values,
                           label=f'Class {class_num}',
                           color=color, linewidth=1, alpha=0.8)
        
        # Set x-axis ticks and labels
        tick_spacing = max(len(timestamp_list) // 10, 1)
        tick_positions = x_categorical[::tick_spacing]
        tick_labels = [timestamp_list[i] for i in range(0, len(timestamp_list), tick_spacing)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        ax.set_title(f"HMA of Min & Max Class Probabilities - {title_suffix}")
        ax.set_xlabel('Time')
        ax.set_ylabel('HMA of Probability')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot normal signals on top subplot
    plot_columns(normal_columns, ax1, "Normal Signals")
    
    # Plot short signals on bottom subplot
    plot_columns(short_columns, ax2, "Short Signals")
    
    # Add overall title
    fig.suptitle(f"{symbol} - {start_date} to {end_date}", y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig

def create_trades_table(trades_df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> Figure:
    """
    Creates a figure containing a table of filtered trades data with improved horizontal layout.
    Handles optional columns: 'is_short', 'trailing_ema_return_long', 'trailing_ema_return_short'
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    filtered_trades = trades_df[
        (trades_df['symbol'] == symbol) &
        (trades_df['buy_timestamp'] >= start_ts) &
        (trades_df['sell_timestamp'] <= end_ts)
    ].copy()
    
    base_columns = [
        'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
        'return_percentage', 'position_size', 'capital', 'hold_time_hours',
        'predicted_class', 'highest_class_probability', 'is_partial', 'return'
    ]
    
    optional_columns = ['is_short', 'trailing_ema_return_long', 'trailing_ema_return_short']
    desired_column_order = base_columns + [col for col in optional_columns if col in filtered_trades.columns]
    filtered_trades = filtered_trades[desired_column_order]
    
    filtered_trades['buy_timestamp'] = filtered_trades['buy_timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    filtered_trades['sell_timestamp'] = filtered_trades['sell_timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    standard_numerical_columns = ['buy_price', 'sell_price', 'return_percentage', 'position_size', 
                                'capital', 'hold_time_hours', 'highest_class_probability', 'return']
    for col in standard_numerical_columns:
        if col in filtered_trades.columns:
            filtered_trades[col] = filtered_trades[col].round(2)
    
    ema_columns = ['trailing_ema_return_long', 'trailing_ema_return_short']
    for col in ema_columns:
        if col in filtered_trades.columns:
            filtered_trades[col] = filtered_trades[col].round(5)
    
    col_widths = {
        'symbol': 0.06,
        'buy_timestamp': 0.08,
        'sell_timestamp': 0.08,
        'buy_price': 0.06,
        'sell_price': 0.06,
        'return_percentage': 0.06,
        'position_size': 0.06,
        'capital': 0.07,
        'hold_time_hours': 0.06,
        'predicted_class': 0.05,
        'highest_class_probability': 0.06,
        'is_short': 0.04,
        'trailing_ema_return_long': 0.07,
        'trailing_ema_return_short': 0.07,
        'is_partial': 0.04,
        'return': 0.06
    }
    
    fig = plt.figure(figsize=(24, len(filtered_trades) * 0.4 + 2))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    widths = [col_widths.get(col, 0.06) for col in filtered_trades.columns]
    table = ax.table(
        cellText=filtered_trades.values,
        colLabels=filtered_trades.columns,
        cellLoc='center',
        loc='center',
        colWidths=widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for (row, col), cell in table._cells.items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
    
    table.scale(1.2, 1.5)
    plt.title(f'Trades for {symbol} ({start_date} to {end_date})', pad=20)
    
    return fig

def create_backtesting_table(backtesting_df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> Figure:
    """
    Creates a figure containing a table of filtered backtesting data.
    
    Args:
        backtesting_df: DataFrame containing all backtesting data
        symbol: Symbol to filter for
        start_date: Start date for filtering (should be timezone-aware)
        end_date: End date for filtering (should be timezone-aware)
    
    Returns:
        matplotlib.figure.Figure: Figure containing the backtesting table
    """
    # Convert dates to timezone-aware timestamps if they aren't already
    if isinstance(start_date, str):
        start_ts = pd.to_datetime(start_date).tz_localize('US/Eastern')
    else:
        start_ts = start_date
        
    if isinstance(end_date, str):
        end_ts = pd.to_datetime(end_date).tz_localize('US/Eastern')
    else:
        end_ts = end_date
    
    # Ensure backtesting_df index is timezone-aware
    if backtesting_df.index.tz is None:
        backtesting_df.index = backtesting_df.index.tz_localize('US/Eastern')
    
    # Filter the backtesting data
    filtered_backtest = backtesting_df[
        (backtesting_df['symbol'] == symbol) &
        (backtesting_df.index >= start_ts) &
        (backtesting_df.index <= end_ts)
    ].copy()
    
    # Select required columns and handle optional class_spread
    required_columns = ['buy_final', 'sell', 'predicted_class_moving_avg']
    if 'class_spread' in filtered_backtest.columns:
        required_columns.append('class_spread')
    
    filtered_backtest = filtered_backtest[required_columns].copy()
    
    # Add the index (datetime) as a column, saving the index first
    datetime_index = filtered_backtest.index
    filtered_backtest = filtered_backtest.reset_index(drop=True)
    filtered_backtest.insert(0, 'datetime', datetime_index.strftime('%Y-%m-%d %H:%M'))
    
    # Round numerical columns
    numerical_columns = ['predicted_class_moving_avg']
    if 'class_spread' in filtered_backtest.columns:
        numerical_columns.append('class_spread')
        
    for col in numerical_columns:
        filtered_backtest[col] = filtered_backtest[col].round(2)
    
    # Create figure and axis with extra space at the top for the title
    fig_height = len(filtered_backtest) * 0.3 + 2.5
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate column widths based on number of columns
    n_columns = len(filtered_backtest.columns)
    if n_columns == 4:  # Without class_spread
        col_widths = [0.3, 0.2, 0.2, 0.3]
    else:  # With class_spread
        col_widths = [0.2, 0.1, 0.1, 0.2, 0.2]
    
    # Create table with adjusted position to make room for title
    table = ax.table(
        cellText=filtered_backtest.values,
        colLabels=filtered_backtest.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 0.95],
        colWidths=col_widths
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Add title with adjusted position
    plt.title(
        f'Backtesting Signals for {symbol}\n({start_ts.strftime("%Y-%m-%d %H:%M")} to {end_ts.strftime("%Y-%m-%d %H:%M")})',
        pad=20
    )
    
    # Adjust layout to prevent title cutoff
    plt.tight_layout()
    
    return fig

def create_backtest_metrics_table(symbol_metrics_df: pd.DataFrame) -> Figure:
    """
    Creates a figure containing a table of backtesting metrics for each symbol.
    
    Args:
        symbol_metrics_df: DataFrame where each row contains metrics for a symbol
            Expected to have 'symbol' column and various metric columns
    
    Returns:
        matplotlib.figure.Figure: Figure containing the metrics table
    """
    # Deep copy to avoid modifying original
    df = symbol_metrics_df.copy()
    
    # Round numerical columns to 2 decimal places
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = df[numerical_columns].round(2)
    
    # Split metrics into groups for better visualization
    metric_groups = {
        'Returns - With Cost': [
            'total_returns_percentage_with_cost',
            'average_percent_return_with_cost',
            'buy_and_hold_return_percentage'
        ],
        'Trade Statistics - With Cost': [
            'number_of_trades',
            'win_loss_ratio_with_cost',
            'sharpe_ratio_with_cost'
        ],
        'Returns': [
            'total_returns_percentage',
            'average_percent_return',
            'buy_and_hold_return_percentage'
        ],
        'Trade Statistics': [
            'number_of_trades',
            'win_loss_ratio',
            'sharpe_ratio'
        ],
        'Position Sizing': [
            'average_position_size',
            'position_size_std',
            'average_kelly_fraction'
        ],
        'Timing': [
            'average_hold_time_hours',
            'trading_hold_time_hours',
            'buy_and_hold_hold_time_hours'
        ],
        'Overnight Trading': [
            'overnight_trades_count',
            'overnight_trades_return_percentage',
            'average_overnight_position_size'
        ]
    }
    
    # Create separate tables for each metric group
    figs = []
    
    for group_name, metrics in metric_groups.items():
        # Filter metrics that exist in the DataFrame
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            continue
            
        # Create subset DataFrame with symbol and current group's metrics
        group_df = df[['symbol'] + available_metrics].copy()
        
        # Create figure with adjusted height based on number of rows
        fig_height = len(group_df) * 0.5 + 2  # Adjust multiplier as needed
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=group_df.values,
            colLabels=group_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2] + [0.8/len(available_metrics)] * len(available_metrics)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title(f'Backtest Metrics - {group_name}')
        
        # Adjust layout
        plt.tight_layout()
        
        figs.append(fig)
    
    return figs

def add_backtest_metrics_to_pdf(symbol_metrics_df: pd.DataFrame, pdf) -> None:
    """
    Adds backtest metrics tables to an existing PDF.
    
    Args:
        symbol_metrics_df: DataFrame containing metrics for each symbol
        pdf: matplotlib.backends.backend_pdf.PdfPages object
    """
    # Generate metric tables
    metric_figs = create_backtest_metrics_table(symbol_metrics_df)
    
    # Add each figure to the PDF
    for fig in metric_figs:
        pdf.savefig(fig)
        plt.close(fig)


def save_all_plots_in_one_pdf(
    trial_number,
    models_info,
    X_test,
    y_test,
    num_classes,
    pdf_output_dir,
    optuna_params,
    trades_df,
    initial_capital: float = 10000, 
    df_train=None,          
    df_test=None,           
    filtered_df=None,  
    backtesting_df=None,
    df_features_master_for_predictions=None,
    start_date='2024-06-01',   # Add new parameter
    end_date='2024-06-05',     # Add new parameter
    device=torch.device('cpu'),
    show_feature_importance=False,
    show_roc_curves=False,
    extra_figures=None,
    debug_output=None, 
    equity_curve_plots=None,
    backtesting_results=None,
    equity_curve_plots_with_cost=None
):
    """
    Creates a single PDF (trial_{trial_number}.pdf) containing:
    A) Confusion matrices for each model in models_info
    B) (Optional) XGBoost feature importance plots
    C) (Optional) ROC-AUC plots for each model
    
    Parameters
    ----------
    trial_number : int
        Identifier for the trial (used for naming the PDF).
    models_info : list of dict
        Each dict should contain:
            'model_name': str,
            'model': trained model object,
            'is_xgboost': bool (optional),
            'is_lightgbm': bool (optional),
            'is_pytorch_nn': bool (optional),
            ... other flags if needed ...
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        True test labels (integer-encoded).
    num_classes : int
        Number of classes in the classification problem.
    pdf_output_dir : str
        Directory where the resulting PDF is saved.
    device : torch.device
        Device to run PyTorch inference on (if needed).
    show_feature_importance : bool
        Whether to include XGBoost feature-importance bar charts in the PDF.
    show_roc_curves : bool
        Whether to include ROC-AUC plots in the PDF.
    """

    if extra_figures is None:
        extra_figures = []
    
    os.makedirs(pdf_output_dir, exist_ok=True)
    pdf_filename = os.path.join(pdf_output_dir, f"trial_{trial_number}.pdf")

    with PdfPages(os.path.join(pdf_output_dir, f"trial_{trial_number}.pdf")) as pdf:

        # ---------------------------------------------------------
        # 0) OPTUNA PARAMETERS (First Page)
        # ---------------------------------------------------------
        fig_params = plt.figure(figsize=(8.5, 11))  # Standard US letter size
        plt.axis('off')  # Hide axes
        
        # Create text content
        text_content = f"Optuna Trial {trial_number} Parameters:\n\n"
        
        # Sort parameters by name for consistent display
        sorted_params = dict(sorted(optuna_params.items()))
        
        # Add parameters with proper formatting
        for param_name, param_value in sorted_params.items():
            # Format floating point numbers to 4 decimal places
            if isinstance(param_value, float):
                param_value = f"{param_value:.4f}"
            text_content += f"{param_name}: {param_value}\n"
        
        # Add the text to the figure
        plt.text(0.1, 0.95, text_content,
                fontsize=10,
                fontfamily='monospace',
                verticalalignment='top',
                transform=plt.gca().transAxes)
        
        # Save the parameters page
        pdf.savefig(fig_params)
        plt.close(fig_params)

        # ---------------------------------------------------------
        # BACKTESTING METRICS 
        # ---------------------------------------------------------

        # Add backtest metrics tables
        if hasattr(backtesting_results, 'symbol_metrics'):
            add_backtest_metrics_to_pdf(backtesting_results.symbol_metrics, pdf)

        # ---------------------------------------------------------
        # EQUITY PLOT
        # ---------------------------------------------------------

        if equity_curve_plots_with_cost is not None:
            for fig in equity_curve_plots_with_cost:
                pdf.savefig(fig)
                plt.close(fig)

        if equity_curve_plots is not None:
            for fig in equity_curve_plots:
                pdf.savefig(fig)
                plt.close(fig)

        # ---------------------------------------------------------
        # Data Prep plots
        # ---------------------------------------------------------

        # Add labels over time plot
        fig_labels = create_labels_over_time_plot(df_features_master_for_predictions)
        pdf.savefig(fig_labels)
        plt.close(fig_labels)

        # Add label distribution plots
        fig_dist = create_label_distribution_plots(df_train, df_test)
        pdf.savefig(fig_dist)
        plt.close(fig_dist)

        # ---------------------------------------------------------
        # LABELING PLOT (Second Page)
        # ---------------------------------------------------------
        if filtered_df is not None and start_date is not None and end_date is not None:
            fig_price = create_price_plot(filtered_df, start_date, end_date)
            pdf.savefig(fig_price)
            plt.close(fig_price)

        # ---------------------------------------------------------
        # BUY AND SELL SIGNAL PLOT
        # ---------------------------------------------------------
        if filtered_df is not None and backtesting_df is not None and start_date is not None and end_date is not None:
            # Get symbol from filtered_df (assuming it's the same for all rows)
            symbol = filtered_df['symbol'].iloc[0] if 'symbol' in filtered_df.columns else None
            
            if symbol is not None:

                # Add signals plot
                fig_signals = create_smoothed_signals_plot(filtered_df, backtesting_df, symbol, start_date, end_date)
                pdf.savefig(fig_signals)
                plt.close(fig_signals)

                # Add HMA probabilities plot
                fig_hma = create_hma_probabilities_minmax_plot(filtered_df, backtesting_df, symbol, start_date, end_date)
                pdf.savefig(fig_hma)
                plt.close(fig_hma)

                # Add HMA probabilities plot
                fig_hma = create_hma_probabilities_plot(filtered_df, backtesting_df, symbol, start_date, end_date)
                pdf.savefig(fig_hma)
                plt.close(fig_hma)

                # Add signals plot
                fig_signals = create_signals_plot(filtered_df, backtesting_df, symbol, start_date, end_date)
                pdf.savefig(fig_signals)
                plt.close(fig_signals)
                
                # Add prediction probabilities plot
                fig_probs = create_prediction_probabilities_plot(filtered_df, backtesting_df, symbol, start_date, end_date)
                pdf.savefig(fig_probs)
                plt.close(fig_probs)

                # Add trades table
                if trades_df is not None and not trades_df.empty:
                    trades_table_fig = create_trades_table(trades_df, symbol, start_date, end_date)
                    if trades_table_fig is not None:
                        pdf.savefig(trades_table_fig)
                        plt.close(trades_table_fig)

                # Add backtesting signals table
                if backtesting_df is not None and not backtesting_df.empty:
                    backtesting_table_fig = create_backtesting_table(backtesting_df, symbol, start_date, end_date)
                    if backtesting_table_fig is not None:
                        pdf.savefig(backtesting_table_fig)
                        plt.close(backtesting_table_fig)


                

        # ---------------------------------------------------------
        # LABELING DEBUG TEXT
        # ---------------------------------------------------------
        # Add this right before the debug_figures creation in save_all_plots_in_one_pdf:

        if debug_output is not None:
            print("\nDEBUG OUTPUT ANALYSIS")
            print("=====================")
            print(f"Number of sections: {len(debug_output.sections)}")
            for section_title, section_content in debug_output.sections:
                content_str = ''.join(section_content)
                line_count = len(content_str.split('\n'))
                print(f"\nSection: {section_title}")
                print(f"Content length (lines): {line_count}")
                print("First few lines of content:")
                preview_lines = content_str.split('\n')[:3]  # Show first 3 lines
                for line in preview_lines:
                    print(f"  {line[:100]}...")  # Show first 100 chars of each line

            debug_figures = create_debug_text_pages(debug_output, max_lines_per_page=200, max_pages=20)
            for fig in debug_figures:
                pdf.savefig(fig)
                plt.close(fig)

        # ---------------------------------------------------------
        # A) CONFUSION MATRICES
        # ---------------------------------------------------------
        for info in models_info:
            model_name = info.get('model_name', 'Unknown Model')
            model = info.get('model', None)
            is_xgboost = info.get('is_xgboost', False)
            is_lightgbm = info.get('is_lightgbm', False)
            is_pytorch_nn = info.get('is_pytorch_nn', False)

            if model is None:
                continue  # skip if no model

            # 1) Get predictions (probabilities)
            if is_xgboost:
                dtest = DMatrix(X_test)
                y_pred_probs = model.predict(dtest)
            elif is_lightgbm:
                try:
                    y_pred_probs = model.predict(X_test, num_iteration=model.best_iteration)
                except:
                    y_pred_probs = model.predict(X_test)
            elif is_pytorch_nn:
                model.eval()
                if isinstance(X_test, pd.DataFrame):
                    X_test_np = X_test.values
                else:
                    X_test_np = X_test
                X_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(X_tensor)
                    y_pred_probs = F.softmax(logits, dim=1).cpu().numpy()
            else:
                # CatBoost, RF, ExtraTrees, ERTBoost, etc.
                y_pred_probs = model.predict_proba(X_test)

            # 2) Handle binary classification (1D array -> 2D)
            if y_pred_probs.ndim == 1:
                y_pred_probs = np.vstack([1 - y_pred_probs, y_pred_probs]).T

            # 3) Argmax for predicted labels
            y_pred = np.argmax(y_pred_probs, axis=1)

            # 4) Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))

            # 4b) Accuracy on y_test
            acc = accuracy_score(y_test, y_pred)  # fraction from 0..1
            acc_percent = acc * 100

            # 5) Plot confusion matrix with seaborn
            fig_cm, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes),
                ax=ax
            )
            ax.set_title(f"Confusion Matrix - {model_name} (Acc={acc_percent:.2f}%)")
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.tight_layout()

            # 6) Save confusion matrix figure to PDF
            pdf.savefig(fig_cm)
            plt.close(fig_cm)

        # ---------------------------------------------------------
        # B) FEATURE IMPORTANCE (XGBoost Example)
        # ---------------------------------------------------------
        if show_feature_importance:
            import xgboost as xgb

            # Find the XGBoost model
            xgb_model_info = None
            for item in models_info:
                if item.get('model_name') == 'XGBoost':
                    xgb_model_info = item
                    break

            if xgb_model_info is not None:
                bst = xgb_model_info['model']  # The XGBoost Booster instance

                # 1) Extract feature importances
                importance_weight = bst.get_score(importance_type='weight')
                importance_gain = bst.get_score(importance_type='gain')
                importance_cover = bst.get_score(importance_type='cover')

                def to_dataframe(imp_dict, col_name):
                    return pd.DataFrame({
                        'Feature': list(imp_dict.keys()),
                        col_name: list(imp_dict.values())
                    })

                df_weight = to_dataframe(importance_weight, 'Weight')
                df_gain = to_dataframe(importance_gain, 'Gain')
                df_cover = to_dataframe(importance_cover, 'Cover')

                # Sort top 60
                df_weight_sorted = df_weight.sort_values('Weight', ascending=False).head(60)
                df_gain_sorted = df_gain.sort_values('Gain', ascending=False).head(60)
                df_cover_sorted = df_cover.sort_values('Cover', ascending=False).head(60)

                # Helper for bar chart
                def plot_feature_importance_matplotlib(df, x_col, y_col, title):
                    num_features = len(df)
                    height = 0.4 * num_features
                    fig, ax = plt.subplots(figsize=(10, max(height, 4)))
                    # Reverse so highest is at the top
                    df = df.iloc[::-1]
                    ax.barh(df[x_col], df[y_col], color='blue')
                    ax.set_title(title, fontsize=14)
                    ax.set_xlabel(y_col, fontsize=12)
                    ax.set_ylabel(x_col, fontsize=12)
                    ax.tick_params(axis='y', labelsize=10)
                    plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.08)
                    return fig

                # Create and save 3 plots: Weight, Gain, Cover
                fig_weight = plot_feature_importance_matplotlib(
                    df_weight_sorted, 'Feature', 'Weight', 'XGBoost Top 60 by Weight'
                )
                pdf.savefig(fig_weight)
                plt.close(fig_weight)

                fig_gain = plot_feature_importance_matplotlib(
                    df_gain_sorted, 'Feature', 'Gain', 'XGBoost Top 60 by Gain'
                )
                pdf.savefig(fig_gain)
                plt.close(fig_gain)

                fig_cover = plot_feature_importance_matplotlib(
                    df_cover_sorted, 'Feature', 'Cover', 'XGBoost Top 60 by Cover'
                )
                pdf.savefig(fig_cover)
                plt.close(fig_cover)

        # ---------------------------------------------------------
        # C) ROC AUC PLOTS
        # ---------------------------------------------------------
        if show_roc_curves:

            for info in models_info:
                model_name = info.get('model_name', 'Unknown Model')
                model = info.get('model', None)
                if model is None:
                    continue
                
                print(f"[INFO] Generating ROC for '{model_name}'...")

                #----------------------------------
                # 1. Get predicted probabilities
                #----------------------------------
                # Distinguish among XGBoost, LightGBM, PyTorch, etc.
                if model_name.lower() == 'xgboost':
                    # XGBoost logic
                    dtest = xgb.DMatrix(X_test, enable_categorical=True)
                    y_pred_probs = model.predict(dtest)
                    n_classes = y_pred_probs.shape[1]
                    classes_ = np.arange(n_classes)
                
                elif model_name.lower() == 'lightgbm':
                    # LightGBM logic
                    try:
                        y_pred_probs = model.predict(X_test, num_iteration=model.best_iteration)
                    except:
                        y_pred_probs = model.predict(X_test)
                    n_classes = y_pred_probs.shape[1]
                    classes_ = np.arange(n_classes)
                
                elif model_name.lower() in ('pytorch nn', 'pytorch_nn'):
                    # PyTorch logic
                    model.eval()
                    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits = model(X_test_tensor)
                        y_pred_probs = F.softmax(logits, dim=1).cpu().numpy()
                    n_classes = y_pred_probs.shape[1]
                    classes_ = np.arange(n_classes)
                
                else:
                    # Generic scikit-learn logic (RandomForest, CatBoost, ExtraTrees, ERTBoost, etc.)
                    # -> These typically have model.classes_
                    if not hasattr(model, 'classes_'):
                        raise AttributeError(
                            f"Model '{model_name}' is missing classes_ attribute; cannot do ROC. "
                            f"(If this is a PyTorch model, please handle it in the PyTorch branch.)"
                        )
                    
                    classes_ = model.classes_
                    n_classes = len(classes_)
                    y_pred_probs = model.predict_proba(X_test)

                #----------------------------------
                # 2. Handle binary => shape (n_samples,)
                #----------------------------------
                if y_pred_probs.ndim == 1:
                    y_pred_probs = np.vstack([1 - y_pred_probs, y_pred_probs]).T
                    n_classes = 2
                    classes_ = np.arange(2)
                
                #----------------------------------
                # 3. Binarize the labels
                #----------------------------------
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc

                y_test_binarized = label_binarize(y_test, classes=classes_)
                if y_test_binarized.shape[1] != n_classes:
                    print(f"[WARNING] Mismatch in shapes. y_test_binarized has {y_test_binarized.shape[1]} columns, "
                        f"expected {n_classes}. Check that classes match the model's classes_.")
                
                #----------------------------------
                # 4. Create a matplotlib figure for ROC
                #----------------------------------
                fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
                
                roc_auc_dict = {}
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
                    roc_auc_val = auc(fpr, tpr)
                    roc_auc_dict[i] = roc_auc_val
                    ax_roc.plot(fpr, tpr, label=f"Class {classes_[i]} (AUC={roc_auc_val:.2f})")
                
                # Diagonal line (random)
                ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
                
                avg_auc = np.mean(list(roc_auc_dict.values()))
                ax_roc.set_title(f"ROC - {model_name} (Avg AUC={avg_auc:.2f})")
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc='best')
                plt.tight_layout()
                
                #----------------------------------
                # 5. Save the figure to PDF
                #----------------------------------
                pdf.savefig(fig_roc)
                plt.close(fig_roc)

        #----------------------------------
        # Add other plots plots if provided
        #----------------------------------

        # Add time series metric plots
        # time_series_figs = generate_time_series_metric_plot_long_short(trades_df, initial_capital=initial_capital)
        # for fig in time_series_figs:
        #     pdf.savefig(fig)
        #     plt.close(fig)

        if len(trades_df) > 0 and 'predicted_class' in trades_df.columns:
            class_metric_figs = generate_class_metric_plots(trades_df, initial_capital=initial_capital)
            for fig in class_metric_figs:
                pdf.savefig(fig)
                plt.close(fig)

            # Add time series metric plots
            time_series_figs = generate_time_series_metric_plot(trades_df, initial_capital=initial_capital)
            for fig in time_series_figs:
                pdf.savefig(fig)
                plt.close(fig)

        for fig in extra_figures:
            pdf.savefig(fig)
            plt.close(fig)


    print(f"[INFO] Saved confusion matrices, feature importance, and ROC AUC (where specified) to: {pdf_filename}")
