from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from visuals import generate_volume_bl_trends

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Data Cleaning
def clean_nop_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace('"', '').str.replace(',', '').str.strip()

    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    df.fillna(0, inplace=True)
    # Step 4: Convert date columns to datetime (if applicable)
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True)  # <--- Add this
            except Exception as e:
                print(f"Failed to parse {col}: {e}")


    # Step 5: Strip strings in object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Convert numeric columns
    for col in ['MKT_VAL_BL', 'VOLUME_BL']:
        if col in df.columns:
            df[col] = df[col].str.replace('"', '').str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
# date
from datetime import datetime
import re

def extract_date_from_filename(filename: str) -> datetime:
    # Example filename: NOP_PW_20241116064501.csv
    match = re.search(r'(\d{14})', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    return None


# card01
def generate_nop_volume_card(df1, df2, file1_name, file2_name, save_path="static/nop_volume_card.png"):
    import matplotlib.pyplot as plt
    import os
    date1 = pd.to_datetime(df1.iloc[0, 0])
    date2 = pd.to_datetime(df2.iloc[0, 0])


    if date1 >= date2:
        selected_df = df1
        selected_date = date1
    else:
        selected_df = df2
        selected_date = date2

    total_volume = selected_df['VOLUME_BL'].sum()
    
    # Plot the card
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')

    text = f"$\\bf{{NOP\ Volume}}$\n{total_volume:,.0f}\n(As of {selected_date.strftime('%d-%b-%Y')})"
    ax.text(
    0.5, 0.5, text,
    fontsize=18,
    ha='center',
    va='center',
    color='#222222',
    fontweight='regular',
    fontname='Arial',
    bbox=dict(
        facecolor='white',
        edgecolor='none',
        boxstyle='round,pad=1.2',
        alpha=0.3
    )
)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    return save_path

# card-2
def generate_delta_volume_card(df1, df2, file1_name, file2_name, save_path="static/delta_volume_card.png"):
    import matplotlib.pyplot as plt

    # Calculate total VOLUME_BL for each file
    volume1 = df1['VOLUME_BL'].sum()
    volume2 = df2['VOLUME_BL'].sum()
    delta = volume2 - volume1
    delta_pct = (delta / volume1 * 100) if volume1 != 0 else 0

    # Determine visual cue color
    color = 'green' if delta >= 0 else 'red'
    sign = "+" if delta >= 0 else "-"

    # Get date from filename (fallback if datetime extraction fails)
    date1 = pd.to_datetime(df1.iloc[0, 0])
    date2 = pd.to_datetime(df2.iloc[0, 0])


    # Plot
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')

    text = (
    f"$\\bf{{Δ\\ Volume\\_BL:}}$\n{sign}{abs(delta):,.0f}\n"
    f"{date1.strftime('%d-%b')} → {date2.strftime('%d-%b')}\n"
    f"({sign}{abs(delta_pct):.2f}%)"
)

    ax.text(
    0.5, 0.5, text,
    fontsize=18,
    ha='center',
    va='center',
    color='#222222',
    fontweight='regular',
    fontname='Arial',
    bbox=dict(
        facecolor='white',
        edgecolor='none',
        boxstyle='round,pad=1.2',
        alpha=0.3
    )
)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    return save_path

# card 3
def generate_market_value_card(df1, df2, file1_name, file2_name, save_path="static/market_value_card.png"):
    import matplotlib.pyplot as plt
    import os
    date1 = pd.to_datetime(df1.iloc[0, 0])
    date2 = pd.to_datetime(df2.iloc[0, 0])


    if date1 >= date2:
        selected_df = df1
        selected_date = date1
    else:
        selected_df = df2
        selected_date = date2

    total_volume = selected_df['MKT_VAL_BL'].sum()
    
    # Plot the card
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')

    text = (
    f"$\\bf{{Market\\ Value}}$\n"
    f"{total_volume:,.0f}\n"
    f"(As of {selected_date.strftime('%d-%b-%Y')})"
)

    ax.text(
    0.5, 0.5, text,
    fontsize=18,
    ha='center',
    va='center',
    color='#222222',
    fontweight='regular',
    fontname='Arial',
    bbox=dict(
        facecolor='white',
        edgecolor='none',
        boxstyle='round,pad=1.2',
        alpha=0.3
    )
)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    return save_path

# card 4        

def generate_delta_market_card(df1, df2, file1_name, file2_name, save_path="static/delta_market_card.png"):
    import matplotlib.pyplot as plt

    # Calculate total VOLUME_BL for each file
    volume1 = df1['MKT_VAL_BL'].sum()
    volume2 = df2['MKT_VAL_BL'].sum()
    delta = volume2 - volume1
    delta_pct = (delta / volume1 * 100) if volume1 != 0 else 0

    # Determine visual cue color
    color = 'green' if delta >= 0 else 'red'
    sign = "+" if delta >= 0 else "-"

    # Get date from filename (fallback if datetime extraction fails)
    date1 = pd.to_datetime(df1.iloc[0, 0])
    date2 = pd.to_datetime(df2.iloc[0, 0])


    # Plot
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')

    text = (
    f"$\\bf{{Δ\\ Market\\_Value:}}$\n{sign}{abs(delta):,.0f}\n"
    f"{date1.strftime('%d-%b')} → {date2.strftime('%d-%b')}\n"
    f"({sign}{abs(delta_pct):.2f}%)"
)

    
    ax.text(
    0.5, 0.5, text,
    fontsize=18,
    ha='center',
    va='center',
    color='#222222',
    fontweight='regular',
    fontname='Arial',
    bbox=dict(
        facecolor='white',
        edgecolor='none',
        boxstyle='round,pad=1.2',
        alpha=0.3
    )
)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    return save_path


def get_exposure_hotspot(df1, df2):
    # Add a column to distinguish
    df1 = df1.copy()
    df2 = df2.copy()
    df1['Date'] = 'Date1'
    df2['Date'] = 'Date2'

    # Combine both
    combined = pd.concat([df1, df2], axis=0)

    # Group by Segment and Book
    group_cols = ['Segment', 'BOOK_ATTR8']
    agg_df = combined.groupby(['Date'] + group_cols)[['VOLUME_BL', 'MKT_VAL_BL']].sum().reset_index()

    # Pivot to compare Date1 vs Date2
    pivot = agg_df.pivot(index=group_cols, columns='Date', values=['VOLUME_BL', 'MKT_VAL_BL'])
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    # Compute change (delta)
    pivot['VOL_DELTA'] = pivot['VOLUME_BL_Date2'] - pivot['VOLUME_BL_Date1']
    pivot['MKT_DELTA'] = pivot['MKT_VAL_BL_Date2'] - pivot['MKT_VAL_BL_Date1']

    # Find the max absolute delta (volume or market)
    pivot['MAX_EXPOSURE'] = pivot[['VOL_DELTA', 'MKT_DELTA']].abs().max(axis=1)
    top_row = pivot.loc[pivot['MAX_EXPOSURE'].idxmax()]

    # Create summary
    summary = {
        'Segment': top_row['Segment'],
        'Book': top_row['BOOK_ATTR8'],
        'VOL_CHANGE': top_row['VOL_DELTA'],
        'MKT_VAL_CHANGE': top_row['MKT_DELTA']
    }

    return summary

def exposure_hotspot_report_text(summary_dict):
    segment = summary_dict['Segment']
    book = summary_dict['Book']
    vol_change = summary_dict['VOL_CHANGE']
    mkt_change = summary_dict['MKT_VAL_CHANGE']

    # Format numbers
    vol_str = f"{vol_change:,.2f}"
    mkt_str = f"₹{mkt_change:,.2f}"

    report_text = f"""
    <strong>🔥Exposure Hotspot Identified:</strong><br>The highest exposure change was detected in the <strong>Segment</strong>: <span style='color:#007acc'>{segment}</span> 
    and <strong>Book</strong>: <span style='color:#007acc'>{book}</span>.<br>
    It saw a <strong>VOLUME_BL change</strong> of <span style='color:{'red' if vol_change < 0 else 'green'}'>{vol_str}</span> 
    and a <strong>Market Value change</strong> of <span style='color:{'red' if mkt_change < 0 else 'green'}'>{mkt_str}</span> 
    between the two dates.
    """
    return report_text
# ai

def generate_ai_summary_for_volume_change(df1, df2):
    import pandas as pd

    include_horizon = 'Horizon' in df1.columns and 'Horizon' in df2.columns

    group_cols = ['Segment', 'BOOK']
    if include_horizon:
        group_cols.append('Horizon')

    df1_grouped = df1.groupby(group_cols)['VOLUME_BL'].sum().reset_index()
    df2_grouped = df2.groupby(group_cols)['VOLUME_BL'].sum().reset_index()

    merged = pd.merge(df1_grouped, df2_grouped, on=group_cols, suffixes=('_df1', '_df2')).fillna(0)
    merged['Delta'] = merged['VOLUME_BL_df2'] - merged['VOLUME_BL_df1']
    merged['DeltaPerc'] = 100 * merged['Delta'] / (merged['VOLUME_BL_df1'].replace(0, 1))

    filtered = merged[merged['DeltaPerc'].abs() > 5]

    if filtered.empty:
        return "", "", False  # No change, return blank tables and False flag

    def row_html(row):
        horizon_html = f"<td>{row['Horizon']}</td>" if include_horizon else ""
        delta_perc = row['DeltaPerc']
        delta_units = int(row['Delta'])
        delta_class = "positive" if delta_perc > 0 else "negative"
        return f"""
            <tr>
                <td>{row['Segment']}</td>
                <td>{row['BOOK']}</td>
                {horizon_html}
                <td class="{delta_class}">{delta_perc:.2f}%</td>
                <td>{delta_units:,}</td>
            </tr>
        """

    rows = [row_html(row) for _, row in filtered.iterrows()]
    short_table = "\n".join(rows[:4])
    full_table = "\n".join(rows)

    return short_table, full_table, include_horizon


def generate_threshold_breach_alerts(df1, df2):
    import pandas as pd

    df1_grouped = df1.groupby('Segment')['VOLUME_BL'].sum().reset_index()
    df2_grouped = df2.groupby('Segment')['VOLUME_BL'].sum().reset_index()

    merged = pd.merge(df1_grouped, df2_grouped, on='Segment', suffixes=('_df1', '_df2')).fillna(0)
    merged['Delta'] = merged['VOLUME_BL_df2'] - merged['VOLUME_BL_df1']
    merged['DeltaPerc'] = 100 * merged['Delta'] / (merged['VOLUME_BL_df1'].replace(0, 1))

    breached = merged[merged['DeltaPerc'].abs() > 5]

    if breached.empty:
        return "", ""

    alerts = []
    for _, row in breached.iterrows():
        segment = row['Segment']
        delta = row['Delta']
        delta_perc = row['DeltaPerc']
        direction = "increase" if delta > 0 else "decrease"
        alerts.append(
            f"⚠️ <strong>{segment}</strong>: {direction} of <strong>{abs(delta_perc):.2f}%</strong> "
            f"({int(delta):,} units). <em>Rebalancing suggested</em>."
        )

    short = "<br>".join(alerts[:3])
    full = "<br>".join(alerts)
    return short, full



# nop
def preprocess_nop_volume(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Ensure relevant columns are present
    required_columns = ['Report Date', 'VOLUME_BL', 'Segment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Required columns 'Report Date', 'VOLUME_BL', and 'Segment' not found.")

    # Step 2: Get max report_date
    max_date = df['Report Date'].max()

    # Step 3: Filter for max report_date
    latest_df = df[df['Report Date'] == max_date]

    # Step 4: Group by segment and sum volume_bl as nop_volume
    result_df = latest_df.groupby('Segment', as_index=False)['VOLUME_BL'].sum()
    result_df.rename(columns={'VOLUME_BL': 'nop_volume'}, inplace=True)

    return result_df

def generate_top5_books_by_mkt_val_delta(df1, df2):
    import matplotlib.pyplot as plt
    import os
    # Ensure MKT_VAL_BL is numeric
    df1['MKT_VAL_BL'] = pd.to_numeric(df1['MKT_VAL_BL'], errors='coerce')
    df2['MKT_VAL_BL'] = pd.to_numeric(df2['MKT_VAL_BL'], errors='coerce')

    # Group by BOOK
    df1_grouped = df1.groupby('BOOK')['MKT_VAL_BL'].sum().reset_index()
    df2_grouped = df2.groupby('BOOK')['MKT_VAL_BL'].sum().reset_index()

    # Merge
    merged = pd.merge(df1_grouped, df2_grouped, on='BOOK', how='outer', suffixes=('_df1', '_df2')).fillna(0)

    # Calculate Delta and AbsDelta
    merged['Delta'] = merged['MKT_VAL_BL_df2'] - merged['MKT_VAL_BL_df1']
    merged['AbsDelta'] = merged['Delta'].abs()

    # Sort and get Top 5
    top5 = merged.sort_values(by='AbsDelta', ascending=False).head(5)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top5['BOOK'], top5['Delta'], color='skyblue')
    ax.set_title('Top 5 Books by MKT_VAL_BL Delta')
    ax.set_xlabel('BOOK')
    ax.set_ylabel('MKT_VAL_BL Delta')
    ax.axhline(0, color='gray', linestyle='--')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom' if yval >= 0 else 'top')

    top5_books_plot_filename = "top5_books_mkt_val_delta.png"

    plt.tight_layout()
    plt.savefig(top5_books_plot_filename)
    plt.close()
    return top5_books_plot_filename


def generate_top5_movers_by_segment_book(df1, df2):
    import matplotlib.pyplot as plt
    import os

    df1_grouped = df1.groupby(['Segment', 'BOOK'])['MKT_VAL_BL'].sum().reset_index()
    df2_grouped = df2.groupby(['Segment', 'BOOK'])['MKT_VAL_BL'].sum().reset_index()

    merged = pd.merge(df1_grouped, df2_grouped, on=['Segment', 'BOOK'], how='outer', suffixes=('_df1', '_df2')).fillna(0)
    merged['Delta'] = merged['MKT_VAL_BL_df2'] - merged['MKT_VAL_BL_df1']
    merged['AbsDelta'] = merged['Delta'].abs()

    # Create a combined label for y-axis
    merged['Label'] = merged['Segment'] + ' | ' + merged['BOOK']
    top5 = merged.sort_values('AbsDelta', ascending=False).head(5)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top5['Label'], top5['Delta'], color='salmon')
    ax.set_title('Top 5 Movers by Segment | Book (Δ MKT_VAL_BL)')
    ax.set_xlabel('Delta Market Value')
    ax.set_ylabel('Segment | Book')
    ax.axvline(0, color='black', linewidth=0.8)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (50 if width >= 0 else -200), bar.get_y() + bar.get_height()/2,
                f'{width:,.0f}', va='center', ha='left' if width >= 0 else 'right')

    plt.tight_layout()
    plot_path = os.path.join('top5_segment_book_movers.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# heatmap
def generate_segment_horizon_heatmap(df1, df2):
    

    df1['Date'] = 'Date1'
    df2['Date'] = 'Date2'
    df = pd.concat([df1, df2], ignore_index=True)

    if 'Horizon' not in df.columns or 'Segment' not in df.columns:
        print("Missing required columns for heatmap. Skipping plot.")
        return None
    
    # df['MKT_VAL_BL'] = df['MKT_VAL_BL'].str.replace('"', '').str.replace(',', '').str.strip()
    df['MKT_VAL_BL'] = pd.to_numeric(df['MKT_VAL_BL'], errors='coerce')

    # Aggregate MKT_VAL_BL
    agg = df.groupby(['Date', 'Segment', 'Horizon'])['MKT_VAL_BL'].sum().reset_index()

    # Pivot each date separately
    pivot1 = agg[agg['Date'] == 'Date1'].pivot(index='Segment', columns='Horizon', values='MKT_VAL_BL')
    pivot2 = agg[agg['Date'] == 'Date2'].pivot(index='Segment', columns='Horizon', values='MKT_VAL_BL')

    # Align and compute delta
    delta = pivot2 - pivot1
    delta = delta.fillna(0)

    # formatted_annot = delta.applymap(lambda x: f"+€{x:.1f}M" if x > 0 else (f"€{x:.1f}M" if x < 0 else "€0.0M"))
    cmap = sns.diverging_palette(10, 145, as_cmap=True)
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(delta, fmt="", cmap=cmap, # annot=formatted_annot
                center=0, linewidths=0.5, cbar=False,
                annot_kws={"size": 10, "weight": "bold"}, square=True)
    # ax.text(0.5, 1.05, "Segment × Horizon Heatmap (Δ MKT_VAL_BL)",
    #     horizontalalignment='center',
    #     verticalalignment='bottom',
    #     transform=ax.transAxes,
    #     fontsize=14, weight='bold')

    # plt.title("Segment × Horizon Heatmap (Δ MKT_VAL_BL)")
    plt.ylabel("Segment")
    plt.xlabel("Horizon")
    plt.tight_layout()

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_path = os.path.join('static', 'segment_horizon_heatmap.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return "segment_horizon_heatmap.png"  # Just the filename!

# Routes

@app.route('/')
def index():
    return render_template('index.html')  # Make sure you have templates/index.html

@app.route('/generate', methods=['POST'])
def generate():
    uploaded_files = {}

    for key in ['csv1', 'csv2']:
        file = request.files.get(key)
        if not file or not file.filename.endswith('.csv'):
            return f"Invalid or missing file: {key}", 400

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)

        try:
            df = pd.read_csv(save_path)
            uploaded_files[key] = {
                'df': df,
                'filename': file.filename
            }
            print(f"{key} uploaded successfully with shape {df.shape}")
        except Exception as e:
            return f"Failed to read {key}: {str(e)}", 500

    # checking validity of files
    df1_cleaned = clean_nop_data(uploaded_files['csv1']['df'])
    df2_cleaned = clean_nop_data(uploaded_files['csv2']['df'])

    nop_volume_card_path = generate_nop_volume_card(df1_cleaned, df2_cleaned, uploaded_files['csv1']['filename'], uploaded_files['csv2']['filename'])

    delta_volume_card_path = generate_delta_volume_card(df1_cleaned, df2_cleaned, uploaded_files['csv1']['filename'], uploaded_files['csv2']['filename'])

    market_value_card_path = generate_market_value_card(df1_cleaned, df2_cleaned, uploaded_files['csv1']['filename'], uploaded_files['csv2']['filename'])

    delta_market_card_path = generate_delta_market_card(df1_cleaned, df2_cleaned, uploaded_files['csv1']['filename'], uploaded_files['csv2']['filename'])

    summary = get_exposure_hotspot(df1_cleaned, df2_cleaned)
    highlight_text = exposure_hotspot_report_text(summary)
    short_volume_table, full_volume_table, include_horizon = generate_ai_summary_for_volume_change(df1_cleaned, df2_cleaned)
    threshold_alerts_short, threshold_alerts_full = generate_threshold_breach_alerts(df1_cleaned, df2_cleaned)


    plot_paths = generate_volume_bl_trends(df1_cleaned, df2_cleaned)
    top5_books_plot = generate_top5_books_by_mkt_val_delta(df1_cleaned, df2_cleaned)
    top5_segment_book_plot = generate_top5_movers_by_segment_book(df1_cleaned, df2_cleaned)
    heatmap_filename = generate_segment_horizon_heatmap(df1_cleaned, df2_cleaned)

    print(heatmap_filename)


    return render_template(
    'report.html',
    nop_volume_card_path=nop_volume_card_path,
    delta_volume_card_path=delta_volume_card_path,
    market_value_card_path=market_value_card_path,
    delta_market_card_path=delta_market_card_path,
    highlight_text=highlight_text,
    short_volume_table=short_volume_table,
    full_volume_table=full_volume_table,
    include_horizon=include_horizon,
    threshold_alerts_short=threshold_alerts_short,
    threshold_alerts_full=threshold_alerts_full,
    plot_paths=plot_paths,
    top5_books_plot=top5_books_plot,
    top5_segment_book_plot=top5_segment_book_plot,
    segment_horizon_heatmap=heatmap_filename,
)

if __name__ == '__main__':
    app.run(debug=True)