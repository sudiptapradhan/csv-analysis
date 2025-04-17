import matplotlib.pyplot as plt
import pandas as pd
import os

# Ensure plots folder exists
os.makedirs("static/plots", exist_ok=True)

def clean_and_prepare_data(df1, df2):
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Drop rows where VOLUME_BL is NaN
    df1.dropna(subset=['VOLUME_BL'], inplace=True)
    df2.dropna(subset=['VOLUME_BL'], inplace=True)

    # Optional: convert to integer if all values are whole numbers
    df1['VOLUME_BL'] = df1['VOLUME_BL'].astype(int)
    df2['VOLUME_BL'] = df2['VOLUME_BL'].astype(int)

    # Handle Report Date
    if 'Report Date' in df1.columns:
        df1['Date'] = pd.to_datetime(df1['Report Date']).dt.date
    if 'Report Date' in df2.columns:
        df2['Date'] = pd.to_datetime(df2['Report Date']).dt.date

    df = pd.concat([df1, df2], ignore_index=True)
    return df

def plot_volume_by_column(df, col):
    # Step 1: Get unique dates (assuming exactly two for comparison)
    unique_dates = df['Date'].dropna().unique()
    if len(unique_dates) != 2:
        print(f"[ERROR] Expected 2 dates, found {len(unique_dates)}: {unique_dates}")
        return None

    date1, date2 = sorted(unique_dates)

    # Step 2: Filter and group by Date and the given column
    grouped = df.groupby(['Date', col])['VOLUME_BL'].sum().reset_index()

    # Step 3: Pivot the data so we get each category with values for the two dates
    pivot_df = grouped.pivot(index=col, columns='Date', values='VOLUME_BL').fillna(0)
    pivot_df = pivot_df[[date1, date2]]  # Ensure order is consistent

    # Ploting side-by-side bars
    ax = pivot_df.plot(kind='bar', figsize=(12, 6))
    ax.set_title(f'VOLUME_BL Comparison by {col} on {date1} vs {date2}')
    ax.set_ylabel('VOLUME_BL')
    ax.set_xlabel(col)
    ax.legend(title='Date')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Saving plot
    plot_path = f"static/plots/volume_diff_by_{col.lower()}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT SAVED] {plot_path}")
    return plot_path


def plot_volume_by_segment(df):
    return plot_volume_by_column(df, 'Segment')


def plot_volume_by_tgroup1(df):
    return plot_volume_by_column(df, 'TGROUP1')


def plot_volume_by_bookattr8(df):
    return plot_volume_by_column(df, 'BOOK_ATTR8')


def plot_volume_by_userval4(df):
    return plot_volume_by_column(df, 'USR_VAL4')


def plot_volume_by_tgroup2(df):
    return plot_volume_by_column(df, 'TGROUP2')


def generate_volume_bl_trends(df1, df2):
    df = clean_and_prepare_data(df1, df2)

    plot_paths = [
        plot_volume_by_segment(df),
        plot_volume_by_tgroup1(df),
        plot_volume_by_bookattr8(df),
        plot_volume_by_userval4(df),
        plot_volume_by_tgroup2(df),
    ]

    # Filter out None in case some columns were missing
    return [path for path in plot_paths if path]
