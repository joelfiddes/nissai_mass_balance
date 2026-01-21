#!/usr/bin/env python3
"""
Script to process and visualize glacier mass balance measurements from Nissai glacier.
The data contains three measurement periods: 2022-23, 2023-24, and 2024-25.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_mass_balance_data(csv_file):
    """
    Read and process the mass balance CSV file.
    The file contains three separate tables for different years.
    """
    # Read the raw CSV file
    df_raw = pd.read_csv(csv_file)
    
    # Find the row indices where each year's data starts
    year_starts = []
    for idx, row in df_raw.iterrows():
        if pd.notna(row['Stake']) and str(row['Stake']).strip() == 'Stake':
            year_starts.append(idx)
    
    # Extract data for each year
    datasets = []
    
    # 2022-23 data (first section)
    if len(year_starts) >= 2:
        df_2022_23 = df_raw.iloc[year_starts[0]+1:year_starts[1]-3].copy()
        df_2022_23 = df_2022_23.dropna(subset=['mass balance 2022-23 (m w.e.)'])
        df_2022_23 = df_2022_23[df_2022_23['mass balance 2022-23 (m w.e.)'] != '']
        df_2022_23['Year'] = '2022-23'
        df_2022_23['Mass_Balance_mwe'] = pd.to_numeric(df_2022_23['mass balance 2022-23 (m w.e.)'], errors='coerce')
        datasets.append(df_2022_23[['Stake', 'elevation', 'Mass_Balance_mwe', 'Year']])
    
    # 2023-24 data (second section)
    if len(year_starts) >= 3:
        df_2023_24 = df_raw.iloc[year_starts[1]+1:year_starts[2]-3].copy()
        df_2023_24 = df_2023_24.dropna(subset=['mass balance 2023-24 (m w.e.)'])
        df_2023_24 = df_2023_24[df_2023_24['mass balance 2023-24 (m w.e.)'] != '']
        df_2023_24['Year'] = '2023-24'
        df_2023_24['Mass_Balance_mwe'] = pd.to_numeric(df_2023_24['mass balance 2023-24 (m w.e.)'], errors='coerce')
        datasets.append(df_2023_24[['Stake', 'elevation', 'Mass_Balance_mwe', 'Year']])
    
    # 2024-25 data (third section)
    if len(year_starts) >= 3:
        df_2024_25 = df_raw.iloc[year_starts[2]+1:].copy()
        df_2024_25 = df_2024_25.dropna(subset=['mass balance 2024-25 (m w.e.)'])
        df_2024_25 = df_2024_25[df_2024_25['mass balance 2024-25 (m w.e.)'] != '']
        df_2024_25['Year'] = '2024-25'
        df_2024_25['Mass_Balance_mwe'] = pd.to_numeric(df_2024_25['mass balance 2024-25 (m w.e.)'], errors='coerce')
        datasets.append(df_2024_25[['Stake', 'elevation', 'Mass_Balance_mwe', 'Year']])
    
    # Combine all datasets
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        combined_df = combined_df.dropna(subset=['Mass_Balance_mwe'])
        combined_df['Stake'] = pd.to_numeric(combined_df['Stake'], errors='coerce')
        combined_df['elevation'] = pd.to_numeric(combined_df['elevation'], errors='coerce')
        return combined_df
    else:
        return pd.DataFrame()

def plot_mass_balance_data(df):
    """
    Create comprehensive plots of the mass balance data.
    """
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nissai Glacier Mass Balance Measurements', fontsize=16, fontweight='bold')
    
    # Plot 1: Mass balance by stake for each year
    ax1 = axes[0, 0]
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        ax1.plot(year_data['Stake'], year_data['Mass_Balance_mwe'], 
                marker='o', linewidth=2, markersize=6, label=year)
    
    ax1.set_xlabel('Stake Number')
    ax1.set_ylabel('Mass Balance (m w.e.)')
    ax1.set_title('Mass Balance by Stake')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mass balance vs elevation
    ax2 = axes[0, 1]
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        ax2.scatter(year_data['elevation'], year_data['Mass_Balance_mwe'], 
                   s=60, alpha=0.7, label=year)
    
    ax2.set_xlabel('Elevation (m)')
    ax2.set_ylabel('Mass Balance (m w.e.)')
    ax2.set_title('Mass Balance vs Elevation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of mass balance by year
    ax3 = axes[1, 0]
    years = df['Year'].unique()
    data_by_year = [df[df['Year'] == year]['Mass_Balance_mwe'].values for year in years]
    box_plot = ax3.boxplot(data_by_year, labels=years, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        
    ax3.set_ylabel('Mass Balance (m w.e.)')
    ax3.set_title('Mass Balance Distribution by Year')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar plot of mean mass balance by year
    ax4 = axes[1, 1]
    yearly_means = df.groupby('Year')['Mass_Balance_mwe'].agg(['mean', 'std']).reset_index()
    bars = ax4.bar(yearly_means['Year'], yearly_means['mean'], 
                   yerr=yearly_means['std'], capsize=5, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax4.set_ylabel('Mean Mass Balance (m w.e.)')
    ax4.set_title('Annual Mean Mass Balance')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, yearly_means['mean']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mean_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_summary_table(df):
    """
    Create a summary table with statistics for each year.
    """
    summary = df.groupby('Year')['Mass_Balance_mwe'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    summary.columns = ['N_measurements', 'Mean_mwe', 'Std_mwe', 'Min_mwe', 'Max_mwe']
    
    print("\n" + "="*50)
    print("MASS BALANCE SUMMARY STATISTICS")
    print("="*50)
    print(summary)
    print("="*50)
    
    return summary

def save_processed_data(df, output_file):
    """
    Save the processed data to a CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

def main():
    """
    Main function to process and visualize mass balance data.
    """
    # Input file path
    input_file = "/Users/joel/Downloads/UPDATE_Nissai_massbalance_Sangvor_PAMIR - Sheet5.csv"
    
    print("Processing Nissai Glacier Mass Balance Data...")
    print(f"Reading data from: {input_file}")
    
    # Read and process data
    df = read_mass_balance_data(input_file)
    
    if df.empty:
        print("No valid data found in the CSV file!")
        return
    
    print(f"\nSuccessfully processed {len(df)} mass balance measurements")
    print(f"Years covered: {', '.join(df['Year'].unique())}")
    
    # Display first few rows
    print("\nFirst 10 rows of processed data:")
    print(df.head(10))
    
    # Create summary statistics
    summary = create_summary_table(df)
    
    # Create plots
    print("\nGenerating plots...")
    fig = plot_mass_balance_data(df)
    
    # Save processed data
    output_file = "/Users/joel/src/TopoPyScale/nissai_mass_balance_processed.csv"
    save_processed_data(df, output_file)
    
    # Save summary statistics
    summary_file = "/Users/joel/src/TopoPyScale/nissai_mass_balance_summary.csv"
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")
    
    return df, summary

if __name__ == "__main__":
    df, summary = main()