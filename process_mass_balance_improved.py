#!/usr/bin/env python3
"""
Improved script to process and visualize glacier mass balance measurements from Nissai glacier.
The data contains three measurement periods: 2022-23, 2023-24, and 2024-25.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def read_mass_balance_data_improved(csv_file):
    """
    Read and process the mass balance CSV file with improved parsing.
    """
    # Read the entire file as text first
    with open(csv_file, 'r') as f:
        content = f.read()
    
    # Split by lines
    lines = content.split('\n')
    
    # Find sections for each year by looking for specific patterns
    datasets = []
    
    # Process 2022-23 data
    data_2022_23 = []
    for i, line in enumerate(lines):
        if 'mass balance 2022-23 (m w.e.)' in line and line.count(',') > 10:
            # Found a data row for 2022-23
            parts = line.split(',')
            try:
                stake = int(float(parts[0])) if parts[0].strip() else None
                elevation = float(parts[4]) if parts[4].strip() else None
                mass_balance = float(parts[11]) if parts[11].strip() else None
                
                if stake is not None and mass_balance is not None and elevation is not None:
                    data_2022_23.append({
                        'Stake': stake,
                        'elevation': elevation,
                        'Mass_Balance_mwe': mass_balance,
                        'Year': '2022-23'
                    })
            except (ValueError, IndexError):
                continue
    
    # Process 2023-24 data
    data_2023_24 = []
    for i, line in enumerate(lines):
        if 'mass balance 2023-24 (m w.e.)' in line and line.count(',') > 10:
            # Found a data row for 2023-24
            parts = line.split(',')
            try:
                stake = int(float(parts[0])) if parts[0].strip() else None
                elevation = float(parts[4]) if parts[4].strip() else None
                mass_balance = float(parts[11]) if parts[11].strip() else None
                
                if stake is not None and mass_balance is not None and elevation is not None:
                    data_2023_24.append({
                        'Stake': stake,
                        'elevation': elevation,
                        'Mass_Balance_mwe': mass_balance,
                        'Year': '2023-24'
                    })
            except (ValueError, IndexError):
                continue
    
    # Process 2024-25 data
    data_2024_25 = []
    for i, line in enumerate(lines):
        if 'mass balance 2024-25 (m w.e.)' in line and line.count(',') > 10:
            # Found a data row for 2024-25
            parts = line.split(',')
            try:
                stake = int(float(parts[0])) if parts[0].strip() else None
                elevation = float(parts[4]) if parts[4].strip() else None
                mass_balance = float(parts[11]) if parts[11].strip() else None
                
                if stake is not None and mass_balance is not None and elevation is not None:
                    data_2024_25.append({
                        'Stake': stake,
                        'elevation': elevation,
                        'Mass_Balance_mwe': mass_balance,
                        'Year': '2024-25'
                    })
            except (ValueError, IndexError):
                continue
    
    # Combine all datasets
    all_data = data_2022_23 + data_2023_24 + data_2024_25
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    else:
        return pd.DataFrame()

def manual_parse_data(csv_file):
    """
    Manually parse the data by reading the CSV and extracting sections.
    """
    df_raw = pd.read_csv(csv_file)
    
    all_data = []
    
    # Look through all rows to find valid mass balance data
    for index, row in df_raw.iterrows():
        # Check for 2022-23 data
        if pd.notna(row.get('mass balance 2022-23 (m w.e.)')):
            try:
                mass_balance = float(row['mass balance 2022-23 (m w.e.)'])
                stake = int(float(row['Stake']))
                elevation = float(row['elevation'])
                
                all_data.append({
                    'Stake': stake,
                    'elevation': elevation,
                    'Mass_Balance_mwe': mass_balance,
                    'Year': '2022-23'
                })
            except (ValueError, TypeError):
                pass
        
        # Check for 2023-24 data
        if pd.notna(row.get('mass balance 2023-24 (m w.e.)')):
            try:
                mass_balance = float(row['mass balance 2023-24 (m w.e.)'])
                stake = int(float(row['Stake']))
                elevation = float(row['elevation'])
                
                all_data.append({
                    'Stake': stake,
                    'elevation': elevation,
                    'Mass_Balance_mwe': mass_balance,
                    'Year': '2023-24'
                })
            except (ValueError, TypeError):
                pass
        
        # Check for 2024-25 data
        if pd.notna(row.get('mass balance 2024-25 (m w.e.)')):
            try:
                mass_balance = float(row['mass balance 2024-25 (m w.e.)'])
                stake = int(float(row['Stake']))
                elevation = float(row['elevation'])
                
                all_data.append({
                    'Stake': stake,
                    'elevation': elevation,
                    'Mass_Balance_mwe': mass_balance,
                    'Year': '2024-25'
                })
            except (ValueError, TypeError):
                pass
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
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
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Mass balance by stake for each year
    ax1 = axes[0, 0]
    for i, year in enumerate(sorted(df['Year'].unique())):
        year_data = df[df['Year'] == year].sort_values('Stake')
        ax1.plot(year_data['Stake'], year_data['Mass_Balance_mwe'], 
                marker='o', linewidth=2, markersize=6, label=year, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Stake Number')
    ax1.set_ylabel('Mass Balance (m w.e.)')
    ax1.set_title('Mass Balance by Stake')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mass balance vs elevation
    ax2 = axes[0, 1]
    for i, year in enumerate(sorted(df['Year'].unique())):
        year_data = df[df['Year'] == year]
        ax2.scatter(year_data['elevation'], year_data['Mass_Balance_mwe'], 
                   s=60, alpha=0.7, label=year, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Elevation (m)')
    ax2.set_ylabel('Mass Balance (m w.e.)')
    ax2.set_title('Mass Balance vs Elevation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of mass balance by year
    ax3 = axes[1, 0]
    years = sorted(df['Year'].unique())
    data_by_year = [df[df['Year'] == year]['Mass_Balance_mwe'].values for year in years]
    box_plot = ax3.boxplot(data_by_year, tick_labels=years, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax3.set_ylabel('Mass Balance (m w.e.)')
    ax3.set_title('Mass Balance Distribution by Year')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar plot of mean mass balance by year
    ax4 = axes[1, 1]
    yearly_means = df.groupby('Year')['Mass_Balance_mwe'].agg(['mean', 'std']).reset_index()
    yearly_means = yearly_means.sort_values('Year')
    
    bars = ax4.bar(yearly_means['Year'], yearly_means['mean'], 
                   yerr=yearly_means['std'], capsize=5, alpha=0.7, 
                   color=colors[:len(yearly_means)])
    
    ax4.set_ylabel('Mean Mass Balance (m w.e.)')
    ax4.set_title('Annual Mean Mass Balance')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, yearly_means['mean']):
        height = bar.get_height()
        y_pos = height + 0.05 if height < 0 else height + 0.05
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
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
    summary = summary.sort_index()
    
    print("\n" + "="*60)
    print("MASS BALANCE SUMMARY STATISTICS")
    print("="*60)
    print(summary)
    print("="*60)
    
    return summary

def save_processed_data(df, output_file):
    """
    Save the processed data to a CSV file.
    """
    df_sorted = df.sort_values(['Year', 'Stake']).reset_index(drop=True)
    df_sorted.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

def main():
    """
    Main function to process and visualize mass balance data.
    """
    # Input file path
    input_file = "/Users/joel/Downloads/UPDATE_Nissai_massbalance_Sangvor_PAMIR - Sheet5.csv"
    
    print("Processing Nissai Glacier Mass Balance Data...")
    print(f"Reading data from: {input_file}")
    
    # Try manual parsing method
    df = manual_parse_data(input_file)
    
    if df.empty:
        print("No valid data found in the CSV file!")
        return
    
    print(f"\nSuccessfully processed {len(df)} mass balance measurements")
    print(f"Years covered: {', '.join(sorted(df['Year'].unique()))}")
    
    # Display data by year
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        print(f"\n{year}: {len(year_data)} measurements")
        print(year_data[['Stake', 'elevation', 'Mass_Balance_mwe']].to_string(index=False))
    
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