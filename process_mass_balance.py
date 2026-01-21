#!/usr/bin/env python3
"""
Final improved script to process and visualize glacier mass balance measurements from Nissai glacier.
Properly parses the three separate sections: 2022-23, 2023-24, and 2024-25.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_mass_balance_sections(csv_file):
    """
    Parse the CSV file by sections based on the header rows.
    """
    # Read the entire file
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    # Find header rows that indicate the start of each section
    section_starts = []
    for i, line in enumerate(lines):
        if line.startswith('Stake,X,Y,GPS name,elevation'):
            section_starts.append(i)
    
    print(f"Found {len(section_starts)} sections at lines: {section_starts}")
    
    all_data = []
    
    # Process each section
    for section_idx, start_line in enumerate(section_starts):
        header = lines[start_line].strip()
        print(f"\nProcessing section {section_idx + 1} starting at line {start_line + 1}")
        print(f"Header: {header[:100]}...")
        
        # Determine the year based on the header
        if 'mass balance 2022-23' in header:
            year = '2022-23'
            mass_balance_col = 'mass balance 2022-23 (m w.e.)'
        elif 'mass balance 2023-24' in header:
            year = '2023-24'
            mass_balance_col = 'mass balance 2023-24 (m w.e.)'
        elif 'mass balance 2024-25' in header:
            year = '2024-25'
            mass_balance_col = 'mass balance 2024-25 (m w.e.)'
        else:
            continue
        
        # Find the end of this section (next header or end of file)
        if section_idx + 1 < len(section_starts):
            end_line = section_starts[section_idx + 1]
        else:
            end_line = len(lines)
        
        # Extract data for this section
        section_lines = lines[start_line:end_line]
        
        # Create a temporary CSV for this section
        temp_csv_content = ''.join(section_lines)
        
        # Write to temporary file and read with pandas
        temp_file = f'/tmp/temp_section_{section_idx}.csv'
        with open(temp_file, 'w') as f:
            f.write(temp_csv_content)
        
        try:
            df_section = pd.read_csv(temp_file)
            print(f"Read {len(df_section)} rows for {year}")
            
            # Filter for rows with valid mass balance data
            valid_rows = df_section[
                pd.notna(df_section[mass_balance_col]) & 
                (df_section[mass_balance_col] != '') &
                pd.notna(df_section['Stake']) &
                (df_section['Stake'] != '') &
                pd.notna(df_section['elevation']) &
                (df_section['elevation'] != '')
            ]
            
            print(f"Found {len(valid_rows)} valid measurements for {year}")
            
            # Convert to our format
            for _, row in valid_rows.iterrows():
                try:
                    stake = int(float(row['Stake']))
                    elevation = float(row['elevation'])
                    mass_balance = float(row[mass_balance_col])
                    
                    all_data.append({
                        'Stake': stake,
                        'elevation': elevation,
                        'Mass_Balance_mwe': mass_balance,
                        'Year': year
                    })
                except (ValueError, TypeError) as e:
                    print(f"Skipping row due to conversion error: {e}")
                    continue
        
        except Exception as e:
            print(f"Error processing section {section_idx + 1}: {e}")
            continue
    
    return pd.DataFrame(all_data)

def plot_mass_balance_comprehensive(df):
    """
    Create comprehensive plots of the mass balance data.
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    years = sorted(df['Year'].unique())
    
    # Plot 1: Mass balance by stake for each year (top left)
    ax1 = plt.subplot(2, 3, 1)
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year].groupby('Stake')['Mass_Balance_mwe'].mean().reset_index()
        ax1.plot(year_data['Stake'], year_data['Mass_Balance_mwe'], 
                marker='o', linewidth=2.5, markersize=7, label=year, color=colors[i])
    
    ax1.set_xlabel('Stake Number', fontsize=12)
    ax1.set_ylabel('Mass Balance (m w.e.)', fontsize=12)
    ax1.set_title('Mass Balance by Stake (Annual Averages)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mass balance vs elevation (top middle)
    ax2 = plt.subplot(2, 3, 2)
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year]
        ax2.scatter(year_data['elevation'], year_data['Mass_Balance_mwe'], 
                   s=80, alpha=0.7, label=year, color=colors[i])
    
    ax2.set_xlabel('Elevation (m)', fontsize=12)
    ax2.set_ylabel('Mass Balance (m w.e.)', fontsize=12)
    ax2.set_title('Mass Balance vs Elevation', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of mass balance by year (top right)
    ax3 = plt.subplot(2, 3, 3)
    data_by_year = [df[df['Year'] == year]['Mass_Balance_mwe'].values for year in years]
    box_plot = ax3.boxplot(data_by_year, labels=years, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax3.set_ylabel('Mass Balance (m w.e.)', fontsize=12)
    ax3.set_title('Distribution by Year', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar plot of mean mass balance by year (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    yearly_stats = df.groupby('Year')['Mass_Balance_mwe'].agg(['mean', 'std', 'count']).reset_index()
    yearly_stats = yearly_stats.sort_values('Year')
    
    bars = ax4.bar(yearly_stats['Year'], yearly_stats['mean'], 
                   yerr=yearly_stats['std'], capsize=5, alpha=0.8, 
                   color=colors[:len(yearly_stats)])
    
    ax4.set_ylabel('Mean Mass Balance (m w.e.)', fontsize=12)
    ax4.set_title('Annual Mean Mass Balance', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val, n_obs in zip(bars, yearly_stats['mean'], yearly_stats['count']):
        height = bar.get_height()
        y_pos = height - 0.3 if height < 0 else height + 0.1
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{mean_val:.2f}\n(n={n_obs})', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # Plot 5: Elevation profile with mass balance (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate mean mass balance by elevation bins
    df['elev_bin'] = pd.cut(df['elevation'], bins=8)
    elev_profile = df.groupby(['elev_bin', 'Year'])['Mass_Balance_mwe'].mean().unstack(fill_value=np.nan)
    
    x_pos = range(len(elev_profile.index))
    width = 0.25
    
    for i, year in enumerate(years):
        if year in elev_profile.columns:
            offset = (i - len(years)/2 + 0.5) * width
            ax5.bar([x + offset for x in x_pos], elev_profile[year], 
                   width=width, alpha=0.8, label=year, color=colors[i])
    
    ax5.set_xlabel('Elevation Bins', fontsize=12)
    ax5.set_ylabel('Mean Mass Balance (m w.e.)', fontsize=12)
    ax5.set_title('Mass Balance by Elevation Bands', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'{int(interval.mid)}m' for interval in elev_profile.index], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Time series trend (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate overall mean for each year
    yearly_means = df.groupby('Year')['Mass_Balance_mwe'].mean()
    yearly_means = yearly_means.sort_index()
    
    # Create x-axis positions for years
    year_positions = list(range(len(yearly_means)))
    ax6.plot(year_positions, yearly_means.values, 'o-', linewidth=3, markersize=10, color='red')
    
    # Add trend line
    if len(year_positions) > 1:
        z = np.polyfit(year_positions, yearly_means.values, 1)
        p = np.poly1d(z)
        ax6.plot(year_positions, p(year_positions), '--', linewidth=2, alpha=0.7, color='gray')
    
    ax6.set_xticks(year_positions)
    ax6.set_xticklabels(yearly_means.index)
    ax6.set_ylabel('Mean Mass Balance (m w.e.)', fontsize=12)
    ax6.set_title('Temporal Trend', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Nissai Glacier Mass Balance Analysis (2022-2025)', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save to PNG
    output_path = "/Users/joel/src/nissai_mass_balance/nissai_mass_balance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")

    plt.show()

    return fig

def create_summary_table(df):
    """
    Create comprehensive summary statistics.
    """
    # Basic statistics by year
    basic_stats = df.groupby('Year')['Mass_Balance_mwe'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    basic_stats.columns = ['N_measurements', 'Mean_mwe', 'Std_mwe', 'Min_mwe', 'Max_mwe']
    
    # Additional statistics
    summary_stats = []
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        stats = {
            'Year': year,
            'N_stakes': year_data['Stake'].nunique(),
            'N_measurements': len(year_data),
            'Mean_mwe': year_data['Mass_Balance_mwe'].mean(),
            'Std_mwe': year_data['Mass_Balance_mwe'].std(),
            'Min_mwe': year_data['Mass_Balance_mwe'].min(),
            'Max_mwe': year_data['Mass_Balance_mwe'].max(),
            'Median_mwe': year_data['Mass_Balance_mwe'].median(),
            'Elev_range': f"{year_data['elevation'].min():.0f}-{year_data['elevation'].max():.0f}m"
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MASS BALANCE SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.round(3).to_string(index=False))
    print("="*80)
    
    return summary_df

def main():
    """
    Main function to process and visualize mass balance data.
    """
    input_file = "/Users/joel/src/nissai_mass_balance/Nissai_massbalance_RAW.csv"
    
    print("Processing Nissai Glacier Mass Balance Data...")
    print(f"Reading data from: {input_file}\n")
    
    # Parse the data
    df = parse_mass_balance_sections(input_file)
    
    if df.empty:
        print("No valid data found!")
        return None, None
    
    print(f"\n{'='*60}")
    print(f"DATA PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total measurements processed: {len(df)}")
    print(f"Years covered: {', '.join(sorted(df['Year'].unique()))}")
    print(f"Stakes measured: {sorted(df['Stake'].unique())}")
    print(f"Elevation range: {df['elevation'].min():.0f} - {df['elevation'].max():.0f} m")
    
    # Show sample data
    print(f"\nSample of processed data:")
    print(df.head(10).to_string(index=False))
    
    # Create comprehensive summary
    summary = create_summary_table(df)
    
    # Create plots
    print("\nGenerating comprehensive visualization...")
    fig = plot_mass_balance_comprehensive(df)
    
    # Save processed data
    output_file = "/Users/joel/src/nissai_mass_balance/Nissai_massbalance_FINAL.csv"
    df_sorted = df.sort_values(['Year', 'Stake', 'elevation']).reset_index(drop=True)
    df_sorted.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Save summary
    summary_file = "/Users/joel/src/nissai_mass_balance/Nissai_massbalance_SUMMARY.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    return df, summary

if __name__ == "__main__":
    df, summary = main()