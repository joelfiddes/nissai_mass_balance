#!/usr/bin/env python3
"""
Simplified visualization script for Nissai glacier mass balance data.
Shows only panels 1, 3, 5, and 6 as requested.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mass_balance_simplified(df):
    """
    Create simplified plots with only panels 1, 3, 5, and 6.
    """
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    years = sorted(df['Year'].unique())
    
    # Panel 1: Mass balance by stake elevation for each year (top left)
    ax1 = axes[0, 0]
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year].groupby('Stake').agg({
            'Mass_Balance_mwe': 'mean', 
            'elevation': 'mean'
        }).reset_index()
        ax1.plot(year_data['elevation'], year_data['Mass_Balance_mwe'], 
                marker='o', linewidth=2.5, markersize=7, label=year, color=colors[i])
    
    ax1.set_xlabel('Stake Elevation (m)', fontsize=12)
    ax1.set_ylabel('Mass Balance (m w.e.)', fontsize=12)
    ax1.set_title('Mass Balance by Stake Elevation (Annual Averages)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 3: Box plot of mass balance by year (top right)
    ax3 = axes[0, 1]
    data_by_year = [df[df['Year'] == year]['Mass_Balance_mwe'].values for year in years]
    box_plot = ax3.boxplot(data_by_year, labels=years, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax3.set_ylabel('Mass Balance (m w.e.)', fontsize=12)
    ax3.set_title('Distribution by Year', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 5: Elevation profile with mass balance (bottom left)
    ax5 = axes[1, 0]
    
    # Calculate mean mass balance by elevation bins
    df['elev_bin'] = pd.cut(df['elevation'], bins=6)
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
    
    # Panel 6: Time series trend (bottom right)
    ax6 = axes[1, 1]
    
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
        
        # Add trend equation
        slope = z[0]
        intercept = z[1]
        ax6.text(0.05, 0.95, f'Trend: {slope:.3f} m w.e./year', 
                transform=ax6.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax6.set_xticks(year_positions)
    ax6.set_xticklabels(yearly_means.index)
    ax6.set_ylabel('Mean Mass Balance (m w.e.)', fontsize=12)
    ax6.set_title('Temporal Trend', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Nissai Glacier Mass Balance Analysis (2022-2025)', fontsize=16, fontweight='bold')
    
    # Add caption
    caption = ("Figure: Mass balance measurements from Nissai glacier showing (a) elevation-dependent mass balance patterns, "
              "(b) annual distribution variability, (c) elevation band analysis, and (d) temporal trends. "
              "Data shows increasingly negative mass balance from 2022-23 (-2.20 m w.e.) through 2024-25 (-3.26 m w.e.), "
              "with elevation-dependent patterns indicating greater losses at lower elevations.")
    
    fig.text(0.1, 0.02, caption, fontsize=10, ha='left', va='bottom', wrap=True, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for caption

    # Save to PNG
    output_path = "/Users/joel/src/nissai_mass_balance/nissai_mass_balance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")

    plt.show()

    return fig

def main():
    """
    Main function to create simplified visualization.
    """
    # Read the processed data
    input_file = "/Users/joel/src/nissai_mass_balance/Nissai_massbalance_FINAL.csv"
    
    print("Creating simplified visualization with panels 1, 3, 5, and 6...")
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} measurements for {len(df['Year'].unique())} years")
        
        # Create simplified plots
        fig = plot_mass_balance_simplified(df)
        
        print("Visualization complete!")
        
        return fig
        
    except FileNotFoundError:
        print(f"Could not find processed data file: {input_file}")
        print("Please run the full processing script first.")
        return None

if __name__ == "__main__":
    fig = main()