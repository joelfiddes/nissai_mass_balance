# Nissai Glacier Mass Balance Analysis

Analysis of glacier mass balance measurements from Nissai glacier (Pamir region) spanning 2022-2025.

![Mass Balance Analysis](nissai_mass_balance_analysis.png)

## Overview

This project processes and visualizes mass balance data collected from 7 measurement stakes on Nissai glacier at elevations ranging from ~4,037m to ~4,398m.

**Key findings:**
- All measurements show negative mass balance (glacier retreat)
- 2022-23: Mean -2.20 m w.e.
- 2023-24: Mean -1.14 m w.e.
- 2024-25: Mean -3.26 m w.e.
- Overall trend: -0.53 m w.e./year

## Files

| File | Description |
|------|-------------|
| `Nissai_massbalance_RAW.csv` | Raw field measurements |
| `Nissai_massbalance_FINAL.csv` | Processed measurements (stake, elevation, mass balance, year) |
| `Nissai_massbalance_SUMMARY.csv` | Summary statistics by year |
| `process_mass_balance.py` | Full processing pipeline with 6-panel output |
| `plot_mass_balance_simplified.py` | Generate 4-panel visualization |

## Usage

```bash
pip install -r requirements.txt
python process_mass_balance.py
```

## License

MIT
