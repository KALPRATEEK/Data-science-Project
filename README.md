# Data-science-Project

A financial data science project focused on collecting, cleaning, aligning, and visualizing market data for exploratory investment analysis.

## Overview

This repository explores how data science can be applied to financial markets using a practical workflow:

- collect historical market data
- clean and align datasets across assets
- compute daily percentage changes
- compare stocks and assets visually
- inspect raw market behavior
- enrich the workflow with company calendar and news data sources

The goal of the project is to build a strong foundation for investment-oriented data analysis and prepare the data pipeline for more advanced forecasting or machine learning work later.

---

## What this project does

The project currently focuses on exploratory analysis and data preparation rather than a final predictive model.

It includes:

- historical market data import with `yfinance`
- cleaning and standardizing CSV files
- aligning timestamps across assets
- percentage-change analysis for:
  - Open
  - High
  - Low
  - Volume
- comparative visualizations across multiple stocks/assets
- per-stock visualizations of both raw values and daily percentage changes
- experiments with additional financial context such as:
  - earnings / calendar data
  - company news import

---

## Why this project is interesting

Investment projects often jump directly into prediction without building a reliable data workflow first.
This project is interesting because it shows the earlier and more important steps:

- collecting real market data
- making different assets comparable on the same timeline
- cleaning noisy datasets
- generating interpretable visual outputs
- understanding behavior before applying machine learning

That makes it useful for:

- financial data science learning
- portfolio / GitHub presentation
- academic project development
- future stock prediction or ranking systems

---

## Assets used in the project
The repository contains data for a mix of major stocks and market assets, including examples such as:

- AAPL
- AMZN
- GOOGL
- META
- MSFT
- NVDA
- PLTR
- TSLA
- UBER
- HOOD
- BTC-USD
- GC=F (Gold Futures)

This mix makes the project more interesting because it compares both equities and non-equity assets in one workflow.

---

## Project structure

- `cleanData.ipynb`  
  Main notebook for cleaning, aligning, and visualizing the data.

- `data/`  
  Historical CSV files used for analysis.

- `import_data/`  
  Python scripts for downloading market data.

- `plots/`  
  Comparative percentage-change plots across assets.

- `plots_per_stock/`  
  Per-stock plots showing multi-metric changes and raw values.

- `Documentation/`  
  Supporting project documents, report material, and presentation files.

---

## Workflow

### 1. Data import
Historical market data is downloaded with Python scripts.

### 2. Date alignment
Because some assets can have different trading calendars, the project aligns timestamps so datasets can be compared more fairly.

### 3. Data cleaning
The notebook removes empty rows, invalid dates, and rows with missing numeric values, then standardizes formatting.

### 4. Feature transformation
The project computes daily percentage changes for key columns such as Open, High, Low, and Volume.

### 5. Visualization
Several chart groups are generated:
- single plots per file and metric
- comparison plots across assets
- combined plots per stock
- raw value plots with price and volume

---

## What is shown in the outputs
The project generates multiple visual outputs that help explain market behavior clearly.

### Comparative plots
These compare daily percentage changes across assets for:
- Open
- High
- Low
- Volume

These are useful for identifying which assets moved more aggressively or more steadily over time.
### Per-stock percentage-change plots
These show how Open, High, Low, and Volume changed day by day for one asset.

These are useful for understanding the internal behavior of a single stock.

### Raw value plots
These show Open, High, Low, and Volume together, helping connect price behavior with trading activity.

---

## Example use cases

This project can be used for:

- comparing volatility across major stocks
- exploring how volume reacts to price changes
- checking whether different assets move similarly over time
- preparing consistent input data for machine learning models
- building future forecasting or stock-ranking pipelines

---

## Running the project

### Requirements

Install the main Python packages used in the project:

```bash
pip install pandas numpy matplotlib yfinance finnhub-python


