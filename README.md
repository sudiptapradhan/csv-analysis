# Power NOP Analyzer
A Flask-based web application to analyze and visualize changes in NOP (Net Open Position) volume and market value between two CSV snapshots. This tool helps identify exposure hotspots, significant volume shifts, and breach thresholds.

## Features
- Upload two CSVs (current and previous snapshots)
- Analyze VOLUME_BL and MKT_VAL_BL changes
- Identify exposure hotspots
- Segment & Horizon-based heatmaps
- Breach alerts for >5% NOP volume changes
- Graphs and summary cards in the report

## Prerequisites
- Python 3.10 or later

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sudiptapradhan/csv-analysis
```
### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
python app.py
```
