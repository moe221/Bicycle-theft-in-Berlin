# Berlin Bicycle Theft Analysis Dashboard

This dashboard visualizes bicycle theft patterns in Berlin, helping users understand high-risk areas and find safer parking spots.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/moe221/Bicycle-theft-in-Berlin.git
cd Bicycle-theft-in-Berlin
```

### 2. Set Up Python Environment
It's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Data Files Structure
Ensure the following data files are present in the `data/` directory:
- `Fahrraddiebstahl.csv` - Bicycle theft data
- `lor_2021-01-01_k3_uebersicht_id_namen.xlsx` - Berlin LOR information
- `Einwohnerregisterstatistik-Berlin-Dec2024.csv` - Population data
- `lor_2021-01-01_k3_shapefiles_nur_id.7z` - Shapefile for map visualization

### 5. Run the Dashboard
```bash
streamlit run dashboard.py
```
The dashboard will open in your default web browser at `http://localhost:8501`

## Features
- Interactive map showing theft density across Berlin
- Time-based heatmap of theft patterns
- Safe parking spot recommendations based on location
- District-level filtering and analysis
