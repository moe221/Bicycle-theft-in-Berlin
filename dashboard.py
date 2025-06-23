import pandas as pd
import numpy as np
import altair as alt
import folium
import geopandas as gpd
import py7zr
import os
from geopy.geocoders import Nominatim
import warnings
from shapely.geometry import Point
import streamlit as st

warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare all required datasets"""
    # Load the Berlin administrative areas data (LOR)
    lor_info = pd.read_excel("data/lor_2021-01-01_k3_uebersicht_id_namen.xlsx", sheet_name="LOR_2023_Übersicht")

    # Load the bicycle theft data
    bike_thefts = pd.read_csv("data/Fahrraddiebstahl.csv", encoding='latin-1')

    # Load berlin population data per LOR
    data_population = pd.read_csv("data/Einwohnerregisterstatistik-Berlin-Dec2024.csv")

    # Clean population data
    data_population = data_population[data_population["Population"] != "–"]
    data_population["Population"] = data_population["Population"].map(float)

    return lor_info, bike_thefts, data_population

def prepare_data(lor_info, bike_thefts, data_population):
    """Clean and prepare the data for analysis"""
    # Join LOR info
    bike_thefts_clean = bike_thefts.merge(lor_info, how="left", left_on="LOR", right_on="PLR_ID")

    # Format dates
    bike_thefts_clean["TATZEIT_ANFANG_DATUM"] = pd.to_datetime(bike_thefts_clean["TATZEIT_ANFANG_DATUM"], format="%d.%m.%Y")
    bike_thefts_clean["TATZEIT_ENDE_DATUM"] = pd.to_datetime(bike_thefts_clean["TATZEIT_ENDE_DATUM"], format="%d.%m.%Y")

    # Add year and month
    bike_thefts_clean["YEAR"] = bike_thefts_clean["TATZEIT_ANFANG_DATUM"].dt.year
    bike_thefts_clean["MONTH"] = bike_thefts_clean["TATZEIT_ANFANG_DATUM"].dt.month

    # Load and prepare shapefile
    shapefile_archive_path = "data/lor_2021-01-01_k3_shapefiles_nur_id.7z"
    shapefile_extract_path = "data/shapefiles"

    # Extract shapefile if not already extracted
    if not os.path.exists(shapefile_extract_path):
        with py7zr.SevenZipFile(shapefile_archive_path, mode='r') as archive:
            archive.extractall(path=shapefile_extract_path)

    # Find and load shapefile
    shapefile_path = ""
    for root, dirs, files in os.walk(shapefile_extract_path):
        for file in files:
            if file.endswith(".shp"):
                shapefile_path = os.path.join(root, file)
                break

    gdf = gpd.read_file(shapefile_path)

    return bike_thefts_clean, gdf

def prepare_theft_data(bike_thefts_clean, data_population):
    """Prepare theft data for visualization"""
    bike_thefts_total = bike_thefts_clean.groupby([
        "LOR", "BLN_ID", "BEZ_ID", "PGR_ID", "PGR",
        "BZR_ID", "BZR", "PLR_ID", "PLR"
    ])[["VERSUCH"]].count().reset_index()

    # Merge with population data
    bike_thefts_total = bike_thefts_total.merge(data_population, how="left", on="LOR")

    # Calculate thefts per 1000 residents
    bike_thefts_total["TheftsPer1K"] = (bike_thefts_total["VERSUCH"] / bike_thefts_total["Population"]) * 1000

    return bike_thefts_total

def prepare_time_data(bike_thefts_clean):
    """Prepare time-based data for visualization"""
    # Add day of week and hour information
    bike_thefts_clean["DOW_Name"] = bike_thefts_clean["TATZEIT_ANFANG_DATUM"].dt.day_name()
    bike_thefts_clean["DOW"] = bike_thefts_clean["TATZEIT_ANFANG_DATUM"].dt.weekday

    # Calculate mid-time for time range
    bike_thefts_clean["StartTimeStamp"] = pd.to_datetime(bike_thefts_clean["TATZEIT_ANFANG_DATUM"]) + pd.to_timedelta(bike_thefts_clean["TATZEIT_ANFANG_STUNDE"], unit='h')
    bike_thefts_clean["EndTimeStamp"] = pd.to_datetime(bike_thefts_clean["TATZEIT_ENDE_DATUM"]) + pd.to_timedelta(bike_thefts_clean["TATZEIT_ENDE_STUNDE"], unit='h')
    bike_thefts_clean["MidTime"] = bike_thefts_clean["StartTimeStamp"] + (bike_thefts_clean["EndTimeStamp"] - bike_thefts_clean["StartTimeStamp"]) / 2
    bike_thefts_clean["MidHour"] = bike_thefts_clean["MidTime"].dt.hour

    return bike_thefts_clean

def prepare_aggregated_data(merged_gdf):
    """Prepare aggregated data at different levels"""
    # PLR level
    plr_gdf = merged_gdf.dissolve(by=["PLR_ID", "PLR"], aggfunc={
        "VERSUCH": "sum",
        "Population": "sum"
    }).reset_index()
    plr_gdf["TheftsPer1K"] = (plr_gdf["VERSUCH"] / plr_gdf["Population"]) * 1000

    # BZR level
    bzr_gdf = merged_gdf.dissolve(by=["BZR_ID", "BZR"], aggfunc={
        "VERSUCH": "sum",
        "Population": "sum"
    }).reset_index()
    bzr_gdf["TheftsPer1K"] = (bzr_gdf["VERSUCH"] / bzr_gdf["Population"]) * 1000

    # PGR level
    pgr_gdf = merged_gdf.dissolve(by=["PGR_ID", "PGR"], aggfunc={
        "VERSUCH": "sum",
        "Population": "sum"
    }).reset_index()
    pgr_gdf["TheftsPer1K"] = (pgr_gdf["VERSUCH"] / pgr_gdf["Population"]) * 1000

    return plr_gdf, bzr_gdf, pgr_gdf

def create_theft_map(plr_gdf, bzr_gdf, pgr_gdf):
    """Create the theft density map with multiple layers"""
    m = folium.Map(
        location=[52.5200, 13.4050],
        zoom_start=10,
        tiles="cartodbpositron"
    )

    # PLR layer (with legend)
    plr_choropleth = folium.Choropleth(
        geo_data=plr_gdf,
        data=plr_gdf,
        columns=["PLR_ID", "TheftsPer1K"],
        key_on="feature.properties.PLR_ID",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        name="Planungsraum Level",
        legend_name="Bike Thefts per 1,000 Residents",
        line_weight=0.5,
        highlight=True
    )
    plr_choropleth.add_to(m)

    # Add tooltips to the PLR layer
    folium.GeoJsonTooltip(
        fields=['PLR', 'TheftsPer1K'],
        aliases=['Area:', 'Thefts per 1,000:'],
        labels=True,
        sticky=False,
        localize=True,
        style="""
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #ccc;
            border-radius: 3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            padding: 5px;
            font-family: Arial, sans-serif;
        """
    ).add_to(plr_choropleth.geojson)

    # Make the legend more transparent and styled
    for key in plr_choropleth._children:
        if key.startswith('color_map'):
            legend = plr_choropleth._children[key]
            legend.add_to(m)
            m.get_root().html.add_child(folium.Element("""
                <style>
                    .legend {
                        background-color: rgba(255, 255, 255, 0.8);
                        padding: 10px;
                        border-radius: 5px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.2);
                        font-family: Arial, sans-serif;
                    }
                    .legend i {
                        width: 18px;
                        height: 18px;
                        float: left;
                        margin-right: 8px;
                        opacity: 0.7;
                    }
                </style>
            """))

    folium.LayerControl().add_to(m)
    return m

def create_time_heatmap(bike_thefts_clean, selected_bez=None):
    """Create the time-based heatmap with optional BEZ filter"""
    # Filter data if BEZ is selected
    if selected_bez:
        bike_thefts_clean = bike_thefts_clean[bike_thefts_clean["BZR"] == selected_bez]

    bike_thefts_heatmap = bike_thefts_clean.groupby(["DOW", "DOW_Name", "MidHour"])[["VERSUCH"]].count().reset_index()
    bike_thefts_heatmap = bike_thefts_heatmap.sort_values(by=["DOW"], ascending=False)

    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    title = f"Bike Thefts by Day of Week and Hour{' in ' + selected_bez if selected_bez else ''}"

    # Create base chart with improved styling
    chart = alt.Chart(bike_thefts_heatmap).mark_rect().encode(
        x=alt.X('DOW_Name:O',
                sort=ordered_days,
                title="Day of Week",
                axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=14)),
        y=alt.Y('MidHour:O',
                sort='ascending',
                title="Hour of Day",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        color=alt.Color('VERSUCH:Q',
                       scale=alt.Scale(scheme='reds'),
                       title="Number of Thefts",
                       legend=alt.Legend(titleFontSize=12, labelFontSize=11)),
        tooltip=[
            alt.Tooltip('DOW_Name:O', title='Day'),
            alt.Tooltip('MidHour:O', title='Hour'),
            alt.Tooltip('VERSUCH:Q', title='Thefts')
        ]
    ).properties(
        width="container",
        height=500,
        title=alt.TitleParams(
            text=title,
            fontSize=16,
            anchor='middle',
            dy=-10
        )
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False
    )

    return chart

def get_risk_level(theft_rate):
    """Determine risk level based on theft rate"""
    if theft_rate < 18:
        return "Very Low"
    elif theft_rate < 36:
        return "Low"
    elif theft_rate < 72:
        return "Medium"
    elif theft_rate < 89:
        return "High"
    else:
        return "Very High"

def find_safe_parking_spots(destination_lat, destination_lon, merged_gdf, radius_meters=1000, n_alternatives=2):
    """Find the safest parking spots within a given radius of a destination."""
    # Convert destination to a GeoDataFrame with the same CRS as our data
    destination_point = gpd.GeoDataFrame(
        {'geometry': [Point(destination_lon, destination_lat)]},
        crs="EPSG:4326"
    )
    destination_point = destination_point.to_crs(merged_gdf.crs)

    # Create a buffer around the destination point
    buffer = destination_point.buffer(radius_meters)

    # Find all LORs that intersect with the buffer
    nearby_lors = merged_gdf[merged_gdf.intersects(buffer[0])]

    if nearby_lors.empty:
        return {
            "error": "No areas found within the specified radius. Try increasing the radius or choosing a different location."
        }

    nearby_lors["risk_level"] = nearby_lors["TheftsPer1K"].apply(get_risk_level)

    # Sort by theft rate (ascending)
    nearby_lors = nearby_lors.sort_values(by='TheftsPer1K')

    # Get the safest LOR and alternatives
    safest_lors = nearby_lors.head(n_alternatives + 1)

    # Calculate the centroid of each LOR as the recommended parking spot
    safest_lors['parking_spot'] = safest_lors.geometry.centroid

    # Calculate the distance from the destination to each parking spot
    destination_geom = destination_point.geometry[0]
    safest_lors['distance_meters'] = safest_lors['parking_spot'].apply(
        lambda x: destination_geom.distance(x)
    )

    # Convert the parking spots to WGS84 for display
    safest_lors_wgs84 = safest_lors.copy()
    safest_lors_wgs84['parking_spot_wgs84'] = safest_lors_wgs84['parking_spot']
    safest_lors_wgs84 = safest_lors_wgs84.set_geometry('parking_spot_wgs84')
    safest_lors_wgs84 = safest_lors_wgs84.to_crs("EPSG:4326")

    # Find the LOR that contains the destination point
    destination_lor = merged_gdf[merged_gdf.contains(destination_point.geometry[0])]

    # If the destination point is not within any LOR, use the nearest one
    if destination_lor.empty:
        merged_gdf['distance_to_dest'] = merged_gdf.geometry.apply(
            lambda x: x.distance(destination_point.geometry[0])
        )
        destination_lor = merged_gdf.loc[[merged_gdf['distance_to_dest'].idxmin()]]

    # Create the result dictionary
    result = {
        "destination": {
            "lat": destination_lat,
            "lon": destination_lon
        },
        "destination_area": {
            "name": destination_lor.iloc[0]['PLR'],
            "thefts_per_1k": destination_lor.iloc[0]['TheftsPer1K'],
            "risk_level": get_risk_level(destination_lor.iloc[0]['TheftsPer1K'])
        },
        "recommendations": []
    }

    # Add each recommendation
    for i, (_, lor) in enumerate(safest_lors.iterrows()):
        # Get the WGS84 coordinates of the parking spot
        parking_spot_wgs84 = safest_lors_wgs84.iloc[i]['parking_spot_wgs84']

        recommendation = {
            "rank": i + 1,
            "name": lor['PLR'],
            "PLR_ID": lor['PLR_ID'],  # Add PLR_ID to the recommendation
            "lat": parking_spot_wgs84.y,
            "lon": parking_spot_wgs84.x,
            "distance_meters": int(lor['distance_meters']),
            "thefts_per_1k": lor['TheftsPer1K'],
            "risk_level": get_risk_level(lor['TheftsPer1K'])
        }
        result["recommendations"].append(recommendation)

    return result

def plot_recommendations(results, radius_meters=1000, merged_gdf=None):
    """Create an interactive map with parking recommendations"""
    from shapely.geometry import Point
    import folium
    from folium.plugins import MarkerCluster

    # Destination coordinates
    lat = results['destination']['lat']
    lon = results['destination']['lon']

    # Create base map with CartoDB Positron style
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")

    # Convert destination to GeoSeries and buffer it in gdf CRS
    destination_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(merged_gdf.crs)
    buffer = destination_point.buffer(radius_meters)

    # Filter LORs within buffer
    nearby_lors = merged_gdf[merged_gdf.intersects(buffer[0])].copy()
    nearby_lors = nearby_lors.to_crs("EPSG:4326")

    # Add choropleth for TheftsPer1K in the filtered LORs
    if not nearby_lors.empty:
        folium.Choropleth(
            geo_data=nearby_lors,
            data=nearby_lors,
            columns=["PLR_ID", "TheftsPer1K"],
            key_on="feature.properties.PLR_ID",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            name="Risk by Area",
            show_legend=False
        ).add_to(m)

    # Add radius circle
    folium.Circle(
        location=[lat, lon],
        radius=radius_meters,
        color='blue',
        fill=False,
        dash_array='5, 5',
        tooltip='Search Radius'
    ).add_to(m)

    # Add destination marker
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='red', icon='star'),
        popup=folium.Popup(f"<b>Destination</b><br>Risk: {results['destination_area']['risk_level']}<br>{results['destination_area']['thefts_per_1k']:.2f} thefts/1K", max_width=250)
    ).add_to(m)

    # Add hover tooltips with area information
    style_function = lambda x: {
        'fillColor': '#00000000',
        'color': '#00000000',
        'fillOpacity': 0.0,
        'weight': 0
    }

    highlight_function = lambda x: {
        'fillColor': '#00000000',
        'color': '#666666',
        'fillOpacity': 0.0,
        'weight': 1
    }



    # Convert merged_gdf to GeoJSON for tooltips
    merged_gdf['risk_level'] = merged_gdf['TheftsPer1K'].apply(get_risk_level)

    # Create GeoJSON layer with tooltips
    folium.GeoJson(
        merged_gdf,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['PLR', 'Population', 'VERSUCH', 'TheftsPer1K', 'risk_level'],
            aliases=['Area', 'Population', 'Total Thefts', 'Thefts per 1,000', 'Risk Level'],
            style=('background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;')
        )
    ).add_to(m)


    # Add recommended parking spots with markers
    marker_cluster = MarkerCluster(name="Recommended Parking").add_to(m)

    for rec in results['recommendations']:
        popup_text = (
            f"<div style='font-size: 12px; padding: 10px;'>"
            f"<h4>Recommended Spot {rec['rank']}</h4>"
            f"<b>{rec['name']}</b><br>"
            f"Risk Level: {rec['risk_level']}<br>"
            f"Thefts per 1K residents: {rec['thefts_per_1k']:.2f}<br>"
            f"Distance to destination: {rec['distance_meters']}m"
            f"</div>"
        )

        folium.Marker(
            location=[rec['lat'], rec['lon']],
            icon=folium.Icon(color='green', icon='bicycle'),
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Recommended Spot {rec['rank']}"
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    return m

def main():
    st.set_page_config(layout="wide", page_title="Berlin Bicycle Theft Analysis")

    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .stMarkdown {
            max-width: 100% !important;
        }
        .element-container {
            max-width: 100% !important;
        }
        .stTitle {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #1E1E1E !important;
            margin-bottom: 2rem !important;
        }
        .stHeader {
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            color: #2C3E50 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        .stSubheader {
            font-size: 1.4rem !important;
            font-weight: 500 !important;
            color: #34495E !important;
        }
        .stSelectbox {
            margin-bottom: 1.5rem !important;
        }
        .stSlider {
            margin-bottom: 1.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title with custom styling
    st.markdown('<h1 class="stTitle">Berlin Bicycle Theft Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Load and prepare data
    lor_info, bike_thefts, data_population = load_data()
    bike_thefts_clean, gdf = prepare_data(lor_info, bike_thefts, data_population)
    bike_thefts_total = prepare_theft_data(bike_thefts_clean, data_population)
    bike_thefts_clean = prepare_time_data(bike_thefts_clean)

    # Merge with geometry for mapping
    bike_thefts_total["PLR_ID"] = bike_thefts_total["PLR_ID"].astype(int)
    gdf["PLR_ID"] = gdf["PLR_ID"].astype(int)
    merged_gdf = gdf.merge(bike_thefts_total, how="right", on="PLR_ID")

    # Prepare aggregated data
    plr_gdf, bzr_gdf, pgr_gdf = prepare_aggregated_data(merged_gdf)

    # Create two columns for the top row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2 class="stHeader">Bicycle Theft Density in Berlin</h2>', unsafe_allow_html=True)
        theft_map = create_theft_map(plr_gdf, bzr_gdf, pgr_gdf)
        st.components.v1.html(theft_map._repr_html_(), height=500)

    with col2:
        st.markdown('<h2 class="stHeader">Theft Patterns by Time</h2>', unsafe_allow_html=True)
        # Add BEZ dropdown with custom styling
        st.markdown('<p style="margin-bottom: 0.5rem;">Select District (Bezirk):</p>', unsafe_allow_html=True)
        bez_options = ["All"] + sorted(bike_thefts_clean["BZR"].unique().tolist())
        selected_bez = st.selectbox("", bez_options, key="bez_select", label_visibility="collapsed")
        selected_bez = None if selected_bez == "All" else selected_bez

        time_heatmap = create_time_heatmap(bike_thefts_clean, selected_bez)
        st.altair_chart(time_heatmap, use_container_width=True)

    # Create a separator with margin
    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)

    # Safe parking finder section
    st.markdown('<h2 class="stHeader">Find Safe Parking Spots</h2>', unsafe_allow_html=True)
    st.markdown('<p style="margin-bottom: 1.5rem;">Enter a location to find the safest parking spots nearby.</p>', unsafe_allow_html=True)

    # Create three columns for the input controls with better spacing
    control_col1, control_col2, control_col3 = st.columns([2, 1, 1])

    with control_col1:
        address = st.text_input("Destination address or PLZ:", "Alexanderplatz, Berlin", key="address")
    with control_col2:
        radius = st.slider("Search radius (meters):", 500, 2000, 1000, 100, key="radius")
    with control_col3:
        alternatives = st.slider("Number of alternatives:", 1, 5, 2, 1, key="alternatives")

    # Create columns for map and recommendations with better proportions
    map_col, rec_col = st.columns([3, 1])

    try:
        geolocator = Nominatim(user_agent="bicycle_theft_berlin_app")
        location = geolocator.geocode(address)

        if location:
            results = find_safe_parking_spots(
                location.latitude,
                location.longitude,
                merged_gdf,
                radius,
                alternatives - 1
            )

            with map_col:
                m = plot_recommendations(results, radius, merged_gdf)
                st.components.v1.html(m._repr_html_(), height=600)

            with rec_col:
                st.markdown('<h3 class="stSubheader">Recommendations</h3>', unsafe_allow_html=True)
                for rec in results['recommendations']:
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #2C3E50;">Area {rec['rank']}: {rec['name']}</h4>
                            <p style="margin: 0; color: #34495E;">
                                <strong>Risk Level:</strong> {rec['risk_level']}<br>
                                <strong>Distance:</strong> {rec['distance_meters']}m<br>
                                <strong>Thefts per 1K:</strong> {rec['thefts_per_1k']:.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("Could not find the specified address. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
