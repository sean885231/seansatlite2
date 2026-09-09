import os
import shutil
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import geopandas as gpd
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# ===================================================================
# --- 1. 資料準備：定義路徑與下載資料 ---
# ===================================================================

# --- 設定路徑與網址 ---
DATA_DIR = "./radar_data"
os.makedirs(DATA_DIR, exist_ok=True)

RADAR_URL = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0059-001.json"
RADAR_JSON_PATH = os.path.join(DATA_DIR, "O-A0059-001.json")
LIGHTNING_URL = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0039-001.kmz"
LIGHTNING_KMZ_PATH = os.path.join(DATA_DIR, "O-A0039-001.kmz")
COUNTY_SHP_PATH = "./COUNTY_MOI_1130718.shp"
TOWN_SHP_PATH = "./TOWN_MOI_1111118.shp"

def download_data(url, save_path):
    print(f"Downloading data from {url}...")
    try:
        if os.path.exists(save_path):
            os.remove(save_path)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Data successfully saved to: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return False
    return True

os.makedirs(DATA_DIR, exist_ok=True)
download_data(RADAR_URL, RADAR_JSON_PATH)
download_data(LIGHTNING_URL, LIGHTNING_KMZ_PATH)

print("-" * 50)
print("Data preparation complete. Starting map generation...")
print("-" * 50)


# ===================================================================
# --- 2. 資料解析函式 ---
# ===================================================================

def read_radar_grid(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    param = data['cwaopendata']['dataset']['datasetInfo']['parameterSet']
    content_str = data['cwaopendata']['dataset']['contents']['content']
    lon_start, lat_start = float(param['StartPointLongitude']), float(param['StartPointLatitude'])
    res, nx, ny = float(param['GridResolution']), int(param['GridDimensionX']), int(param['GridDimensionY'])
    content_flat = np.array(content_str.strip().split(','), dtype=np.float32)
    content_2d = content_flat.reshape((ny, nx))
    lons = lon_start + np.arange(nx) * res
    lats = lat_start + np.arange(ny) * res
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    content_2d = np.ma.masked_where((content_2d < 0), content_2d)
    return lon_grid, lat_grid, content_2d, param['DateTime']

def read_lightning_data(kmz_file):
    lightning_strikes = []
    try:
        with zipfile.ZipFile(kmz_file, 'r') as kmz:
            kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
            with kmz.open(kml_filename, 'r') as kml_file:
                tree = ET.parse(kml_file)
                root = tree.getroot()
                namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
                for placemark in root.findall('.//kml:Placemark', namespace):
                    timestamp_str = placemark.find('.//kml:TimeStamp/kml:when', namespace).text
                    strike_time = datetime.fromisoformat(timestamp_str)
                    coords_text = placemark.find('.//kml:coordinates', namespace).text
                    coords = coords_text.strip().split(',')
                    style_url = placemark.find('kml:styleUrl', namespace).text
                    strike_type = 'cg' if 'cg' in style_url else 'ic'
                    if len(coords) >= 2:
                        lightning_strikes.append({'lon': float(coords[0]), 'lat': float(coords[1]), 'time': strike_time, 'type': strike_type})
    except Exception as e:
        print(f"⚠ Error parsing lightning data: {e}")

    if lightning_strikes:
        print(f"Successfully parsed {len(lightning_strikes)} lightning strikes.")
    else:
        print("No lightning strikes were detected in this period.")
    return lightning_strikes

# ===================================================================
# --- 3. 最終版繪圖函式 (視覺化微調) ---
# ===================================================================

def plot_final_map(radar_json, lightning_kmz, county_shp, town_shp):
    lon_grid, lat_grid, radar_value, timestamp_str = read_radar_grid(radar_json)
    map_time = datetime.fromisoformat(timestamp_str)

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([119, 123.5, 21.5, 25.5], crs=ccrs.PlateCarree())

    from matplotlib.colors import ListedColormap, BoundaryNorm

    # --- 1️⃣ 建立色碼列表 ---
    colors = [
        "#FFFFFF",  # 0
        "#000FFF", "#00ECFF", "#00DAFF", "#00C8FF", "#00B6FF",  # 1-5
        "#00A3FF", "#0091FF", "#007FFF", "#006DFF", "#005BFF",  # 6-10
        "#0048FF", "#0036FF", "#0024FF", "#0012FF", "#0000FF",  # 11-15
        "#00FF00", "#00F400", "#00E900", "#00DE00", "#00D300",  # 16-20
        "#00C800", "#00BE00", "#00B400", "#00AA00", "#00A000",  # 21-25
        "#009600", "#33AB00", "#66C000", "#99D500", "#CCEA00",  # 26-30
        "#FFFF00", "#FFF400", "#FFE900", "#FFDE00", "#FFD300",  # 31-35
        "#FFC800", "#FFB800", "#FFA800", "#FF9800", "#FF8800",  # 36-40
        "#FF7800", "#FF6000", "#FF4800", "#FF3000", "#FF1800",  # 41-45
        "#FF0000", "#F40000", "#E90000", "#DE0000", "#D30000",  # 46-50
        "#C80000", "#BE0000", "#B40000", "#AA0000", "#A00000",  # 51-55
        "#960000", "#AB0033", "#C00066", "#D50099", "#FF00FF",  # 56-60
        "#EA00FF", "#D500FF", "#C000FF", "#AB00FF", "#9600FF"   # 61-66
    ]

    # --- 2️⃣ 對應 dBZ 等級 ---
    levels = list(range(0, 67))  # 0 ~ 66 dBZ

    # --- 3️⃣ 建立 Colormap 與 Norm ---
    custom_cmap = ListedColormap(colors, name='custom_radar')
    custom_norm = BoundaryNorm(levels, ncolors=custom_cmap.N, clip=True)

    # --- 4️⃣ 套用到 pcolormesh ---
    radar_map = ax.pcolormesh(
        lon_grid, lat_grid, radar_value,
        cmap=custom_cmap,
        norm=custom_norm, # 修正: 加上 norm 讓顏色更準確對應
        shading='auto',
        zorder=1,
        alpha=0.8
    )
    plt.colorbar(radar_map, ax=ax, orientation='vertical', pad=0.06, label='Reflectivity (dBZ)')

    if os.path.exists(town_shp):
        gpd.read_file(town_shp).to_crs(epsg=4326).plot(ax=ax, facecolor='none', edgecolor='#555555', linewidth=0.4, zorder=1)
    if os.path.exists(county_shp):
        gpd.read_file(county_shp).to_crs(epsg=4326).plot(ax=ax, facecolor='none', edgecolor='#000000', linewidth=1.5, zorder=2)

    # --- 繪製閃電資料 ---
    lightning_strikes = read_lightning_data(lightning_kmz)
    if lightning_strikes:
        plot_data = {
            'cg': {'red': [], 'yellow': [], 'green': [], 'blue': []},
            'ic': {'red': [], 'yellow': [], 'green': [], 'blue': []}
        }
        for strike in lightning_strikes:
            age = map_time - strike['time']
            color_key = None
            if timedelta(minutes=0) <= age < timedelta(minutes=5): color_key = 'red'
            elif timedelta(minutes=5) <= age < timedelta(minutes=10): color_key = 'yellow'
            elif timedelta(minutes=10) <= age < timedelta(minutes=30): color_key = 'green'
            elif timedelta(minutes=30) <= age < timedelta(minutes=60): color_key = 'blue'
            if color_key:
                plot_data[strike['type']][color_key].append((strike['lon'], strike['lat']))
        
        for strike_type, colors_dict in plot_data.items():
            for color, coords in colors_dict.items():
                if coords:
                    lons, lats = zip(*coords)
                    if strike_type == 'cg':
                        ax.scatter(lons, lats, marker='+', color=color, s=40, zorder=10)
                    else:
                        ax.scatter(lons, lats, marker='o', facecolor='none', edgecolor=color, s=30, linewidth=1, zorder=10)

    # --- 圖例 ---
    legend_elements = [
        mlines.Line2D([], [], color='black', marker='o', mfc='none', linestyle='None', markersize=6, label='Cloud-to-Cloud'),
        mlines.Line2D([], [], color='black', marker='+', linestyle='None', markersize=7, label='Cloud-to-Ground'),
        mlines.Line2D([0], [0], marker='o', color='w', label='By Age:', markerfacecolor='k', markersize=0),
        mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=8, label='  0-5 min'),
        mlines.Line2D([], [], color='yellow', marker='s', linestyle='None', markersize=8, label='  5-10 min'),
        mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=8, label='  10-30 min'),
        mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=8, label='  30-60 min'),
    ]
    legend = ax.legend(handles=legend_elements, title="Lightning Legend", loc='upper right', fontsize='small')
    legend.get_frame().set_alpha(0.6)

    ax.set_title(f"Radar Reflectivity and Lightning Distribution\nTime: {timestamp_str}", fontsize=14)
    ax.gridlines(dms=True, x_inline=False, y_inline=False)
    
    # ==================== 新增 Logo 區塊 (修正版：錨定於 Axes 下方) ====================
    logo_path = "./logo1.png"  # 假設 logo 在根目錄
    if os.path.exists(logo_path):
        try:
            img = mpimg.imread(logo_path)
            
            # 設定縮放比例: 縮小 30 倍 (1/30)
            zoom_factor = 1.0 / 30.0
            
            # 建立圖片框
            imagebox = OffsetImage(img, zoom=zoom_factor, alpha=1.0)
            
            # 設定位置與對齊方式 (關鍵修改):
            # 1. xycoords='axes fraction': 使用地圖座標軸為基準
            # 2. xy=(0.5, -0.12): 水平置中(0.5)，垂直位於地圖底線下方 12% 處 (-0.12)
            #    (如果覺得太低或太高，請微調 -0.12 這個數字，例如改為 -0.1 或 -0.15)
            # 3. box_alignment=(0.5, 1.0): 將 Logo 圖片本身的「中上方」對齊到設定的 xy 點
            #    這樣 Logo 就像是掛在地圖下緣一樣
            ab = AnnotationBbox(imagebox, (0.5, -0.02), xycoords='axes fraction',
                                box_alignment=(0.5, 1.0), frameon=False, pad=0)
            
            ax.add_artist(ab)
            print("Logo added successfully (anchored below axes).")
        except Exception as e:
            print(f"Error adding logo: {e}")
    # ==============================================================================

    output_filename = "outputs/Radar_Lightning_Map.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Map successfully saved to: {output_filename}")
    plt.close()

# ===================================================================
# --- 4. 主程式執行區 ---
# ===================================================================

if os.path.exists(RADAR_JSON_PATH) and os.path.exists(LIGHTNING_KMZ_PATH):
    plot_final_map(RADAR_JSON_PATH, LIGHTNING_KMZ_PATH, COUNTY_SHP_PATH, TOWN_SHP_PATH)
else:

    print("⚠ Could not proceed with plotting because main data files are missing.")


