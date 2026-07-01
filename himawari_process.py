#!/usr/bin/env python3
import os
import re
import glob
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
import ftplib
import getpass
from tqdm import tqdm
from datetime import timezone # 請確認 datetime import 這行有 timezone
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
# ==================== 新增 Import ====================
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# ==================== 修改配置區 ====================
# 定義裁切區域列表 (名稱, 範圍 [lon_min, lon_max, lat_min, lat_max])
REGIONS = [
    ("EastAsia", [90, 150, 0, 50]),
    ("Taiwan", [116, 126, 19, 28])
]

# Logo 路徑
LOGO_PATH = "./logo1.png"  # 請確保此檔案存在於根目錄

# 指定要輸出的波段與模式設定
# 格式: Band_ID: (是否輸出灰階, 是否輸出彩色)
TARGET_BANDS = {
    "B03": (True, False),  # 只輸出 Original (灰階)
    "B08": (False, True),  # 只輸出 Color
    "B13": (False, True),  # 只輸出 Color
}
# 設定學術字體
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman','DejaVu Serif']
plt.rcParams['font.size'] = 11

# ==================== 配置區 ====================
INPUT_DIR ="./input"
OUTPUT_DIR ="./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)


# 東亞裁切範圍 [lon_min, lon_max, lat_min, lat_max]
EAST_ASIA_EXTENT = [90, 150, 0, 50]

# 完整的 Himawari-9 波段定義
BAND_INFO = {
    # 可見光與近紅外波段 (使用 albedo)
    "B01": {"var": "albedo_01", "name": "Band 1 (0.47 μm - Blue)", "type": "visible", "invert": False},
    "B02": {"var": "albedo_02", "name": "Band 2 (0.51 μm - Green)", "type": "visible", "invert": False},
    "B03": {"var": "albedo_03", "name": "Band 3 (0.64 μm - Red)", "type": "visible", "invert": False},
    "B04": {"var": "albedo_04", "name": "Band 4 (0.86 μm - Vegetation)", "type": "visible", "invert": False},
    "B05": {"var": "albedo_05", "name": "Band 5 (1.6 μm - Snow/Ice)", "type": "visible", "invert": False},
    "B06": {"var": "albedo_06", "name": "Band 6 (2.3 μm - Cloud Particle)", "type": "visible", "invert": False},
    
    # 紅外波段 (使用 tbb - 亮度溫度)
    "B07": {"var": "tbb_07", "name": "Band 7 (3.9 μm - Shortwave IR)", "type": "infrared", "invert": True},
    "B08": {"var": "tbb_08", "name": "Band 8 (6.2 μm - Upper Water Vapor)", "type": "water_vapor", "invert": True},
    "B09": {"var": "tbb_09", "name": "Band 9 (6.9 μm - Mid Water Vapor)", "type": "water_vapor", "invert": True},
    "B10": {"var": "tbb_10", "name": "Band 10 (7.3 μm - Lower Water Vapor)", "type": "water_vapor", "invert": True},
    "B11": {"var": "tbb_11", "name": "Band 11 (8.6 μm - Cloud Phase)", "type": "infrared", "invert": True},
    "B12": {"var": "tbb_12", "name": "Band 12 (9.6 μm - Ozone)", "type": "infrared", "invert": True},
    "B13": {"var": "tbb_13", "name": "Band 13 (10.4 μm - Clean IR)", "type": "infrared", "invert": True},
    "B14": {"var": "tbb_14", "name": "Band 14 (11.2 μm - IR Window)", "type": "infrared", "invert": True},
    "B15": {"var": "tbb_15", "name": "Band 15 (12.4 μm - Dirty IR)", "type": "infrared", "invert": True},
    "B16": {"var": "tbb_16", "name": "Band 16 (13.3 μm - CO2)", "type": "infrared", "invert": True},
}

# 色調強化參數
ENHANCEMENT_PARAMS = {
    "visible": {
        "default": {"brightness": 1.15, "contrast": 1.25},
        "enhanced": {"brightness": 1.25, "contrast": 1.40},
    },
    "water_vapor": {
        "default": {"brightness": 1.15, "contrast": 1.25},
        "enhanced": {"brightness": 1.30, "contrast": 1.50},
    },
    "infrared": {
        "default": {"brightness": 1.15, "contrast": 1.25},
        "enhanced": {"brightness": 1.25, "contrast": 1.45},
    }
}

# ==================== 配置區 (修改部分) ====================

def create_custom_ir_cmap():
    """建立自定義的紅外線強化色階 (User Defined)"""
    # 定義節點與顏色: [數值(Kelvin), [R, G, B]] (0-255)
    bounds = [172, 320]
    points = [
        [172, [0, 0, 0]],
        [187, [94, 94, 94]],
        [207, [255, 255, 255]],
        [208, [0, 0, 0]],       # 劇烈變化邊界
        [222, [255, 0, 0]],
        [237, [255, 255, 0]],
        [247, [0, 255, 0]],
        [252, [50, 50, 140]],
        [262, [17, 192, 233]],
        [267, [52, 229, 248]],
        [273, [255, 255, 255]],
        [320, [0, 0, 0]]
    ]
    
    # 正規化位置 (0.0 - 1.0) 並轉換顏色 (0.0 - 1.0)
    cdict = {'red': [], 'green': [], 'blue': []}
    min_val, max_val = bounds
    span = max_val - min_val
    
    for val, rgb in points:
        pos = (val - min_val) / span
        r, g, b = [c/255.0 for c in rgb]
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))
        
    return LinearSegmentedColormap('custom_ir', cdict)

# 氣象常用色彩映射
COLORMAP_SETTINGS = {
    "water_vapor": {
        "cmap": "nipy_spectral",
        "name": "Water Vapor",
        "use_raw_data": False  # 水氣維持原狀
    },
    "infrared": {
        "cmap": create_custom_ir_cmap(), # 使用自定義色階
        "name": "Infrared Enhanced",
        "use_raw_data": True,   # 紅外線使用原始 Kelvin 數據繪圖
        "vmin": 150,            # 色階下限 (K)
        "vmax": 320             # 色階上限 (K)
    }
}

# RGB 合成配置
RGB_COMPOSITES = {
    "true_color": {
        "R": "albedo_03",  # Band 3 - Red
        "G": "albedo_02",  # Band 2 - Green
        "B": "albedo_01",  # Band 1 - Blue
        "name": "True Color RGB"
    }
}

# 雲頂高度分類 (使用 Band 13 的亮度溫度)
CLOUD_TOP_HEIGHT_CLASSES = {
    "name": "Cloud Top Height Classification",
    "variable": "tbb_13",  # Band 13 (10.4 μm - Clean IR)
    "classes": [
        {"range": (270, 320), "label": "Low Cloud (Warm)", "color": "#8B4513"},      # 褐色
        {"range": (240, 270), "label": "Mid-level Cloud", "color": "#FFD700"},       # 金色
        {"range": (220, 240), "label": "High Cloud (Cold)", "color": "#FF6347"},     # 番茄紅
        {"range": (190, 220), "label": "Very High/Deep Convection", "color": "#FF0000"},  # 紅色
        {"range": (-100, 190), "label": "Extreme Cold (Overshooting Tops)", "color": "#8B0000"},  # 深紅色
    ],
    "description": """
Cloud Top Height Classification based on brightness temperature (Band 13):
- Low Cloud (270-320K): Warm clouds, typically stratocumulus or fog
- Mid-level Cloud (240-270K): Altocumulus, altostratus
- High Cloud (220-240K): Cirrus, cirrostratus
- Very High/Deep Convection (190-220K): Deep convective clouds, severe storms
- Overshooting Tops (<190K): Extremely cold cloud tops, intense convection
"""
}

# 雲分類 (使用多波段組合)
CLOUD_TYPE_CLASSIFICATION = {
    "name": "Cloud Type Classification",
    "variables": {
        "vis": "albedo_03",      # Band 3 (可見光)
        "ir_window": "tbb_13",   # Band 13 (IR Window)
        "wv": "tbb_08",          # Band 8 (Water Vapor)
    },
    "classes": [
        {"label": "Clear Sky", "color": "#87CEEB"},           # 天藍色
        {"label": "Water Cloud", "color": "#F0F8FF"},         # 淡藍白色
        {"label": "Ice Cloud (Cirrus)", "color": "#FFFFFF"},  # 白色
        {"label": "Mixed Phase Cloud", "color": "#FFE4B5"},   # 淺橙色
        {"label": "Deep Convective Cloud", "color": "#FF4500"}, # 橙紅色
        {"label": "Fog/Low Stratus", "color": "#D3D3D3"},     # 淺灰色
    ],
    "description": """
Cloud Type Classification using multi-spectral approach:
- Clear Sky: Low reflectance, warm temperature
- Water Cloud: High reflectance, warm/moderate temperature  
- Ice Cloud (Cirrus): Moderate reflectance, very cold temperature
- Mixed Phase Cloud: High reflectance, cold temperature
- Deep Convective Cloud: Very high reflectance, extremely cold temperature
- Fog/Low Stratus: Moderate/high reflectance, very warm temperature

Algorithm uses thresholds on:
1. Visible reflectance (albedo_03)
2. IR window temperature (tbb_13) 
3. Water vapor channel (tbb_08)
"""
}

# --- FTP 下載配置 ---
FTP_ENABLED = True
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = "guosean.weather_gmail.com"
FTP_PASS = "SP+wari8"  # <--- 在這裡新增您的密碼
FTP_REMOTE_BASE_PATH = "/jma/netcdf"
SEARCH_WINDOW_MINUTES = 60  # 從當前時間往前搜尋多久的範圍
SEARCH_START_OFFSET_MINUTES = 10 # 從多久前開始搜尋 (JAXA資料通常有延遲)
# -----

# ==================== 函式定義 ====================

def clear_folder(folder_path):
    """清空指定資料夾內的所有檔案"""
    print(f"清空資料夾: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'無法刪除 {file_path}. 原因: {e}')

def download_file_with_progress(ftp, remote_path, local_path):
    """使用 tqdm 顯示進度條下載檔案"""
    try:
        total_size = ftp.size(remote_path)
        filename = os.path.basename(remote_path)
        
        with open(local_path, 'wb') as f, tqdm(
            desc=f"下載 {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            def callback(chunk):
                f.write(chunk)
                pbar.update(len(chunk))
            
            ftp.retrbinary(f"RETR {remote_path}", callback)
        return True
    except Exception as e:
        print(f"\n下載失敗: {e}")
        if os.path.exists(local_path):
            os.remove(local_path) # 刪除不完整的檔案
        return False
def download_file_with_progress(ftp, remote_path, local_path):
    """使用 tqdm 顯示進度條下載檔案 (已修改為不預先查詢大小)"""
    try:
        filename = os.path.basename(remote_path)
        
        # total 設為 None，tqdm 會自動顯示進度但沒有百分比
        with open(local_path, 'wb') as f, tqdm(
            desc=f"下載 {filename}",
            total=None,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            def callback(chunk):
                f.write(chunk)
                pbar.update(len(chunk))
            
            ftp.retrbinary(f"RETR {remote_path}", callback)
        return True
    except Exception as e:
        print(f"\n下載失敗: {e}")
        if os.path.exists(local_path):
            os.remove(local_path) # 刪除不完整的檔案
        return False

def find_and_download_latest_file():
    """連線FTP，尋找並下載最新的檔案 (已改用 NLST 檢查檔案是否存在)"""
    try:
        password = FTP_PASS
        
        with ftplib.FTP(FTP_HOST, FTP_USER, password, timeout=60) as ftp:
            print(f"✓ FTP 連線成功: {FTP_HOST}")
            ftp.encoding = 'utf-8'
            ftp.set_pasv(True)

            start_time_utc = datetime.now(timezone.utc) - timedelta(minutes=SEARCH_START_OFFSET_MINUTES)
            print(f"從 {start_time_utc.strftime('%Y-%m-%d %H:%M')} UTC 開始，在 {SEARCH_WINDOW_MINUTES} 分鐘內搜尋最新檔案...")

            current_minute = start_time_utc.minute
            if current_minute >= 50: search_minute = 50
            elif current_minute >= 40: search_minute = 30
            elif current_minute >= 30: search_minute = 30
            elif current_minute >= 20: search_minute = 20
            elif current_minute >= 10: search_minute = 10
            else: search_minute = 0
            
            search_time = start_time_utc.replace(minute=search_minute, second=0, microsecond=0)
            
            for _ in range(SEARCH_WINDOW_MINUTES // 10 + 2):
                yyyymm = search_time.strftime('%Y%m')
                dd = search_time.strftime('%d')
                yyyymmdd = search_time.strftime('%Y%m%d')
                hhmm = search_time.strftime('%H%M')
                
                remote_dir = f"{FTP_REMOTE_BASE_PATH}/{yyyymm}/{dd}"
                filename = f"NC_H09_{yyyymmdd}_{hhmm}_R21_FLDK.07001_06001.nc"
                
                try:
                    ftp.cwd(remote_dir)
                    # ==================== 核心修改點 ====================
                    # 改用 NLST 列出檔名列表來檢查，而不是用 SIZE
                    print(f"  在目前資料夾內搜尋檔案: {filename} ... ", end="")
                    file_list = ftp.nlst()
                    if filename in file_list:
                        print("找到了!")
                    else:
                        print("不存在")
                        raise ftplib.error_perm("File not found in NLST list") # 手動觸發例外以便跳到 except 區塊
                    # ====================================================

                    clear_folder(INPUT_DIR)
                    
                    remote_filepath_for_download = f"{remote_dir}/{filename}"
                    local_filepath = os.path.join(INPUT_DIR, filename)
                    if download_file_with_progress(ftp, remote_filepath_for_download, local_filepath):
                        print(f"✓ 下載完成: {local_filepath}")
                        return True
                    else:
                        return False

                except ftplib.error_perm:
                    # 這段現在會捕捉到 CWD 失敗或上面我們手動觸發的例外
                    search_time -= timedelta(minutes=10)
                    if search_time.minute == 40:
                        search_time -= timedelta(minutes=10)
                    continue
                except Exception as e:
                    print(f"發生錯誤: {e}")
                    break
            
            print("⚠ 在指定時間範圍內未找到任何檔案。")
            return False

    except ftplib.all_errors as e:
        print(f"FTP 錯誤: {e}")
        return False
    except Exception as e:
        print(f"發生未知錯誤: {e}")
        return False

def find_nc_files(input_dir):
    """自動搜尋符合格式的 NC 檔案"""
    pattern = os.path.join(input_dir, "NC_H09_*_*_R21_FLDK.*.nc")
    files = glob.glob(pattern)
    return sorted(files)

def parse_filename(filepath):
    """解析檔案名稱，提取日期時間資訊"""
    filename = os.path.basename(filepath)
    # 格式: NC_H08_YYYYMMDD_HHMM_R21_FLDK.06001_06001.nc
    match = re.match(r'NC_H09_(\d{8})_(\d{4})_R21_FLDK\..*\.nc', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMM
        
        # 轉換為 datetime 物件
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        return dt, date_str, time_str
    return None, None, None

def normalize_uint8(arr, invert=False):
    """將陣列正規化為 0-255 uint8"""
    arr = np.where(np.isfinite(arr), arr, np.nan)
    minv, maxv = np.nanmin(arr), np.nanmax(arr)
    if maxv == minv:
        out = np.full(arr.shape, 127, dtype=np.uint8)
    else:
        out = ((arr - minv) / (maxv - minv) * 255)
    if invert:
        out = 255 - out
    return out.astype(np.uint8)

def gamma_correct(arr, gamma=1/2.2):
    """對 0-1 範圍的浮點數陣列進行 Gamma 校正"""
    return np.power(np.clip(arr, 0, 1), gamma)

def add_title_to_map(ax, band_name, dt_utc):
    """加入正式化標題 (UTC+8)"""
    # 時間轉換: UTC -> UTC+8 (CST)
    dt_local = dt_utc + timedelta(hours=8)
    date_str = dt_local.strftime('%Y-%m-%d')
    time_str = dt_local.strftime('%H:%M CST')
    
    # 左上角: 衛星名稱與波段
    # 使用學術風格字體，字體大小可依需求微調
    main_text = f"Himawari-9 AHI\n{band_name}"
    ax.text(0.02, 0.98, main_text, 
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', family='serif',
            ha='left', va='top', color='white',
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # 右上角: 日期與時間
    time_text = f"{date_str}\n{time_str}"
    ax.text(0.98, 0.98, time_text,
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', family='monospace', # 時間使用等寬字體較整齊
            ha='right', va='top', color='yellow',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])

def add_logo_to_map(ax):
    """在右下角加入 Logo (縮小 11 倍)"""
    if os.path.exists(LOGO_PATH):
        try:
            logo_img = mpimg.imread(LOGO_PATH)
            
            # 計算縮放比例: 1 / 11
            zoom_factor = 1.0 / 35.0 
            
            # 建立圖片方塊
            imagebox = OffsetImage(logo_img, zoom=zoom_factor, alpha=1.0) # alpha=1.0 為不透明
            
            # 設定位置: xy=(1, 0) 為 Axes 右下角
            # box_alignment=(1.0, 0.0) 代表圖片本身的右下角對齊 xy 點
            # 這樣可以確保圖片剛好貼在右下角邊緣
            ab = AnnotationBbox(imagebox, (1, 0), xycoords='axes fraction',
                                box_alignment=(1.0, 0.0),
                                frameon=False, pad=0.1) # pad 可微調邊距
            ax.add_artist(ab)
        except Exception as e:
            print(f"無法載入 Logo: {e}")


# ==================== 函式定義 (修改部分) ====================

def process_single_band(ds, band_id, band_info, lon, lat, dt_utc, output_folder):
    """處理單一波段 (含多區域與特定輸出模式)"""
    
    # --- 設定區: 指定要輸出的波段與模式 ---
    # 格式: Band_ID: (輸出灰階?, 輸出彩色?)
    TARGET_BANDS = {
        "B03": (True, False),  # Band 3: 只要灰階
        "B08": (False, True),  # Band 8: 只要彩色 (攝氏)
        "B13": (False, True),  # Band 13: 只要彩色 (攝氏)
    }
    
    # 檢查是否為目標波段
    if band_id not in TARGET_BANDS:
        return

    vname = band_info["var"]
    band_name = band_info["name"]
    band_type = band_info["type"]
    invert = band_info["invert"]
    need_gray, need_color = TARGET_BANDS[band_id]

    if vname not in ds.variables:
        return
    
    print(f"  處理 {band_id}...")
    
    # 讀取數據
    arr = ds[vname].values.squeeze() # 移除多餘維度
    
    # 準備繪圖數據
    # 1. 灰階圖用的正規化數據 (0-255)
    img_uint8 = normalize_uint8(arr, invert=invert)
    
    # 定義要執行的任務列表
    tasks = []
    if need_gray: tasks.append("gray")
    if need_color: tasks.append("color")

    # 針對每個區域進行繪圖
    for region_name, extent in REGIONS:
        
        for mode in tasks:
            # 初始化繪圖參數
            plot_data = img_uint8
            cmap = "gray"
            vmin, vmax = None, None
            cbar_label = ""
            suffix = ""
            coast_color = "cyan"
            
            # 彩色模式參數設定
            if mode == "color":
                suffix = "_color"
                coast_color = "white"
                settings = COLORMAP_SETTINGS.get(band_type)
                
                if settings:
                    cmap = settings["cmap"]
                    # 如果設定為使用原始數據 (如紅外線溫度)，則不使用 uint8
                    if settings.get("use_raw_data", False):
                        plot_data = arr
                        vmin = settings.get("vmin")
                        vmax = settings.get("vmax")
                    
                    if band_type == "infrared":
                        cbar_label = "Temperature (°C)"
                    elif band_type == "water_vapor":
                        cbar_label = "Water Vapor"

            # 建立圖表
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, ccrs.PlateCarree())
            
            # 繪製影像
            im = ax.imshow(plot_data, origin='upper',
                           extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)],
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax)
            
            # 加入地圖特徵
            ax.coastlines(resolution='10m', color=coast_color, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', edgecolor=coast_color)
            
            # 加入色條 (僅彩色模式)
            if mode == "color":
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
                cbar.ax.tick_params(colors='white', labelsize=8)
                cbar.set_label(cbar_label, color='white', fontsize=9)
                
                # 如果是紅外線，將 Kelvin 轉為 Celsius 顯示
                if band_type == "infrared":
                    cbar.ax.yaxis.set_major_formatter(
                        ticker.FuncFormatter(lambda x, pos: f"{x - 273.15:.0f}")
                    )

            # 加入標題與 Logo
            add_title_to_map(ax, band_name, dt_utc)
            add_logo_to_map(ax)
            
            # 儲存檔案
            # 檔名格式: Bxx_original[_color][_Taiwan].png
            reg_suffix = f"_{region_name}" if region_name != "EastAsia" else ""
            filename = f"{band_id}_original{suffix}{reg_suffix}.png"
            out_path = os.path.join(output_folder, filename)
            
            bg_color = 'black' if mode == "gray" else '#1a1a1a'
            plt.savefig(out_path, bbox_inches='tight', dpi=200, facecolor=bg_color)
            plt.close()
            print(f"    ✓ {filename}")

def create_rgb_composite(ds, composite_name, composite_info, lon, lat, dt_utc, output_folder):
    """建立 RGB 合成圖 (修改版：支援多區域、Logo)"""
    
    # 只處理 True Color
    if composite_name != "true_color":
        return

    print(f"  建立 RGB 合成: {composite_info['name']}")
    
    # (讀取與處理 RGB 數據的代碼保持不變...)
    try:
        r = ds[composite_info['R']].values.squeeze()
        g = ds[composite_info['G']].values.squeeze()
        b = ds[composite_info['B']].values.squeeze()
    except KeyError:
        return

    # 正規化與 Gamma 校正
    def process_channel(c):
        c = np.nan_to_num(c)
        norm = (c - np.min(c)) / (np.max(c) - np.min(c))
        return gamma_correct(norm)

    rgb_image = np.stack([process_channel(r), process_channel(g), process_channel(b)], axis=-1)

    # 針對區域迴圈
    for region_name, extent in REGIONS:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        
        ax.coastlines(resolution='10m', linewidth=0.8, color='white')
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, edgecolor='white')
        
        ax.imshow(rgb_image, origin='upper',
                 extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)],
                 transform=ccrs.PlateCarree())
        
        # 標題與 Logo
        add_title_to_map(ax, "True Color RGB", dt_utc)
        add_logo_to_map(ax)
        
        # 檔名
        region_suffix = "" if region_name == "EastAsia" else f"_{region_name}"
        out_path = os.path.join(output_folder, f"RGB_{composite_name}{region_suffix}.png")
        
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"    ✓ {os.path.basename(out_path)} 完成")

def process_nc_file(filepath):
    """處理單一 NC 檔案"""
    print(f"\n{'='*60}")
    print(f"處理檔案: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # 解析檔名取得時間
    dt_utc, date_str, time_str = parse_filename(filepath)
    if dt_utc is None:
        print("⚠ 無法解析檔案名稱，跳過此檔案")
        return
    
    print(f"觀測時間: {dt_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    
    # 建立輸出資料夾
    output_folder = os.path.join(OUTPUT_DIR)
    os.makedirs(output_folder, exist_ok=True)
    
    # 開啟 NetCDF
    try:
        ds = xr.open_dataset(filepath, engine="netcdf4")
    except Exception as e:
        print(f"⚠ 無法開啟檔案: {e}")
        return
    
    # 取得經緯度
    lon, lat = None, None
    for n in ("lon", "longitude", "x"):
        if n in ds.variables:
            lon = ds[n].values
            break
    for n in ("lat", "latitude", "y"):
        if n in ds.variables:
            lat = ds[n].values
            break
    
    if lon is None or lat is None:
        print("⚠ 找不到經緯度資訊")
        ds.close()
        return
    
    # 轉換為 2D 網格
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    
    print(f"\n處理 {len(BAND_INFO)} 個波段...")
    
    # 處理所有波段
    for band_id, band_info in BAND_INFO.items():
        process_single_band(ds, band_id, band_info, lon, lat, dt_utc, output_folder)
    
    print(f"\n建立 RGB 合成圖...")
    
    # 建立 RGB 合成
    for composite_name, composite_info in RGB_COMPOSITES.items():
        create_rgb_composite(ds, composite_name, composite_info, lon, lat, dt_utc, output_folder)

    ds.close()
    print(f"\n✓ 完成! 所有輸出已儲存至: {output_folder}")

# ==================== 主程式 ====================

def main():
    print("Himawari-9 全波段影像處理程式")
    print(f"輸入資料夾: {INPUT_DIR}")
    print(f"輸出資料夾: {OUTPUT_DIR}")
    
    # --- 新增: 執行 FTP 下載 ---
    # (您可以在配置區將 FTP_ENABLED 設為 False 來跳過此步驟)
    if 'FTP_ENABLED' in globals() and FTP_ENABLED:
        print("\n" + "="*25 + " FTP 下載 " + "="*25)
        download_successful = find_and_download_latest_file()
        if not download_successful:
            print("\n⚠ FTP 下載失敗或未找到檔案，無法繼續執行分析。")
            return
        print("="*64)
    # ---------------------------

    # 搜尋本地檔案 (下載後)
    nc_files = find_nc_files(INPUT_DIR)
    
    if not nc_files:
        print(f"\n⚠ 在 {INPUT_DIR} 中找不到符合格式的 NC 檔案")
        print("檔案格式應為: NC_H09_YYYYMMDD_HHMM_R21_FLDK.*.nc")
        return
    
    print(f"\n找到 {len(nc_files)} 個檔案進行分析:")
    for i, f in enumerate(nc_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    # 處理所有檔案
    for nc_file in nc_files:
        try:
            process_nc_file(nc_file)
        except Exception as e:
            print(f"\n⚠ 處理檔案時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有檔案處理完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
