import requests
import os
import urllib3
from datetime import datetime, timedelta
from PIL import Image, ImageDraw

# 關閉 SSL 不安全連線的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== 設定區 (修改部分) ====================

# 1. 路徑設定
OUTPUT_DIR = "./outputs"

# 【修改點】分別定義兩張底圖的路徑
BASE_IMAGE_PATH_24 = "./Real-time_rainfall_background_24.png"  # 原本的底圖 (24hr 用)
BASE_IMAGE_PATH_01 = "./Real-time_rainfall_background_1.png"   # 新的底圖 (1hr 用)

# 【修改點】在產品列表中新增 'base_path' 欄位，指定該產品要用的底圖
PRODUCTS = [
    {
        'type': '24hr',
        'code': '24', 
        'input_name': 'rain_input_24.png',
        'output_name': 'NCDR_Rain_Composite_24hr.png',
        'base_path': BASE_IMAGE_PATH_24  # 指定使用 24hr 底圖
    },
    {
        'type': '1hr',
        'code': '01', 
        'input_name': 'rain_input_01.png',
        'output_name': 'NCDR_Rain_Composite_1hr.png',
        'base_path': BASE_IMAGE_PATH_01  # 指定使用 1hr 底圖
    }
]

# NCDR URL 樣板 (TYPE 會被替換成 24 或 01)
URL_TEMPLATE = "https://watch.ncdr.nat.gov.tw/00_Wxmap/7R1_KRID_RAINMAP/{YYYYMM}/{YYYYMMDDHHmm}/raingauge_{TYPE}_{YYYYMMDDHHmm}.png"

# 2. 圖片定位與尺寸設定 (全體共用)
IMAGE_CONFIG = {
    'x': 279,     # 原始 X: 278.5
    'y': 57,      # 原始 Y: 57.1
    'w': 1039,    # 原始 W: 1038.9
    'h': 1812     # 原始 H: 1811.6
}

# 3. 要「挖掉」的區塊設定 (Global Coordinates)
MASK_BLOCKS = [
    {'x': 279, 'y': 57, 'w': 669, 'h': 103},   # 區塊 1
    {'x': 1202, 'y': 669, 'w': 115, 'h': 1200} # 區塊 2
]

# ==================== 程式邏輯區 ====================

def get_utc8_now():
    """取得目前的 UTC+8 時間"""
    return datetime.utcnow() + timedelta(hours=8)

def round_down_to_10min(dt):
    """將時間的分鐘數無條件捨去至最近的 10 分位"""
    minute = dt.minute - (dt.minute % 10)
    return dt.replace(minute=minute, second=0, microsecond=0)

def download_images():
    """
    尋找並下載圖片
    邏輯：找到一個有效的時間點，然後嘗試下載該時間點的所有產品(24hr & 1hr)
    """
    print("步驟 1: 尋找可用雨量圖時間點...")
    
    now_utc8 = get_utc8_now()
    rounded_now = round_down_to_10min(now_utc8)
    
    # 規則：從最近時間-10分開始，最多回推 1 小時
    start_time = rounded_now - timedelta(minutes=10)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://watch.ncdr.nat.gov.tw/"
    }

    found_time = None

    # 1. 先找時間點 (以 24hr 為基準來測試連結是否有效)
    for i in range(7):
        check_time = start_time - timedelta(minutes=10 * i)
        yyyymm = check_time.strftime("%Y%m")
        yyyymmddhhmm = check_time.strftime("%Y%m%d%H%M")
        
        # 測試 24hr 的連結
        test_url = URL_TEMPLATE.format(YYYYMM=yyyymm, YYYYMMDDHHmm=yyyymmddhhmm, TYPE='24')
        print(f"  偵測時間: {check_time.strftime('%H:%M')} ...", end="")
        
        try:
            # 只用 HEAD 請求快速檢查，或者 GET
            response = requests.get(test_url, headers=headers, verify=False, timeout=5)
            if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                print(" [有效]")
                found_time = check_time
                break
            else:
                print(" [無效]")
        except:
            print(" [錯誤]")

    if not found_time:
        print("錯誤：找不到任何可用的影像時間點。")
        return False

    # 2. 確定時間後，下載所有產品 (24hr 和 1hr)
    print(f"\n步驟 2: 開始下載鎖定時間 [{found_time.strftime('%H:%M')}] 的圖片...")
    yyyymm = found_time.strftime("%Y%m")
    yyyymmddhhmm = found_time.strftime("%Y%m%d%H%M")
    
    success_count = 0
    
    for prod in PRODUCTS:
        url = URL_TEMPLATE.format(YYYYMM=yyyymm, YYYYMMDDHHmm=yyyymmddhhmm, TYPE=prod['code'])
        save_path = os.path.join(OUTPUT_DIR, prod['input_name'])
        
        print(f"  下載 [{prod['type']}] : {url}")
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=15)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                print(f"    -> 成功儲存: {prod['input_name']}")
                success_count += 1
            else:
                print(f"    -> 下載失敗 (Status {r.status_code})")
        except Exception as e:
            print(f"    -> 下載錯誤: {e}")

    return success_count > 0

def process_single_image(input_filename, output_filename, base_path):
    """處理單張圖片：縮放 -> 挖空 -> 合成"""
    input_path = os.path.join(OUTPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    if not os.path.exists(input_path):
        print(f"  略過: 找不到輸入檔 {input_filename}")
        return

    try:
        # 1. 載入底圖與雨量圖
        base_img = Image.open(base_path).convert("RGBA")
        rain_img = Image.open(input_path).convert("RGBA")
        
        # 2. 縮放
        target_size = (IMAGE_CONFIG['w'], IMAGE_CONFIG['h'])
        rain_img = rain_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 3. 挖空
        draw = ImageDraw.Draw(rain_img)
        origin_x = IMAGE_CONFIG['x']
        origin_y = IMAGE_CONFIG['y']
        
        for block in MASK_BLOCKS:
            # 轉換為相對座標
            rect = [
                block['x'] - origin_x,
                block['y'] - origin_y,
                block['x'] - origin_x + block['w'],
                block['y'] - origin_y + block['h']
            ]
            draw.rectangle(rect, fill=(0, 0, 0, 0))

        # 4. 合成
        paste_pos = (origin_x, origin_y)
        base_img.paste(rain_img, paste_pos, rain_img)
        
        # 5. 存檔
        base_img.save(output_path)
        print(f"  完成: {output_filename}")

    except Exception as e:
        print(f"  處理失敗 {output_filename}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 簡單檢查底圖是否存在
    for prod in PRODUCTS:
        if not os.path.exists(prod['base_path']):
            print(f"嚴重錯誤：找不到底圖 {prod['base_path']} (給 {prod['type']} 使用)")
            return

    # 1. 下載
    if download_images():
        print("\n步驟 3: 進行圖片合成...")
        
        # 2. 針對每個產品進行處理
        for prod in PRODUCTS:
            # 【修改點】呼叫時傳入該產品專屬的 base_path
            process_single_image(prod['input_name'], prod['output_name'], prod['base_path'])
            
        print("\n全部作業完成！")

if __name__ == "__main__":
    main()