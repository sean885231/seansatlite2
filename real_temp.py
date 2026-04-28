import requests
import zipfile
import io
import os
import numpy as np
from PIL import Image

# ==================== 設定區 ====================

# 1. 檔案路徑設定
WORK_DIR = "./outputs"
OUTPUT_FILENAME = "Final_Composite_NewLayout.png"

# KMZ 下載網址
KMZ_URL = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0038-002.kmz"

# 底圖路徑 (請修改為您的底圖路徑)
BASE_CANVAS_PATH = "./Real-time_temp_background.png"

# 疊圖檔案路徑
# 請注意：依據您的說明，疊圖1是中層，疊圖2是最上層
MIDDLE_LAYER_PATH = "./town.jpeg"  # 疊圖1 (中層)
TOP_LAYER_PATH = "./coutry.jpeg"     # 疊圖2 (最上層)

# 2. 圖層參數設定 (W, H, X, Y) - 自動四捨五入
# 溫度圖 (最下層)
TEMP_LAYER_CONFIG = {
    'w': 1043, 'h': 1857, 'x': 234, 'y': 48  # 原始: 1042.8, 1856.7, 234.1, 48.2
}

# 疊圖1 (中層) - 需要透明度 0.52
MIDDLE_LAYER_CONFIG = {
    'w': 1066, 'h': 1776, 'x': 234, 'y': 129, # 原始: 1065.7, 1776.2, 234.1, 128.6
    'opacity': 0.52
}

# 疊圖2 (最上層)
TOP_LAYER_CONFIG = {
    'w': 1043, 'h': 1834, 'x': 234, 'y': 109  # 原始: 1042.8, 1833.5, 234.1, 108.8
}

# ==================== 程式邏輯區 ====================

def download_extract_and_crop_kmz(url):
    """下載 KMZ -> 提取圖片 -> 執行指定裁切"""
    print(f"1. 下載並處理 KMZ 溫度圖...")
    try:
        # 下載
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        
        # 讀取
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            target_file = "0/0/0.png"
            if target_file not in z.namelist():
                print(f"   [錯誤] KMZ 中找不到 {target_file}")
                return None
            
            img_data = z.read(target_file)
            img = Image.open(io.BytesIO(img_data))
            
            # --- 執行您指定的裁切邏輯 ---
            # 裁切範圍 (left, upper, right, lower)
            # 原始邏輯: (160, 80, img.width - 310, img.height - 70)
            crop_box = (160, 80, img.width - 310, img.height - 70)
            cropped_img = img.crop(crop_box)
            print(f"   -> 裁切完成 (原始 {img.size} -> 裁切後 {cropped_img.size})")
            
            return cropped_img
            
    except Exception as e:
        print(f"   [錯誤] KMZ 處理失敗: {e}")
        return None

def make_white_transparent(img, threshold=200):
    """將白色背景轉為透明"""
    img = img.convert("RGBA")
    data = np.array(img)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    # 判斷白色 (R,G,B 都大於閥值)
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    
    # 將符合條件的像素 Alpha 設為 0
    data[..., 3][white_mask] = 0
    
    return Image.fromarray(data)

def apply_opacity(img, opacity):
    """調整圖片透明度 (opacity: 0.0 ~ 1.0)"""
    img = img.convert("RGBA")
    alpha = img.split()[3]
    # 將 Alpha 通道數值乘以 opacity
    alpha = alpha.point(lambda p: int(p * opacity))
    img.putalpha(alpha)
    return img

def main():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    
    final_output_path = os.path.join(WORK_DIR, OUTPUT_FILENAME)

    # 1. 準備底圖
    print("準備底圖...")
    if os.path.exists(BASE_CANVAS_PATH):
        canvas = Image.open(BASE_CANVAS_PATH).convert("RGBA")
    else:
        print(f"   [警告] 找不到底圖 {BASE_CANVAS_PATH}，建立空白透明畫布代替。")
        canvas = Image.new("RGBA", (3000, 2500), (0, 0, 0, 0))

    # ---------------------------------------------------------
    # Layer 1 (最下層): 溫度圖 (KMZ -> Crop -> Resize -> Paste)
    # ---------------------------------------------------------
    temp_img = download_extract_and_crop_kmz(KMZ_URL)
    if temp_img:
        cfg = TEMP_LAYER_CONFIG
        target_size = (cfg['w'], cfg['h'])
        # 縮放
        temp_resized = temp_img.resize(target_size, Image.Resampling.LANCZOS)
        # 貼上
        canvas.paste(temp_resized, (cfg['x'], cfg['y']), temp_resized)
        print(f"   -> [最下層] 溫度圖合成完畢。")

    # ---------------------------------------------------------
    # Layer 2 (中層): 疊圖 1 (Load -> White2Trans -> Opacity -> Resize -> Paste)
    # ---------------------------------------------------------
    if os.path.exists(MIDDLE_LAYER_PATH):
        print(f"2. 處理中層 (疊圖1)...")
        mid_img = Image.open(MIDDLE_LAYER_PATH).convert("RGBA")
        
        # 去白底
        mid_img = make_white_transparent(mid_img)
        
        # 調整透明度 (0.52)
        cfg = MIDDLE_LAYER_CONFIG
        mid_img = apply_opacity(mid_img, cfg['opacity'])
        print(f"   -> 透明度已設為 {cfg['opacity']}")
        
        # 縮放
        target_size = (cfg['w'], cfg['h'])
        mid_resized = mid_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 貼上
        canvas.paste(mid_resized, (cfg['x'], cfg['y']), mid_resized)
        print(f"   -> [中層] 疊圖1 合成完畢。")
    else:
        print(f"   [警告] 找不到中層圖片: {MIDDLE_LAYER_PATH}")

    # ---------------------------------------------------------
    # Layer 3 (最上層): 疊圖 2 (Load -> White2Trans -> Resize -> Paste)
    # ---------------------------------------------------------
    if os.path.exists(TOP_LAYER_PATH):
        print(f"3. 處理最上層 (疊圖2)...")
        top_img = Image.open(TOP_LAYER_PATH).convert("RGBA")
        
        # 去白底
        top_img = make_white_transparent(top_img)
        
        # 縮放
        cfg = TOP_LAYER_CONFIG
        target_size = (cfg['w'], cfg['h'])
        top_resized = top_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 貼上
        canvas.paste(top_resized, (cfg['x'], cfg['y']), top_resized)
        print(f"   -> [最上層] 疊圖2 合成完畢。")
    else:
        print(f"   [警告] 找不到最上層圖片: {TOP_LAYER_PATH}")

    # 4. 存檔
    canvas.save(final_output_path)
    print(f"\n全部完成！圖片已儲存至: {final_output_path}")

if __name__ == "__main__":
    main()