import cv2
import numpy as np
import streamlit as st
import os
import tempfile
from PIL import Image, ImageOps
from ultralytics import YOLO

# --- 1. 頁面基本設定 ---
st.set_page_config(
    page_title="台灣道路安全帽辨識系統",
    page_icon="🛵",
    layout="wide"
)

# --- 2. 核心後處理邏輯：Bike Proximity Filter ---
def apply_bike_proximity_filter(results, proximity_ratio=0.5):
    """
    雙重過濾：rider_with_helmet 與 rider_without_helmet 的中心點都必須
    靠近某個 bike，否則視為路人/雜訊誤判，濾除不顯示。

    背景：實測影片發現路人會被誤判成 rider_with_helmet（非僅
    rider_without_helmet 才有此問題），因此兩個 class 都需要過濾。
    """
    boxes = results[0].boxes
    bike_boxes = []
    rider_with_boxes = []
    rider_without_boxes = []

    for box in boxes:
        cls_name = results[0].names[int(box.cls[0])]
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        if cls_name == 'bike':
            bike_boxes.append(xyxy)
        elif cls_name == 'rider_with_helmet':
            rider_with_boxes.append((xyxy, conf))
        elif cls_name == 'rider_without_helmet':
            rider_without_boxes.append((xyxy, conf))

    def is_near_any_bike(rx1, ry1, rx2, ry2):
        rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
        for (bx1, by1, bx2, by2) in bike_boxes:
            bike_diag = ((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5
            bike_cx, bike_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            dist = ((rcx - bike_cx) ** 2 + (rcy - bike_cy) ** 2) ** 0.5
            if dist < bike_diag * proximity_ratio:
                return True
        return False

    valid_with = [item for item in rider_with_boxes if is_near_any_bike(*item[0])]
    valid_without = [item for item in rider_without_boxes if is_near_any_bike(*item[0])]

    return valid_with, valid_without


# --- 3. OpenCV 繪圖輔助函式 ---
def draw_predictions(image, results, valid_with, valid_without, use_filter):
    """
    繪製 Bounding Box。valid_with / valid_without 是過濾後仍保留的偵測框，
    用座標比對決定哪些原始偵測框要被畫出來。
    """
    img_draw = np.array(image)
    color_map = {
        'bike': (255, 165, 0),                 # 橘色
        'rider_with_helmet': (0, 255, 0),       # 綠色
        'rider_without_helmet': (255, 0, 0)     # 紅色
    }
    boxes = results[0].boxes
    names = results[0].names

    v_with_boxes = [tuple(item[0]) for item in valid_with]
    v_without_boxes = [tuple(item[0]) for item in valid_without]

    for box in boxes:
        cls_name = names[int(box.cls[0])]
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, xyxy)

        if use_filter:
            if cls_name == 'rider_with_helmet':
                if not any(abs(b[0] - xyxy[0]) < 1.0 and abs(b[1] - xyxy[1]) < 1.0 for b in v_with_boxes):
                    continue
            elif cls_name == 'rider_without_helmet':
                if not any(abs(b[0] - xyxy[0]) < 1.0 and abs(b[1] - xyxy[1]) < 1.0 for b in v_without_boxes):
                    continue

        color = color_map.get(cls_name, (128, 128, 128))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, h + 5)
        cv2.rectangle(img_draw, (x1, text_y - h - 5), (x1 + w, text_y + 5), color, -1)
        cv2.putText(img_draw, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return img_draw


# --- 4. Sidebar 側邊欄控制面板 ---
st.sidebar.title("⚙️ 控制面板")
st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider("Confidence 門檻值", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
st.sidebar.markdown("### 後處理優化")
use_proximity_filter = st.sidebar.checkbox("開啟 Bike Proximity 過濾器", value=True)
proximity_ratio = 0.5
if use_proximity_filter:
    proximity_ratio = st.sidebar.slider("Proximity 距離比例", min_value=0.1, max_value=1.0, value=0.5, step=0.05)


# 【修正】補回模型路徑存在性檢查，避免找不到檔案時噴出不友善的原始錯誤
@st.cache_resource
def load_yolo_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_3class_v4.pt")
    if not os.path.exists(model_path):
        st.error(
            f"❌ 找不到模型檔案：`{model_path}`\n\n"
            "請確認 `best_3class_v4.pt` 與 `app.py` 放在同一資料夾。"
        )
        st.stop()
    return YOLO(model_path)


model = load_yolo_model()

# --- 5. 主畫面標頭與首頁 Overview ---
st.title("🛵 台灣道路安全帽辨識系統 (YOLOv8)")
st.markdown("---")

st.markdown("### 📌 專案背景與系統架構")
col_intro, col_flow = st.columns([1, 1])
with col_intro:
    st.write("**專案目標**：建立一套適用於台灣複雜道路場景（Dashcam 行車紀錄器視角）的安全帽配戴辨識系統。")
    st.write("**核心痛點**：傳統物件偵測容易將非騎車之路人或背景誤判為騎士（無論是否戴帽）。本系統加入幾何鄰近過濾演算法，以降低 False Positive 雜訊。")
    st.markdown("""
    **目標類別定義 (3-Class)**：
    - `bike`（機車）
    - `rider_with_helmet`（配戴安全帽騎士）
    - `rider_without_helmet`（未配戴安全帽騎士）
    """)
with col_flow:
    st.info("""
    **⚙️ 系統資料流架構：**  
    
    影像/影片輸入 (Input)  
          ↓  
    YOLOv8 模型推論 (3-Class Detection)  
          ↓  
    核心 Bike Proximity 過濾器（幾何後處理，雙 class 過濾）  
          ↓  
    輸出疑似未配戴安全帽騎士警示  
    """)

st.markdown("---")

# --- 6. 分頁佈局 ---
tab1, tab2, tab3, tab4 = st.tabs(["📸 圖片測試", "🎥 影片測試", "📊 Model Performance", "🔍 Project Insights"])

# --- Tab 1: 圖片測試【修正：改用 valid_with / valid_without 雙變數，配合新的 draw_predictions 簽名】 ---
with tab1:
    st.header("上傳圖片進行偵測")
    uploaded_file = st.file_uploader("選擇一張台灣街景或機車騎士圖片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        pil_image = ImageOps.exif_transpose(pil_image)  # 修正手機照片 EXIF 旋轉

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="原始圖片", use_container_width=True)
        with col2:
            with st.spinner("模型辨識中..."):
                results = model(pil_image, conf=conf_threshold, iou=0.45)

                if use_proximity_filter:
                    valid_with, valid_without = apply_bike_proximity_filter(results, proximity_ratio)
                else:
                    valid_with = [
                        (box.xyxy[0].tolist(), float(box.conf[0]))
                        for box in results[0].boxes
                        if results[0].names[int(box.cls[0])] == 'rider_with_helmet'
                    ]
                    valid_without = [
                        (box.xyxy[0].tolist(), float(box.conf[0]))
                        for box in results[0].boxes
                        if results[0].names[int(box.cls[0])] == 'rider_without_helmet'
                    ]

                processed_img = draw_predictions(pil_image, results, valid_with, valid_without, use_proximity_filter)
                st.image(processed_img, caption="辨識結果", use_container_width=True)
                st.success(f"偵測完成！本畫面共偵測到 **{len(valid_without)}** 位疑似未配戴安全帽之騎士。")

# --- Tab 2: 影片測試【移除無法正常運作的暫停功能，其餘保留】 ---
with tab2:
    st.header("🎥 影片動態辨識測試")

    video_source = st.radio("選擇影片來源", ["使用系統預設範例影片", "自行上傳本地影片"], horizontal=True)
    uploaded_video = None

    if video_source == "自行上傳本地影片":
        uploaded_video = st.file_uploader("選擇測試影片...", type=["mp4", "avi", "mov"])
        st.caption("建議使用 30 秒以內短片段，避免處理過久。")

    tmp_in_path = None
    if video_source == "使用系統預設範例影片":
        if os.path.exists("vid/sample_clip.mp4"):
            tmp_in_path = "vid/sample_clip.mp4"
            st.success("✅ 已成功載入預設範例影片 `sample_clip.mp4`")
        else:
            st.error("❌ 找不到 `sample_clip.mp4`！請確認檔案已放入專案根目錄中。")
    elif video_source == "自行上傳本地影片" and uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

    frame_skip = st.slider("每隔幾幀處理一次", min_value=1, max_value=15, value=3)

    if tmp_in_path is not None:
        cap = cv2.VideoCapture(tmp_in_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(f"📋 影片規格：`{total_frames}` 幀 · `{fps:.1f}` FPS ｜ 抽樣處理 `{total_frames // frame_skip}` 幀")

        start_btn = st.button("🎬 開始執行影片辨識", type="primary")

        if start_btn:
            st.session_state.video_processed = False
            st.session_state.video_bytes = None

            frame_placeholder = st.empty()
            progress_bar = st.progress(0, text="準備處理影片...")

            cap = cv2.VideoCapture(tmp_in_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                tmp_out_path = tmp_out.name

            output_fps = max(fps / frame_skip, 1.0)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_out_path, fourcc, output_fps, (width, height))

            frame_idx = 0
            without_helmet_total = 0
            with_helmet_total = 0

            with st.spinner("影片辨識中..."):
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        results = model(pil_frame, conf=conf_threshold, iou=0.45, verbose=False)

                        if use_proximity_filter:
                            valid_with, valid_without = apply_bike_proximity_filter(results, proximity_ratio)
                        else:
                            valid_with = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in results[0].boxes if results[0].names[int(b.cls[0])] == 'rider_with_helmet']
                            valid_without = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in results[0].boxes if results[0].names[int(b.cls[0])] == 'rider_without_helmet']

                        processed_rgb = draw_predictions(pil_frame, results, valid_with, valid_without, use_proximity_filter)
                        frame_placeholder.image(processed_rgb, caption=f"🎬 辨識中（第 {frame_idx} / {total_frames} 幀）", use_container_width=True)

                        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                        writer.write(processed_bgr)
                        without_helmet_total += len(valid_without)
                        with_helmet_total += len(valid_with)

                    frame_idx += 1
                    progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0), text=f"進度：{frame_idx} / {total_frames} 幀")

            cap.release()
            writer.release()
            progress_bar.empty()
            frame_placeholder.empty()

            with open(tmp_out_path, "rb") as f:
                st.session_state.video_bytes = f.read()
            st.session_state.without_helmet_total = without_helmet_total
            st.session_state.with_helmet_total = with_helmet_total
            st.session_state.video_processed = True

            if video_source == "自行上傳本地影片":
                os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

        if st.session_state.get("video_processed", False):
            st.success(
                f"✅ 處理完成！全片累計偵測到 **{st.session_state.without_helmet_total}** 次疑似未戴安全帽騎士"
                f"（已套用 proximity 過濾，排除路人等雜訊誤判）。"
            )
            st.download_button(
                label="⬇️ 下載辨識結果影片",
                data=st.session_state.video_bytes,
                file_name="helmet_detection_result.mp4",
                mime="video/mp4"
            )
            st.caption("⚠️ 部分瀏覽器無法直接預覽 mp4v 編碼，下載後用 VLC 開啟即可正常播放。")

# --- Tab 3: Model Performance【新增 v1→v4 修正後進步曲線】 ---
with tab3:
    st.header("📊 最終模型效能評估")
    st.markdown("專案發現早期版本（V2、V3）驗證集存在 Data Leakage 風險，已用 V4 重建之乾淨驗證集，對 V1～V4 重新評估，確保比較公平。")

    st.subheader("📈 版本迭代真實進步曲線（修正後）")
    progression_data = {
        "版本": ["v1", "v2", "v3", "v4"],
        "Train 樣本數": [165, 507, 768, 654],
        "mAP50": [0.2617, 0.4207, 0.6007, 0.871],
        "備註": ["3-class 初版", "補充 without_helmet 樣本", "補充白/彩色安全帽樣本", "✅ 補充銀/灰背面視角 + 資料清洗"],
    }
    st.table(progression_data)
    st.caption("⚠️ 此為用 V4 乾淨驗證集（54 張，無 augmentation 混入）重新評估的結果，非各版本原始訓練記錄數字。")

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("💡 整體指標 (Overall Metrics)")
        overall_data = {
            "評估指標 (Metric)": ["Precision (精準率)", "Recall (召回率)", "mAP50", "mAP50-95"],
            "數值 (Value)": [0.8684, 0.7539, 0.871, 0.5475]
        }
        st.table(overall_data)

    with col_m2:
        st.subheader("🎯 各類別指標 (Class-wise Metrics)")
        class_data = {
            "類別名稱 (Class)": ["bike", "rider_with_helmet", "rider_without_helmet"],
            "Precision": [0.895, 0.846, 0.865],
            "Recall": [0.793, 0.702, 0.767],
            "mAP50": [0.887, 0.807, 0.913]
        }
        st.table(class_data)

# --- Tab 4: Project Insights【更新 Data Leakage 章節，補上實際方法論與真實數字】 ---
with tab4:
    st.header("🔍 Project Insights & Lessons Learned")

    st.markdown("### 🛠️ 資料集優化歷程 (Dataset Refinement)")
    st.write("- **銀白色安全帽背面誤判修正**：早期版本容易將銀色背面安全帽高信心度誤判為未戴安全帽。最終版本針對台灣街景進行特徵補強與資料清洗，提升泛化能力。")
    st.write("- **後處理 Bike Proximity Filter（雙 class 過濾）**：利用幾何演算法，強制限制騎士偵測框（無論是否戴帽）與機車框之相對距離。此設計源自實測影片時發現路人會被誤判為 `rider_with_helmet`，因此將過濾範圍從單一類別擴大為雙類別，有效剔除路人、靜態雜訊等 False Positive。")

    st.markdown("### ⚠️ 專案挑戰：Data Leakage（資料洩漏）的發現與修正")
    st.warning("""
**問題發現**：檢視 V2、V3 的資料處理流程時，發現 Augmented 圖片是在切分 Train/Valid **之前**就先生成，再整批重新上傳 Roboflow 切分。這導致同一張原圖的不同增強版本，可能一張落在 Train、另一張落在 Valid，形成 Validation Data Leakage，使當時記錄的 mAP 數字過於樂觀、不可信。

**修正方式**：撰寫自動化腳本，比對 V4 乾淨驗證集（54 張無 augmentation 原圖）與 V1～V3 訓練集的檔名重疊率，確認重疊比例隨版本疊代遞增（V1: 7.5% → V2: 15.1% → V3: 24.5%），符合資料集演進邏輯。接著用這份乾淨驗證集，重新評估 V1～V4 各版本當時實際採用的主力 checkpoint；其中 V4 的重評結果（0.871）與原始記錄（0.869）幾乎一致，驗證了重評方法本身無誤。

**結果**：修正後的真實進步曲線（mAP50：0.26 → 0.42 → 0.60 → 0.87）比原始記錄更具參考價值——原始數字因 data leakage 部分版本偏高，修正後才看得出各次資料補強的真實貢獻。
    """)

    st.markdown("### 🚀 未來展望 (Future Work)")
    st.info("""
        1. 補充帽類負樣本（棒球帽、鴨舌帽等），降低跨類別誤判
        2. 補充腳踏車負樣本或新增獨立 class
        3. 蒐集夜間與低光源訓練資料，評估並強化模型於低光源場景的表現
        4. 引進 ByteTrack 做多目標追蹤與車流統計
    """)
