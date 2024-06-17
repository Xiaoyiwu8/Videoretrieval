import streamlit as st
import cv2
import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline
from ultralytics import YOLO
import speech_recognition as sr

# 禁用Hugging Face缓存系统警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 初始化 YOLO 和 CLIP 模型
device = 'cpu'  # 或者 'cuda' 如果有 GPU
yolo_model = YOLO('yolov8n.pt').to(device)  # 使用自动下载的模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_frames(video_path, output_dir, frame_rate=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // frame_rate)

    frame_id = 0
    saved_frame_count = 0  # 增加一个计数器来记录保存的帧数
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1  # 每保存一帧，计数器增加

        frame_id += 1

    cap.release()
    return saved_frame_count

def extract_features(frame_dir):
    features = []
    detections = []  # 定义 detections 列表来存储所有帧的检测结果
    frame_id = 0  # 增加一个 frame_id 计数器

    for frame_filename in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_filename)
        frame = cv2.imread(frame_path)

        # 使用 YOLO 检测对象
        results = yolo_model(frame)
        for result in results:
            detection_boxes = result.boxes.xyxy.cpu().numpy()  # 获取检测框

            # 使用 CLIP 提取图像特征
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
            clip_features = clip_model.get_image_features(**inputs)

            # 假设结果包括对象名称和处理时间
            object_names = [result.names[int(cls)] for cls in result.boxes.cls]  # 获取对象名称
            processing_time = result.speed  # 获取处理时间

            height, width, _ = frame.shape
            counts = {obj: object_names.count(obj) for obj in set(object_names)}

            detections.append([frame_id, height, width, list(object_names), list(counts.values()), processing_time])
            features.append((frame_filename, detection_boxes, clip_features.cpu().detach().numpy()))

        frame_id += 1

    return features, detections

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("请说话...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='zh-CN')
        st.write(f"你说的是: {text}")
        return text
    except sr.UnknownValueError:
        st.write("无法识别音频")
    except sr.RequestError as e:
        st.write(f"无法请求结果; {e}")

    return ""

st.title("视频检索系统")

uploaded_video = st.file_uploader("上传一个视频", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    video_path = os.path.join("temp", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    output_dir = os.path.join("temp", "frames")
    num_frames = extract_frames(video_path, output_dir)
    st.write(f"提取了 {num_frames} 帧")

    video_features, detections = extract_features(output_dir)

    # 将检测结果转换为DataFrame
    df = pd.DataFrame(detections, columns=['Frame', 'Height', 'Width', 'Objects', 'Counts', 'ProcessingTime'])
    st.dataframe(df)

    # 将DataFrame保存到CSV文件
    output_csv = os.path.join("temp", "detection_results.csv")
    df.to_csv(output_csv, index=False)
    st.write(f"检测结果已保存到 {output_csv}")

    # 打印提取的特征信息
    for feature in video_features:
        frame_filename, detection_boxes, clip_features = feature
        st.write(f"Frame: {frame_filename}")
        st.write(f"Detections: {detection_boxes}")
        st.write(f"CLIP Features: {clip_features}")

search_text = st.text_input("输入查询文本")
if st.button("查询"):
    entities = ner(search_text)
    st.write("识别的实体：")
    for entity in entities:
        st.write(f"{entity['word']} ({entity['entity']})")

if st.button("语音输入"):
    speech_text = recognize_speech()
    if speech_text:
        entities = ner(speech_text)
        st.write("识别的实体：")
        for entity in entities:
            st.write(f"{entity['word']} ({entity['entity']})")














