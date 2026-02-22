import os
import pickle
import numpy as np
import pandas as pd
import cv2
import hashlib
from typing import Dict, List, Tuple, Any
from collections import Counter
from sklearn.preprocessing import StandardScaler


class SensorPreprocessor:
    def __init__(self, window_size: int = 50, stride: int = 25):
        self.window_size = window_size
        self.stride = stride
    
    def engineer_window_features(self, window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        _min = window.min(axis=0)
        _max = window.max(axis=0)
        delta = np.diff(window, axis=0).mean(axis=0)
        energy = (window ** 2).sum(axis=0)
        freq_ratio = (np.diff(window, axis=0) ** 2).sum(axis=0) / (np.abs(window).sum(axis=0) + eps)
        
        features = np.concatenate([mean, std, _min, _max, delta, energy, freq_ratio])
        features = np.where(np.isfinite(features), features, 0.0)
        return features.astype(np.float32)
    
    @staticmethod
    def compute_hash(features: np.ndarray) -> str:
        return hashlib.md5(','.join(f'{v:.6f}' for v in features[:10]).encode()).hexdigest()


class VideoProcessor:
    def __init__(self, img_size: Tuple[int, int] = (224, 224), frames_per_video: int = 30):
        self.img_size = img_size
        self.frames_per_video = frames_per_video
    
    def extract_frames(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        neutral = np.full((self.frames_per_video, *self.img_size, 3), 0.5, dtype=np.float32)
        
        if not cap.isOpened():
            cap.release()
            return neutral
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return neutral
        
        N, L = self.frames_per_video, total
        indices = [int(round(n * (L - 1) / (N - 1))) for n in range(N)] if N > 1 else [0]
        
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return neutral
        
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])
        
        return np.array(frames[:self.frames_per_video], dtype=np.float32)


class DataLoader:
    def __init__(self, base_path: str, uncalibrated_cols: List[str]):
        self.base_path = base_path
        self.uncalibrated_cols = uncalibrated_cols
    
    def load_folders(self, corrupted_folders: List[str] = None) -> Dict[str, Dict[str, Any]]:
        if corrupted_folders is None:
            corrupted_folders = []
        
        folders = sorted(
            [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))],
            key=lambda x: int(x) if x.isdigit() else 0
        )
        folders = [f for f in folders if f not in corrupted_folders]
        
        all_data = {}
        for folder in folders:
            try:
                csv_path = os.path.join(self.base_path, folder, f"{folder}.csv")
                video_path = os.path.join(self.base_path, folder, f"{folder}.mp4")
                
                if os.path.exists(csv_path) and os.path.exists(video_path):
                    df = pd.read_csv(csv_path)
                    df = df.drop(columns=self.uncalibrated_cols, errors='ignore')
                    df = df[df['annotation_text'].notna()]
                    df = df[df['annotation_text'].str.strip() != '']
                    all_data[folder] = {'csv': df, 'video_path': video_path}
            except Exception as e:
                print(f"Error loading {folder}: {e}")
        
        return all_data


class ScalerManager:
    @staticmethod
    def fit_scaler(train_data: pd.DataFrame, sensor_cols: List[str]) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(train_data[sensor_cols].values)
        return scaler
    
    @staticmethod
    def save_scaler(scaler: StandardScaler, sensor_cols: List[str], 
                    label_mapping: Dict[str, int], output_path: str):
        with open(output_path, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'sensor_cols': sensor_cols,
                'label_mapping': label_mapping
            }, f)
    
    @staticmethod
    def load_scaler(path: str) -> Tuple[StandardScaler, List[str], Dict[str, int]]:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['scaler'], data['sensor_cols'], data['label_mapping']
