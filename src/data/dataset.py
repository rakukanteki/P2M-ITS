import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Dict, Any
from .preprocessing import SensorPreprocessor, VideoProcessor


class FolderDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = np.transpose(s['frames'], (0, 3, 1, 2))
        return {
            'video': torch.FloatTensor(frames),
            'sensor': torch.FloatTensor(s['sensor']),
            'label': torch.LongTensor([s['label']])[0],
        }


class WindowedSampleGenerator:
    def __init__(self, sensor_preprocessor: SensorPreprocessor, 
                 video_processor: VideoProcessor):
        self.sensor_preprocessor = sensor_preprocessor
        self.video_processor = video_processor
    
    def create_samples(self, folder_id: str, folder_data: Dict[str, Any],
                      sensor_cols: List[str], label_mapping: Dict[str, int],
                      sensor_scaler) -> List[Dict[str, Any]]:
        df = folder_data['csv'].copy()
        video_path = folder_data['video_path']
        
        raw = df[sensor_cols].values.astype(np.float32)
        norm = sensor_scaler.transform(raw)
        labels_raw = df['annotation_text'].values
        
        W, S = self.sensor_preprocessor.window_size, self.sensor_preprocessor.stride
        T = len(norm)
        
        frames = self.video_processor.extract_frames(video_path)
        
        samples = []
        for start in range(0, T - W + 1, S):
            window = norm[start:start + W]
            ann_slice = labels_raw[start:start + W]
            
            counts = Counter(ann_slice)
            label_txt = counts.most_common(1)[0][0]
            if label_txt not in label_mapping:
                continue
            
            label = label_mapping[label_txt]
            feat_vec = self.sensor_preprocessor.engineer_window_features(window)
            h = self.sensor_preprocessor.compute_hash(feat_vec)
            
            samples.append({
                'sensor': feat_vec,
                'frames': frames,
                'label': label,
                'folder_id': folder_id,
                'hash': h
            })
        
        return samples


class DataLeakageChecker:
    @staticmethod
    def check(train_samples: List[Dict], val_samples: List[Dict], 
              test_samples: List[Dict], train_folders: List[str],
              val_folders: List[str], test_folders: List[str]) -> Dict[str, Any]:
        train_hashes = {s['hash'] for s in train_samples}
        val_hashes = {s['hash'] for s in val_samples}
        test_hashes = {s['hash'] for s in test_samples}
        
        tv_overlap = len(train_hashes & val_hashes)
        tt_overlap = len(train_hashes & test_hashes)
        vt_overlap = len(val_hashes & test_hashes)
        
        tf_set = set(train_folders)
        vf_set = set(val_folders)
        ef_set = set(test_folders)
        
        return {
            'train_val_overlap': tv_overlap,
            'train_test_overlap': tt_overlap,
            'val_test_overlap': vt_overlap,
            'train_val_folder_overlap': len(tf_set & vf_set),
            'train_test_folder_overlap': len(tf_set & ef_set),
            'zero_leakage': tv_overlap == 0 and tt_overlap == 0
        }
