import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config, DataConfig, ModelConfig, TrainingConfig, FederatedConfig
from data.preprocessing import (
    DataLoader, SensorPreprocessor, VideoProcessor, ScalerManager
)
from data.dataset import WindowedSampleGenerator, FolderDataset, DataLeakageChecker
from models.multimodal import MultimodalModel
from training.loss import FocalLoss
from training.trainer import CentralizedTrainer
from training.evaluator import ModelEvaluator
from federated.server import FLServer
from utils.logger import Logger
from utils.metrics import Visualizer
from xai.manager import XAIManager


def main():
    Logger.section("MULTIMODAL FEDERATED LEARNING PIPELINE")
    
    config = Config()
    config.data.OUTPUT_DIR = os.path.expanduser(config.data.OUTPUT_DIR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Logger.info(f"Device: {device}")
    
    Logger.section("STEP 1: LOAD DATA")
    data_loader = DataLoader(config.data.BASE_PATH, config.data.UNCALIBRATED_COLS)
    corrupted = [str(i) for i in range(85, 104)]
    all_data = data_loader.load_folders(corrupted)
    Logger.progress(f"Loaded {len(all_data)} folders")
    
    label_mapping = {}
    label_set = set()
    for folder_data in all_data.values():
        label_set.update(folder_data['csv']['annotation_text'].unique())
    label_mapping = {label: idx for idx, label in enumerate(sorted(label_set))}
    
    Logger.section("STEP 2: FOLDER-LEVEL SPLIT")
    folder_ids = list(all_data.keys())
    np.random.seed(config.data.RANDOM_STATE)
    np.random.shuffle(folder_ids)
    
    n = len(folder_ids)
    n_train = int(n * config.data.TRAIN_RATIO)
    n_val = int(n * config.data.VAL_RATIO)
    
    train_folders = folder_ids[:n_train]
    val_folders = folder_ids[n_train:n_train + n_val]
    test_folders = folder_ids[n_train + n_val:]
    
    Logger.progress(f"Train: {len(train_folders)} | Val: {len(val_folders)} | Test: {len(test_folders)}")
    
    Logger.section("STEP 3: FIT SCALER ON TRAINING DATA")
    train_sensor_data = []
    for folder in train_folders:
        df = all_data[folder]['csv']
        sensor_cols = df.select_dtypes(include=[np.number]).columns
        sensor_cols = [c for c in sensor_cols if 'seconds_elapsed' not in c]
        train_sensor_data.append(df[sensor_cols].values)
    
    train_sensor_data_all = np.vstack(train_sensor_data)
    sensor_scaler = ScalerManager.fit_scaler(
        pd.DataFrame(train_sensor_data_all, columns=sensor_cols), sensor_cols
    )
    scaler_path = os.path.join(config.data.OUTPUT_DIR, 'sensor_scaler.pkl')
    ScalerManager.save_scaler(sensor_scaler, sensor_cols, label_mapping, scaler_path)
    Logger.progress(f"Scaler fitted on {len(train_sensor_data_all):,} rows")
    
    Logger.section("STEP 4-7: CREATE WINDOWED DATASETS")
    sensor_preprocessor = SensorPreprocessor(
        config.data.WINDOW_SIZE, config.data.STRIDE
    )
    video_processor = VideoProcessor(config.data.IMG_SIZE, config.data.FRAMES_PER_VIDEO)
    sample_generator = WindowedSampleGenerator(sensor_preprocessor, video_processor)
    
    def build_split(folder_list, label):
        samples = []
        for idx, folder in enumerate(folder_list, 1):
            s = sample_generator.create_samples(
                folder, all_data[folder], sensor_cols, label_mapping, sensor_scaler
            )
            samples.extend(s)
            if idx % 10 == 0:
                Logger.progress(f"   {label} {idx}/{len(folder_list)} ({len(samples)} windows)")
        return samples
    
    train_samples = build_split(train_folders, 'Train')
    val_samples = build_split(val_folders, 'Val')
    test_samples = build_split(test_folders, 'Test')
    
    Logger.progress(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    
    Logger.section("STEP 8: DATA LEAKAGE VERIFICATION")
    leakage_report = DataLeakageChecker.check(
        train_samples, val_samples, test_samples,
        train_folders, val_folders, test_folders
    )
    Logger.progress(f"Train-Val overlap: {leakage_report['train_val_overlap']}")
    Logger.progress(f"Train-Test overlap: {leakage_report['train_test_overlap']}")
    if leakage_report['zero_leakage']:
        Logger.progress("✅ ZERO DATA LEAKAGE CONFIRMED")
    else:
        Logger.progress("⚠️ WARNING: POTENTIAL DATA LEAKAGE")
    
    Logger.section("STEP 9: PYTORCH DATASET & DATALOADER")
    train_ds = FolderDataset(train_samples)
    val_ds = FolderDataset(val_samples)
    test_ds = FolderDataset(test_samples)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.training.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.training.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.training.BATCH_SIZE, shuffle=False, num_workers=2
    )
    
    Logger.section("STEP 10-12: BUILD CENTRALIZED MODEL")
    sensor_feat_dim = train_samples[0]['sensor'].shape[0]
    Logger.progress(f"Sensor feature dim: {sensor_feat_dim}")
    
    model = MultimodalModel(
        sensor_feat_dim,
        config.model.NUM_CLASSES,
        config.model.BACKBONE,
        config.model.SENSOR_EMBEDDING_DIM,
        config.model.VIDEO_EMBEDDING_DIM,
        config.model.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger.progress(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    train_labels = [s['label'] for s in train_samples]
    class_weights_np = compute_class_weight(
        'balanced', classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights_np).to(device)
    
    criterion = FocalLoss(gamma=config.training.FOCAL_GAMMA, weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.LEARNING_RATE,
        weight_decay=config.training.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    Logger.section("STEP 13: CENTRALIZED TRAINING")
    trainer = CentralizedTrainer(model, device, optimizer, scheduler, criterion, config)
    best_val_acc = trainer.train(train_loader, val_loader, config.data.OUTPUT_DIR)
    Logger.progress(f"Best centralized val accuracy: {best_val_acc:.2f}%")
    
    model.load_state_dict(torch.load(
        os.path.join(config.data.OUTPUT_DIR, 'best_model.pth')
    ))
    Visualizer.plot_training_history(
        trainer.history,
        os.path.join(config.data.OUTPUT_DIR, 'training_history.png')
    )
    
    Logger.section("STEP 14: CENTRALIZED EVALUATION")
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader, class_names)
    
    ci_results = evaluator.bootstrap_ci(
        results['labels'], results['predictions'], results['probabilities']
    )
    
    Logger.progress(f"Centralized Accuracy: {results['accuracy']:.2f}%")
    Logger.progress(f"  95% CI: [{ci_results['accuracy'][1]:.2f}%, {ci_results['accuracy'][2]:.2f}%]")
    Logger.progress(f"Macro F1: {results['macro_f1']:.4f}")
    
    Visualizer.plot_confusion_matrix(
        results['confusion_matrix'], class_names,
        os.path.join(config.data.OUTPUT_DIR, 'confusion_matrix.png')
    )
    
    Logger.section("STEP 15: EXPLAINABLE AI (XAI) - SHAP & GradCAM")
    xai_manager = XAIManager(model, device, config.data.OUTPUT_DIR, class_names)
    
    Logger.progress("Generating SHAP explanations for sensor features...")
    shap_features = xai_manager.generate_shap_explanations(
        train_loader, test_loader, num_background=50, num_test=10
    )
    Logger.progress(f"Top sensor features: {list(shap_features.keys())[:5]}")
    
    Logger.progress("Generating GradCAM explanations for video frames...")
    gradcam_results = xai_manager.generate_gradcam_explanations(
        test_loader, num_samples=5
    )
    Logger.progress(f"GradCAM samples generated: {gradcam_results.get('samples_generated', 0)}")
    
    xai_manager.save_xai_report()
    Logger.progress("XAI report saved to outputs/xai/xai_report.json")
    
    Logger.section("STEP 16-17: SCAFFOLD FEDERATED LEARNING")
    samples_by_client = {}
    folders_per_client = np.array_split(train_folders, config.federated.NUM_CLIENTS)
    
    for cid, flist in enumerate(folders_per_client):
        client_samples = [s for s in train_samples if s['folder_id'] in set(flist)]
        samples_by_client[cid] = client_samples
        Logger.progress(f"Client {cid}: {len(list(flist))} folders | {len(client_samples)} windows")
    
    fed_model = MultimodalModel(
        sensor_feat_dim, config.model.NUM_CLASSES,
        config.model.BACKBONE, config.model.SENSOR_EMBEDDING_DIM,
        config.model.VIDEO_EMBEDDING_DIM, config.model.DROPOUT
    ).to(device)
    
    fl_server = FLServer(
        MultimodalModel, fed_model, device, config, criterion
    )
    
    best_fed_acc = fl_server.train(
        list(samples_by_client.values()), val_loader, config.data.OUTPUT_DIR
    )
    Logger.progress(f"Best federated val accuracy: {best_fed_acc:.2f}%")
    
    Visualizer.plot_federated_convergence(
        fl_server.history,
        os.path.join(config.data.OUTPUT_DIR, 'federated_history.png')
    )
    
    Logger.section("STEP 18: CLIENT-LEVEL PERFORMANCE VARIANCE")
    fed_model.load_state_dict(torch.load(
        os.path.join(config.data.OUTPUT_DIR, 'best_federated_model.pth')
    ))
    
    client_accs = []
    for cid, c_samples in samples_by_client.items():
        if len(c_samples) == 0:
            continue
        c_ds = FolderDataset(c_samples)
        c_loader = torch.utils.data.DataLoader(
            c_ds, batch_size=config.training.BATCH_SIZE, num_workers=0
        )
        c_correct = c_total = 0
        fed_model.eval()
        with torch.no_grad():
            for batch in c_loader:
                vid = batch['video'].to(device)
                sen = batch['sensor'].to(device)
                labs = batch['label'].to(device)
                out = fed_model(vid, sen)
                _, p = out.max(1)
                c_correct += p.eq(labs).sum().item()
                c_total += labs.size(0)
        c_acc = 100. * c_correct / max(c_total, 1)
        client_accs.append(c_acc)
        Logger.progress(f"Client {cid}: {c_acc:.2f}% ({c_total} windows)")
    
    Logger.progress(f"Mean client accuracy: {np.mean(client_accs):.2f}%")
    Logger.progress(f"Std (variance): {np.std(client_accs):.2f}%")
    
    Visualizer.plot_client_accuracy(
        client_accs,
        os.path.join(config.data.OUTPUT_DIR, 'client_accuracy.png')
    )
    
    Logger.section("STEP 19: SAVE FINAL RESULTS")
    final_results = {
        'data_leakage': leakage_report,
        'centralized': {
            'accuracy': results['accuracy'],
            'accuracy_ci_95': [ci_results['accuracy'][1], ci_results['accuracy'][2]],
            'macro_f1': results['macro_f1'],
            'macro_f1_ci_95': [ci_results['macro_f1'][1], ci_results['macro_f1'][2]],
            'inference_time_ms': results['inference_time_ms'],
        },
        'federated': {
            'strategy': config.federated.STRATEGY,
            'best_val_accuracy': best_fed_acc,
            'client_accuracies': client_accs,
            'client_acc_mean': float(np.mean(client_accs)),
            'client_acc_std': float(np.std(client_accs)),
        }
    }
    
    with open(os.path.join(config.data.OUTPUT_DIR, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    Logger.section("PIPELINE COMPLETE")
    Logger.progress(f"Outputs in: {config.data.OUTPUT_DIR}")
    
    return model, fed_model, config, final_results


if __name__ == '__main__':
    main()
