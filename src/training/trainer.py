import numpy as np
import torch
import time
import os
from typing import Dict, List, Tuple


class CentralizedTrainer:
    def __init__(self, model, device, optimizer, scheduler, criterion, config):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 
            'lr': [], 'epoch_time': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        tr_loss = tr_correct = tr_total = 0
        
        for batch in train_loader:
            video = batch['video'].to(self.device)
            sensor = batch['sensor'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(video, sensor)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                          self.config.training.GRAD_CLIP)
            self.optimizer.step()
            
            tr_loss += loss.item()
            _, preds = outputs.max(1)
            tr_total += labels.size(0)
            tr_correct += preds.eq(labels).sum().item()
        
        return tr_loss / len(train_loader), 100. * tr_correct / tr_total
    
    def validate(self, val_loader):
        self.model.eval()
        va_loss = va_correct = va_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(self.device)
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(video, sensor)
                loss = self.criterion(outputs, labels)
                
                va_loss += loss.item()
                _, preds = outputs.max(1)
                va_total += labels.size(0)
                va_correct += preds.eq(labels).sum().item()
        
        return va_loss / len(val_loader), 100. * va_correct / va_total
    
    def train(self, train_loader, val_loader, output_dir: str):
        best_val_acc = 0.0
        patience_ctr = 0
        
        for epoch in range(self.config.training.EPOCHS):
            t0 = time.time()
            
            tr_loss, tr_acc = self.train_epoch(train_loader)
            va_loss, va_acc = self.validate(val_loader)
            ep_time = time.time() - t0
            
            cur_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(va_acc)
            
            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(va_loss)
            self.history['val_acc'].append(va_acc)
            self.history['lr'].append(cur_lr)
            self.history['epoch_time'].append(ep_time)
            
            if (epoch + 1) % 5 == 0 or epoch < 5:
                gap = tr_acc - va_acc
                print(f"Ep {epoch+1:3d}/{self.config.training.EPOCHS} | "
                      f"Train {tr_acc:6.2f}% | Val {va_acc:6.2f}% | "
                      f"Gap {gap:+.2f}% | Loss T={tr_loss:.4f} V={va_loss:.4f} | "
                      f"LR {cur_lr:.2e}")
            
            if va_acc > best_val_acc + self.config.training.MIN_DELTA:
                best_val_acc = va_acc
                patience_ctr = 0
                torch.save(self.model.state_dict(), 
                          os.path.join(output_dir, 'best_model.pth'))
                print(f"   → Best model saved! val_acc={va_acc:.2f}%")
            else:
                patience_ctr += 1
                if patience_ctr >= self.config.training.PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return best_val_acc
