import torch
import torch.nn.functional as F
import time
import os
from typing import Dict, List, Tuple
from .strategies.fedavg import FedAvg
from .strategies.scaffold import SCAFFOLD
from .strategies.fedprox import FedProx
from .strategies.fedadam import FedAdam
from .strategies.fednova import FedNova
from ..training.loss import FocalLoss


class FLServer:
    def __init__(self, model_class, global_model, device, config, criterion):
        self.model_class = model_class
        self.global_model = global_model
        self.device = device
        self.config = config
        self.criterion = criterion
        self.global_params = {k: v.cpu() for k, v in global_model.state_dict().items()}
        
        strategy_name = config.federated.STRATEGY.lower()
        if strategy_name == 'fedavg':
            self.strategy = FedAvg(config)
        elif strategy_name == 'scaffold':
            self.strategy = SCAFFOLD(config)
            self.strategy.initialize_controls(self.global_params)
        elif strategy_name == 'fedprox':
            self.strategy = FedProx(config)
        elif strategy_name == 'fedadam':
            self.strategy = FedAdam(config)
        elif strategy_name == 'fednova':
            self.strategy = FedNova(config)
        else:
            raise ValueError(f"Unknown FL strategy: {strategy_name}")
        
        self.history = {'round': [], 'val_loss': [], 'val_acc': []}
    
    def federated_round(self, client_samples_list: List[List[Dict]], 
                       val_loader, selected_clients: List[int]):
        all_deltas = []
        all_sizes = []
        all_ctrl_deltas = [] if isinstance(self.strategy, SCAFFOLD) else None
        all_local_updates = [] if isinstance(self.strategy, FedNova) else None
        
        for client_id in selected_clients:
            if isinstance(self.strategy, SCAFFOLD):
                delta, ctrl_delta, size = self.strategy.local_train(
                    client_id, client_samples_list[client_id],
                    self.global_params, self.device,
                    self.model_class, self.criterion, self.config
                )
                all_deltas.append(delta)
                all_ctrl_deltas.append(ctrl_delta)
                self.strategy.update_client_control(client_id, ctrl_delta)
                all_sizes.append(size)
            
            elif isinstance(self.strategy, FedNova):
                delta, size, local_updates = self.strategy.local_train(
                    client_id, client_samples_list[client_id],
                    self.global_params, self.device,
                    self.model_class, self.criterion, self.config
                )
                all_deltas.append(delta)
                all_sizes.append(size)
                all_local_updates.append(local_updates)
            
            else:
                delta, size = self.strategy.local_train(
                    client_id, client_samples_list[client_id],
                    self.global_params, self.device,
                    self.model_class, self.criterion, self.config
                )
                all_deltas.append(delta)
                all_sizes.append(size)
        
        if isinstance(self.strategy, SCAFFOLD):
            self.global_params, _ = self.strategy.aggregate(
                self.global_params, all_deltas, all_ctrl_deltas, all_sizes
            )
        elif isinstance(self.strategy, FedNova):
            self.global_params = self.strategy.aggregate(
                self.global_params, all_deltas, all_sizes, all_local_updates
            )
        else:
            self.global_params = self.strategy.aggregate(
                self.global_params, all_deltas, all_sizes
            )
        
        self.global_model.load_state_dict(
            {k: v.to(self.device) for k, v in self.global_params.items()}
        )
        
        val_loss, val_acc = self._validate(val_loader)
        
        return val_loss, val_acc
    
    def _validate(self, val_loader):
        self.global_model.eval()
        val_loss = val_correct = val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(self.device)
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.global_model(video, sensor)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
        
        return val_loss / len(val_loader), 100. * val_correct / val_total
    
    def train(self, client_samples_list: List[List[Dict]], val_loader, 
             output_dir: str) -> float:
        best_val_acc = 0.0
        
        for rnd in range(self.config.federated.NUM_ROUNDS):
            rnd_start = time.time()
            
            selected = list(range(len(client_samples_list)))
            val_loss, val_acc = self.federated_round(client_samples_list, val_loader, selected)
            
            self.history['round'].append(rnd + 1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if (rnd + 1) % 5 == 0 or rnd < 3:
                print(f"Round {rnd+1:3d}/{self.config.federated.NUM_ROUNDS} | Val {val_acc:.2f}% | "
                      f"Loss {val_loss:.4f} | Time {time.time()-rnd_start:.1f}s")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.global_model.state_dict(),
                          os.path.join(output_dir, 'best_federated_model.pth'))
        
        return best_val_acc
