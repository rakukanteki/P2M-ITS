import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from .base import FLStrategy


class SCAFFOLD(FLStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.client_ctrls = {}
        self.global_ctrl = None
    
    def initialize_controls(self, global_params):
        self.global_ctrl = self._zeros_like_params(global_params)
    
    def set_client_control(self, client_id: int, params: Dict):
        self.client_ctrls[client_id] = params
    
    def local_train(self, client_id: int, client_samples, global_params, 
                   device, model_class, criterion, config) -> Tuple[Dict, Dict, int]:
        if len(client_samples) == 0:
            return (self._zeros_like_params(global_params), 
                   self._zeros_like_params(global_params), 0)
        
        from ..client import FLClient
        client = FLClient(client_id, client_samples, config)
        
        local_model = model_class.to(device)
        local_model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, local_model.parameters()),
            lr=config.training.LEARNING_RATE,
            weight_decay=config.training.WEIGHT_DECAY
        )
        
        K = config.federated.LOCAL_EPOCHS
        eta = config.training.LEARNING_RATE
        
        client_ctrl = self.client_ctrls.get(client_id, 
                                            self._zeros_like_params(global_params))
        
        local_model.train()
        for _ in range(K):
            for batch in client.loader:
                video = batch['video'].to(device)
                sensor = batch['sensor'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = local_model(video, sensor)
                loss = criterion(outputs, labels)
                loss.backward()
                
                for name, param in local_model.named_parameters():
                    if param.requires_grad and param.grad is not None and name in self.global_ctrl:
                        correction = (self.global_ctrl[name] - client_ctrl[name]).to(device)
                        param.grad.data.add_(correction)
                
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 
                                              config.training.GRAD_CLIP)
                optimizer.step()
        
        local_params = self._params_to_vec(local_model.state_dict())
        delta_params = {k: local_params[k].cpu() - global_params[k].cpu() 
                       for k in global_params}
        
        delta_ctrl = {
            k: (-self.global_ctrl[k] + 
                (global_params[k].cpu() - local_params[k].cpu()) / (K * eta))
            for k in global_params
        }
        
        return delta_params, delta_ctrl, len(client_samples)
    
    def aggregate(self, global_params, all_deltas, all_ctrl_deltas, 
                 client_sizes) -> Tuple[Dict, Dict]:
        total_samples = max(sum(client_sizes), 1)
        K = len(client_sizes)
        
        agg_delta = self._zeros_like_params(global_params)
        for delta, ns in zip(all_deltas, client_sizes):
            w = ns / total_samples
            agg_delta = self._add_params(agg_delta, delta, scale=w)
        
        new_global_params = self._add_params(global_params, agg_delta, scale=1.0)
        
        agg_ctrl_delta = self._zeros_like_params(self.global_ctrl)
        for ctrl_delta in all_ctrl_deltas:
            agg_ctrl_delta = self._add_params(agg_ctrl_delta, ctrl_delta, 
                                             scale=1.0 / K)
        
        new_global_ctrl = self._add_params(self.global_ctrl, agg_ctrl_delta)
        
        self.global_ctrl = new_global_ctrl
        
        return new_global_params, new_global_ctrl
    
    def update_client_control(self, client_id: int, ctrl_delta: Dict):
        if client_id not in self.client_ctrls:
            self.client_ctrls[client_id] = self._zeros_like_params(ctrl_delta)
        
        self.client_ctrls[client_id] = self._add_params(
            self.client_ctrls[client_id], ctrl_delta
        )
