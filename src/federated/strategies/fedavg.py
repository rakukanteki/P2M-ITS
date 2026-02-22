import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List
from .base import FLStrategy


class FedAvg(FLStrategy):
    def local_train(self, client_id: int, client_samples, global_params, 
                   device, model_class, criterion, config) -> Tuple[Dict, int]:
        if len(client_samples) == 0:
            return self._zeros_like_params(global_params), 0
        
        from ..client import FLClient
        client = FLClient(client_id, client_samples, config)
        
        local_model = model_class.to(device)
        local_model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, local_model.parameters()),
            lr=config.training.LEARNING_RATE,
            weight_decay=config.training.WEIGHT_DECAY
        )
        
        local_model.train()
        for _ in range(config.federated.LOCAL_EPOCHS):
            for batch in client.loader:
                video = batch['video'].to(device)
                sensor = batch['sensor'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = local_model(video, sensor)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 
                                              config.training.GRAD_CLIP)
                optimizer.step()
        
        local_params = self._params_to_vec(local_model.state_dict())
        delta = {k: local_params[k].cpu() - global_params[k].cpu() for k in global_params}
        
        return delta, len(client_samples)
    
    def aggregate(self, global_params, all_deltas, client_sizes) -> Dict:
        total_samples = max(sum(client_sizes), 1)
        agg_delta = self._zeros_like_params(global_params)
        
        for delta, ns in zip(all_deltas, client_sizes):
            w = ns / total_samples
            agg_delta = self._add_params(agg_delta, delta, scale=w)
        
        return self._add_params(global_params, agg_delta, scale=1.0)
