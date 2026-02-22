import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod


class FLStrategy(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def local_train(self, client_id: int, client_samples, global_params, 
                   device) -> Tuple[Dict, int]:
        pass
    
    @abstractmethod
    def aggregate(self, global_params, all_deltas, client_sizes) -> Dict:
        pass
    
    def _params_to_vec(self, state_dict):
        return {k: v.clone().float() for k, v in state_dict.items()}
    
    def _zeros_like_params(self, state_dict):
        return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dict.items()}
    
    def _add_params(self, a, b, scale=1.0):
        return {k: a[k] + scale * b[k] for k in a}
    
    def _scale_params(self, a, scale):
        return {k: a[k] * scale for k in a}
