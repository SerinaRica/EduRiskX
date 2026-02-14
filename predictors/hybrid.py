from typing import List, Dict, Any
import numpy as np
import torch
from .base_predictor import BasePredictor

class HybridPredictor(BasePredictor):
    def __init__(self, base_predictor, reasoner, feature_cols: List[str]):
        self.base = base_predictor
        self.reasoner = reasoner
        self.feature_cols = feature_cols
        
    def fit(self, X_train, y_train, **kwargs):
        # Delegate training to the base neural model
        return self.base.fit(X_train, y_train, **kwargs)
        
    def predict_proba(self, X, lengths=None, batch_size=128, device=None):
        # 1. Get neural predictions
        probs, hidden = self.base.predict_proba(X, lengths, batch_size, device)
        
        # 2. Apply F-Logic Reasoning
        # We need to reconstruct the feature dictionary for each student
        # X shape: [batch, max_len, input_dim]
        # We use the last valid timestep for reasoning
        
        final_probs = []
        
        if lengths is None:
             lengths = np.maximum(1, np.sum((X!=0).any(axis=2), axis=1))
        
        for i in range(len(probs)):
            neural_prob = probs[i]
            length = int(lengths[i])
            
            # Get features from the last valid timestep (length-1)
            # Or should we use aggregated features? 
            # F-Logic usually works on the current state.
            # Let's use the features at the last available week.
            last_step_features = X[i, length-1, :]
            
            student_features = {}
            for j, col in enumerate(self.feature_cols):
                if j < len(last_step_features):
                    student_features[col] = float(last_step_features[j])
            
            # Reason
            # risk_prob from neural model is used as input belief
            final_belief, _ = self.reasoner.reason(student_features, neural_prob)
            final_probs.append(final_belief)
            
        return np.array(final_probs), hidden
