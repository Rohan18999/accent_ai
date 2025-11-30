"""HuBERT model loading and prediction utilities"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# Label names (same order as training)
# Label names (same order as training)
LABEL_NAMES = ['Andhra Pradesh', 'Gujrat', 'Jharkhand', 'Karnataka', 'Kerala', 'Tamil']


class SimpleClassifier(nn.Module):
    """Classifier for HuBERT embeddings"""
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# Register SimpleClassifier in __main__ namespace for pickle compatibility
sys.modules['__main__'].SimpleClassifier = SimpleClassifier

class HuBERTAccentPredictor:
    """Complete HuBERT-based accent prediction system"""
    
    def __init__(self, layer_results_path, best_layer=None):
        """
        Initialize HuBERT predictor
        
        Args:
            layer_results_path: Path to saved layer_results.pth
            best_layer: Layer index to use (if None, automatically selects best)
        """
        # Load HuBERT model
        model_name = "facebook/hubert-base-ls960"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.hubert_model = HubertModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(DEVICE)
        self.hubert_model.eval()
        
        # Load layer results with weights_only=False for PyTorch 2.6+
        layer_results = torch.load(layer_results_path, map_location=DEVICE, weights_only=False)
        
        # Determine best layer if not specified
        if best_layer is None:
            self.best_layer = max(layer_results.keys(), 
                                  key=lambda l: layer_results[l]['test_acc'])
        else:
            self.best_layer = best_layer
        
        # Load classifier for best layer
        self.classifier = layer_results[self.best_layer]['model']
        self.classifier.eval()
        
        print(f"HuBERT Predictor initialized")
        print(f"Using Layer: {self.best_layer}")
        print(f"Layer Test Accuracy: {layer_results[self.best_layer]['test_acc']:.4f}")
    
    def extract_embedding(self, audio_path):
        """Extract HuBERT embedding from audio file"""
        # Load audio
        wav, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        
        # Prepare input for HuBERT
        inputs = self.feature_extractor(
            wav,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        # Extract embeddings from all layers
        with torch.no_grad():
            outputs = self.hubert_model(**inputs)
            hidden_states = outputs.hidden_states
            
            # Get embedding from best layer
            layer_output = hidden_states[self.best_layer]  # (1, time, hidden_size)
            embedding = layer_output.mean(dim=1).squeeze(0)  # Mean pooling
        
        return embedding
    
    def predict(self, audio_path):
        """
        Predict accent from audio file
        
        Returns:
            predicted_accent: Name of predicted accent
            confidence: Confidence score (0-1)
            all_probs: Probability distribution over all accents
        """
        # Extract embedding
        embedding = self.extract_embedding(audio_path)
        
        # Classify
        with torch.no_grad():
            logits = self.classifier(embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = logits.argmax(dim=1).item()
            confidence = probs[pred_idx].item()
        
        predicted_accent = LABEL_NAMES[pred_idx]
        
        return predicted_accent, confidence, probs.cpu().numpy()

def load_hubert_predictor(model_path, best_layer=None):
    """
    Load HuBERT predictor
    
    Args:
        model_path: Path to hubert_layer_analysis.pth
        best_layer: Specific layer to use (None = auto-select best)
    
    Returns:
        HuBERTAccentPredictor instance
    """
    return HuBERTAccentPredictor(model_path, best_layer)
