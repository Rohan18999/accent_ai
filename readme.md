# Indian Accent Classification using MFCC-CNN and HuBERT

## Project Overview

This project implements two approaches for classifying Indian accents:
1. **MFCC-CNN**: Traditional feature extraction with Convolutional Neural Network
2. **HuBERT**: Transformer-based pre-trained model with layer-wise analysis

**Dataset**: [IndicAccentDb on Hugging Face](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Dataset Information](#dataset-information)
- [File Structure](#file-structure)
- [Step-by-Step Execution](#step-by-step-execution)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python Version**: Python 3.8 or higher (tested on Python 3.8-3.10)
- **RAM**: Minimum 8GB (16GB recommended for HuBERT)
- **GPU**: CUDA-compatible GPU recommended (optional but speeds up processing)
- **Disk Space**: At least 5GB free space for dataset and models

### Required Accounts
- **Hugging Face Account** (optional but recommended): [Sign up here](https://huggingface.co/join)

---

## Installation Guide

### Step 1: Set Up Python Environment

It's recommended to use a virtual environment to avoid package conflicts.

**Option A: Using venv**
```bash
# Create virtual environment
python -m venv accent_classification_env

# Activate virtual environment
# On Windows:
accent_classification_env\Scripts\activate
# On macOS/Linux:
source accent_classification_env/bin/activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n accent_classification python=3.9
conda activate accent_classification
```

### Step 2: Install Core Dependencies

Install PyTorch first (visit [PyTorch Official Site](https://pytorch.org/get-started/locally/) for system-specific installation):

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### Step 3: Install Required Packages

Run the following commands in order:

```bash
# Audio processing libraries
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install python_speech_features==0.6

# Machine learning libraries
pip install scikit-learn==1.3.2
pip install numpy==1.24.3

# Dataset and transformers
pip install datasets==3.6.0
pip install transformers==4.35.2

# Visualization
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Progress bars
pip install tqdm==4.66.1
```

### Step 4: Verify Installation

To verify all packages are installed correctly, open a Python interpreter or notebook and run:

```python
import torch
import librosa
import datasets
import transformers
from python_speech_features import mfcc
import sklearn

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ Librosa version:", librosa.__version__)
print("✓ Datasets version:", datasets.__version__)
print("✓ Transformers version:", transformers.__version__)
print("✓ Scikit-learn version:", sklearn.__version__)
print("\nAll packages installed successfully!")
```

If all imports work without errors, you're ready to proceed!

---

## Dataset Information

**Dataset Name**: IndicAccentDb  
**Hugging Face Link**: https://huggingface.co/datasets/DarshanaS/IndicAccentDb  
**Description**: Collection of Indian English speech samples with accent labels  
**Number of Classes**: 6 accent categories  
**Audio Format**: WAV files, various sampling rates (automatically handled)

The dataset will be automatically downloaded when you run the code for the first time.

---

## File Structure

Your project directory should look like this:

```
accent_classification/
│
├── mfcc_cnn.pth                # Pre-trained MFCC-CNN model (you need this)
├── hubert_layer_analysis.pth   # Generated after running HuBERT script
└── README.md                   # This file
```

---

## Step-by-Step Execution

### Part 1: MFCC-CNN Model Evaluation

This approach uses traditional audio features (MFCCs) with a CNN classifier.

#### Step 1.1: Ensure Model File Exists

**Important**: You need the pre-trained model file `mfcc_cnn.pth` in the same directory. This file should contain the trained model weights.

If you don't have this file, you'll need to train the model first. Contact the project maintainer or train it yourself.

#### Step 1.2: Run MFCC Evaluation

Execute the MFCC evaluation code provided in your notebook or script.

**Expected Runtime**: 5-15 minutes depending on hardware

#### Step 1.3: View Results

The script will output:
- Test accuracy percentage
- Detailed classification report (precision, recall, F1-score)
- Confusion matrix visualization

---

### Part 2: HuBERT Layer-wise Analysis

This approach uses a pre-trained transformer model to analyze which layers best capture accent information.

#### Step 2.1: Run HuBERT Analysis

Execute the HuBERT layer-wise analysis code provided in your notebook or script.

**Expected Runtime**: 
- With GPU: 30-60 minutes
- Without GPU: 2-4 hours

#### Step 2.2: Monitor Progress

The script shows progress bars for:
- Embedding extraction (train/val/test sets)
- Classifier training for each layer

You'll see output like:
```
Extracting embeddings from train set...
100%|████████████| 125/125 [05:23<00:00,  2.59s/it]

Training classifier for Layer 0...
  Layer 0: Val Acc = 0.7234, Test Acc = 0.7156

Training classifier for Layer 1...
  Layer 1: Val Acc = 0.7456, Test Acc = 0.7389
...
```

#### Step 2.3: View Results

After completion, you'll see:
- Layer-wise performance plot
- Best layer identification
- Detailed classification report for the best layer

---

## Expected Outputs

### MFCC-CNN Outputs

**Console Output:**
```
Device: cuda
Labels: ['accent1', 'accent2', 'accent3', 'accent4', 'accent5', 'accent6']
Test set size: XXX
Model loaded successfully!
Test dataset ready!
Evaluating model...

==================================================
TEST ACCURACY: 0.XXXX (XX.XX%)
==================================================

Detailed Classification Report:
              precision    recall  f1-score   support
   accent1       0.XX      0.XX      0.XX       XX
   accent2       0.XX      0.XX      0.XX       XX
   ...
```

**Visual Output:**
- Confusion matrix heatmap showing prediction patterns

### HuBERT Outputs

**Console Output:**
```
Device: cuda
Labels: ['accent1', 'accent2', 'accent3', 'accent4', 'accent5', 'accent6']
Number of classes: 6
Train/Val/Test sizes: XXX XXX XXX
HuBERT model loaded: facebook/hubert-base-ls960
Number of layers: 12
Hidden size: 768

============================================================
LAYER-WISE ANALYSIS - Training classifiers for each layer
============================================================

Training classifier for Layer 0...
  Layer 0: Val Acc = 0.XXXX, Test Acc = 0.XXXX
...

============================================================
BEST LAYER FOR ACCENT CLASSIFICATION: Layer X
Test Accuracy: 0.XXXX (XX.XX%)
============================================================
```

**Visual Output:**
- Layer-wise performance comparison plot
- Shows validation and test accuracy across all layers

**Saved File:**
- `hubert_layer_analysis.pth`: Contains all layer results, models, and predictions

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in the code
- For MFCC: Change `batch_size=32` to `batch_size=16` or `batch_size=8`
- For HuBERT: Change `batch_size=8` to `batch_size=4` or `batch_size=2`

### Issue 2: Dataset Download Fails

**Error**: Connection timeout or HTTP errors

**Solution**:
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/large/disk

# Or manually download dataset
from datasets import load_dataset
dataset = load_dataset("DarshanaS/IndicAccentDb", cache_dir="/your/path")
```

### Issue 3: Model File Not Found

**Error**: `FileNotFoundError: mfcc_cnn.pth`

**Solution**:
- Ensure `mfcc_cnn.pth` is in the same directory as your script
- Check file name spelling and case sensitivity
- If you don't have the file, you need to train the model first

### Issue 4: Import Errors

**Error**: `ModuleNotFoundError: No module named 'XXX'`

**Solution**:
```bash
# Reinstall specific package
pip install --upgrade package_name

# Or reinstall all packages
pip install -r requirements.txt --force-reinstall
```

### Issue 5: Slow Execution Without GPU

**Solution**:
- Reduce dataset size for testing: `test_hf = test_hf.select(range(100))`
- Use smaller batch sizes
- Consider using Google Colab with free GPU: https://colab.research.google.com/

---

## Package Versions Summary

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8-3.10 | Base language |
| torch | ≥1.13.0 | Deep learning framework |
| librosa | 0.10.1 | Audio processing |
| soundfile | 0.12.1 | Audio I/O |
| python_speech_features | 0.6 | MFCC extraction |
| scikit-learn | 1.3.2 | Metrics and evaluation |
| datasets | 3.6.0 | Dataset loading |
| transformers | 4.35.2 | HuBERT model |
| matplotlib | 3.8.2 | Visualization |
| seaborn | 0.13.0 | Enhanced plots |
| numpy | 1.24.3 | Numerical operations |
| tqdm | 4.66.1 | Progress bars |

---

## Useful Links

- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets/
- **IndicAccentDb Dataset**: https://huggingface.co/datasets/DarshanaS/IndicAccentDb
- **HuBERT Model**: https://huggingface.co/facebook/hubert-base-ls960
- **Librosa Documentation**: https://librosa.org/doc/latest/index.html
- **Transformers Documentation**: https://huggingface.co/docs/transformers/

---

## Additional Notes

### Using Google Colab

If you don't have a GPU locally, you can use Google Colab:

1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → GPU → Save
4. Install packages using `!pip install` commands
5. Copy and paste the code
6. Run cells sequentially

### Memory Management Tips

- Close other applications when running scripts
- Clear Python kernel between runs: `Kernel → Restart`
- Monitor GPU memory: `nvidia-smi` (if using NVIDIA GPU)

### Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all packages are correctly installed
3. Ensure you're using compatible Python and package versions
4. Check dataset availability on Hugging Face

---

## Citation

If you use this code, please cite:

```bibtex
@dataset{indicaccentdb,
  title={IndicAccentDb},
  author={Darshana S},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/DarshanaS/IndicAccentDb}
}
```

---

**Last Updated**: November 2024  
**Maintainer**: [Your Name/Contact]  
**License**: [Specify License]