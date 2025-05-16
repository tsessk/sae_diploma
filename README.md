# Sparse Autoencoder Analysis of Fine-tuned Language Models

This repository contains the code for my bachelor's diploma project on analyzing internal representations of language models using Sparse Autoencoders (SAE).

## Project Overview

The goal of this project is to investigate the differences between a base language model and its fine-tuned version by:

1. Performing Supervised Fine-Tuning (SFT) on a language model for a specific task
2. Using Sparse Autoencoders (SAE) to inspect and analyze the internal weights
3. Comparing the internal representations between the original and fine-tuned models

For a detailed explanation of the research methodology, please see the [Methodology](METHODOLOGY.md) document.

## Repository Structure

```
sae_diploma/
├── src/                     # Source code
│   ├── models/              # Model implementations
│   │   └── sae.py           # Sparse Autoencoder implementation
│   ├── data_processing/     # Data preprocessing utilities
│   │   └── extract_activations.py  # Script to extract model activations
│   ├── training/            # Training scripts
│   │   ├── run_sft.py       # Script for Supervised Fine-Tuning
│   │   └── train_sae.py     # Script for training SAE models
│   ├── analysis/            # Analysis tools
│   │   └── compare_models.py  # Script for model comparison
│   └── utils/               # Utility functions
│       └── activation_utils.py  # Utilities for handling activations
├── notebooks/               # Jupyter notebooks for experiments
├── data/                    # Data directory (created on demand)
├── models/                  # Saved model checkpoints (created on demand)
├── figures/                 # Visualizations and diagrams
├── requirements.txt         # Project dependencies
├── METHODOLOGY.md           # Detailed research methodology
└── README.md                # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sae_diploma.git
cd sae_diploma

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Extract Activations

To extract activations from a model:

```bash
python src/data_processing/extract_activations.py \
    --model_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "wikimedia/wikipedia" \
    --text_column "text" \
    --layer_idx -1 \
    --num_samples 1000 \
    --output_dir "./data/activations"
```

### 2. Supervised Fine-Tuning (SFT)

To fine-tune a model:

```bash
python src/training/run_sft.py \
    --model_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "cnn_dailymail" \
    --text_column "article" \
    --target_column "highlights" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --output_dir "./models/sft_model" \
    --use_chat_format
```

### 3. Train Sparse Autoencoder

To train an SAE on model activations:

```bash
python src/training/train_sae.py \
    --activations_path "./data/activations/SmolLM2-1.7B_layer-1_activations.pt" \
    --architecture "topk" \
    --d_sae 4096 \
    --l1_coefficient 1e-3 \
    --k 100 \
    --num_epochs 20 \
    --output_dir "./models/sae_model"
```

### 4. Compare Models

To compare the original and fine-tuned models using SAE:

```bash
python src/analysis/compare_models.py \
    --base_model "HuggingFaceTB/SmolLM2-1.7B" \
    --ft_model "your-username/SmolLM2-FT-Summarization" \
    --base_sae_path "./models/sae_model/sae_final_model.pt" \
    --output_dir "./analysis_results"
```

## Experiment Notebooks

Detailed experiments and analysis can be found in the notebooks directory:

- `01_data_preparation.ipynb` - Preparing datasets for training
- `02_supervised_fine_tuning.ipynb` - Fine-tuning the base model
- `03_sae_training.ipynb` - Training Sparse Autoencoders
- `04_model_comparison.ipynb` - Comparing the internal representations
- `05_feature_visualization.ipynb` - Visualizing learned features

## Model Architecture

The Sparse Autoencoder (SAE) consists of:

- **Encoder**: Linear projection from input dimension to SAE features
- **Activation**: ReLU or TopK activation for sparsity
- **Decoder**: Linear projection back to input dimension

The architecture is designed to learn a sparse, over-complete representation of neural network activations.

## Results

The analysis shows how the internal representations of language models change after fine-tuning. Key findings include:

- Different activation patterns between base and fine-tuned models
- New features emerging in the fine-tuned model
- Specific neurons that change their behavior after fine-tuning

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for providing models and libraries
- The research community for development of Sparse Autoencoders 