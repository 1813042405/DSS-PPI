# DSS-PPI: A Self-Supervised Graph Learning Framework for Protein-Protein Interaction Prediction via Multimodal Sequence Semantics

This repository contains the code for our paper. DSS-PPI is a deep learning-based framework for protein-protein interaction (PPI) prediction. By integrating ProTrek sequence features, ProtT5 sequence features, and GAT graph features, the framework significantly enhances the accuracy of PPI predictions.

## Framework Overview
<img width="1589" height="647" alt="模型图1" src="https://github.com/user-attachments/assets/30b36841-a6b6-46bd-9e9e-36a5ae29846a" />



## Project Structure

* **`main.py`**: The main entry point of the program, responsible for orchestrating the entire pipeline.
* **`trainer.py`**: Contains the core training loop and evaluation logic for PPI prediction.
* **`feature_extraction.py`**: Implements various methods for protein feature extraction.
* **`models.py`**: Defines model architectures, including GAT, DGI, and the classifier.
* **`utils.py`**: Provides utility functions, including data preprocessing and pre-training functionalities.
* **`features/`**: Stores feature records for the human dataset.

## Data Preparation

The system requires the following data files:

* **`human.csv`**: A CSV file containing protein IDs and their corresponding sequences.
* **`human_pos.edgelist`**: A list of positive protein-protein interaction (PPI) edges.
* **`human_neg.edgelist`**: A list of negative protein-protein interaction (PPI) edges.

### CSV File Format
The `human.csv` file must contain two columns:
1.  **First Column**: Protein ID
2.  **Second Column**: Amino acid sequence

| Protein ID | Sequence             |
| :--------- | :------------------- |
| P12345     | MKVLLRLICFIALLISS... |
| Q67890     | MSSLPVPYKLV...       |

### Edgelist Format
Each line in the `.edgelist` files should represent an interaction between two proteins, typically formatted as:
`Protein_A Protein_B`

## Quick Start

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run training:

   ```
   python main.py
   ```

## Model Architecture

The DSS-PPI framework consists of the following key components:

* **ProTrek Feature Extractor**: Extracts deep representations of protein sequences.
* **ProtT5 Feature Extractor**: Performs transformer-based protein sequence feature extraction.
* **GAT (Graph Attention Network)**: Learns graph representations by incorporating sequence similarity scores.
* **DGI Self-Supervised Pre-training**: Utilizes **Deep Graph Infomax (DGI)** for robust unsupervised graph pre-training.
* **Multi-Feature Fusion Classifier**: Integrates diverse feature sets (ProTrek, ProtT5, and GAT) to perform final PPI prediction.

## Contact

For questions, please contact the author.
