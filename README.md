# Bio_ClinicalBERT-fine-tuned

This model is a fine-tuned version of emilyalsentzer/Bio_ClinicalBERT on an duxprajapati/symptom-disease-dataset. 
It achieves the following results on the evaluation set:

Loss: 0.1920
Precision: 0.7879
Recall: 0.8752
F1: 0.8292
Accuracy: 0.9456
Model description
More information needed

Intended uses & limitations
More information needed

Training and evaluation data
More information needed

Training procedure
Training hyperparameters
The following hyperparameters were used during training:

Training Configuration
Model: emilyalsentzer/Bio_ClinicalBERT
Learning Rate: 2e-5
Train Batch Size: 8
Eval Batch Size: 8
Seed: 42
Optimizer: Used AdamW (from PyTorch) with:
Betas: (0.9, 0.999)
Epsilon: 1e-08
Additional Optimizer Arguments: None
Learning Rate Scheduler Type: linear
Number of Epochs: 3
Weight Decay: 0.01
Evaluation Strategy: epoch
Max Sequence Length: 512
Gradient Accumulation Steps: 1
Mixed Precision: Disabled (since running on CPU or MPS)
