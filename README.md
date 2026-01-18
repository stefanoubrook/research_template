# MNIST Research Lab: Singular Learning Theory & Interpretability

This repository serves as a research environment for investigating the geometric properties of neural networks during training, specifically focusing on Singular Learning Theory (SLT) and Mechanistic Interpretability

## Research Context
This project explores the transition of neural networks through the lens of the Real Log Canonical Threshold (RLCT). By tracking the Local Learning Coefficient (LLC), I monitor how the effective dimension of the model shifts as it converges on the MNIST manifold.

## Technical Stack
- **Architecture**: 2-layer MLP with customizable hidden dimensions
- **Optimizer**: `AdamW` with tuned weight decay to encourage the discovery of "flat" local minima.
- **Environment**: `uv` for reproducible, high-performance dependency management.
- **Tensor Logic**: `einops` for verifiable, readable tensor rearrangements, replacing standard `view()` calls to improve mechanistic transparency.

## Interpretability & SLT Metrics
- **LLC Estimation**: Implements a weight-perturbation estimator to track the local free energy of the model.
- **Weight Visualizations**: Utilizes a diverging `RdBu` colormap to isolate inhibitory and excitatory feature detectors.
- **Observations**: 
  - Identified **Dead Neurons** (pure white filters) as signatures of overparameterization.
  - Observed **Phase Transitions** (transient spikes in LLC) during the resolution of complex data clusters.

## Experiment Tracking
All runs are logged to **Weights & Biases (W&B)**, synchronizing loss, accuracy, and geometric complexity on a single temporal axis for rapid analysis of phase transitions.

## Observations
- Observed a final LLC of ~9 for 128-neuron width vs ~6 for 64-neuron width.
- This empirical evidence supports the theory of simplicity bias and functional compression in constrained architectures.