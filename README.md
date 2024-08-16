# Privacy_vs_Fairness: A Federated Learning Framework for Evaluating the Relationship of Privacy Preservation and Fairness

## Overview

This repository provides a robust and flexible framework for evaluating the trade-offs between **Privacy Preservation (PP)** and **Fairness** in Federated Learning (FL). It implements three primary privacy-preserving techniques:
- **Differential Privacy (DP)**
- **Homomorphic Encryption (HE)**
- **Secure Multi-Party Computation (MPC)**

The framework is equipped with various fairness-enhancing methods and privacy attack models, allowing researchers and developers to conduct comprehensive experiments to understand the balance between privacy, fairness, and performance in machine learning models. The experiments are primarily conducted using the **MNIST** dataset, but the framework is extendable to other datasets as well.

## Repository Structure

The repository is organized as follows:

.
├── data/                             # Contains dataset files
├── Extra/                            # Additional resources, scripts, or experimental files
├── Fairness_Enhancing_Methods/        # Algorithms that improve fairness in federated learning
├── Privacy_Attack_Models/             # Implementations of attack models such as Membership Inference Attacks (MIA) and Gradient Inversion Attacks (GIA)
├── Privacy_Preservation_Techniques/   # Implementations of Differential Privacy, Homomorphic Encryption, and MPC
├── README.md                         # Project overview and instructions
├── requirements.txt                  # Python dependencies



## Key Features

- **Modular and Extendable Framework**: Easily test and modify different privacy-preserving techniques (DP, HE, MPC) and fairness algorithms.
- **Evaluation Metrics**:
  - **Fairness**: Measures such as *Average Variance* and *Euclidean Distance* to quantify fairness across clients.
  - **Privacy**: Metrics such as *MIA Success Rate* and *GIA Success Rate* to evaluate privacy leakage.
  - **Performance**: Standard metrics like *Classification Accuracy* to assess overall model effectiveness.
- **Privacy-Preserving Techniques**: 
  - **DP**: Adds noise to gradients to ensure privacy.
  - **HE (via TenSEAL)**: Encrypts gradients using CKKS for secure computation.
  - **MPC**: Securely splits the data among clients to prevent centralized exposure.

## Requirements

The project uses Python 3.6+ and requires several libraries such as `tenseal`, `torch`, and `numpy`. To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage
## Contributions
## Liscence
