# Empirical Analysis of Privacy-Fairness-Accuracy Trade-offs in Federated Learning  
### A Step Towards Responsible AI  
**Published in AAAI/ACM AIES 2025**

## 🧠 Overview

This repository provides a modular and extensible framework to empirically analyze the trade-offs between **Privacy**, **Fairness**, and **Accuracy** in **Federated Learning (FL)**. We implement and benchmark three key privacy-preserving mechanisms:

- **Differential Privacy (DP)**
- **Homomorphic Encryption (HE)**
- **Secure Multi-Party Computation (SMC)**

This framework is compatible with a variety of fairness-aware optimization algorithms and privacy attack models. It supports both synthetic (MNIST, Fashion-MNIST) and real-world datasets (e.g., Alzheimer's MRI, Credit Card Fraud), and is ideal for researchers and practitioners aiming to evaluate robust, privacy-preserving, and fair FL systems.

---

## 📁 Repository Structure

```
.
├── data/                    # Preprocessed datasets or scripts for downloading them
├── attack_models/          # Code for simulating membership and gradient inference attacks
├── privfair_fl/            # Algorithms for private and fair federated learning
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

---

## ✨ Key Features

- **Privacy-Preserving Techniques**:
  - **DP**: Noise injection via Gaussian/Laplace mechanisms.
  - **HE (via TenSEAL)**: Gradient encryption using the CKKS scheme.
  - **SMC**: Secure computation with secret sharing across clients.

- **Fairness-Enhancing Algorithms**:
  - Implementations of q-FedAvg, Ditto, q-MAML, etc.
  - Client-wise disparity measurement and fairness constraints.

- **Attack Models**:
  - *Membership Inference Attack (MIA)*: To evaluate leakage under DP.
  - *Gradient Inference Attack (GIA)*: To assess representational leakage.

- **Evaluation Metrics**:
  - **Accuracy**: Standard classification metrics.
  - **Privacy**: MIA/GIA success rates.
  - **Fairness**: Average variance and Euclidean distance between client performances.

- **Data Configurations**:
  - IID and non-IID scenarios supported.
  - Supports multiple datasets including MNIST, Fashion-MNIST, Alzheimer’s MRI, and financial fraud detection.

---

## ⚙️ Requirements

- Python ≥ 3.6
- [TenSEAL](https://github.com/OpenMined/TenSEAL) for homomorphic encryption
- PyTorch
- NumPy
- Matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/<your-username>/privfair-fl.git
cd privfair-fl
```

### ▶️ Running Privacy Attacks

```bash
cd attack_models
# Follow specific instructions in README inside this folder
```

### ▶️ Running Private or Fair Federated Learning

```bash
cd privfair_fl
# Select your algorithm and execute corresponding training scripts
```

You can configure your training by editing the relevant config files inside each algorithm folder.

---

## 📊 Citation

If you use this repository or any part of our codebase in your work, **please cite our paper**:

> **Dawood Wasif, Dian Chen, Sindhuja Madabushi, Nithin Alluru, Terrence J Moore, Jin-Hee Cho**  
> *Empirical Analysis of Privacy-Fairness-Accuracy Trade-offs in Federated Learning: A Step Towards Responsible AI*  
> Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025.  

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgment

This research was conducted at the Department of Computer Science, Virginia Tech and supported in part by the U.S. Army Research Laboratory.

---

## 🧩 Final Notes

This repository is intended to help the community understand and optimize the delicate balance between **privacy**, **fairness**, and **performance** in federated systems. Contributions, suggestions, and extensions to additional datasets or algorithms are welcome.

**👉 Please cite our work if you use this repository.**
