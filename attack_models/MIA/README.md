
# Privacy vs Fairness in Federated Learning (MIA Attack Models)

This repository contains experiments and code for running **Membership Inference Attacks (MIA)** on models trained with three privacy-preserving techniques in a federated learning setup: **Differential Privacy (DP)**, **Homomorphic Encryption (HE)**, and **Secure Multi-Party Computation (SMC)**. The main goal is to evaluate the trade-offs between privacy and fairness.

## Getting Started

### Requirements
To run the code, you'll need:

- Python 3.6+
- Required packages can be installed via:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure you have `pytorch` installed for neural network computations.

## MIA Attack Models

You can run the MIA attack models using the following scripts based on the privacy-preserving technique:

- **Differential Privacy (DP):**
  ```bash
  python mia_dp.py
  ```

- **Homomorphic Encryption (HE):**
  ```bash
  python mia_he.py
  ```

- **Secure Multi-Party Computation (SMC):**
  ```bash
  python mia_smc.py
  ```

Each script performs a Membership Inference Attack (MIA) on the models trained with the respective privacy-preserving technique. The results help in understanding the success of attacks under different privacy guarantees.

## Federated Learning Algorithms

The following scripts allow you to run federated learning algorithms, varying different privacy parameters based on the method used:

- **q-FedAvg with Differential Privacy (DP)** (vary `epsilon`):
  ```bash
  python main_qfed_dp_fair.py --epsilon 0.1
  ```

- **q-FedAvg with Homomorphic Encryption (HE)** (vary `poly_modulus_degree`):
  ```bash
  python main_qfed_he_fair.py --poly_modulus_degree 16384
  ```

- **q-FedAvg with Secure Multi-Party Computation (SMC)** (vary `num_shares`):
  ```bash
  python main_qfed_smc_fair.py --num_shares 5
  ```

### Example Commands

For **DP**:
```bash
python main_qfed_dp_fair.py --dataset mnist --model cnn --epochs 50 --epsilon 0.5
```

For **HE**:
```bash
python main_qfed_he_fair.py --dataset mnist --model cnn --epochs 50 --poly_modulus_degree 8192
```

For **SMC**:
```bash
python main_qfed_smc_fair.py --dataset mnist --model cnn --epochs 50 --num_shares 3
```

### Experiment Parameters

You can modify the following parameters for all federated learning runs:

- `--dataset`: Dataset to use (e.g., `mnist`, `cifar10`).
- `--model`: Model type (e.g., `mlp`, `cnn`).
- `--epochs`: Number of training epochs.
- `--gpu`: GPU index (use `0` for the first GPU).

## Results

The results from running MIA attacks or federated learning algorithms will be logged to the `log/` directory. You can analyze the trade-offs between privacy and fairness for each method by reviewing the attack success rates and model accuracy.

## References

If you use this code, please cite the following papers:

```
@article{mcmahan2016communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, H Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and others},
  journal={arXiv preprint arXiv:1602.05629},
  year={2016}
}

@article{Li2020privacy,
  title={Privacy-Preserving Federated Learning Framework Based on Chained Secure Multi-party Computing},
  author={Li, Yong and Zhou, Yipeng and Jolfaei, Alireza and Yu, Dongjin and Xu, Gaochao and Zheng, Xi},
  journal={IEEE Internet of Things Journal},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
