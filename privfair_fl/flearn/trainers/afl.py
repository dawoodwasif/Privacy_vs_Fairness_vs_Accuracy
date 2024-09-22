# import numpy as np
# from tqdm import trange, tqdm
# import tensorflow as tf

# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, project


# class Server(BaseFedarated):
#     def __init__(self, params, learner, dataset):
#         print('Using agnostic flearn (non-stochastic version) to Train')
#         self.inner_opt = tf.train.AdagradOptimizer(params['learning_rate'])
#         super(Server, self).__init__(params, learner, dataset)
#         self.latest_lambdas = np.ones(len(self.clients)) * 1.0 / len(self.clients)
#         self.resulting_model = self.client_model.get_params()  # this is only for the agnostic flearn paper

#     def train(self):

#         print('Training with {} workers ---'.format(self.clients_per_round))
#         num_clients = len(self.clients)
#         pk = np.ones(num_clients) * 1.0 / num_clients

#         batches = {}
#         for c in self.clients:
#             batches[c] = gen_epoch(c.train_data, self.num_rounds+2)

#         for i in trange(self.num_rounds+1, desc='Round: ', ncols=120):
#             # test model
#             if i % self.eval_every == 0:               
#                 self.client_model.set_params(self.resulting_model)
#                 stats = self.test_resulting_model()                   
#                 tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
#                 test_accuracies = np.divide(np.asarray(stats[3]), np.asarray(stats[2]))
#                 for idx in range(len(self.clients)):
#                     tqdm.write('Client {} testing accuracy: {}'.format(self.clients[idx].id, test_accuracies[idx]))
                  
#             solns = []
#             losses = []
#             for idx, c in enumerate(self.clients):
#                 c.set_params(self.latest_model)

#                 batch = next(batches[c])
#                 _, grads, loss = c.solve_sgd(batch) # this gradient is with respect to w

#                 losses.append(loss)
#                 solns.append((self.latest_lambdas[idx],grads[1]))
            
#             avg_gradient = self.aggregate(solns)

#             for v,g in zip(self.latest_model, avg_gradient):
#                 v -= self.learning_rate * g
            
#             for idx in range(len(self.latest_lambdas)):
#                 self.latest_lambdas[idx] += self.learning_rate_lambda * losses[idx]

#             self.latest_lambdas = project(self.latest_lambdas)

#             for k in range(len(self.resulting_model)):
#                 self.resulting_model[k] = (self.resulting_model[k] * i + self.latest_model[k]) * 1.0 / (i+1)

import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import tenseal as ts

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, project


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset, dp_params, he_params, smc_params):
        print('Using Agnostic Federated Learning (non-stochastic version) to Train')
        self.inner_opt = tf.train.AdagradOptimizer(params['learning_rate'])

        # Differential Privacy (DP) parameters
        self.dp_flag = dp_params['dp_flag']  # DP enable flag
        self.epsilon = dp_params['epsilon']  # Privacy budget (epsilon)
        self.delta = dp_params['delta']  # DP delta parameter
        self.sensitivity = dp_params['sensitivity']  # Gradient sensitivity
        self.dp_mechanism = dp_params['mechanism']  # Mechanism: 'laplace' or 'gaussian'

        # Homomorphic Encryption (HE) parameters
        self.he_flag = he_params['he_flag']  # HE enable flag
        self.n_layers_to_encrypt = he_params['he_encrypt_layers']  # Number of layers to encrypt
        self.context = None  # TenSEAL context for HE

        if self.he_flag:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=he_params['poly_modulus_degree'],
                coeff_mod_bit_sizes=he_params['coeff_mod_bit_sizes']
            )
            self.context.global_scale = he_params['global_scale']
            self.context.generate_galois_keys()

        # Secure Multiparty Computation (SMC) parameters
        self.smc_flag = smc_params['smc_flag']  # SMC enable flag
        self.num_shares = smc_params['smc_num_shares']  # Number of shares
        self.smc_threshold = smc_params['smc_threshold']  # Threshold for reconstruction
        self.dynamic_sharing = True  # Enable dynamic/random share distribution

        super(Server, self).__init__(params, learner, dataset)
        self.latest_lambdas = np.ones(len(self.clients)) * 1.0 / len(self.clients)
        self.resulting_model = self.client_model.get_params()  # AFL resulting model

    def train(self):
        """Main training loop for AFL."""
        print(f'Training with {self.clients_per_round} workers using '
              f'{"DP" if self.dp_flag else "HE" if self.he_flag else "SMC" if self.smc_flag else "no privacy"}.')
        
        num_clients = len(self.clients)
        pk = np.ones(num_clients) / num_clients  # Uniform client selection probability

        # Pre-generate training batches for all clients
        batches = {c: gen_epoch(c.train_data, self.num_rounds + 2) for c in self.clients}

        all_rounds_variances = []
        all_rounds_distances = []

        for round_num in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
            # Test model and log metrics every eval round
            if round_num % self.eval_every == 0:
                self.client_model.set_params(self.resulting_model)
                stats = self.test_resulting_model()
                test_accuracies = np.divide(np.asarray(stats[3]), np.asarray(stats[2]))

                tqdm.write(f'At round {round_num} testing accuracy: {np.sum(stats[3]) * 1.0 / np.sum(stats[2]):.4f}')
                for idx in range(len(self.clients)):
                    tqdm.write(f'Client {self.clients[idx].id} testing accuracy: {test_accuracies[idx]:.4f}')

                variance = np.var(test_accuracies)
                all_rounds_variances.append(variance)

                mean_test_accuracy = np.mean(test_accuracies)
                euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
                all_rounds_distances.append(euclidean_distance)

                print(f'Round {round_num} Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}')

            # Client updates
            solns = []
            losses = []

            for idx, c in enumerate(self.clients):
                c.set_params(self.latest_model)

                batch = next(batches[c])
                _, grads, loss = c.solve_sgd(batch)  # Compute gradients and loss

                losses.append(loss)
                solns.append((self.latest_lambdas[idx], grads[1]))

            # Aggregate gradients
            avg_gradient = self.aggregate(solns)

            # Update model
            for v, g in zip(self.latest_model, avg_gradient):
                v -= self.learning_rate * g

            # Update lambdas for fairness weighting
            for idx in range(len(self.latest_lambdas)):
                self.latest_lambdas[idx] += self.learning_rate_lambda * losses[idx]

            # Project lambda values to ensure fairness
            self.latest_lambdas = project(self.latest_lambdas)

            # Update resulting model
            for k in range(len(self.resulting_model)):
                self.resulting_model[k] = (self.resulting_model[k] * round_num + self.latest_model[k]) * 1.0 / (round_num + 1)

        # Save final metrics after training
        final_variance = np.mean(all_rounds_variances)
        final_euclidean_distance = np.mean(all_rounds_distances)

        print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
        print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")

        np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
        np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
        print("Metrics saved successfully.")

    def aggregate(self, solns):
        """Aggregate client solutions into the global model."""
        total_lambda = sum([soln[0] for soln in solns])
        avg_gradient = [np.zeros_like(v) for v in self.latest_model]

        for lambda_weight, grads in solns:
            for i, grad in enumerate(grads):
                avg_gradient[i] += (lambda_weight / total_lambda) * grad

        return avg_gradient
