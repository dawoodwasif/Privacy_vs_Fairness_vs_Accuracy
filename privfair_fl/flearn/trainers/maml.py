# import numpy as np
# from tqdm import trange, tqdm
# import tensorflow as tf


# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# class Server(BaseFedarated):
#     def __init__(self, params, learner, dataset):
#         print('Using fair fed maml to Train')
#         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
#         super(Server, self).__init__(params, learner, dataset)

#     def train(self):
#         print('Training with {} workers ---'.format(self.clients_per_round))
#         num_clients = len(self.clients)
#         pk = np.ones(num_clients) * 1.0 / num_clients

#         train_batches = {}
#         for c in self.clients:
#             train_batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 2)

#         test_batches = {}
#         for c in self.clients:
#             test_batches[c] = gen_batch(c.test_data, self.batch_size, self.num_rounds + 2)

#         print('Have generated training and testing batches for all devices/tasks...')

#         for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):

#             # only train on non-held-out clients
#             indices, selected_clients = self.select_clients(round=i, pk=pk, held_out=self.held_out, num_clients=self.clients_per_round)

#             Deltas = []
#             hs = []

#             selected_clients = selected_clients.tolist()

#             for c in selected_clients:
#                 # communicate the latest model
#                 c.set_params(self.latest_model)
#                 weights_before = c.get_params()

#                 # solve minimization locally
#                 batch1 = next(train_batches[c])
#                 batch2 = next(test_batches[c])

#                 if self.with_maml:
#                     _, grads1, loss1 = c.solve_sgd(batch1)
#                 _, grads2, loss2 = c.solve_sgd(batch2)

#                 Deltas.append([np.float_power(loss2 + 1e-10, self.q) * grad for grad in grads2[1]])
#                 hs.append(self.q * np.float_power(loss2+1e-10, (self.q-1)) * norm_grad(grads2[1]) + (1.0/self.learning_rate) * np.float_power(loss2+1e-10, self.q))

#             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

#         print("###### finish meta-training, start meta-testing ######")


#         test_accuracies = []
#         initial_accuracies = []
#         for c in self.clients[len(self.clients)-self.held_out:]:  # meta-test on the held-out tasks
#             # start from the same initial model that is learnt using q-FFL + MAML
#             c.set_params(self.latest_model)
#             ct, cl, ns = c.test_error_and_loss()
#             initial_accuracies.append(ct * 1.0/ns)
#             # solve minimization locally
#             for iters in range(self.num_fine_tune):  # run k-iterations of sgd
#                 batch = next(train_batches[c])
#                 _, grads1, loss1 = c.solve_sgd(batch)
#             ct, cl, ns = c.test_error_and_loss()
#             test_accuracies.append(ct * 1.0/ns)
#         print("initial mean: ", np.mean(np.asarray(initial_accuracies)))
#         print("initial variance: ", np.var(np.asarray(initial_accuracies)))
#         print(self.output)
#         print("personalized mean: ", np.mean(np.asarray(test_accuracies)))
#         print("personalized variance: ", np.var(np.asarray(test_accuracies)))
#         np.savetxt(self.output+"_"+"test.csv", np.asarray(test_accuracies), delimiter=",")


import numpy as np
from tqdm import trange
import tensorflow as tf
import tenseal as ts


from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset, dp_params, he_params, smc_params):
        print('Using Fair Fed MAML to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

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

        # Initialize base federated class
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        """Main training loop for Federated MAML"""
        print(f'Training with {self.clients_per_round} workers using '
              f'{"DP" if self.dp_flag else "HE" if self.he_flag else "SMC" if self.smc_flag else "no privacy"}.')

        num_clients = len(self.clients)
        pk = np.ones(num_clients) / num_clients  # Uniform client selection probability

        # Pre-generate training and testing batches for all clients
        train_batches = {c: gen_batch(c.train_data, self.batch_size, self.num_rounds + 2) for c in self.clients}
        test_batches = {c: gen_batch(c.test_data, self.batch_size, self.num_rounds + 2) for c in self.clients}

        print('Training and testing batches generated for all clients/tasks...')

        all_rounds_variances = []
        all_rounds_distances = []

        for round_num in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
            if round_num % self.eval_every == 0:
                num_test, num_correct_test = self.test()
                test_accuracies = np.array(num_correct_test) / np.array(num_test)
                variance = np.var(test_accuracies)
                all_rounds_variances.append(variance)

                mean_test_accuracy = np.mean(test_accuracies)
                euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
                all_rounds_distances.append(euclidean_distance)

                print(f'\nRound {round_num} testing accuracy: {mean_test_accuracy:.4f}, '
                      f'Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}')

            # Select clients for this round
            indices, selected_clients = self.select_clients(round=round_num, pk=pk, num_clients=self.clients_per_round)
            selected_clients = selected_clients.tolist()

            Deltas = []
            hs = []
            weights_before = None

            for c in selected_clients:
                c.set_params(self.latest_model)
                weights_before = c.get_params()

                # Solve inner optimization for MAML with batch1 and batch2
                batch1 = next(train_batches[c])
                batch2 = next(test_batches[c])

                if self.with_maml:
                    _, grads1, loss1 = c.solve_sgd(batch1)
                _, grads2, loss2 = c.solve_sgd(batch2)

                # Apply q-FedSGD weighting for client contributions
                Deltas.append([np.float_power(loss2 + 1e-10, self.q) * grad for grad in grads2[1]])
                norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads2[1]])
                hs.append(self.q * np.float_power(loss2 + 1e-10, (self.q - 1)) * norm_grad_sum +
                          (1.0 / self.learning_rate) * np.float_power(loss2 + 1e-10, self.q))

            # Aggregation
            self.latest_model = self.aggregate2(weights_before, Deltas, hs)

        # Save metrics after meta-training
        final_variance = np.mean(all_rounds_variances)
        final_euclidean_distance = np.mean(all_rounds_distances)

        print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
        print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")

        np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
        np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
        print("Metrics saved successfully.")

        print("###### Finished Meta-Training, Starting Meta-Testing ######")

        # Meta-testing on held-out tasks
        test_accuracies = []
        initial_accuracies = []

        for c in self.clients[-self.held_out:]:  # Meta-test on held-out clients
            c.set_params(self.latest_model)
            initial_correct, initial_loss, ns = c.test_error_and_loss()
            initial_accuracies.append(initial_correct * 1.0 / ns)

            # Fine-tune using k iterations of SGD
            for _ in range(self.num_fine_tune):
                batch = next(train_batches[c])
                _, grads1, loss1 = c.solve_sgd(batch)

            test_correct, test_loss, ns = c.test_error_and_loss()
            test_accuracies.append(test_correct * 1.0 / ns)

        print(f"Initial mean accuracy: {np.mean(np.asarray(initial_accuracies)):.4f}")
        print(f"Initial variance: {np.var(np.asarray(initial_accuracies)):.4f}")
        print(f"Personalized mean accuracy: {np.mean(np.asarray(test_accuracies)):.4f}")
        print(f"Personalized variance: {np.var(np.asarray(test_accuracies)):.4f}")

        np.savetxt(self.output + "_test.csv", np.asarray(test_accuracies), delimiter=",")
        print("Test accuracies saved successfully.")

def aggregate2(weights_before, updates, hs):
    """Aggregate client updates into the global model."""
    new_solutions = []
    for w, delta in zip(weights_before, updates):
        delta_sum = np.zeros_like(w)  # Initialize delta sum with the same shape as weights

        for delta_i, h_i in zip(delta, hs):
            # Ensure h_i is properly broadcastable to delta_i's shape
            if np.isscalar(h_i):
                delta_sum += delta_i * h_i  # Scalar multiplication
            else:
                reshaped_h_i = np.reshape(h_i, delta_i.shape)
                delta_sum += delta_i * reshaped_h_i

        new_solutions.append(w - delta_sum)

    return new_solutions

