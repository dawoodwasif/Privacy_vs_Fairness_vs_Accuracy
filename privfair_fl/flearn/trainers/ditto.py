# import numpy as np
# from tqdm import trange, tqdm
# import tensorflow as tf
# import copy

# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba
# from flearn.utils.language_utils import letter_to_vec, word_to_indices


# def process_x(raw_x_batch):
#     x_batch = [word_to_indices(word) for word in raw_x_batch]
#     x_batch = np.array(x_batch)
#     return x_batch

# def process_y(raw_y_batch):
#     y_batch = [letter_to_vec(c) for c in raw_y_batch]
#     return y_batch


# class Server(BaseFedarated):
#     def __init__(self, params, learner, dataset):
#         print('Using global-regularized multi-task learning to Train')
#         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
#         super(Server, self).__init__(params, learner, dataset)

#     def train(self):
#         print('---{} workers per communication round---'.format(self.clients_per_round))

#         np.random.seed(1234567+self.seed)
#         corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
#         print(corrupt_id)

#         if self.dataset == 'shakespeare':
#             for c in self.clients:
#                 c.train_data['y'], c.train_data['x'] = process_y(c.train_data['y']), process_x(c.train_data['x'])
#                 c.test_data['y'], c.test_data['x'] = process_y(c.test_data['y']), process_x(c.test_data['x'])

#         batches = {}
#         for idx, c in enumerate(self.clients):
#             if idx in corrupt_id:
#                 c.train_data['y'] = np.asarray(c.train_data['y'])
#                 if self.dataset == 'celeba':
#                     c.train_data['y'] = 1 - c.train_data['y']
#                 elif self.dataset == 'femnist':
#                     c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
#                 elif self.dataset == 'shakespeare':
#                     c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
#                 elif self.dataset == "vehicle":
#                     c.train_data['y'] = c.train_data['y'] * -1
#                 elif self.dataset == "fmnist":
#                     c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

#             if self.dataset == 'celeba':
#                 # due to a different data storage format
#                 batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters)
#             else:
#                 batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)


#         for i in range(self.num_rounds + 1):
#             if i % self.eval_every == 0 and i > 0:
#                 tmp_models = []
#                 for idx in range(len(self.clients)):
#                     tmp_models.append(self.local_models[idx])

#                 num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
#                 avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
#                 num_test, num_correct_test, _ = self.test(tmp_models)
#                 tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
#                 tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
#                 non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
#                 tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
#                 tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
#                 print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))


#             # weighted sampling
#             indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

#             csolns = []
#             losses = []

#             for idx in indices:
#                 w_global_idx = copy.deepcopy(self.global_model)
#                 c = self.clients[idx]
#                 for _ in range(self.local_iters):
#                     data_batch = next(batches[c])

#                     # local
#                     self.client_model.set_params(self.local_models[idx])
#                     _, grads, _ = c.solve_sgd(data_batch)  


#                     if self.dynamic_lam:

#                         model_tmp = copy.deepcopy(self.local_models[idx])
#                         model_best = copy.deepcopy(self.local_models[idx])
#                         tmp_loss = 10000
#                         # pick a lambda locally based on validation data
#                         for lam_id, candidate_lam in enumerate([0.1, 1, 2]):
#                             for layer in range(len(grads[1])):
#                                 eff_grad = grads[1][layer] + candidate_lam * (self.local_models[idx][layer] - self.global_model[layer])
#                                 model_tmp[layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

#                             c.set_params(model_tmp)
#                             l = c.get_val_loss()
#                             if l < tmp_loss:
#                                 tmp_loss = l
#                                 model_best = copy.deepcopy(model_tmp)

#                         self.local_models[idx] = copy.deepcopy(model_best)

#                     else:
#                         for layer in range(len(grads[1])):
#                             eff_grad = grads[1][layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
#                             self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

#                     # global
#                     self.client_model.set_params(w_global_idx)
#                     loss = c.get_loss() 
#                     losses.append(loss)
#                     _, grads, _ = c.solve_sgd(data_batch)
#                     w_global_idx = self.client_model.get_params()


#                 # get the difference (global model updates)
#                 diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]


#                 # send the malicious updates
#                 if idx in corrupt_id:
#                     if self.boosting:
#                         # scale malicious updates
#                         diff = [self.clients_per_round * u for u in diff]
#                     elif self.random_updates:
#                         # send random updates
#                         stdev_ = get_stdev(diff)
#                         diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

#                 if self.q == 0:
#                     csolns.append(diff)
#                 else:
#                     csolns.append((np.exp(self.q * loss), diff))

#             if self.q != 0:
#                 avg_updates = self.aggregate(csolns)
#             else:
#                 if self.gradient_clipping:
#                     csolns = l2_clip(csolns)

#                 expected_num_mali = int(self.clients_per_round * self.num_corrupted / len(self.clients))

#                 if self.median:
#                     avg_updates = self.median_average(csolns)
#                 elif self.k_norm:
#                     avg_updates = self.k_norm_average(self.clients_per_round - expected_num_mali, csolns)
#                 elif self.krum:
#                     avg_updates = self.krum_average(self.clients_per_round - expected_num_mali - 2, csolns)
#                 elif self.mkrum:
#                     m = self.clients_per_round - expected_num_mali
#                     avg_updates = self.mkrum_average(self.clients_per_round - expected_num_mali - 2, m, csolns)
#                 else:
#                     avg_updates = self.simple_average(csolns)

#             # update the global model
#             for layer in range(len(avg_updates)):
#                 self.global_model[layer] += avg_updates[layer]

# import tenseal as ts

# import numpy as np
# from tqdm import trange, tqdm
# import tensorflow.compat.v1 as tf
# import copy

# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch
# from flearn.utils.language_utils import letter_to_vec, word_to_indices


# # DP function to add noise
# def add_dp_noise(gradient, epsilon, delta, sensitivity, mechanism, flag):
#     if not flag:
#         return gradient
    
#     if mechanism == 'laplace':
#         noise_scale = sensitivity / epsilon
#         noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
#     elif mechanism == 'gaussian':
#         noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
#         noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
#     else:
#         raise ValueError("Unsupported DP mechanism: {}".format(mechanism))

#     return gradient + noise

# # Homomorphic Encryption functions
# def apply_he(gradient, context, flag):
#     if flag and context is not None:
#         flattened_grad = gradient.flatten()
#         return ts.ckks_vector(context, flattened_grad)
#     return gradient

# def decrypt_he(gradient, original_shape, flag):
#     if flag:
#         decrypted_grad = np.array(gradient.decrypt())
#         return decrypted_grad.reshape(original_shape)
#     return gradient

# # SMC functions
# def apply_smc(gradient, num_shares, flag):
#     if flag:
#         shares = [np.random.random(gradient.shape) for _ in range(num_shares - 1)]
#         final_share = gradient - sum(shares)
#         shares.append(final_share)
#         return shares
#     return gradient

# def reconstruct_smc(shares, flag):
#     if flag:
#         return sum(shares)
#     return shares

# class Server(BaseFedarated):
#     def __init__(self, params, learner, dataset, dp_params, he_params, smc_params):
#         print('Using Ditto with DP, HE, and SMC')
#         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

#         # DP parameters
#         self.epsilon = dp_params['epsilon']
#         self.delta = dp_params['delta']
#         self.sensitivity = dp_params['sensitivity']
#         self.mechanism = dp_params['mechanism']
#         self.dp_flag = dp_params['dp_flag']

#         # HE parameters
#         self.he_flag = he_params['he_flag']
#         self.n_layers_to_encrypt = he_params['he_encrypt_layers']
#         self.context = None

#         if self.he_flag:
#             self.context = ts.context(
#                 ts.SCHEME_TYPE.CKKS, 
#                 poly_modulus_degree=he_params['poly_modulus_degree'], 
#                 coeff_mod_bit_sizes=he_params['coeff_mod_bit_sizes']
#             )
#             self.context.global_scale = he_params['global_scale']
#             self.context.generate_galois_keys()
#             print("HE Context Initialized")

#         # SMC parameters
#         self.smc_flag = smc_params['smc_flag']
#         self.num_shares = smc_params['smc_num_shares']

#         # Initialize the federated learning base class
#         super(Server, self).__init__(params, learner, dataset)

#         # Metrics tracking
#         self.all_rounds_variances = []
#         self.all_rounds_distances = []
#         self.all_rounds_accuracies = []

#     def train(self):
#         print('---{} workers per communication round---'.format(self.clients_per_round))

#         np.random.seed(1234567 + self.seed)
#         corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
#         print(corrupt_id)

#         batches = {}
#         for idx, c in enumerate(self.clients):
#             if idx in corrupt_id:
#                 c.train_data['y'] = np.asarray(c.train_data['y'])
#                 if self.dataset == 'celeba':
#                     c.train_data['y'] = 1 - c.train_data['y']
#                 elif self.dataset == 'femnist':
#                     c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
#                 elif self.dataset == 'shakespeare':
#                     c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
#                 elif self.dataset == "vehicle":
#                     c.train_data['y'] = c.train_data['y'] * -1
#                 elif self.dataset == "fmnist":
#                     c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

#             batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)

#         for i in range(self.num_rounds + 1):
#             if i % self.eval_every == 0 and i > 0:
#                 tmp_models = []
#                 for idx in range(len(self.clients)):
#                     tmp_models.append(self.local_models[idx])

#                 num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
#                 avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
#                 num_test, num_correct_test, _ = self.test(tmp_models)
#                 test_accuracies = np.array(num_correct_test) / np.array(num_test)

#                 # Calculate variance, Euclidean distance, and mean accuracy
#                 variance = np.var(test_accuracies)
#                 mean_test_accuracy = np.mean(test_accuracies)
#                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)

#                 # Append metrics for the round
#                 self.all_rounds_variances.append(variance)
#                 self.all_rounds_distances.append(euclidean_distance)
#                 self.all_rounds_accuracies.append(mean_test_accuracy)

#                 tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
#                 tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
#                 tqdm.write(f"Round {i} Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}")
#                 non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
#                 tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
#                 tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
#                 print("Variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))

#             indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

#             csolns = []
#             losses = []

#             for idx in indices:
#                 w_global_idx = copy.deepcopy(self.global_model)
#                 c = self.clients[idx]
#                 for _ in range(self.local_iters):
#                     data_batch = next(batches[c])

#                     self.client_model.set_params(self.local_models[idx])
#                     _, grads, _ = c.solve_sgd(data_batch)

#                     # Apply HE, DP, and SMC to gradients
#                     grad_shapes = [grad.shape for grad in grads[1]]
#                     grads_encrypted = [apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, grad in enumerate(grads[1])]
#                     grads_decrypted = [decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))]
#                     grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag) for grad in grads_decrypted]
#                     grads_smc = [apply_smc(grad, self.num_shares, self.smc_flag) for grad in grads_with_noise]
#                     grads_reconstructed = [reconstruct_smc(grad, self.smc_flag) for grad in grads_smc]

#                     for layer in range(len(grads_reconstructed)):
#                         eff_grad = grads_reconstructed[layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
#                         self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

#                     # Global model update
#                     self.client_model.set_params(w_global_idx)
#                     loss = c.get_loss()
#                     losses.append(loss)

#                 diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

#                 if idx in corrupt_id:
#                     if self.boosting:
#                         diff = [self.clients_per_round * u for u in diff]
#                     elif self.random_updates:
#                         stdev_ = get_stdev(diff)
#                         diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

#                 csolns.append(diff)

#             # Aggregation
#             avg_updates = self.aggregate(csolns)

#             for layer in range(len(avg_updates)):
#                 self.global_model[layer] += avg_updates[layer]

#         # Final metrics
#         final_variance = np.mean(self.all_rounds_variances)
#         final_euclidean_distance = np.mean(self.all_rounds_distances)

#         print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
#         print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")

#         # Save the metrics to CSV
#         np.savetxt(self.output + "_final_variances.csv", np.array(self.all_rounds_variances), delimiter=",")
#         np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(self.all_rounds_distances), delimiter=",")
#         np.savetxt(self.output + "_final_accuracies.csv", np.array(self.all_rounds_accuracies), delimiter=",")

#         print("Metrics saved successfully.")


import tenseal as ts
import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
import copy

tf.disable_eager_execution()


# Assuming these modules are properly defined elsewhere in your project.
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch
from flearn.utils.language_utils import letter_to_vec, word_to_indices

# DP function to add noise
def add_dp_noise(gradient, epsilon, delta, sensitivity, mechanism, flag):
    if not flag:
        return gradient
    
    if mechanism == 'laplace':
        noise_scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
    elif mechanism == 'gaussian':
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
    else:
        raise ValueError("Unsupported DP mechanism: {}".format(mechanism))

    return gradient + noise

# Homomorphic Encryption functions
def apply_he(gradient, context, flag):
    if flag and context is not None:
        flattened_grad = gradient.flatten()
        return ts.ckks_vector(context, flattened_grad)
    return gradient

def decrypt_he(gradient, original_shape, flag):
    if flag:
        decrypted_grad = np.array(gradient.decrypt())
        return decrypted_grad.reshape(original_shape)
    return gradient

# SMC functions
def apply_smc(gradient, num_shares, flag):
    if flag:
        shares = [np.random.random(gradient.shape) for _ in range(num_shares - 1)]
        final_share = gradient - sum(shares)
        shares.append(final_share)
        return shares
    return gradient

def reconstruct_smc(shares, flag):
    if flag:
        return sum(shares)
    return shares

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset, dp_params, he_params, smc_params):
        print('Using Ditto with DP, HE, and SMC')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        # DP parameters
        self.epsilon = dp_params['epsilon']
        self.delta = dp_params['delta']
        self.sensitivity = dp_params['sensitivity']
        self.mechanism = dp_params['mechanism']
        self.dp_flag = dp_params['dp_flag']

        # HE parameters
        self.he_flag = he_params['he_flag']
        self.n_layers_to_encrypt = he_params['he_encrypt_layers']
        self.context = None

        if self.he_flag:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS, 
                poly_modulus_degree=he_params['poly_modulus_degree'], 
                coeff_mod_bit_sizes=he_params['coeff_mod_bit_sizes']
            )
            self.context.global_scale = he_params['global_scale']
            self.context.generate_galois_keys()
            print("HE Context Initialized")

        # SMC parameters
        self.smc_flag = smc_params['smc_flag']
        self.num_shares = smc_params['smc_num_shares']

        # Initialize the federated learning base class
        super(Server, self).__init__(params, learner, dataset)

        # Metrics tracking
        self.all_rounds_variances = []
        self.all_rounds_distances = []
        self.all_rounds_accuracies = []
        self.all_rounds_disparity_losses = []  # Add this line

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        np.random.seed(1234567 + self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print(corrupt_id)

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
                elif self.dataset == 'shakespeare':
                    c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
                elif self.dataset == "vehicle":
                    c.train_data['y'] = c.train_data['y'] * -1
                elif self.dataset == "fmnist":
                    c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

            batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                tmp_models = []
                for idx in range(len(self.clients)):
                    tmp_models.append(self.local_models[idx])

                num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
                avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
                num_test, num_correct_test, _ = self.test(tmp_models)
                test_accuracies = np.array(num_correct_test) / np.array(num_test)

                # Calculate variance, Euclidean distance, mean accuracy, and disparity loss
                variance = np.var(test_accuracies)
                mean_test_accuracy = np.mean(test_accuracies)
                euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
                disparity_loss = np.max(test_accuracies) - np.min(test_accuracies)  # Example of disparity loss

                # Append metrics for the round
                self.all_rounds_variances.append(variance)
                self.all_rounds_distances.append(euclidean_distance)
                self.all_rounds_accuracies.append(mean_test_accuracy)
                self.all_rounds_disparity_losses.append(disparity_loss)  # Save disparity loss

                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                tqdm.write(f"Round {i} Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}, Disparity Loss: {disparity_loss:.4f}")  # Output disparity loss
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                print("Variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))

            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

            csolns = []
            losses = []

            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for _ in range(self.local_iters):
                    data_batch = next(batches[c])

                    self.client_model.set_params(self.local_models[idx])
                    _, grads, _ = c.solve_sgd(data_batch)

                    # Apply HE, DP, and SMC to gradients
                    grad_shapes = [grad.shape for grad in grads[1]]
                    grads_encrypted = [apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, grad in enumerate(grads[1])]
                    grads_decrypted = [decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))]
                    grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag) for grad in grads_decrypted]
                    grads_smc = [apply_smc(grad, self.num_shares, self.smc_flag) for grad in grads_with_noise]
                    grads_reconstructed = [reconstruct_smc(grad, self.smc_flag) for grad in grads_smc]

                    for layer in range(len(grads_reconstructed)):
                        eff_grad = grads_reconstructed[layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
                        self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                    # Global model update
                    self.client_model.set_params(w_global_idx)
                    loss = c.get_loss()
                    losses.append(loss)

                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

                if idx in corrupt_id:
                    if self.boosting:
                        diff = [self.clients_per_round * u for u in diff]
                    elif self.random_updates:
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                csolns.append(diff)

            # Aggregation
            avg_updates = self.aggregate(csolns)

            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]

        # Final metrics
        final_variance = np.mean(self.all_rounds_variances)
        final_euclidean_distance = np.mean(self.all_rounds_distances)
        final_disparity_loss = np.mean(self.all_rounds_disparity_losses)  # Calculate final average disparity loss

        print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
        print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")
        print(f"Final Average Disparity Loss: {final_disparity_loss:.4f}")  # Print final disparity loss

        # Save the metrics to CSV
        np.savetxt(self.output + "_final_variances.csv", np.array(self.all_rounds_variances), delimiter=",")
        np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(self.all_rounds_distances), delimiter=",")
        np.savetxt(self.output + "_final_accuracies.csv", np.array(self.all_rounds_accuracies), delimiter=",")
        np.savetxt(self.output + "_final_disparity_losses.csv", np.array(self.all_rounds_disparity_losses), delimiter=",")  # Save disparity losses to CSV

        print("Metrics saved successfully.")
