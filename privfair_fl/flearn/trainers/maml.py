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


import tenseal as ts
import numpy as np
from tqdm import trange
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # Disable eager execution for compatibility with TensorFlow 1.x API

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch

# Differential Privacy function to add noise
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
        print('Using Fair Fed MAML to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        # Differential Privacy (DP) parameters
        self.dp_flag = dp_params['dp_flag']
        self.epsilon = dp_params['epsilon']
        self.delta = dp_params['delta']
        self.sensitivity = dp_params['sensitivity']
        self.dp_type = dp_params['scope'] # LDP or GDP
        self.mechanism = dp_params['mechanism']

        # Homomorphic Encryption (HE) parameters
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

        # Secure Multiparty Computation (SMC) parameters
        self.smc_flag = smc_params['smc_flag']
        self.num_shares = smc_params['smc_num_shares']
        self.smc_threshold = smc_params['smc_threshold']
        self.dynamic_sharing = True

        # Initialize base federated class
        super(Server, self).__init__(params, learner, dataset)

        # Initialize arrays for metric tracking
        self.all_rounds_accuracies = []
        self.final_accuracies = []
        self.all_rounds_loss_disparities = []

    def train(self):
        print(f'Training with {self.clients_per_round} workers using '
              f'{"DP" if self.dp_flag else "HE" if self.he_flag else "SMC" if self.smc_flag else "no privacy"}.')

        num_clients = len(self.clients)
        pk = np.ones(num_clients) / num_clients  # Uniform client selection probability

        train_batches = {c: gen_batch(c.train_data, self.batch_size, self.num_rounds + 2) for c in self.clients}
        test_batches = {c: gen_batch(c.test_data, self.batch_size, self.num_rounds + 2) for c in self.clients}

        all_rounds_variances = []
        all_rounds_distances = []

        for round_num in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
            if round_num % self.eval_every == 0:
                num_test, num_correct_test, client_losses = self.test()
                test_accuracies = np.array(num_correct_test) / np.array(num_test)
                variance = np.var(test_accuracies)
                all_rounds_variances.append(variance)

                mean_test_accuracy = np.mean(test_accuracies)
                euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
                all_rounds_distances.append(euclidean_distance)

                loss_disparity = np.var(client_losses)
                self.all_rounds_loss_disparities.append(loss_disparity)
                
                self.all_rounds_accuracies.append(mean_test_accuracy)

                print(f'\nRound {round_num} testing accuracy: {mean_test_accuracy:.4f}, '
                      f'Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}, Loss Disparity: {loss_disparity:.4f}')

            indices, selected_clients = self.select_clients(round=round_num, pk=pk, num_clients=self.clients_per_round)
            selected_clients = selected_clients.tolist()

            Deltas = []
            hs = []
            weights_before = None

            for c in selected_clients:
                c.set_params(self.latest_model)
                weights_before = c.get_params()

                batch1 = next(train_batches[c])
                batch2 = next(test_batches[c])

                if self.with_maml:
                    _, grads1, loss1 = c.solve_sgd(batch1)
                _, grads2, loss2 = c.solve_sgd(batch2)

                # Homomorphic Encryption (HE) - apply only to encrypted layers
                grad_shapes = [grad.shape for grad in grads2[1]]
                grads_encrypted = [apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) 
                                   for idx, grad in enumerate(grads2[1])]
                
                grads_decrypted = [decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False) 
                                   for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))]

                # Differential Privacy (DP)
                if self.dp_type == 'LDP':
                    grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag)
                                        for grad in grads_decrypted]
                else:
                    grads_with_noise = grads_decrypted

                # Secure Multiparty Computation (SMC)
                grads_smc = [apply_smc(grad, self.num_shares, self.smc_flag) for grad in grads_with_noise]
                grads_reconstructed = [reconstruct_smc(grad, self.smc_flag) for grad in grads_smc]

                Deltas.append([np.float_power(loss2 + 1e-10, self.q) * grad for grad in grads_reconstructed])
                norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_reconstructed])
                hs.append(self.q * np.float_power(loss2 + 1e-10, (self.q - 1)) * norm_grad_sum +
                          (1.0 / self.learning_rate) * np.float_power(loss2 + 1e-10, self.q))

            self.latest_model = self.aggregate3(weights_before, Deltas, hs)

        # Save and log metrics after training
        final_variance = np.mean(all_rounds_variances)
        final_euclidean_distance = np.mean(all_rounds_distances)
        final_loss_disparity = np.mean(self.all_rounds_loss_disparities)

        print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
        print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")
        print(f"Final Average Loss Disparity: {final_loss_disparity:.4f}")

        np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
        np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
        np.savetxt(self.output + "_final_accuracies.csv", np.array(self.all_rounds_accuracies), delimiter=",")
        np.savetxt(self.output + "_final_loss_disparities.csv", np.array(self.all_rounds_loss_disparities), delimiter=",")
        print("Metrics saved successfully.")

    def aggregate3(self, weights_before, Deltas, hs): 
        
        demominator = np.sum(np.asarray(hs))
        num_clients = len(Deltas)
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]
        
        if self.dp_type == 'GDP':
            newer_solutions=[]
            for solution in new_solutions:
                newer_solutions.append(add_dp_noise(solution,self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag))
        else:
            newer_solutions = new_solutions

        return newer_solutions
