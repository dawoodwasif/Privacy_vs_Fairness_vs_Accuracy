# # # # # import numpy as np
# # # # # from tqdm import trange, tqdm
# # # # # import tensorflow as tf

# # # # # from .fedbase import BaseFedarated
# # # # # from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# # # # # from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# # # # # class Server(BaseFedarated):
# # # # #     def __init__(self, params, learner, dataset):
# # # # #         print('Using fair fed avg to Train')
# # # # #         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
# # # # #         super(Server, self).__init__(params, learner, dataset)

# # # # #     def train(self):
# # # # #         print('Training with {} workers ---'.format(self.clients_per_round))

# # # # #         num_clients = len(self.clients)
# # # # #         pk = np.ones(num_clients) * 1.0 / num_clients

# # # # #         for i in range(self.num_rounds+1):
# # # # #             if i % self.eval_every == 0:
# # # # #                 num_test, num_correct_test = self.test() # have set the latest model for all clients
# # # # #                 num_train, num_correct_train = self.train_error()  
# # # # #                 num_val, num_correct_val = self.validate()  
# # # # #                 tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(np.array(num_correct_test)) * 1.0 / np.sum(np.array(num_test))))
# # # # #                 tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))))
# # # # #                 tqdm.write('At round {} validating accuracy: {}'.format(i, np.sum(np.array(num_correct_val)) * 1.0 / np.sum(np.array(num_val))))
            
            
# # # # #             if i % self.log_interval == 0 and i > int(self.num_rounds/2):                
# # # # #                 test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
# # # # #                 np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
# # # # #                 train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
# # # # #                 np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
# # # # #                 validation_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
# # # # #                 np.savetxt(self.output + "_" + str(i) + "_validation.csv", validation_accuracies, delimiter=",")

            
# # # # #             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

# # # # #             Deltas = []
# # # # #             hs = []

# # # # #             selected_clients = selected_clients.tolist()

# # # # #             for c in selected_clients:                
# # # # #                 # communicate the latest model
# # # # #                 c.set_params(self.latest_model)
# # # # #                 weights_before = c.get_params()
# # # # #                 loss = c.get_loss() # compute loss on the whole training data, with respect to the starting point (the global model)
# # # # #                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
# # # # #                 new_weights = soln[1]

# # # # #                 # plug in the weight updates into the gradient
# # # # #                 grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]
                
# # # # #                 Deltas.append([np.float_power(loss+1e-10, self.q) * grad for grad in grads])
                
# # # # #                 # estimation of the local Lipchitz constant
# # # # #                 hs.append(self.q * np.float_power(loss+1e-10, (self.q-1)) * norm_grad(grads) + (1.0/self.learning_rate) * np.float_power(loss+1e-10, self.q))

# # # # #             # aggregate using the dynamic step-size
# # # # #             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

# # # # import numpy as np
# # # # from tqdm import trange, tqdm
# # # # import tensorflow as tf

# # # # from .fedbase import BaseFedarated
# # # # from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# # # # from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# # # # class Server(BaseFedarated):
# # # #     def __init__(self, params, learner, dataset):
# # # #         print('Using fair fed avg to Train')
# # # #         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
# # # #         super(Server, self).__init__(params, learner, dataset)

# # # #     def train(self):
# # # #         print('Training with {} workers ---'.format(self.clients_per_round))

# # # #         num_clients = len(self.clients)
# # # #         pk = np.ones(num_clients) * 1.0 / num_clients

# # # #         all_rounds_variances = []
# # # #         all_rounds_distances = []

# # # #         for i in range(self.num_rounds + 1):
# # # #             if i % self.eval_every == 0:
# # # #                 num_test, num_correct_test = self.test()  # have set the latest model for all clients
# # # #                 num_train, num_correct_train = self.train_error()
# # # #                 num_val, num_correct_val = self.validate()

# # # #                 test_accuracies = np.array(num_correct_test) * 1.0 / np.array(num_test)
# # # #                 train_accuracies = np.array(num_correct_train) * 1.0 / np.array(num_train)
# # # #                 val_accuracies = np.array(num_correct_val) * 1.0 / np.array(num_val)

# # # #                 tqdm.write('At round {} testing accuracy: {}'.format(i, np.mean(test_accuracies)))
# # # #                 tqdm.write('At round {} training accuracy: {}'.format(i, np.mean(train_accuracies)))
# # # #                 tqdm.write('At round {} validating accuracy: {}'.format(i, np.mean(val_accuracies)))

# # # #                 # Calculate variance in testing accuracies
# # # #                 variance = np.var(test_accuracies)
# # # #                 all_rounds_variances.append(variance)
# # # #                 tqdm.write('At round {} variance in testing accuracy: {:.4f}'.format(i, variance))

# # # #                 # Calculate Euclidean distance between clients' accuracies and the mean accuracy
# # # #                 mean_test_accuracy = np.mean(test_accuracies)
# # # #                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
# # # #                 all_rounds_distances.append(euclidean_distance)
# # # #                 tqdm.write('At round {} Euclidean distance between local accuracies: {:.4f}'.format(i, euclidean_distance))

# # # #             if i % self.log_interval == 0 and i > int(self.num_rounds / 2):
# # # #                 np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
# # # #                 np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
# # # #                 np.savetxt(self.output + "_" + str(i) + "_validation.csv", val_accuracies, delimiter=",")

# # # #             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

# # # #             Deltas = []
# # # #             hs = []

# # # #             selected_clients = selected_clients.tolist()

# # # #             for c in selected_clients:
# # # #                 # communicate the latest model
# # # #                 c.set_params(self.latest_model)
# # # #                 weights_before = c.get_params()
# # # #                 loss = c.get_loss()  # compute loss on the whole training data, with respect to the starting point (the global model)
# # # #                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
# # # #                 new_weights = soln[1]

# # # #                 # plug in the weight updates into the gradient
# # # #                 grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]

# # # #                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads])

# # # #                 # estimation of the local Lipchitz constant
# # # #                 hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad(grads) + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

# # # #             # aggregate using the dynamic step-size
# # # #             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

# # # #         # Final variance and Euclidean distance
# # # #         final_variance = np.mean(all_rounds_variances)
# # # #         final_euclidean_distance = np.mean(all_rounds_distances)
# # # #         print("\nFinal Average Variance in Testing Accuracy: {:.4f}".format(final_variance))
# # # #         print("Final Average Euclidean Distance in Testing Accuracy: {:.4f}".format(final_euclidean_distance))
                    

# # # import numpy as np
# # # from tqdm import trange, tqdm
# # # import tensorflow as tf

# # # from .fedbase import BaseFedarated
# # # from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# # # from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# # # def add_dp_noise(gradient, epsilon, delta, sensitivity, mechanism, flag):
# # #     if not flag:
# # #         return gradient
    
# # #     # Generate noise with the same shape as the gradient
# # #     if mechanism == 'laplace':
# # #         noise_scale = sensitivity / epsilon
# # #         noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
# # #     elif mechanism == 'gaussian':
# # #         noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
# # #         noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
# # #     elif mechanism == 'randomized_response':
# # #         # Assuming gradient is binary (0/1) for simplicity
# # #         noise = tf.random.uniform(shape=gradient.shape, minval=0, maxval=2, dtype=tf.int32)
# # #         noise = tf.cast(noise, tf.float32) * epsilon
# # #     elif mechanism == 'exponential':
# # #         # Exponential mechanism for non-numeric queries, here applied as a noise generator
# # #         score_function = tf.norm(gradient, ord=1)  # Example score function
# # #         noise = tf.exp(epsilon * score_function / (2 * sensitivity))
# # #         noise = tf.random.uniform(shape=gradient.shape, minval=-noise, maxval=noise)
# # #     else:
# # #         raise ValueError("Unsupported mechanism: {}".format(mechanism))

# # #     return gradient + noise

# # # class Server(BaseFedarated):
# # #     def __init__(self, params, learner, dataset, dp_params):
# # #         print('Using fair fed avg with Differential Privacy to Train')
# # #         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
# # #         self.epsilon = dp_params['epsilon']
# # #         self.delta = dp_params['delta']
# # #         self.sensitivity = dp_params['sensitivity']
# # #         self.mechanism = dp_params['mechanism']
# # #         self.dp_flag = dp_params['dp_flag']
# # #         super(Server, self).__init__(params, learner, dataset)

# # #     def train(self):
# # #         print('Training with {} workers ---'.format(self.clients_per_round))

# # #         num_clients = len(self.clients)
# # #         pk = np.ones(num_clients) * 1.0 / num_clients

# # #         all_rounds_variances = []
# # #         all_rounds_distances = []

# # #         for i in range(self.num_rounds + 1):
# # #             if i % self.eval_every == 0:
# # #                 num_test, num_correct_test = self.test()  
# # #                 num_train, num_correct_train = self.train_error()
# # #                 num_val, num_correct_val = self.validate()

# # #                 test_accuracies = np.array(num_correct_test) * 1.0 / np.array(num_test)
# # #                 train_accuracies = np.array(num_correct_train) * 1.0 / np.array(num_train)
# # #                 val_accuracies = np.array(num_correct_val) * 1.0 / np.array(num_val)

# # #                 tqdm.write('At round {} testing accuracy: {}'.format(i, np.mean(test_accuracies)))
# # #                 tqdm.write('At round {} training accuracy: {}'.format(i, np.mean(train_accuracies)))
# # #                 tqdm.write('At round {} validating accuracy: {}'.format(i, np.mean(val_accuracies)))

# # #                 variance = np.var(test_accuracies)
# # #                 all_rounds_variances.append(variance)
# # #                 tqdm.write('At round {} variance in testing accuracy: {:.4f}'.format(i, variance))

# # #                 mean_test_accuracy = np.mean(test_accuracies)
# # #                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
# # #                 all_rounds_distances.append(euclidean_distance)
# # #                 tqdm.write('At round {} Euclidean distance between local accuracies: {:.4f}'.format(i, euclidean_distance))

# # #             if i % self.log_interval == 0 and i > int(self.num_rounds / 2):
# # #                 np.savetxt(self.output + "_{}_test.csv".format(i), test_accuracies, delimiter=",")
# # #                 np.savetxt(self.output + "_{}_train.csv".format(i), train_accuracies, delimiter=",")
# # #                 np.savetxt(self.output + "_{}_validation.csv".format(i), val_accuracies, delimiter=",")

# # #             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

# # #             Deltas = []
# # #             hs = []
# # #             weights_before = None

# # #             selected_clients = selected_clients.tolist()

# # #             if not selected_clients:
# # #                 print("Warning: No clients selected in round {}.".format(i))
# # #                 continue

# # #             for c in selected_clients:
# # #                 c.set_params(self.latest_model)
# # #                 weights_before = c.get_params()
# # #                 if weights_before is None:
# # #                     print("Error: weights_before is None for client {}.".format(c))
# # #                     continue

# # #                 loss = c.get_loss()
# # #                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
# # #                 new_weights = soln[1]

# # #                 grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]

# # #                 # Add noise to each gradient with matching shape
# # #                 grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag) for grad in grads]

# # #                 # Calculate the delta for each gradient
# # #                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_with_noise])

# # #                 # Calculate the norm of the gradient with noise
# # #                 norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_with_noise])

# # #                 # Append to hs for aggregation
# # #                 hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

# # #             if weights_before is not None:
# # #                 self.latest_model = self.aggregate2(weights_before, Deltas, hs)
# # #             else:
# # #                 print("Warning: weights_before not set for round {}. Skipping aggregation.".format(i))

# # #         final_variance = np.mean(all_rounds_variances)
# # #         final_euclidean_distance = np.mean(all_rounds_distances)
# # #         print("\nFinal Average Variance in Testing Accuracy: {:.4f}".format(final_variance))
# # #         print("Final Average Euclidean Distance in Testing Accuracy: {:.4f}".format(final_euclidean_distance))

# # # import tenseal as ts

# # # import numpy as np
# # # from tqdm import trange, tqdm
# # # import tensorflow as tf

# # # from .fedbase import BaseFedarated
# # # from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# # # from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# # # def add_dp_noise(gradient, epsilon, delta, sensitivity, mechanism, flag):
# # #     if not flag:
# # #         return gradient
    
# # #     # Generate noise with the same shape as the gradient
# # #     if mechanism == 'laplace':
# # #         noise_scale = sensitivity / epsilon
# # #         noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
# # #     elif mechanism == 'gaussian':
# # #         noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
# # #         noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
# # #     else:
# # #         raise ValueError("Unsupported mechanism: {}".format(mechanism))

# # #     return gradient + noise

# # # def apply_he(gradient, context, flag):
# # #     """Encrypt the gradient if HE is enabled, else return the gradient."""
# # #     if flag and context is not None:
# # #         flattened_grad = gradient.flatten()  # Flatten the gradient to a 1D vector
# # #         return ts.ckks_vector(context, flattened_grad)
# # #     return gradient

# # # def decrypt_he(gradient, original_shape, flag):
# # #     """Decrypt the gradient if HE is enabled and reshape to original."""
# # #     if flag:
# # #         decrypted_grad = np.array(gradient.decrypt())  # Decrypt and convert to numpy array
# # #         return decrypted_grad.reshape(original_shape)  # Reshape back to the original shape
# # #     return gradient

# # # class Server(BaseFedarated):
# # #     def __init__(self, params, learner, dataset, dp_params, he_params):
# # #         print('Using fair fed avg with Differential Privacy and Homomorphic Encryption to Train')
# # #         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
# # #         self.epsilon = dp_params['epsilon']
# # #         self.delta = dp_params['delta']
# # #         self.sensitivity = dp_params['sensitivity']
# # #         self.mechanism = dp_params['mechanism']
# # #         self.dp_flag = dp_params['dp_flag']

# # #         self.he_flag = he_params['he_flag']
# # #         self.n_layers_to_encrypt = he_params['he_encrypt_layers'] # Number of layers to encrypt
# # #         self.context = None  # Initialize context to None

# # #         print(f"HE Flag: {self.he_flag}, n_layers_to_encrypt: {self.n_layers_to_encrypt}")  # Debug

# # #         if self.he_flag:
# # #             # Setup the TenSEAL context for encryption if HE is enabled
# # #             self.context = ts.context(
# # #                 ts.SCHEME_TYPE.CKKS, 
# # #                 poly_modulus_degree=he_params['poly_modulus_degree'], 
# # #                 coeff_mod_bit_sizes=he_params['coeff_mod_bit_sizes']
# # #             )
# # #             self.context.global_scale = he_params['global_scale']
# # #             self.context.generate_galois_keys()
# # #             print("HE Context Initialized")  # Debug
# # #         else:
# # #             print("HE not enabled")  # Debug

# # #         super(Server, self).__init__(params, learner, dataset)

# # #     def train(self):
# # #         print('Training with {} workers ---'.format(self.clients_per_round))

# # #         num_clients = len(self.clients)
# # #         pk = np.ones(num_clients) * 1.0 / num_clients

# # #         all_rounds_variances = []
# # #         all_rounds_distances = []
        
# # #         # **Initialize lists to store global accuracies and variances**
# # #         all_rounds_global_accuracies = []
# # #         all_rounds_local_variances = []

# # #         for i in range(self.num_rounds + 1):
# # #             if i % self.eval_every == 0:
# # #                 num_test, num_correct_test = self.test()
# # #                 num_train, num_correct_train = self.train_error()
# # #                 num_val, num_correct_val = self.validate()

# # #                 test_accuracies = np.array(num_correct_test) * 1.0 / np.array(num_test)
# # #                 train_accuracies = np.array(num_correct_train) * 1.0 / np.array(num_train)
# # #                 val_accuracies = np.array(num_correct_val) * 1.0 / np.array(num_val)

# # #                 tqdm.write('At round {} testing accuracy: {}'.format(i, np.mean(test_accuracies)))
# # #                 tqdm.write('At round {} training accuracy: {}'.format(i, np.mean(train_accuracies)))
# # #                 tqdm.write('At round {} validating accuracy: {}'.format(i, np.mean(val_accuracies)))

# # #                 variance = np.var(test_accuracies)
# # #                 all_rounds_variances.append(variance)
# # #                 tqdm.write('At round {} variance in testing accuracy: {:.4f}'.format(i, variance))

# # #                 mean_test_accuracy = np.mean(test_accuracies)
# # #                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
# # #                 all_rounds_distances.append(euclidean_distance)
# # #                 tqdm.write('At round {} Euclidean distance between local accuracies: {:.4f}'.format(i, euclidean_distance))
                
# # #                 # **Append global accuracy and variance to the lists**
# # #                 all_rounds_global_accuracies.append(mean_test_accuracy)
# # #                 all_rounds_local_variances.append(variance)

# # #             if i % self.log_interval == 0 and i > int(self.num_rounds / 2):
# # #                 np.savetxt(self.output + "_{}_test.csv".format(i), test_accuracies, delimiter=",")
# # #                 np.savetxt(self.output + "_{}_train.csv".format(i), train_accuracies, delimiter=",")
# # #                 np.savetxt(self.output + "_{}_validation.csv".format(i), val_accuracies, delimiter=",")

# # #             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

# # #             Deltas = []
# # #             hs = []
# # #             weights_before = None

# # #             selected_clients = selected_clients.tolist()

# # #             if not selected_clients:
# # #                 print("Warning: No clients selected in round {}.".format(i))
# # #                 continue

# # #             for c in selected_clients:
# # #                 c.set_params(self.latest_model)
# # #                 weights_before = c.get_params()
# # #                 if weights_before is None:
# # #                     print("Error: weights_before is None for client {}.".format(c))
# # #                     continue

# # #                 loss = c.get_loss()
# # #                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
# # #                 new_weights = soln[1]

# # #                 # Calculate gradient
# # #                 grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]

# # #                 # Encrypt the first `n_layers_to_encrypt` gradients
# # #                 grad_shapes = [grad.shape for grad in grads]  # Store original shapes
# # #                 # for idx, grad in enumerate(grads):
# # #                 #     he_flag_layer = self.he_flag if idx < self.n_layers_to_encrypt else False
# # #                 #     print(f"Layer {idx}, Applying HE: {he_flag_layer}")  # Debug

# # #                 grads_encrypted = [
# # #                     apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) 
# # #                     for idx, grad in enumerate(grads)
# # #                 ]

# # #                 # Decrypt gradients before adding DP noise
# # #                 grads_decrypted = [
# # #                     decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False)
# # #                     for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))
# # #                 ]

# # #                 # Add noise to each gradient
# # #                 grads_with_noise = [
# # #                     add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag) 
# # #                     for grad in grads_decrypted
# # #                 ]

# # #                 # Calculate the delta for each gradient
# # #                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_with_noise])

# # #                 # Calculate the norm of the gradient with noise
# # #                 norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_with_noise])

# # #                 # Append to hs for aggregation
# # #                 hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

# # #             if weights_before is not None:
# # #                 self.latest_model = self.aggregate2(weights_before, Deltas, hs)
# # #             else:
# # #                 print("Warning: weights_before not set for round {}. Skipping aggregation.".format(i))

# # #         # **After training, save the collected metrics**
# # #         # Convert lists to numpy arrays
# # #         global_accuracies_array = np.array(all_rounds_global_accuracies)
# # #         local_variances_array = np.array(all_rounds_local_variances)

# # #         # Save the global testing accuracies
# # #         np.savetxt(self.output + "_global_accuracy.csv", global_accuracies_array, delimiter=",", header="Global_Test_Accuracy", comments='')
        
# # #         # Save the variance of local testing accuracies
# # #         np.savetxt(self.output + "_variance_local_accuracy.csv", local_variances_array, delimiter=",", header="Variance_Local_Test_Accuracy", comments='')

# # #         # Optionally, you can also save all_rounds_distances if needed
# # #         # distances_array = np.array(all_rounds_distances)
# # #         # np.savetxt(self.output + "_euclidean_distance.csv", distances_array, delimiter=",", header="Euclidean_Distance_Local_Accuracies", comments='')

# # #         final_variance = np.mean(all_rounds_variances)
# # #         final_euclidean_distance = np.mean(all_rounds_distances)
# # #         print("\nFinal Average Variance in Testing Accuracy: {:.4f}".format(final_variance))
# # #         print("Final Average Euclidean Distance in Testing Accuracy: {:.4f}".format(final_euclidean_distance))






# # # import numpy as np
# # # from cryptography.fernet import Fernet
# # # import pickle  # To serialize the gradient array while preserving the shape

# # # # Generate a Fernet key and cipher for encryption and decryption
# # # key = Fernet.generate_key()
# # # cipher = Fernet(key)

# # # # SMC functions with encryption and decryption
# # # def apply_smc(gradient, num_shares, flag):
# # #     if flag:
# # #         # Generate random shares
# # #         shares = [np.random.random(gradient.shape) for _ in range(num_shares - 1)]
# # #         final_share = gradient - sum(shares)
# # #         shares.append(final_share)

# # #         # Encrypt each share using Fernet encryption
# # #         encrypted_shares = []
# # #         for share in shares:
# # #             # Serialize the numpy array with pickle to preserve shape
# # #             share_serialized = pickle.dumps(share)
# # #             encrypted_share = cipher.encrypt(share_serialized)  # Encrypt the serialized share
# # #             encrypted_shares.append(encrypted_share)

# # #         return encrypted_shares
# # #     return gradient

# # # def reconstruct_smc(shares, flag):
# # #     if flag:
# # #         decrypted_shares = []

# # #         # Decrypt each share and deserialize using pickle
# # #         for encrypted_share in shares:
# # #             decrypted_share_serialized = cipher.decrypt(encrypted_share)  # Decrypt the share
# # #             decrypted_share = pickle.loads(decrypted_share_serialized)  # Deserialize the numpy array
# # #             decrypted_shares.append(decrypted_share)

# # #         # Reconstruct the original gradient by summing the decrypted shares
# # #         return sum(decrypted_shares)
    
# # #     return shares


# # import tenseal as ts
# # import numpy as np
# # from tqdm import trange
# # import tensorflow.compat.v1 as tf

# # from .fedbase import BaseFedarated
# # from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# # from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch

# # # Differential Privacy function to add noise
# # def add_dp_noise(gradient, epsilon, delta, sensitivity, mechanism, flag):
# #     if not flag:
# #         return gradient
    
# #     if mechanism == 'laplace':
# #         noise_scale = sensitivity / epsilon
# #         noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
# #     elif mechanism == 'gaussian':
# #         noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
# #         noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
# #     else:
# #         raise ValueError("Unsupported DP mechanism: {}".format(mechanism))

# #     return gradient + noise

# # # Homomorphic Encryption functions
# # def apply_he(gradient, context, flag):
# #     if flag and context is not None:
# #         flattened_grad = gradient.flatten()
# #         return ts.ckks_vector(context, flattened_grad)
# #     return gradient

# # def decrypt_he(gradient, original_shape, flag):
# #     if flag:
# #         decrypted_grad = np.array(gradient.decrypt())
# #         return decrypted_grad.reshape(original_shape)
# #     return gradient

# # # SMC functions
# # def apply_smc(gradient, num_shares, flag):
# #     if flag:
# #         shares = [np.random.random(gradient.shape) for _ in range(num_shares - 1)]
# #         final_share = gradient - sum(shares)
# #         shares.append(final_share)
# #         return shares
# #     return gradient

# # def reconstruct_smc(shares, flag):
# #     if flag:
# #         return sum(shares)
# #     return shares

# # class Server(BaseFedarated):
# #     def __init__(self, params, learner, dataset, dp_params, he_params, smc_params):
# #         print('Using q-FedAvg with DP, HE, and SMC')
# #         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

# #         # DP parameters
# #         self.epsilon = dp_params['epsilon']
# #         self.delta = dp_params['delta']
# #         self.sensitivity = dp_params['sensitivity']
# #         self.mechanism = dp_params['mechanism']
# #         self.dp_flag = dp_params['dp_flag']

# #         # HE parameters
# #         self.he_flag = he_params['he_flag']
# #         self.n_layers_to_encrypt = he_params['he_encrypt_layers']
# #         self.context = None

# #         if self.he_flag:
# #             self.context = ts.context(
# #                 ts.SCHEME_TYPE.CKKS, 
# #                 poly_modulus_degree=he_params['poly_modulus_degree'], 
# #                 coeff_mod_bit_sizes=he_params['coeff_mod_bit_sizes']
# #             )
# #             self.context.global_scale = he_params['global_scale']
# #             self.context.generate_galois_keys()
# #             print("HE Context Initialized")

# #         # SMC parameters
# #         self.smc_flag = smc_params['smc_flag']
# #         self.num_shares = smc_params['smc_num_shares']

# #         # Initialize the federated learning base class
# #         super(Server, self).__init__(params, learner, dataset)

# #     def train(self):
# #         print('Training with {} workers ---'.format(self.clients_per_round))

# #         num_clients = len(self.clients)
# #         pk = np.ones(num_clients) / num_clients  # Uniform probability for client selection

# #         all_rounds_variances = []
# #         all_rounds_distances = []
# #         all_rounds_accuracies = []
# #         all_rounds_loss_disparities = []  # Collecting loss disparities

# #         for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
# #             if i % self.eval_every == 0:
# #                 num_test, num_correct_test, client_losses = self.test()  # Modified to get losses too
# #                 test_accuracies = np.array(num_correct_test) / np.array(num_test)
# #                 variance = np.var(test_accuracies)
# #                 all_rounds_variances.append(variance)

# #                 mean_test_accuracy = np.mean(test_accuracies)
# #                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
# #                 all_rounds_distances.append(euclidean_distance)
# #                 all_rounds_accuracies.append(mean_test_accuracy)

# #                 # Calculate and store loss disparity
# #                 loss_disparity = np.var(client_losses)
# #                 all_rounds_loss_disparities.append(loss_disparity)

# #                 print(f'\nRound {i} testing accuracy: {mean_test_accuracy:.4f}, Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}, Loss Disparity: {loss_disparity:.4f}')

# #             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)
# #             Deltas = []
# #             hs = []
# #             weights_before = None
# #             client_losses = []  # To store losses for disparity calculation

# #             selected_clients = selected_clients.tolist()

# #             for c in selected_clients:
# #                 c.set_params(self.latest_model)
# #                 weights_before = c.get_params()

# #                 if weights_before is None:
# #                     print(f"Error: weights_before is None for client {c}.")
# #                     continue

# #                 loss = c.get_loss()
# #                 client_losses.append(loss)  # Collect losses for disparity calculation
# #                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
# #                 new_weights = soln[1]

# #                 # Calculate gradient
# #                 grads = [(u - v) / self.learning_rate for u, v in zip(weights_before, new_weights)]

# #                 # Encrypt the first `n_layers_to_encrypt` gradients
# #                 grad_shapes = [grad.shape for grad in grads]
# #                 grads_encrypted = [apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, grad in enumerate(grads)]

# #                 # Decrypt gradients before adding DP noise
# #                 grads_decrypted = [decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))]

# #                 # Add DP noise to gradients
# #                 grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag) for grad in grads_decrypted]

# #                 # Apply SMC
# #                 grads_smc = [apply_smc(grad, self.num_shares, self.smc_flag) for grad in grads_with_noise]
# #                 grads_reconstructed = [reconstruct_smc(grad, self.smc_flag) for grad in grads_smc]

# #                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_reconstructed])

# #                 norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_reconstructed])
# #                 hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

# #             # Aggregation
# #             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

# #         # Saving the metrics for all rounds
# #         np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
# #         np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
# #         np.savetxt(self.output + "_final_accuracies.csv", np.array(all_rounds_accuracies), delimiter=",")
# #         np.savetxt(self.output + "_final_loss_disparities.csv", np.array(all_rounds_loss_disparities), delimiter=",")

# #         print("Metrics saved successfully.")

# # def aggregate2(weights_before, updates, hs):
# #     new_solutions = []
# #     for w, delta in zip(weights_before, updates):
# #         delta_sum = np.zeros_like(w)

# #         for delta_i, h_i in zip(delta, hs):
# #             if np.isscalar(h_i):
# #                 delta_sum += delta_i * h_i
# #             else:
# #                 reshaped_h_i = np.reshape(h_i, delta_i.shape)
# #                 delta_sum += delta_i * reshaped_h_i

# #         new_solutions.append(w - delta_sum)

# #     return new_solutions

# import tenseal as ts
# import numpy as np
# from tqdm import trange
# import tensorflow.compat.v1 as tf

# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch

# # Differential Privacy function to add noise
# def add_dp_noise(gradient, epsilon, delta, sensitivity, mechanism, dp_flag, dp_type):
#     if not dp_flag:
#         return gradient
    
#     if mechanism == 'laplace':
#         noise_scale = sensitivity / epsilon
#         noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
#     elif mechanism == 'gaussian':
#         noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
#         noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
#     else:
#         raise ValueError("Unsupported DP mechanism: {}".format(mechanism))

#     if dp_type == 'LDP':
#         # Apply noise at the client side (before sending to server)
#         return gradient + noise
#     elif dp_type == 'GDP':
#         # For GDP, noise will be applied globally (later on the aggregated result)
#         return gradient  # Do nothing here, noise will be added at the server
#     else:
#         raise ValueError("Invalid DP type. Choose 'LDP' or 'GDP'.")

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
#         print('Using q-FedAvg with DP, HE, and SMC')
#         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

#         # DP parameters
#         self.epsilon = dp_params['epsilon']
#         self.delta = dp_params['delta']
#         self.sensitivity = dp_params['sensitivity']
#         self.mechanism = dp_params['mechanism']
#         self.dp_flag = dp_params['dp_flag']
#         self.dp_type = dp_params.get('dp_type', 'GDP')  # Use 'GDP' as default, can also be 'LDP'

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

#     def train(self):
#         print(f'Training with {self.clients_per_round} workers ---')

#         num_clients = len(self.clients)
#         pk = np.ones(num_clients) / num_clients  # Uniform probability for client selection

#         all_rounds_variances = []
#         all_rounds_distances = []
#         all_rounds_accuracies = []
#         all_rounds_loss_disparities = []  # Collecting loss disparities

#         for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
#             if i % self.eval_every == 0:
#                 num_test, num_correct_test, client_losses = self.test()  # Modified to get losses too
#                 test_accuracies = np.array(num_correct_test) / np.array(num_test)
#                 variance = np.var(test_accuracies)
#                 all_rounds_variances.append(variance)

#                 mean_test_accuracy = np.mean(test_accuracies)
#                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
#                 all_rounds_distances.append(euclidean_distance)
#                 all_rounds_accuracies.append(mean_test_accuracy)

#                 # Calculate and store loss disparity
#                 loss_disparity = np.var(client_losses)
#                 all_rounds_loss_disparities.append(loss_disparity)

#                 print(f'\nRound {i} testing accuracy: {mean_test_accuracy:.4f}, Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}, Loss Disparity: {loss_disparity:.4f}')

#             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)
#             Deltas = []
#             hs = []
#             weights_before = None
#             client_losses = []  # To store losses for disparity calculation

#             selected_clients = selected_clients.tolist()

#             for c in selected_clients:
#                 c.set_params(self.latest_model)
#                 weights_before = c.get_params()

#                 if weights_before is None:
#                     print(f"Error: weights_before is None for client {c}.")
#                     continue

#                 loss = c.get_loss()
#                 client_losses.append(loss)  # Collect losses for disparity calculation
#                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
#                 new_weights = soln[1]

#                 # Calculate gradient
#                 grads = [(u - v) / self.learning_rate for u, v in zip(weights_before, new_weights)]

#                 # Encrypt the first `n_layers_to_encrypt` gradients
#                 grad_shapes = [grad.shape for grad in grads]
#                 grads_encrypted = [apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, grad in enumerate(grads)]

#                 # Decrypt gradients before adding DP noise
#                 grads_decrypted = [decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))]

#                 # Add DP noise to gradients (LDP or GDP based on dp_type)
#                 grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag, self.dp_type) for grad in grads_decrypted]

#                 # Apply SMC
#                 grads_smc = [apply_smc(grad, self.num_shares, self.smc_flag) for grad in grads_with_noise]
#                 grads_reconstructed = [reconstruct_smc(grad, self.smc_flag) for grad in grads_smc]

#                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_reconstructed])

#                 norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_reconstructed])
#                 hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

#             # Aggregation
#             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

#         # Saving the metrics for all rounds
#         np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
#         np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
#         np.savetxt(self.output + "_final_accuracies.csv", np.array(all_rounds_accuracies), delimiter=",")
#         np.savetxt(self.output + "_final_loss_disparities.csv", np.array(all_rounds_loss_disparities), delimiter=",")

#         print("Metrics saved successfully.")

# def aggregate2(weights_before, updates, hs):
#     new_solutions = []
#     for w, delta in zip(weights_before, updates):
#         delta_sum = np.zeros_like(w)

#         for delta_i, h_i in zip(delta, hs):
#             if np.isscalar(h_i):
#                 delta_sum += delta_i * h_i
#             else:
#                 reshaped_h_i = np.reshape(h_i, delta_i.shape)
#                 delta_sum += delta_i * reshaped_h_i

#         new_solutions.append(w - delta_sum)

#     return new_solutions

import tenseal as ts
import numpy as np
from tqdm import trange
import tensorflow.compat.v1 as tf

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
        print('Using q-FedAvg with DP, HE, and SMC')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        # DP parameters
        self.epsilon = dp_params['epsilon']
        self.delta = dp_params['delta']
        self.sensitivity = dp_params['sensitivity']
        self.mechanism = dp_params['mechanism']
        self.dp_type = dp_params['scope'] # LDP or GDP
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

    def train(self):
        print(f'Training with {self.clients_per_round} workers using {self.dp_type} DP.')

        num_clients = len(self.clients)
        pk = np.ones(num_clients) / num_clients  # Uniform probability for client selection

        all_rounds_variances = []
        all_rounds_distances = []
        all_rounds_accuracies = []
        all_rounds_loss_disparities = []

        for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
            if i % self.eval_every == 0:
                num_test, num_correct_test, client_losses = self.test()
                test_accuracies = np.array(num_correct_test) / np.array(num_test)
                variance = np.var(test_accuracies)
                all_rounds_variances.append(variance)

                mean_test_accuracy = np.mean(test_accuracies)
                euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
                all_rounds_distances.append(euclidean_distance)
                all_rounds_accuracies.append(mean_test_accuracy)

                loss_disparity = np.var(client_losses)
                all_rounds_loss_disparities.append(loss_disparity)

                print(f'\nRound {i} testing accuracy: {mean_test_accuracy:.4f}, Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}, Loss Disparity: {loss_disparity:.4f}')

            indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)
            Deltas = []
            hs = []
            weights_before = None
            client_losses = []

            selected_clients = selected_clients.tolist()

            for c in selected_clients:
                c.set_params(self.latest_model)
                weights_before = c.get_params()

                loss = c.get_loss()
                client_losses.append(loss)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                new_weights = soln[1]

                grads = [(u - v) / self.learning_rate for u, v in zip(weights_before, new_weights)]

                grad_shapes = [grad.shape for grad in grads]
                grads_encrypted = [apply_he(grad, self.context, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, grad in enumerate(grads)]

                grads_decrypted = [decrypt_he(grad, original_shape, self.he_flag if idx < self.n_layers_to_encrypt else False) for idx, (grad, original_shape) in enumerate(zip(grads_encrypted, grad_shapes))]

                # Apply LDP in the train function if dp_type is 'LDP'
                if self.dp_type == 'LDP':
                    grads_with_noise = [add_dp_noise(grad, self.epsilon, self.delta, self.sensitivity, self.mechanism, self.dp_flag) for grad in grads_decrypted]
                else:
                    grads_with_noise = grads_decrypted

                grads_smc = [apply_smc(grad, self.num_shares, self.smc_flag) for grad in grads_with_noise]
                grads_reconstructed = [reconstruct_smc(grad, self.smc_flag) for grad in grads_smc]

                Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_reconstructed])

                norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_reconstructed])
                hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

            self.latest_model = self.aggregate3(weights_before, Deltas, hs)

        # Log final metrics
        print(f"Final Average Variance: {np.mean(all_rounds_variances):.4f}")
        print(f"Final Average Euclidean Distance: {np.mean(all_rounds_distances):.4f}")

        np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
        np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
        np.savetxt(self.output + "_final_accuracies.csv", np.array(all_rounds_accuracies), delimiter=",")
        np.savetxt(self.output + "_final_loss_disparities.csv", np.array(all_rounds_loss_disparities), delimiter=",")

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
