# import numpy as np
# from tqdm import trange, tqdm
# import tensorflow as tf

# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


# class Server(BaseFedarated):
#     def __init__(self, params, learner, dataset):
#         print('Using fair fed SGD to Train')
#         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
#         super(Server, self).__init__(params, learner, dataset)

#     def train(self):
#         print('Training with {} workers ---'.format(self.clients_per_round))
#         num_clients = len(self.clients)
#         pk = np.ones(num_clients) * 1.0 / num_clients

#         batches = {}
#         for c in self.clients:
#             batches[c] = gen_epoch(c.train_data, self.num_rounds + 2)

#         print('Have generated training batches for all clients...')

#         all_rounds_variances = []
#         all_rounds_distances = []

#         for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
#             # test model
#             if i % self.eval_every == 0:
#                 num_test, num_correct_test = self.test()  # have set the latest model for all clients
#                 num_train, num_correct_train = self.train_error()
#                 num_val, num_correct_val = self.validate()
#                 tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(np.array(num_correct_test)) * 1.0 / np.sum(np.array(num_test))))
#                 tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))))
#                 tqdm.write('At round {} validating accuracy: {}'.format(i, np.sum(np.array(num_correct_val)) * 1.0 / np.sum(np.array(num_val))))
                
#                 test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
#                 train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
#                 val_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))

#                 # Calculate variance in testing accuracies
#                 variance = np.var(test_accuracies)
#                 all_rounds_variances.append(variance)
#                 tqdm.write('At round {} variance in testing accuracy: {:.4f}'.format(i, variance))

#                 # Calculate Euclidean distance between clients' accuracies and the mean accuracy
#                 mean_test_accuracy = np.mean(test_accuracies)
#                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
#                 all_rounds_distances.append(euclidean_distance)
#                 tqdm.write('At round {} Euclidean distance between local accuracies: {:.4f}'.format(i, euclidean_distance))

#                 if self.track_individual_accuracy == 1:
#                     for idx in range(len(self.clients)):
#                         tqdm.write('Client {} testing accuracy: {}'.format(self.clients[idx].id, test_accuracies[idx]))

#             if i % self.log_interval == 0 and i > int(self.num_rounds / 2):
#                 np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
#                 np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
#                 np.savetxt(self.output + "_" + str(i) + "_validation.csv", val_accuracies, delimiter=",")

#             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

#             Deltas = []
#             hs = []

#             selected_clients = selected_clients.tolist()

#             for c in selected_clients:
#                 # communicate the latest model
#                 c.set_params(self.latest_model)
#                 weights_before = c.get_params()

#                 # solve minimization locally
#                 batch = next(batches[c])
#                 _, grads, loss = c.solve_sgd(batch)

#                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads[1]])
#                 if self.static_step_size:
#                     hs.append(1.0 / self.learning_rate)
#                 else:
#                     hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad(grads[1]) + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

#             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

#         # Final variance and Euclidean distance
#         final_variance = np.mean(all_rounds_variances)
#         final_euclidean_distance = np.mean(all_rounds_distances)
#         print("\nFinal Average Variance in Testing Accuracy: {:.4f}".format(final_variance))
#         print("Final Average Euclidean Distance in Testing Accuracy: {:.4f}".format(final_euclidean_distance))

# import numpy as np
# from tqdm import trange, tqdm
# import tensorflow as tf
# import random
# import functools

# from .fedbase import BaseFedarated
# from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
# from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch

# # Library for Shamir's Secret Sharing
# from secretsharing import SecretSharer
# import base64



# class Server(BaseFedarated):
#     def __init__(self, params, learner, dataset, dp_params, he_params):
#         # Initialize learning rate
#         self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        
#         # Differential Privacy parameters
#         self.dp_flag = dp_params['dp_flag']    # Enable DP?
#         self.epsilon = dp_params['epsilon']    # Privacy budget
#         self.delta = dp_params['delta']        # DP delta value
#         self.sensitivity = dp_params['sensitivity']  # Gradient sensitivity
#         self.mechanism = dp_params['mechanism']  # Laplace or Gaussian
        
#         # Homomorphic Encryption parameters
#         self.he_flag = he_params['he_flag']    # Enable HE?
#         self.n_layers_to_encrypt = he_params['he_encrypt_layers']  # Number of layers to encrypt
#         self.context = None    # Initialize TenSEAL context for HE

#         # Secure Multiparty Computation (SMC) parameters
#         self.smc_flag = True #smc_params['smc_flag']  # Enable SMC?
#         self.num_shares = 5 #smc_params['num_shares']  # Number of shares in secret sharing

#         # Initialize the federated learning base class
#         super(Server, self).__init__(params, learner, dataset)

#         # Set up TenSEAL context for HE
#         if self.he_flag:
#             self.context = ts.context(
#                 ts.SCHEME_TYPE.CKKS, 
#                 poly_modulus_degree=he_params['poly_modulus_degree'], 
#                 coeff_mod_bit_sizes=he_params['coeff_mod_bit_sizes']
#             )
#             self.context.global_scale = he_params['global_scale']
#             self.context.generate_galois_keys()
#             print("HE Context Initialized")

#     # Differential Privacy function
#     def add_dp_noise(self, gradient):
#         """Add DP noise to gradients if DP is enabled."""
#         if not self.dp_flag:
#             return gradient

#         if self.mechanism == 'laplace':
#             noise_scale = self.sensitivity / self.epsilon
#             noise = np.random.laplace(loc=0.0, scale=noise_scale, size=gradient.shape)
#         elif self.mechanism == 'gaussian':
#             noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
#             noise = np.random.normal(loc=0.0, scale=noise_scale, size=gradient.shape)
#         else:
#             raise ValueError(f"Unsupported DP mechanism: {self.mechanism}")

#         return gradient + noise

#     # Homomorphic Encryption (HE) functions
#     def apply_he(self, gradient):
#         """Encrypt gradient using HE if enabled."""
#         if self.he_flag and self.context is not None:
#             flattened_grad = gradient.flatten()  # Flatten the gradient to 1D vector
#             return ts.ckks_vector(self.context, flattened_grad)
#         return gradient

#     def decrypt_he(self, gradient, original_shape):
#         """Decrypt the gradient if HE is enabled and reshape."""
#         if self.he_flag:
#             decrypted_grad = np.array(gradient.decrypt())  # Decrypt and convert to numpy array
#             return decrypted_grad.reshape(original_shape)  # Reshape to original shape
#         return gradient

#     # Secure Multiparty Computation (SMC) functions using Shamir's Secret Sharing
#     def apply_smc(self, gradient):
#         """Apply secret sharing to the gradient if SMC is enabled."""
#         if self.smc_flag:
#             shares = self.split_secret(gradient, self.num_shares)  # Split gradient into shares
#             return shares
#         return gradient

#     def reconstruct_smc(self, shares):
#         """Reconstruct the gradient from shares if SMC is enabled."""
#         if self.smc_flag:
#             return self.reconstruct_secret(shares)  # Reconstruct original gradient from shares
#         return shares
    

#     def split_secret(self, gradient, n_shares):
#         """
#         Splits the gradient into smaller chunks, encodes each chunk as a hexadecimal string,
#         and splits it into secret shares using Shamir's Secret Sharing.
#         """
#         gradient_bytes = gradient.tobytes()  # Convert gradient to bytes
#         chunk_size = 128  # Define a chunk size (adjust as needed)

#         # Split the gradient into smaller chunks
#         gradient_chunks = [gradient_bytes[i:i + chunk_size] for i in range(0, len(gradient_bytes), chunk_size)]

#         shares_list = []

#         # Apply secret sharing to each chunk
#         for chunk in gradient_chunks:
#             secret_str = chunk.hex()  # Convert each chunk to a hexadecimal string

#             # Validate hexadecimal format
#             if not all(c in '0123456789abcdefABCDEF' for c in secret_str):
#                 print(f"Invalid hexadecimal string before sharing: {secret_str}")
#                 raise ValueError("Invalid hexadecimal string before sharing")

#             try:
#                 # Debugging: Log the hexadecimal string to ensure it's valid
#                 # print(f"Hexadecimal string before sharing: {secret_str}")
                
#                 # Split the secret
#                 shares = SecretSharer.split_secret(secret_str, 3, num_shares=n_shares)  # Adjust threshold as needed
#                 shares_list.append(shares)
#             except ValueError as e:
#                 print(f"Error splitting secret: {e}")
#                 raise e

#         return shares_list  # Return all shares of the chunks


#     # def reconstruct_secret(self, shares):
#     #     """
#     #     Reconstructs the original gradient from the secret shares.
#     #     """
#     #     reconstructed_chunks = []

#     #     # Iterate over each set of shares to reconstruct the original secret
#     #     for share_set in shares:
#     #         try:
#     #             # Recover the secret string from shares
#     #             secret_str = SecretSharer.recover_secret(share_set) 
                
#     #             # Debugging: Print the reconstructed secret string
#     #             print(f"Reconstructed hexadecimal string: {secret_str}")

#     #             # Validate if the string is a valid hexadecimal string
#     #             if not all(c in '0123456789abcdefABCDEF' for c in secret_str):
#     #                 print(f"Invalid reconstructed hex string at position: {secret_str}")
#     #                 raise ValueError("Reconstructed string is not valid hexadecimal")


#     #             print(f"Reconstructed string: {secret_str}")
#     #             print(f"Length of the string: {len(secret_str)}")
#     #             if len(secret_str) % 2 != 0:
#     #                 # Log the error and fix the string (either pad or investigate further)
#     #                 secret_str = secret_str[:-1]  # Remove the last character as a temporary fix
#     #                 print("Fixed string length!")


#     #             # Attempt to convert the string back to bytes
#     #             secret_bytes = bytes.fromhex(secret_str)  # Convert back to bytes
#     #             reconstructed_chunks.append(secret_bytes)

#     #         except ValueError as e:
#     #             print(f"Error converting secret to bytes: {e}")
#     #             raise e

#     #     # Concatenate all the reconstructed chunks into the original gradient bytes
#     #     reconstructed_gradient_bytes = b''.join(reconstructed_chunks)

#     #     # Convert back to the original gradient shape
#     #     gradient = np.frombuffer(reconstructed_gradient_bytes, dtype=np.float32)

#     #     return gradient
    
#     def reconstruct_secret(self, shares):
#         """
#         Reconstructs the original gradient from the secret shares.
#         """
#         reconstructed_chunks = []

#         # Iterate over each set of shares to reconstruct the original secret
#         for share_set in shares:
#             try:
#                 # Recover the secret string from shares
#                 secret_str = SecretSharer.recover_secret(share_set)
                
#                 # Debugging: Print the reconstructed secret string
#                 # print(f"Reconstructed hexadecimal string: {secret_str}")

#                 # Validate if the string is a valid hexadecimal string
#                 if not all(c in '0123456789abcdefABCDEF' for c in secret_str):
#                     invalid_chars = [c for c in secret_str if c not in '0123456789abcdefABCDEF']
#                     print(f"Invalid characters found in reconstructed string: {invalid_chars}")
#                     raise ValueError(f"Invalid hexadecimal string found: {secret_str}")

#                 # # Ensure string length is even for hex conversion
#                 # if len(secret_str) % 2 != 0:
#                 #     # Fix odd-length string by trimming or padding
#                 #     print(f"Odd-length hexadecimal string detected: {len(secret_str)}. Trimming...")
#                 #     secret_str = secret_str[:-1]  # Trimming the last character to make it even

#                 if len(secret_str) % 2 != 0:
#                     # Log the issue and pad with a zero to ensure even length
#                     print(f"Odd-length hexadecimal string detected: {len(secret_str)}. Padding...")
#                     secret_str = '0' + secret_str  # Pad the string with a leading zero
                



#                 # Attempt to convert the string back to bytes
#                 secret_bytes = bytes.fromhex(secret_str)
#                 reconstructed_chunks.append(secret_bytes)

#             except ValueError as e:
#                 print(f"Error converting secret to bytes: {e}")
#                 raise e

#         # Concatenate all the reconstructed chunks into the original gradient bytes
#         reconstructed_gradient_bytes = b''.join(reconstructed_chunks)

#         # Ensure the buffer size is a multiple of 4 before converting to float32
#         if len(reconstructed_gradient_bytes) % 4 != 0:
#             padding_needed = 4 - (len(reconstructed_gradient_bytes) % 4)
#             print(f"Warning: Buffer size {len(reconstructed_gradient_bytes)} is not a multiple of 4. Padding with {padding_needed} zero bytes...")
#             reconstructed_gradient_bytes += b'\x00' * padding_needed
        
#         # print(f"Reconstructed gradient byte length: {len(reconstructed_gradient_bytes)}")


#         # Convert back to the original gradient shape
#         try:
#             gradient = np.frombuffer(reconstructed_gradient_bytes, dtype=np.float32)
#             print(f"Reconstructed gradient shape: {gradient.shape}")
#         except ValueError as e:
#             print(f"Error in frombuffer: {e}")
#             raise e

#         return gradient







#     def train(self):
#         print(f'Training with {self.clients_per_round} workers using {"DP" if self.dp_flag else "HE" if self.he_flag else "SMC" if self.smc_flag else "no privacy"}.')

#         num_clients = len(self.clients)
#         pk = np.ones(num_clients) * 1.0 / num_clients

#         all_rounds_variances = []
#         all_rounds_distances = []

#         for i in trange(self.num_rounds + 1, desc='Round: ', ncols=120):
#             if i % self.eval_every == 0:
#                 num_test, num_correct_test = self.test()
#                 test_accuracies = np.array(num_correct_test) / np.array(num_test)
#                 variance = np.var(test_accuracies)
#                 all_rounds_variances.append(variance)

#                 mean_test_accuracy = np.mean(test_accuracies)
#                 euclidean_distance = np.linalg.norm(test_accuracies - mean_test_accuracy)
#                 all_rounds_distances.append(euclidean_distance)

#                 print(f'Round {i} testing accuracy: {mean_test_accuracy:.4f}, Variance: {variance:.4f}, Euclidean Distance: {euclidean_distance:.4f}')

#             indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

#             Deltas = []
#             hs = []
#             weights_before = None

#             selected_clients = selected_clients.tolist()

#             for c in selected_clients:
#                 c.set_params(self.latest_model)
#                 weights_before = c.get_params()

#                 if weights_before is None:
#                     print(f"Error: weights_before is None for client {c}.")
#                     continue

#                 loss = c.get_loss()
#                 soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
#                 new_weights = soln[1]

#                 grads = [(u - v) / self.learning_rate for u, v in zip(weights_before, new_weights)]

#                 # Apply Homomorphic Encryption (HE)
#                 grad_shapes = [grad.shape for grad in grads]  # Store original shapes
#                 grads_encrypted = [self.apply_he(grad) for grad in grads]
#                 grads_decrypted = [self.decrypt_he(grad, original_shape) for grad, original_shape in zip(grads_encrypted, grad_shapes)]

#                 # Apply Differential Privacy (DP) Noise
#                 grads_with_noise = [self.add_dp_noise(grad) for grad in grads_decrypted]

#                 # Apply Secure Multiparty Computation (SMC)
#                 grads_smc = [self.apply_smc(grad) for grad in grads_with_noise]
#                 grads_reconstructed = [self.reconstruct_smc(grad) for grad in grads_smc]

#                 Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_reconstructed])

#                 norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_reconstructed])
#                 hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

#             self.latest_model = self.aggregate2(weights_before, Deltas, hs)

#         final_variance = np.mean(all_rounds_variances)
#         final_euclidean_distance = np.mean(all_rounds_distances)

#         print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
#         print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")

#         # Save metrics
#         np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
#         np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
#         print("Metrics saved successfully.")

import numpy as np
from tqdm import trange
import tensorflow as tf
import tenseal as ts

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset, dp_params, he_params, smc_params):
        # Initialize optimizer
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

    # SMC functions
    def apply_smc(self, gradient, round_num):
        """Split gradient into random shares for SMC with round-based randomness."""
        if self.smc_flag:
            np.random.seed(round_num)  # Use round number as seed for randomness
            shares = [np.random.random(gradient.shape) for _ in range(self.num_shares - 1)]
            final_share = gradient - sum(shares)
            shares.append(final_share)

            # Optionally randomize share distribution between rounds
            if self.dynamic_sharing:
                shares = self.randomize_share_distribution(shares)

            return shares
        return gradient

    def randomize_share_distribution(self, shares):
        """Randomly shuffle the shares to prevent correlation across rounds."""
        np.random.shuffle(shares)
        return shares

    def reconstruct_smc(self, shares):
        """Reconstruct the gradient from its shares."""
        if self.smc_flag:
            return sum(shares)
        return shares

    def train(self):
        """Main training loop for federated learning."""
        print(f'Training with {self.clients_per_round} clients using '
              f'{"DP" if self.dp_flag else "HE" if self.he_flag else "SMC" if self.smc_flag else "no privacy"}.')
        
        num_clients = len(self.clients)
        pk = np.ones(num_clients) / num_clients  # Uniform probability distribution for client selection

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

            for client in selected_clients:
                client.set_params(self.latest_model)
                weights_before = client.get_params()

                if weights_before is None:
                    print(f"Error: weights_before is None for client {client}.")
                    continue

                loss = client.get_loss()
                soln, stats = client.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                new_weights = soln[1]

                # Compute gradients
                grads = [(w_before - w_after) / self.learning_rate for w_before, w_after in zip(weights_before, new_weights)]

                # Apply SMC (if enabled)
                grads_smc = [self.apply_smc(grad, round_num=round_num) for grad in grads]
                grads_reconstructed = [self.reconstruct_smc(grad) for grad in grads_smc]

                # q-FedSGD weighting for each client's contribution
                Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads_reconstructed])

                # Compute norm for the aggregation weighting
                norm_grad_sum = np.sum([np.sum(np.square(grad)) for grad in grads_reconstructed])
                hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad_sum + (1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))

            # Aggregate updated gradients
            self.latest_model = self.aggregate2(weights_before, Deltas, hs)

        # Log final variance and Euclidean distance
        final_variance = np.mean(all_rounds_variances)
        final_euclidean_distance = np.mean(all_rounds_distances)

        print(f"\nFinal Average Variance in Testing Accuracy: {final_variance:.4f}")
        print(f"Final Average Euclidean Distance in Testing Accuracy: {final_euclidean_distance:.4f}")

        # Save metrics to files
        np.savetxt(self.output + "_final_variances.csv", np.array(all_rounds_variances), delimiter=",")
        np.savetxt(self.output + "_final_euclidean_distances.csv", np.array(all_rounds_distances), delimiter=",")
        print("Metrics saved successfully.")

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
