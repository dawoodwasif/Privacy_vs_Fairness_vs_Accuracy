import numpy as np
import argparse
import importlib
import random
import os
import tensorflow.compat.v1 as tf
from flearn.utils.model_utils import read_data


# GLOBAL PARAMETERS
OPTIMIZERS = ['qffedsgd', 'qffedavg', 'afl', 'maml', 'ditto']
DATASETS = [ 'synthetic', 'vehicle', 'sent140', 'shakespeare',
'synthetic_iid', 'synthetic_hybrid', 
'fmnist', 'adult', 'omniglot', 'mnist', 'mri_iid', 'mri_non_iid']   


MODEL_PARAMS = {
    'adult.lr': (2, ), # num_classes,
    'adult.lr_afl': (2, ), # num_classes,
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'fmnist.lr': (3,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'mri.cnn': (2,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, num_class num_hidden
    'synthetic.mclr': (10, ), # num_classes
    'vehicle.svm':(2, ), # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                    help='name of optimizer;',
                    type=str,
                    choices=OPTIMIZERS,
                    default='qffedavg')
    parser.add_argument('--dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='nist')
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval_every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients_per_round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--num_epochs', 
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1) 
    parser.add_argument('--learning_rate',
                    help='learning rate for inner solver;',
                    type=float,
                    default=0.003)
    parser.add_argument('--seed',
                    help='seed for random initialization;',
                    type=int,
                    default=0)
    parser.add_argument('--sampling',
                    help='client sampling methods',
                    type=int,
                    default='5') # uniform sampling + weighted average
    parser.add_argument('--q',
                    help='reweighting factor',
                    type=float,
                    default='0.0') # no weighting, the same as fedavg
    parser.add_argument('--output',
                    help='file to save the final accuracy across all devices',
                    type=str,
                    default='output ') 
    parser.add_argument('--learning_rate_lambda',
                    help='learning rate for lambda in agnostic flearn',
                    type=float,
                    default=0)
    parser.add_argument('--log_interval',
                    help='intervals (how many rounds) to output accuracy distribution (data dependent',
                    type=int,
                    default=10)
    parser.add_argument('--data_partition_seed',
                    help='seed for splitting data into train/test/validation',
                    type=int,
                    default=1)
    parser.add_argument('--static_step_size',
                    help='whether to use our method or use a best tuned step size FedSGD to solve q-FFL',
                    type=int,
                    default=0)  # default is using our method
    parser.add_argument('--track_individual_accuracy',
                    help='whether to track each device\'s accuracy, only true when comparing with AFL',
                    type=int,
                    default=0)  
    parser.add_argument('--held_out',
                    help="number of held out devices/tasks",
                    type=int,
                    default=0)
    parser.add_argument('--num_fine_tune',
                    help="number of fine-tuning iterations",
                    type=int,
                    default=0)
    parser.add_argument('--with_maml',
                    help="whether to learn better intializations or use finetuning baseline",
                    type=int,
                    default=0)
    parser.add_argument('--run_number',
                    help="ru number of mri datasets",
                    type=int,
                    default=1)
    


    # Differential Privacy (DP) parameters
    parser.add_argument('--dp_epsilon',
                        help='epsilon value for differential privacy;',
                        type=float,
                        default=16.0)
    parser.add_argument('--dp_delta',
                        help='delta value for differential privacy;',
                        type=float,
                        default=1e-5)
    parser.add_argument('--dp_sensitivity',
                        help='sensitivity value for differential privacy;',
                        type=float,
                        default=1.0)
    parser.add_argument('--dp_mechanism',
                        help='mechanism for differential privacy (laplace, gaussian, randomized_response, exponential);',
                        type=str,
                        default='gaussian')
    parser.add_argument('--dp_flag',
                        help='flag for dp;',
                        type=bool,
                        default=False)
    

    # Homomorphic Encryption (HE) parameters
    parser.add_argument('--he_flag',
                        help='flag to enable homomorphic encryption (HE);',
                        type=bool,
                        default=False)
    parser.add_argument('--he_poly_modulus_degree',
                        help='degree of the polynomial modulus (e.g., 4096, 8192, 16384);',
                        type=int,
                        default=16384
                        )
    parser.add_argument('--he_coeff_mod_bit_sizes',
                        help='bit sizes for each coefficient modulus; should be a list (e.g., [60, 40, 40, 60]);',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default=[60, 40, 40, 60]
                        )
    parser.add_argument('--he_global_scale',
                        help='global scale used in CKKS (usually a power of 2, e.g., 2^40);',
                        type=float,  
                        default=2 ** 40)
    parser.add_argument('--he_encrypt_layers',
                        help='number of layers to encrypt ;',
                        type=int,
                        default=5)                      
    parser.add_argument('--he_galois_keys',
                        help='flag to generate galois keys for HE rotations;',
                        type=bool,
                        default=True)
    

        # Secure Multi Party Computation (SMC) parameters
    parser.add_argument('--smc_flag',
                        help='flag to enable smc;',
                        type=bool,
                        default=False)
    parser.add_argument('--smc_threshold',
                        help='minimum shares are needed to reconstruct the secret,',
                        type=int,
                        default=7
                        )
    parser.add_argument('--smc_num_shares',
                        help='number of parties to distribute data among,',
                        type=int,
                        default=10
                        )
    
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))




    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    elif parsed['dataset'].startswith("mri"):  # MRI_AD Dataset has iid and non iid versions
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'mri', parsed['model'])
        parsed['dataset'] += f"/run{parsed['run_number']}" # change slash for each OS
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    # load selected trainer
    if parsed['optimizer'] in ['l2sgd', 'ditto', 'apfl', 'mapper', 'ewc', 'meta', 'kl']:
        opt_path = 'flearn.trainers_MTL.%s' % parsed['optimizer']
    else:
        opt_path = 'flearn.trainers.%s' % parsed['optimizer']

    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]


    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)


    dp_params = {
        'epsilon': parsed['dp_epsilon'],
        'delta': parsed['dp_delta'],
        'sensitivity': parsed['dp_sensitivity'],
        'mechanism': parsed['dp_mechanism'],
        'dp_flag': parsed['dp_flag']
    }

    he_params = {
        'he_flag': parsed['he_flag'],
        'poly_modulus_degree': parsed['he_poly_modulus_degree'],
        'coeff_mod_bit_sizes': parsed['he_coeff_mod_bit_sizes'],
        'global_scale': parsed['he_global_scale'],
        'he_encrypt_layers': parsed['he_encrypt_layers'],
        'galois_keys': parsed['he_galois_keys']
    }

    smc_params = {
        'smc_flag': parsed['smc_flag'],
        'smc_threshold': parsed['smc_threshold'],
        'smc_num_shares': parsed['smc_num_shares']
    }


    parsed['dp_params'] = dp_params

    parsed['he_params'] = he_params
     
    parsed['smc_params'] = smc_params

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(options, learner, dataset, dp_params=options['dp_params'], he_params=options['he_params'], smc_params=options['smc_params'])
    t.train()

    # Save the trained model
    save_model(t, options)

def save_model(trainer, options):
    """Saves the trained model to a specified file."""
    model_save_path = os.path.join("weights", "{}_model".format(options['optimizer']))
    
    # Check if the trainer has a method to get the latest model
    if hasattr(trainer, 'latest_model'):
        model_params = trainer.latest_model
    else:
        raise AttributeError("Trainer does not have 'latest_model' attribute.")

    # Save the model
    np.save(model_save_path, model_params)
    print("Model weights saved at {}".format(model_save_path))

if __name__ == '__main__':
    main()





