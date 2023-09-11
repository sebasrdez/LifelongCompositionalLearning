import os
import argparse
import torch

from lifelong_experiment import main as lifelong_experiment

algorithms = ['er_compositional', 'ewc_compositional', 'nft_compositional']
algorithms += ['er_joint', 'ewc_joint', 'nft_joint']
algorithms += ['er_nocomponents', 'ewc_nocomponents', 'nft_nocomponents']
algorithms += ['er_dynamic', 'ewc_dynamic', 'nft_dynamic']

num_seeds = 3
num_epochs = 180
update_frequency = 180
ewc_lambda = 1e-4
layer_size = 64
num_layers = 3
num_init_tasks = 1
freeze_encoder = True

initial_seed = 0
num_tasks = 3
batch_size = 128
init_mode = 'random'
results_root = './results/results_gated'
architecture = 'mlp_gated'
save_frequency = 1
replay_size = -1
regression = [True, True, True, True, True, True, False, False]
num_train = -1

for a in algorithms:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=num_tasks, type=int)
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=num_epochs, type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=batch_size, type=int)
    parser.add_argument('-f', '--update_frequency', dest='component_update_frequency', default=update_frequency, type=int)
    parser.add_argument('--lambda', dest='ewc_lambda', default=ewc_lambda, type=float) 
    parser.add_argument('--replay', dest='replay_size', default=replay_size, type=int)
    parser.add_argument('-s', '--layer_size', dest='layer_size', default=layer_size, type=int)
    parser.add_argument('-l', '--num_layers', dest='num_layers', default=num_layers, type=int)
    parser.add_argument('-k', '--init_tasks', dest='num_init_tasks', default=num_init_tasks, type=int)
    parser.add_argument('-i', '--init_mode', dest='init_mode', default=init_mode, choices=['random_onehot', 'one_module_per_task', 'random'])
    parser.add_argument('-arc', '--architecture', dest='arch', default=architecture, choices=['mlp', 'mlp_gated', 'linear'])
    parser.add_argument('-alg', '--algorithm', dest='algo', default=a)
    parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=num_seeds, type=int)
    parser.add_argument('-r', '--results_root', dest='results_root', default=results_root)
    parser.add_argument('-sf', '--save_frequency', dest='save_frequency', default=save_frequency, type=int)
    parser.add_argument('--initial_seed', dest='initial_seed', default=initial_seed, type=int)
    parser.add_argument('--num_train', dest='num_train', default=num_train, type=int)
    parser.add_argument('--freeze_encoder', dest='freeze_encoder', default=freeze_encoder, type=bool)
    parser.add_argument('--regression', dest='regression', default=regression, type=bool)
    args = parser.parse_args()

    print('\n\tAlgorithm: {}'.format(args.algo))

    lifelong_experiment(num_tasks=args.num_tasks,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        component_update_frequency=args.component_update_frequency,
        ewc_lambda=args.ewc_lambda,
        replay_size=args.replay_size,
        layer_size=args.layer_size,
        num_layers=args.num_layers,
        num_init_tasks=args.num_init_tasks,
        init_mode=args.init_mode,
        architecture=args.arch,
        algorithm=args.algo,
        num_seeds=args.num_seeds,
        results_root=args.results_root,
        save_frequency=args.save_frequency,
        initial_seed=args.initial_seed,
        num_train=args.num_train,
        freeze_encoder=args.freeze_encoder,
        regression=args.regression)


