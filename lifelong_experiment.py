import numpy as np
import torch
import argparse
import os

from datasets import dataset

from models.mlp import MLP
from models.mlp_soft_lifelong import MLPSoftLL
from models.mlp_soft_lifelong_dynamic import MLPSoftLLDynamic
from models.mlp_soft_gated_lifelong import MLPSoftGatedLL
from models.mlp_soft_gated_lifelong_dynamic import MLPSoftGatedLLDynamic
from models.linear import Linear
from models.linear_factored import LinearFactored

# Explicitly compositional with dynamic module number
from learners.er_dynamic import CompositionalDynamicER
from learners.ewc_dynamic import CompositionalDynamicEWC
from learners.nft_dynamic import CompositionalDynamicNFT

# Explicitly compositional
from learners.er_compositional import CompositionalER
from learners.nft_compositional import CompositionalNFT
from learners.ewc_compositional import CompositionalEWC

# Implicitly compositional baselines (composition in the model, not in training)
from learners.ewc_joint import JointEWC
from learners.er_joint import JointER
from learners.nft_joint import JointNFT

# No-components baselines (no composition in the model or in training)
from learners.ewc_nocomponents import NoComponentsEWC
from learners.er_nocomponents import NoComponentsER
from learners.nft_nocomponents import NoComponentsNFT


def main(num_tasks=10,
        num_epochs=100,
        batch_size=64,
        component_update_frequency=100,
        ewc_lambda=1e-5,
        replay_size=-1,
        layer_size=64,
        num_layers=4,
        num_init_tasks=4,
        init_mode='random_onehot',
        architecture='mlp',
        algorithm='er_compositional',
        num_seeds=1,
        results_root='./tmp/results',
        save_frequency=1,
        initial_seed=0,
        num_train=-1,
        regression=True,
        freeze_encoder=True,
        paths=['./datasets/gripper_and_low_friction_large_arm.csv', 
                './datasets/gripper_and_low_friction.csv', 
                './datasets/no_gripper_and_high_friction.csv']):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seed in range(initial_seed, initial_seed + num_seeds):
        print('\nSEED: ', seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        torch_dataset = dataset.MyDataset(num_tasks=num_tasks, paths=paths)
        freeze_encoder = freeze_encoder
        
        if architecture == 'mlp':
            if 'dynamic' in algorithm:
                net = MLPSoftLLDynamic(torch_dataset.features,
                                size=layer_size,
                                depth=num_layers,
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                max_components=-1,
                                init_ordering_mode=init_mode,
                                device=device,
                                regression=regression,
                                freeze_encoder=freeze_encoder)
            elif 'nocomponents' in algorithm:
                net = MLP(torch_dataset.features,
                        size=layer_size,
                        depth=num_layers,
                        num_tasks=num_tasks,
                        num_init_tasks=num_init_tasks,
                        device=device,
                        regression=regression,
                        freeze_encoder=freeze_encoder)
            else:
                net = MLPSoftLL(torch_dataset.features,
                                size=layer_size,
                                depth=num_layers,
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                init_ordering_mode=init_mode,
                                device=device,
                                regression=regression,
                                freeze_encoder=freeze_encoder)
        elif architecture == 'mlp_gated':
            if 'dynamic' in algorithm:
                net = MLPSoftGatedLLDynamic(torch_dataset.features,
                                size=layer_size,
                                depth=num_layers,
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                max_components=-1,
                                init_ordering_mode=init_mode,
                                device=device,
                                regression=regression,
                                freeze_encoder=freeze_encoder)
            else:
                net = MLPSoftGatedLL(torch_dataset.features,
                                size=layer_size,
                                depth=num_layers,
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                init_ordering_mode=init_mode,
                                device=device,
                                regression=regression,
                                freeze_encoder=freeze_encoder)
        elif architecture == 'linear':
            if 'nocomponents' in algorithm:
                net = Linear(torch_dataset.features,
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                regression=regression,
                                device=device)
            else:
                net = LinearFactored(torch_dataset.features,
                                depth=num_layers, 
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                init_ordering_mode=init_mode,
                                regression=regression,
                                device=device)
            # Ignore the batch_size in the arguments (batch learning)
            batch_size = torch_dataset.max_batch_size
            
        else:
            raise NotImplementedError('{} architecture is not supported'.format(architecture))

        net.train()     # training mode
        kwargs = {}

        if num_train == - 1:
            results_dir=os.path.join(results_root, algorithm, 'seed_{}'.format(seed))
        else:
            results_dir=os.path.join(results_root, algorithm, 'seed_{}'.format(seed), 'numtrain_{}'.format(num_train))

        if algorithm == 'er_compositional':
            if replay_size == -1:
                replay_size = batch_size
            agent = CompositionalER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_compositional':
            agent = CompositionalEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'nft_compositional':
            agent = CompositionalNFT(net, results_dir=results_dir)

        elif algorithm == 'er_joint':
            if replay_size == -1:
                replay_size = batch_size
            agent = JointER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_joint':
            agent = JointEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'nft_joint':
            agent = JointNFT(net, results_dir=results_dir)

        elif algorithm == 'er_nocomponents':
            if replay_size == -1:
                replay_size = batch_size
            agent = NoComponentsER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_nocomponents':
            agent = NoComponentsEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'nft_nocomponents':
            agent = NoComponentsNFT(net, results_dir=results_dir)

        elif algorithm == 'er_dynamic':
            if replay_size == -1:
                replay_size = batch_size
            agent = CompositionalDynamicER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_dynamic':
            agent = CompositionalDynamicEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'nft_dynamic':
            agent = CompositionalDynamicNFT(net, results_dir=results_dir)

        else:
            raise NotImplementedError('{} algorithm is not supported'.format(algorithm))

        for task_id, trainset in enumerate(torch_dataset.trainset):
            trainloader = (
                torch.utils.data.DataLoader(trainset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    ))
            testloaders = {task: torch.utils.data.DataLoader(testset,
                                batch_size=torch_dataset.max_batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                ) for task, testset in enumerate(torch_dataset.testset[:(task_id+1)])}

            if 'dynamic' in algorithm:
                valloader = torch.utils.data.DataLoader(torch_dataset.valset[task_id],
                                batch_size=torch_dataset.max_batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                )
                kwargs = {'valloader': valloader}

            agent.train(trainloader, 
                task_id, 
                component_update_freq=component_update_frequency,
                num_epochs=num_epochs,
                testloaders=testloaders,
                save_freq=save_frequency,
                **kwargs)
            

if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong compositional learning')
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=3, type=int)
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=30, type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('-f', '--update_frequency', dest='component_update_frequency', default=30, type=int)
    parser.add_argument('--lambda', dest='ewc_lambda', default=1e-4, type=float) 
    parser.add_argument('--replay', dest='replay_size', default=-1, type=int)
    parser.add_argument('-s', '--layer_size', dest='layer_size', default=64, type=int)
    parser.add_argument('-l', '--num_layers', dest='num_layers', default=4, type=int)
    parser.add_argument('-k', '--init_tasks', dest='num_init_tasks', default=1, type=int)
    parser.add_argument('-i', '--init_mode', dest='init_mode', default='random', choices=['random_onehot', 'one_module_per_task', 'random'])
    parser.add_argument('-arc', '--architecture', dest='arch', default='mlp_gated', 
        choices=['mlp', 'mlp_gated', 'linear'])
    parser.add_argument('-alg', '--algorithm', dest='algo', default='er_compositional', 
        choices=['er_compositional', 'ewc_compositional', 'nft_compositional',
                'er_joint', 'ewc_joint', 'nft_joint',
                'er_nocomponents', 'ewc_nocomponents', 'nft_nocomponents',
                'er_dynamic', 'ewc_dynamic', 'nft_dynamic'])
    parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=1, type=int)
    parser.add_argument('-r', '--results_root', dest='results_root', default='./results/results_gated')
    parser.add_argument('-sf', '--save_frequency', dest='save_frequency', default=1, type=int)
    parser.add_argument('--initial_seed', dest='initial_seed', default=0, type=int)
    parser.add_argument('--num_train', dest='num_train', default=-1, type=int)
    parser.add_argument('--freeze_encoder', dest='freeze_encoder', default=True, type=bool)
    parser.add_argument('--regression', dest='regression', default=[True, True, True, True, True, True, False, False], type=bool)
    args = parser.parse_args()

    print('Will train on {} tasks for {} epochs.'.format(args.num_tasks, args.num_epochs))
    print('The batch size will be {} and the modules will be updated every {} iterations'.format(args.batch_size, args.component_update_frequency))
    print('The network will contain {} layers of size {}'.format(args.num_layers, args.layer_size))
    print('The first {} tasks will be used to initialize the modules in mode {}'.format(args.num_init_tasks, args.init_mode))
    print('Experiments will be repeated for {} random seeds, starting at {}'.format(args.num_seeds, args.initial_seed))
    print('Results will be stored in {}'.format(os.path.join(args.results_root, args.algo)))

    main(num_tasks=args.num_tasks,
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