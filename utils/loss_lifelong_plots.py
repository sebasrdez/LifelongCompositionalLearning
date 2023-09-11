import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=34)
matplotlib.rc('ytick', labelsize=34)
font = {'family': 'normal', 'size': 34}
import numpy as np
import os

def main(num_tasks_all, datasets, algorithms, num_seeds, num_init_tasks, num_epochs, save_frequency, results_root):
    """Create the loss evolution graphs throughout the training in each task."""

    plot_tasks = num_tasks_all

    num_tasks_all = [num_tasks_all]

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    title_map = {
        'loss1': 'Cylinder Distance',
        'loss2': 'Cylinder Angle',
        'loss3': 'Cylinder Diameter',
        'loss4': 'Box Distance',
        'loss5': 'Box Angle',
        'loss6': 'Box Diameter',
        'loss7': 'Ball in Left Hand',
        'loss8': 'Ball in Right Hand'
    }
    
    ylabel_map = {
        'loss1': 'RMSE',
        'loss2': 'RMSE',
        'loss3': 'RMSE',
        'loss4': 'RMSE',
        'loss5': 'RMSE',
        'loss6': 'RMSE',
        'loss7': 'BCEWithLogits',
        'loss8': 'BCEWithLogits'
    }

    for algorithm in algorithms:
        for i, dataset in enumerate(datasets):
            learning_curves = {}
            num_tasks = num_tasks_all[i]
            task_step = num_tasks // plot_tasks
            num_iter = num_epochs * (num_tasks - num_init_tasks + 1) + 1   # bundle all init tasks together, add 1 epoch to show final tasks' final performance
            x_axis = np.arange(0, num_iter, save_frequency)
            for seed in range(num_seeds):
                iter_cnt = 0
                prev_components = 3
                for task_id in range(num_init_tasks - 1, num_tasks):
                    results_dir = os.path.join(results_root, algorithm, 'seed_{}'.format(seed), 'task_{}'.format(task_id))
                    if 'dynamic' in algorithm and task_id >= num_init_tasks:
                        with open(os.path.join(results_dir, 'num_components.txt')) as f:
                            line = f.readline()
                            curr_components = int(line.lstrip('final components: '))
                            keep_component = curr_components > prev_components
                            prev_components = curr_components
                    with open(os.path.join(results_dir, 'log.txt')) as f:
                        for epoch in range(0, num_epochs + (1 if task_id == num_tasks - 1 else 0), save_frequency):
                            try:
                                next(f)    # epochs: 100, training task: 9
                            except StopIteration:
                                print(algorithm, seed, task_id, epoch)
                                raise
                            for task in range(task_id + 1):
                                line = f.readline()
                                if task % task_step != 0:
                                    continue
                                line = line.rstrip('\n')
                                i_0 = len('\ttask: {}\t'.format(task))
                                while i_0 != -1:
                                    i_f = line.find(':', i_0)
                                    key = line[i_0: i_f]
                                    if key not in learning_curves and iter_cnt == 0 and seed == 0:
                                        learning_curves[key] = np.full((num_seeds, plot_tasks, num_iter // save_frequency), np.nan)
                                    i_0 = line.find(key + ': ', i_0) + len(key + ': ')
                                    i_f = line.find('\t', i_0)
                                    substr = line[i_0: i_f] if i_f != 0 else line[i_0:]
                                    try:
                                        val = float(substr)
                                    except:
                                        if keep_component:
                                            val = float(substr.split(',')[0].lstrip('('))
                                        else:
                                            val = float(substr.split(',')[1].rstrip(')'))
                                    learning_curves[key][seed, task // task_step, iter_cnt] = val
                                    i_0 = i_f if i_f == -1 else i_f + 1
                            iter_cnt += 1

            for key in learning_curves:
                mean_val = learning_curves[key].mean(axis=0).T      # each column is treated as a dataset
                plt.figure(figsize=(12, 7.8))
                plt.subplot(111)
                plt.plot(x_axis, mean_val, linewidth=4)
                plt.grid(axis='y')
                plt.ylabel(ylabel_map[key], fontsize=34)
                plt.xlabel('# epochs', fontsize=34)
                plt.title(algorithm.split('_')[0].upper() + algorithm.split('_')[1].title() + ' - ' + title_map[key], fontsize=34, y=1.08)
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.savefig(os.path.join(results_root, algorithm, title_map[key] + '.pdf'), transparent=True)
                plt.close()

            # Plot mean of all variables
            mean_all = np.mean(np.stack([learning_curves[key].mean(axis=0).T for key in learning_curves]), axis=0)
            std_all = np.std(np.stack([learning_curves[key].mean(axis=0).T for key in learning_curves]), axis=0)
            plt.figure(figsize=(12, 7.8))
            plt.subplot(111)
            for i in range(mean_all.shape[1]):
                plt.plot(x_axis, mean_all[:, i], linewidth=4)
                plt.fill_between(x_axis, mean_all[:, i] - std_all[:, i], mean_all[:, i] + std_all[:, i], alpha=0.2)
            plt.grid(axis='y')
            plt.ylabel('Mean Loss', fontsize=34)
            plt.xlabel('# epochs', fontsize=34)
            plt.ylim(0, 1)
            plt.yticks(np.arange(0, 1, 0.1))
            plt.gca().yaxis.grid(True)
            plt.title(algorithm.split('_')[0].upper() + algorithm.split('_')[1].title(), fontsize=34, y=1.08)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.savefig(os.path.join(results_root, algorithm, 'mean_loss.pdf'), transparent=True)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss vs number of training iterations for lifelong compositional learning')
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=3, type=int, nargs='+')
    parser.add_argument('-d', '--datasets', dest='datasets', default='MyDataset', nargs='+')
    parser.add_argument('-alg', '--algorithms', dest='algos',
                        default=['er_compositional', 'ewc_compositional', 'nft_compositional', 
                                 'er_joint', 'ewc_joint', 'nft_joint', 
                                 'er_nocomponents', 'ewc_nocomponents', 'nft_nocomponents', 
                                 'er_dynamic', 'ewc_dynamic', 'nft_dynamic'], nargs='+')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=180, type=int)
    parser.add_argument('-sf', '--save_frequency', dest='save_frequency', default=1, type=int)
    parser.add_argument('-k', '--init_tasks', dest='num_init_tasks', default=1, type=int)
    parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=5, type=int)
    parser.add_argument('-r', '--results_root', dest='results_root', default='./tmp/results_gated')
    args = parser.parse_args()

    main(args.num_tasks, args.datasets, args.algos, args.num_seeds, args.num_init_tasks, args.num_epochs,
         args.save_frequency, args.results_root)
