import argparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22)
font = {'family' : 'normal',
        'size'   : 22}
import numpy as np
import os

def main(num_tasks_all, datasets, algorithms, num_seeds, num_init_tasks, num_epochs, save_frequency, results_root):
    """Creates a bar chart to compare the different methods on the test set."""
    
    num_tasks_all = [num_tasks_all]

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    name_order = {'ER Dynamic': 0,
                  'ER Compositional': 1,
                  'ER Joint': 2,
                  'ER Nocomponents': 3,
                  'EWC Dynamic': 4,
                  'EWC Compositional': 5,
                  'EWC Joint': 6,
                  'EWC Nocomponents': 7,
                  'NFT Dynamic': 8,
                  'NFT Compositional': 9,
                  'NFT Joint': 10,
                  'NFT Nocomponents': 11}

    version_map = {'Dynamic': 'Dyn. + Comp.',
                   'Compositional': 'Compositional',
                   'Joint': 'Joint',
                   'Nocomponents': 'No Comp.'}

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

    save_map = {
        'loss1': 'Cylinder_Distance',
        'loss2': 'Cylinder_Angle',
        'loss3': 'Cylinder_Diameter',
        'loss4': 'Box_Distance',
        'loss5': 'Box_Angle',
        'loss6': 'Box_Diameter',
        'loss7': 'Ball_in_Left_Hand',
        'loss8': 'Ball_in_Right_Hand'
    }

    jumpstart_vals_all_datasets = {}
    finetuning_vals_all_datasets = {}
    forward_transfer_vals_all_datasets = {}
    final_vals_all_datasets = {}
    jumpstart_errs_all_datasets = {}
    finetuning_errs_all_datasets = {}
    forward_transfer_errs_all_datasets = {}
    final_errs_all_datasets = {}

    for i, dataset in enumerate(datasets):
        num_tasks = num_tasks_all[i]
        jumpstart_vals = {}
        finetuning_vals = {}
        forward_transfer_vals = {}
        final_vals = {}
        jumpstart_vals_all_algos = {}
        finetuning_vals_all_algos = {}
        forward_transfer_vals_all_algos = {}
        final_vals_all_algos = {}
        jumpstart_errs_all_algos = {}
        finetuning_errs_all_algos = {}
        forward_transfer_errs_all_algos = {}
        final_errs_all_algos = {}
        names = []

        for algorithm in algorithms:
            jumpstart_vals[algorithm] = {}
            finetuning_vals[algorithm] = {}
            forward_transfer_vals[algorithm] = {}
            final_vals[algorithm] = {}

            for seed in range(num_seeds):
                prev_components = 3

                for task_id in range(num_tasks):
                    results_dir = os.path.join(results_root, algorithm, 'seed_{}'.format(seed),
                                               'task_{}'.format(task_id))

                    if 'dynamic' in algorithm and task_id >= num_init_tasks:
                        with open(os.path.join(results_dir, 'num_components.txt')) as f:
                            line = f.readline()
                            curr_components = int(line.lstrip('final components: '))
                            keep_component = curr_components > prev_components
                            prev_components = curr_components

                    with open(os.path.join(results_dir, 'log.txt')) as f:
                        ##### JUMPSTART #########
                        next(f)
                        for task in range(task_id):
                            next(f)
                        line = f.readline()
                        line = line.rstrip('\n')
                        i_0 = len('\ttask: {}\t'.format(task_id))

                        while i_0 != -1:
                            i_f = line.find(':', i_0)
                            key = line[i_0: i_f]

                            if task_id == 0 and seed == 0:
                                jumpstart_vals[algorithm][key] = np.zeros((num_seeds, num_tasks))
                                finetuning_vals[algorithm][key] = np.zeros((num_seeds, num_tasks))
                                forward_transfer_vals[algorithm][key] = np.zeros((num_seeds, num_tasks))
                                final_vals[algorithm][key] = np.zeros((num_seeds, num_tasks))

                                if key not in jumpstart_vals_all_algos:
                                    jumpstart_vals_all_algos[key] = []
                                    finetuning_vals_all_algos[key] = []
                                    forward_transfer_vals_all_algos[key] = []
                                    final_vals_all_algos[key] = []
                                    jumpstart_errs_all_algos[key] = []
                                    finetuning_errs_all_algos[key] = []
                                    forward_transfer_errs_all_algos[key] = []
                                    final_errs_all_algos[key] = []

                                if key not in jumpstart_vals_all_datasets:
                                    jumpstart_vals_all_datasets[key] = {}
                                    finetuning_vals_all_datasets[key] = {}
                                    forward_transfer_vals_all_datasets[key] = {}
                                    final_vals_all_datasets[key] = {}
                                    jumpstart_errs_all_datasets[key] = {}
                                    finetuning_errs_all_datasets[key] = {}
                                    forward_transfer_errs_all_datasets[key] = {}
                                    final_errs_all_datasets[key] = {}

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

                            jumpstart_vals[algorithm][key][seed, task_id] = val
                            i_0 = i_f if i_f == - 1 else i_f + 1

                        if task_id < num_init_tasks - 1:
                            continue

                        ###### IGNORE FINTEUNING PROCESS #########
                        if '_compositional' in algorithm or '_dynamic' in algorithm:
                            stop_at = num_epochs - save_frequency
                        else:
                            stop_at = num_epochs

                        for epoch in range(1, stop_at, save_frequency):
                            try:
                                next(f)    # epochs: 100, training task: 9
                            except StopIteration:
                                print(dataset, algorithm, seed, task_id, epoch)
                                raise
                            for task in range(task_id + 1):
                                next(f)

                        ###### FETUNING ###########
                        next(f)

                        if task_id == num_init_tasks - 1:
                            start_loop_at = 0
                        elif task_id == num_tasks - 1 and '_compositional' not in algorithm and '_dynamic' not in algorithm:
                            start_loop_at = 0
                        else:
                            start_loop_at = task_id

                        for task in range(start_loop_at):
                            next(f)

                        for task in range(start_loop_at, task_id + 1):
                            line = f.readline()
                            line = line.rstrip('\n')
                            i_0 = len('\ttask: {}\t'.format(task))

                            while i_0 != -1:
                                i_f = line.find(':', i_0)
                                key = line[i_0: i_f]
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

                                if task == task_id or task_id == num_init_tasks - 1:
                                    finetuning_vals[algorithm][key][seed, task] = val

                                if task_id == num_tasks - 1 and '_compositional' not in algorithm and '_dynamic' not in algorithm:
                                    final_vals[algorithm][key][seed, task] = val

                                i_0 = i_f if i_f == - 1 else i_f + 1

                        ####### FORWARD TRANSFER #######
                        if ('_compositional' in algorithm or '_dynamic' in algorithm) and task_id != num_init_tasks - 1:
                            if task_id == num_tasks - 1:
                                start_loop_at = 0

                            next(f)

                            for task in range(start_loop_at):
                                next(f)

                            for task in range(start_loop_at, task_id + 1):
                                line = f.readline()
                                line = line.rstrip('\n')
                                i_0 = len('\ttask: {}\t'.format(task))

                                while i_0 != -1:
                                    i_f = line.find(':', i_0)
                                    key = line[i_0: i_f]
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

                                    if task == task_id:
                                        forward_transfer_vals[algorithm][key][seed, task] = val

                                    if task_id == num_tasks - 1:
                                        final_vals[algorithm][key][seed][task] = val

                                    i_0 = i_f if i_f == - 1 else i_f + 1
                        else:
                            for task in range(start_loop_at, task_id + 1):
                                for key in finetuning_vals[algorithm]:
                                    forward_transfer_vals[algorithm][key][seed, task] = finetuning_vals[algorithm][key][seed, task]

            for key in jumpstart_vals[algorithm]:
                jumpstart_vals_all_algos[key].append(jumpstart_vals[algorithm][key].mean())
                jumpstart_errs_all_algos[key].append(jumpstart_vals[algorithm][key].mean(axis=1).std())
                finetuning_vals_all_algos[key].append(finetuning_vals[algorithm][key].mean())
                finetuning_errs_all_algos[key].append(finetuning_vals[algorithm][key].mean(axis=1).std())
                forward_transfer_vals_all_algos[key].append(forward_transfer_vals[algorithm][key].mean())
                forward_transfer_errs_all_algos[key].append(forward_transfer_vals[algorithm][key].mean(axis=1).std())
                final_vals_all_algos[key].append(final_vals[algorithm][key].mean())
                final_errs_all_algos[key].append(final_vals[algorithm][key].mean(axis=1).std())

            names.append(algorithm.split('_')[0].upper() + ' ' + algorithm.split('_')[1].title())

        idx = [x[0] for x in sorted(enumerate(names), key=lambda x: name_order[x[1]])]
        names = np.array(names)[idx]

        for key in jumpstart_vals_all_algos:
            # Sort by names to group by base algorithm
            jumpstart_vals_all_algos[key] = np.array(jumpstart_vals_all_algos[key])[idx]
            jumpstart_errs_all_algos[key] = np.array(jumpstart_errs_all_algos[key])[idx] / np.sqrt(num_seeds)
            finetuning_vals_all_algos[key] = np.array(finetuning_vals_all_algos[key])[idx]
            finetuning_errs_all_algos[key] = np.array(finetuning_errs_all_algos[key])[idx] / np.sqrt(num_seeds)
            forward_transfer_vals_all_algos[key] = np.array(forward_transfer_vals_all_algos[key])[idx]
            forward_transfer_errs_all_algos[key] = np.array(forward_transfer_errs_all_algos[key])[idx] / np.sqrt(num_seeds)
            final_vals_all_algos[key] = np.array(final_vals_all_algos[key])[idx]
            final_errs_all_algos[key] = np.array(final_errs_all_algos[key])[idx] / np.sqrt(num_seeds)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

        for key in jumpstart_vals_all_algos:
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(111)
            w = 0.3
            y_pos = np.arange(len(names))

            # Group by base algorithm
            base_counts = np.array([sum(x.startswith('ER') for x in names),
                                    sum(x.startswith('EWC') for x in names),
                                    sum(x.startswith('NFT') for x in names)])

            ax.bar(y_pos - w, forward_transfer_vals_all_algos[key], yerr=forward_transfer_errs_all_algos[key],
                   label='Forward', width=w, align='edge', capsize=10, color=colors[2])
            ax.bar(y_pos, final_vals_all_algos[key], yerr=final_errs_all_algos[key],
                   label='Final', width=w, align='edge', capsize=10, color=colors[3], alpha=0.7)
            plt.xticks(y_pos, [version_map[x.split(' ')[1]] for x in names], rotation=45, ha='right')

            for i, cnt in enumerate(np.cumsum(base_counts)):
                if i < len(base_counts) - 1:
                    ax.axvline(x=cnt - 0.5, c='red', linestyle='--', linewidth=4)
                x = cnt - (base_counts[i] / 2) - 0.75
                _, top = ax.get_ylim()
                y = top * 1.01
                ax.text(x, y, ['ER', 'EWC', 'NFT'][i], fontdict={'fontsize': 16})

            plt.ylabel(ylabel_map[key], fontsize=22, labelpad=50)
            plt.title(save_map[key], fontsize=22, y=1.1)
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            ax.tick_params(axis='y', which='both', right=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
            plt.ylim(0, 1)
            plt.legend(fontsize=16, loc='upper right')
            plt.tight_layout()

            save_dir = os.path.join(results_root, 'plots')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, '{}.pdf'.format(save_map[key])))
            plt.close()

        jumpstart_vals_all_datasets[dataset] = jumpstart_vals_all_algos
        finetuning_vals_all_datasets[dataset] = finetuning_vals_all_algos
        forward_transfer_vals_all_datasets[dataset] = forward_transfer_vals_all_algos
        final_vals_all_datasets[dataset] = final_vals_all_algos
        jumpstart_errs_all_datasets[dataset] = jumpstart_errs_all_algos
        finetuning_errs_all_datasets[dataset] = finetuning_errs_all_algos
        forward_transfer_errs_all_datasets[dataset] = forward_transfer_errs_all_algos
        final_errs_all_datasets[dataset] = final_errs_all_algos

    # Plot the average of all variables
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    w = 0.3
    y_pos = np.arange(len(names))

    mean_forward_transfer_vals = np.mean(list(forward_transfer_vals_all_algos.values()), axis=0)
    mean_final_vals = np.mean(list(final_vals_all_algos.values()), axis=0)
    mean_forward_transfer_errs = np.mean(list(forward_transfer_errs_all_algos.values()), axis=0)
    mean_final_errs = np.mean(list(final_errs_all_algos.values()), axis=0)

    ax.bar(y_pos - w, mean_forward_transfer_vals, yerr=mean_forward_transfer_errs, label='Forward', width=w,
           align='edge', capsize=10, color=colors[2])
    ax.bar(y_pos, mean_final_vals, yerr=mean_final_errs, label='Final', width=w, align='edge', capsize=10,
           color=colors[3], alpha=0.7)
    plt.xticks(y_pos, [version_map[x.split(' ')[1]] for x in names], rotation=45, ha='right')

    for i, cnt in enumerate(np.cumsum(base_counts)):
        if i < len(base_counts) - 1:
            ax.axvline(x=cnt - 0.5, c='red', linestyle='--', linewidth=4)
        x = cnt - (base_counts[i] / 2) - 0.75
        _, top = ax.get_ylim()
        y = top * 1.01
        ax.text(x, y, ['ER', 'EWC', 'NFT'][i], fontdict={'fontsize': 16})

    plt.ylabel('Average Loss', fontsize=22, labelpad=50)
    plt.title('Average Loss of All Variables', fontsize=22, y=1.1)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', right=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1)
    plt.legend(fontsize=16, loc='upper right')

    save_dir = os.path.join(results_root, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'mean_loss.pdf'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot forward and final loss for lifelong compositional learning')
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=3, type=int, nargs='+')
    parser.add_argument('-d', '--datasets', dest='datasets', default='MyDataset', nargs='+')
    parser.add_argument('-alg', '--algorithms', dest='algorithms', default=['er_compositional', 'ewc_compositional', 'nft_compositional',
                                                                        'er_joint', 'ewc_joint', 'nft_joint',
                                                                        'er_nocomponents', 'ewc_nocomponents', 'nft_nocomponents',
                                                                        'er_dynamic', 'ewc_dynamic', 'nft_dynamic'], nargs='+')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=180, type=int)
    parser.add_argument('-sf', '--save_frequency', dest='save_frequency', default=1, type=int)
    parser.add_argument('-k', '--init_tasks', dest='num_init_tasks', default=1, type=int)
    parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=1, type=int)
    parser.add_argument('-r', '--results_root', dest='results_root', default='./results/results_gated')
    args = parser.parse_args()

    main(args.num_tasks,
        args.datasets,
        args.algorithms,
        args.num_seeds,
        args.num_init_tasks,
        args.num_epochs,
        args.save_frequency,
        args.results_root)
