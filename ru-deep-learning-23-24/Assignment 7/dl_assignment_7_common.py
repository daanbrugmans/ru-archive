# This is a file where you should put your own functions
import math
import csv
from typing import Dict, Any, Iterator, List
from dataclasses import dataclass

import numpy as np
from d2l.torch import try_gpu
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn.utils.prune
from d2l import torch as d2l


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
def get_datasets(name: str, batch_size=60, used_data=1.0) -> dict[str, DataLoader[Any]]:
    """
    :param int batch_size: batch size to use for the DataLoaders
    :param str name: name of the dataset to load. Available are 'mnist' and 'fashionmnist'
    :param float used_data: percentage of total dataset to use
    """

    image_transformations = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda image: torch.flatten(image))
    ])

    if name == "mnist":
        train_and_val_data = MNIST(root=".", train=True, download=True, transform=image_transformations)
        test_data = MNIST(root=".", train=False, download=True, transform=image_transformations)
    elif name == "fashionmnist":
        train_and_val_data = FashionMNIST(root=".", train=True, download=True, transform=image_transformations)
        test_data = FashionMNIST(root=".", train=False, download=True, transform=image_transformations)
    elif name == 'cifar10':
        train_and_val_data = CIFAR10(root=".", train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_data = CIFAR10(root=".", train=False, download=True, transform=torchvision.transforms.ToTensor())
    else:
        raise ValueError(f"Dataset name {name} was not recognized")

    train_data_count = math.floor(0.9 * len(train_and_val_data) * used_data)
    validation_data_count = math.ceil(0.1 * len(train_and_val_data) * used_data)
    unused_data_count = len(train_and_val_data) - train_data_count - validation_data_count

    train_data, validation_data, _ = torch.utils.data.random_split(
        train_and_val_data, [train_data_count, validation_data_count, unused_data_count]
    )

    test_count = math.ceil(len(test_data) * used_data)
    unused_data_count = len(test_data) - test_count
    test_data, _ = torch.utils.data.random_split(
        test_data, [test_count, unused_data_count]
    )

    return {
        "train": DataLoader(train_data, batch_size, shuffle=True),
        "val": DataLoader(validation_data, batch_size),
        "test": DataLoader(test_data, batch_size)
    }


# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------

# TODO: Define network architectures here
def _create_lenet(in_features=784, num_classes=10):
    return nn.Sequential(
        # I could not find the activation function used in the paper, but other implementations
        # of LeNet seem to use sigmoid
        nn.Linear(in_features, 300), nn.Sigmoid(),
        nn.Linear(300, 100), nn.Sigmoid(),
        nn.Linear(100, num_classes)  # , nn.Softmax(dim=0)
    )


def _create_resnet18():
    return torchvision.models.resnet18(num_classes=10)


def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'lenet':
        net = _create_lenet(**kwargs)
    elif arch == 'resnet18':
        net = _create_resnet18()
    else:
        raise ValueError(f"Architecture name {arch} was not recognized")
    return net


# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

# TODO: Define training, testing and model loading here
def evaluate_loss(net: nn, data_iter, loss_fn, device):
    """Evaluate the loss of a model on the given dataset.

    Copied from d2l and slightly modified"""
    metric = d2l.Accumulator(2)  # Sum of losses, no. Of examples

    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)

        y_hat = net(x)

        loss = loss_fn(y_hat, y)

        metric.add(d2l.reduce_sum(loss), d2l.size(loss))

    return metric[0] / metric[1]


def evaluate(net, evaluation_set, loss_fn, device):
    loss = evaluate_loss(net, evaluation_set, loss_fn, device)
    accuracy = d2l.evaluate_accuracy_gpu(net, evaluation_set, device=device)

    return loss, accuracy


def train(net: nn.Module, data_loaders: Dict[str, DataLoader], optimizer, loss_fn,
          device: str, model_file_name: str, learning_rate: float = 0.0012, iterations: int = 50000,
          eval_every_n_iterations=100, graph=True, keep_checkpoints=True):
    # inspired by assignment 5
    """
    Trains the model net with data from the data_loaders['train'], data_loaders['val'], data_loaders['test'].
    """
    net = net.to(device)
    optimizer = optimizer(net.parameters(), lr=learning_rate)

    if graph:
        training_progression_animator = d2l.Animator(
            xlabel='iteration',
            legend=['train loss', 'train accuracy', 'validation loss', 'validation accuracy'],
            figsize=(10, 5)
        )
    min_val_loss = float("inf")
    iteration_min_val_loss = 0
    iteration_count = 0

    with open(f'checkpoints/{model_file_name}.csv', 'x', newline='') as csvfile:
        fieldnames = ['iteration', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        while True:
            # monitor loss, accuracy, number of samples
            # metrics = {'train': d2l.Accumulator(3), 'val': d2l.Accumulator(3)}
            for _, (x, y) in enumerate(data_loaders['train']):
                iteration_count += 1
                if iteration_count % eval_every_n_iterations == 0:
                    test_loss, test_acc = evaluate(net, data_loaders['test'], loss_fn, device)
                    val_loss, val_acc = evaluate(net, data_loaders['val'], loss_fn, device)
                    train_loss, train_acc = evaluate(net, data_loaders['train'], loss_fn, device)

                    csvwriter.writerow({'iteration': iteration_count, 'train_loss': train_loss, 'train_acc': train_acc,
                                        'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss,
                                        'test_acc': test_acc})

                    if keep_checkpoints:
                        path = f"checkpoints/model-{model_file_name}-{iteration_count}.pth"
                        torch.save(net.state_dict(), path)

                    if val_loss < min_val_loss:
                        path = f"checkpoints/model-{model_file_name}-best.pth"
                        torch.save(net.state_dict(), path)
                        min_val_loss = val_loss
                        iteration_min_val_loss = iteration_count

                    if graph:
                        training_progression_animator.add(iteration_count, (train_loss, train_acc, val_loss, val_acc))

                x = x.to(device)
                y = y.to(device)

                y_hat = net(x)

                loss = loss_fn(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration_count >= iterations:
                    break
            else:
                continue
            break

        test_loss, test_acc = evaluate(net, data_loaders['test'], loss_fn, device)
        val_loss, val_acc = evaluate(net, data_loaders['val'], loss_fn, device)
        train_loss, train_acc = evaluate(net, data_loaders['train'], loss_fn, device)

        csvwriter.writerow({'iteration': iteration_count, 'train_loss': train_loss, 'train_acc': train_acc,
                            'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss,
                            'test_acc': test_acc})

    path = f"checkpoints/model-{model_file_name}-final.pth"
    torch.save(net.state_dict(), path)

    print(f'train loss {train_loss:.3f}, train accuracy {train_acc:.3f}, '
          f'val loss {val_loss:.3f}, val accuracy {val_acc:.3f}, '
          f'test loss {test_loss:.3f}, test accuracy {test_acc:.3f}, '
          f'min val loss at iteration {iteration_min_val_loss}')

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

def prune_random(net: nn.Sequential, p: float):
    for layer, last in is_last(net.children()):
        if last:
            if isinstance(layer, nn.Linear):
                nn.utils.prune.random_unstructured(layer, 'weight', amount=p / 2)
        if isinstance(layer, nn.Linear):
            nn.utils.prune.random_unstructured(layer, 'weight', amount=p)


# TODO: Put functions related to pruning here
def copy_prune(mask_net: nn.Sequential, apply_net: nn.Sequential):
    for mask_layer, apply_layer in zip(mask_net.children(), apply_net.children()):
        if isinstance(mask_layer, nn.Linear):
            weight_mask = mask_layer.state_dict().get('weight_mask')
            nn.utils.prune.custom_from_mask(apply_layer, 'weight', weight_mask)


def prune(net: nn.Sequential, p: float):
    """
    Prunes a network in-place, i.e., does not return a new network.

    :param net: Network to prune
    :param p: percentage of weights to pune
    """
    for layer, last in is_last(net.children()):
        if last:
            if isinstance(layer, nn.Linear):
                nn.utils.prune.l1_unstructured(layer, 'weight', amount=p / 2)
        if isinstance(layer, nn.Linear):
            nn.utils.prune.l1_unstructured(layer, 'weight', amount=p)


def is_last(i: Iterator):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Copied from https://stackoverflow.com/a/1630350/21752100
    # Get an iterator and pull the first value.
    last = next(i)
    # Run the iterator to exhaustion (starting from the second value).
    for val in i:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False


def experiment_section_1(arch: str, dataset: str, optim, lr, fc_pruning_rate, conv_pruning_rate, trails):
    datasets = get_datasets(dataset)
    device = try_gpu()
    common_name = f'experiment1-{arch}-{dataset}-{fc_pruning_rate:.3f}-{conv_pruning_rate:.3f}'
    iterations = 50_000

    for trail in range(trails):
        torch.manual_seed(trail)
        net = create_network(arch)
        name = f'{common_name}-one-shot-{trail}'
        prune_random(net, fc_pruning_rate)
        train(net, datasets, optim, CrossEntropyLoss(), device, name, lr, iterations=iterations, graph=False,
              keep_checkpoints=False)

        torch.manual_seed(trail + trails)
        net = create_network(arch)
        name = f'{common_name}-l1-source-{trail}'
        train(net, datasets, optim, CrossEntropyLoss(), device, name, lr, iterations=int(iterations / 10), graph=False,
              keep_checkpoints=False)
        prune(net, fc_pruning_rate)
        net_sparse = create_network(arch)
        net_sparse = net_sparse.to(device)
        copy_prune(net, net_sparse)
        name = f'{common_name}-l1-target-{trail}'
        train(net, datasets, optim, CrossEntropyLoss(), device, name, lr, iterations=iterations, graph=False,
              keep_checkpoints=False)


@dataclass
class StatisticsExperiment1:
    arch: str
    dataset: str
    prune_strategy: str
    pruned_weights: float
    early_stop_iteration: int
    acc_at_early_stop: float


def read_stats_ex1(pruned_weights, arch: str, dataset: str, prune_strategy: str, trail: int):
    file = f'checkpoints/experiment1-{arch}-{dataset}-{pruned_weights:.3f}-0.000-{prune_strategy}-{trail}.csv'
    with open(file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        min_val_loss = float('inf')
        early_stop_iteration = 0
        acc_at_early_stop = 0
        for row in reader:
            if float(row['val_loss']) < min_val_loss:
                early_stop_iteration = int(row['iteration'])
                min_val_loss = float(row['val_loss'])
                acc_at_early_stop = float(row['test_acc'])

    return StatisticsExperiment1(arch, dataset, prune_strategy, pruned_weights, early_stop_iteration, acc_at_early_stop)


def plot_experiment_1():
    import matplotlib.pyplot as plt

    pr_rates = [1, 0.411, 0.169, 0.070, 0.029, 0.012, 0.005, 0.002]
    pr_rates_percentage = [pr_rate * 100 for pr_rate in pr_rates]
    pr_rates_percentage_str = [f'{pr_rate:.2f}' for pr_rate in pr_rates_percentage]
    num_trails = 5

    fig, (iter_plt, acc_plt) = plt.subplots(1, 2, figsize=(12, 5))

    iter_plt.set_ylabel('Early-Stop Iteration (Val.)')
    iter_plt.set_xlabel('Percent of Weights Remaining')
    iter_plt.set_xlim(110, .1)
    iter_plt.set_xscale('log')
    iter_plt.set_xticks(pr_rates_percentage, pr_rates_percentage_str)

    acc_plt.set_ylabel('Accuracy at Early-Stop (Test)')
    acc_plt.set_xlabel('Percent of Weights Remaining')
    acc_plt.set_xlim(110, .1)
    acc_plt.set_ylim(.9, 1)
    acc_plt.set_xscale('log')
    acc_plt.set_xticks(pr_rates_percentage, pr_rates_percentage_str)

    for strategy in ['one-shot', 'l1-target']:
        x = []
        early_stop = []
        early_stop_err = [[], []]
        acc = []
        acc_err = [[], []]
        for pr_rate in pr_rates:
            sum_early_stop = 0
            min_early_stop = float('inf')
            max_early_stop = float('-inf')

            sum_acc = 0
            min_acc = float('inf')
            max_acc = float('-inf')
            for trail in range(num_trails):
                stats = read_stats_ex1(1 - pr_rate, 'lenet', 'mnist', strategy, trail)

                sum_early_stop += stats.early_stop_iteration
                min_early_stop = min(stats.early_stop_iteration, min_early_stop)
                max_early_stop = max(stats.early_stop_iteration, max_early_stop)

                sum_acc += stats.acc_at_early_stop
                min_acc = min(stats.acc_at_early_stop, min_acc)
                max_acc = max(stats.acc_at_early_stop, min_acc)

            mean_early_stop = sum_early_stop / num_trails
            mean_acc = sum_acc / num_trails

            x.append(pr_rate * 100)
            acc.append(mean_acc)
            acc_err[0].append(mean_acc - min_acc)
            acc_err[1].append(max_acc - mean_acc)
            early_stop.append(mean_early_stop)
            early_stop_err[0].append(mean_early_stop - min_early_stop)
            early_stop_err[1].append(max_early_stop - mean_early_stop)

        iter_plt.errorbar(x, early_stop, early_stop_err, label=strategy)
        acc_plt.plot(x, acc, acc_err, label=strategy)

    iter_plt.legend()
    acc_plt.legend()


def iterative_pruning(arch: str, data: str, j: int, n: int, pruning_rate: float, trials=5):
    from torch.optim.adam import Adam

    for trial in range(trials):
        torch.manual_seed(trial)
        net = create_network(arch)
        datasets = get_datasets(data)
        device = try_gpu()
        common_name = f'experiment2-{arch}-{data}-{pruning_rate:.3f}-trial{trial}'

        for i in range(n):
            name = f'{common_name}-prune_iter{i}'
            train(net, datasets, Adam, CrossEntropyLoss(), device, name, iterations=j, keep_checkpoints=False,
                  graph=False)
            # FIXME I guess we do not add additional pruning here, but just reevaluate a new pruning
            prune(net, pruning_rate / n)
            net_old = net
            net = create_network(arch)
            net = net.to(device)
            copy_prune(net_old, net)

        name = f'{common_name}-pruned'
        train(net, datasets, Adam, CrossEntropyLoss(), device, name, iterations=20_000, keep_checkpoints=False,
              graph=False)


@dataclass
class StatisticsExperiment2:
    arch: str
    dataset: str
    pruned_weights: float
    iterations: np.array
    test_acc: np.array


def read_stats_ex2(pruned_weights, arch: str, dataset: str, trail: int):
    file = f'checkpoints/experiment2-{arch}-{dataset}-{pruned_weights:.3f}-trail{trail}-pruned.csv'
    # file = f'checkpoints/experiment2-{arch}-{dataset}-{pruned_weights:.3f}-pruned.csv'
    with open(file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        iterations = []
        test_acc = []
        for row in reader:
            iterations.append(row['iteration'])
            test_acc.append(row['test_acc'])

    return StatisticsExperiment2(arch, dataset, pruned_weights, np.array(iterations), np.array(test_acc, dtype=float))


def plot_iterative_pruning():
    import matplotlib.pyplot as plt

    pr_rates = [1, 0.513, 0.211, 0.07, 0.036, 0.019]
    # pr_rates = [1, 0.411, 0.169, 0.070, 0.029, 0.012, 0.005, 0.002]
    num_trails = 5
    num_train_iterations = 20_000
    data_points = int(num_train_iterations / 100 + 1)

    fig, (plt1) = plt.subplots(1, 1, figsize=(12, 12))

    plt1.set_ylabel('Test accuracy')
    plt1.set_xlabel('Training Iterations')
    plt1.set_ylim(0.94, 0.99)

    for pr_rate in pr_rates:
        x = []
        acc_err = [[], []]
        acc_err_cleared = [np.zeros(data_points, dtype=float), np.zeros(data_points, dtype=float)]

        sum_acc = np.zeros(data_points, dtype=float)
        min_acc = np.ones(data_points, dtype=float) * float('inf')
        max_acc = np.ones(data_points, dtype=float) * float('-inf')
        for trail in range(num_trails):
            stats = read_stats_ex2(1 - pr_rate, 'lenet', 'mnist', trail)

            sum_acc = np.add(sum_acc, stats.test_acc)
            min_acc = np.fmin(stats.test_acc, min_acc)
            max_acc = np.fmax(stats.test_acc, max_acc)
            x = stats.iterations

        mean_acc = sum_acc / num_trails

        acc_err[0] = mean_acc - min_acc
        acc_err[1] = max_acc - mean_acc
        acc_err_cleared[0][::10] = acc_err[0][::10]
        acc_err_cleared[1][::10] = acc_err[1][::10]

        plt1.errorbar(x, mean_acc, acc_err_cleared, label=f'{(pr_rate * 100):.3f}')

    plt1.legend()


def global_pruning(net: nn.Module, p: float):
    parameters_to_prune = []
    for layer in net.children():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            parameters_to_prune.append((layer, 'weight'))

    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=p,
    )


# TODO merge with iterative_pruning method
def iterative_global_pruning(arch: str, data: str, j: int, n: int, pruning_rate: float, trials=5, rand_reinit=False):
    from torch.optim.adam import Adam

    for trial in range(trials):
        torch.manual_seed(trial)
        net = create_network(arch)
        datasets = get_datasets(data)
        device = try_gpu()
        common_name = f'experiment3-{arch}-{data}-{pruning_rate:.3f}-trial{trial}-rand_reinit={rand_reinit}'

        for i in range(n):
            name = f'{common_name}-prune_iter{i}'
            train(net, datasets, Adam, CrossEntropyLoss(), device, name, iterations=j, keep_checkpoints=False,
                  graph=False)
            # FIXME I guess we do not add additional pruning here, but just reevaluate a new pruning
            global_pruning(net, pruning_rate / n)
            net_old = net
            if rand_reinit:
                torch.manual_seed(trial + trials)
            net = create_network(arch)
            net = net.to(device)
            copy_prune(net_old, net)

        name = f'{common_name}-pruned'
        train(net, datasets, Adam, CrossEntropyLoss(), device, name, iterations=30_000, keep_checkpoints=False,
              graph=False)


def read_stats_ex3(arch: str, dataset: str, pr_rate: float, trail: int, rand_reinit: bool):
    filename = f'checkpoints/experiment3-{arch}-{dataset}-{pr_rate:.3f}-trial{trail}-rand_reinit={rand_reinit}-pruned.csv'
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        iterations = []
        test_acc = []
        for row in reader:
            iterations.append(row['iteration'])
            test_acc.append(row['test_acc'])

    return StatisticsExperiment2(arch, dataset, pr_rate, np.array(iterations), np.array(test_acc, dtype=float))


def plot_global_pruning():
    import matplotlib.pyplot as plt

    pr_rates = [1, 0.417, 0.178, 0.08]
    pr_rates_inverted = [1 - pr_rate for pr_rate in pr_rates]
    num_trails = 3
    plot_iterations = [10_000, 20_000, 30_000]
    pr_rates_percentage = [pr_rate * 100 for pr_rate in pr_rates]
    pr_rates_percentage_str = [f'{pr_rate:.2f}' for pr_rate in pr_rates_percentage]

    fig, plts = plt.subplots(3, 1, figsize=(6, 18))

    for i, plt in enumerate(plts):
        plt.set_ylabel(f'Test accuracy {plot_iterations[i]}')
        plt.set_xlabel('Percent of Weights Remaining')
        plt.set_xscale('log')
        print(pr_rates_percentage_str)
        plt.set_xticks(pr_rates, pr_rates_percentage_str)
        # plt.set_ylim(0.8, 1.)
        x = []
        mean_acc = []
        acc_err = [[], []]
        for pr_rate in pr_rates:
            sum_acc = 0
            min_acc = float('inf')
            max_acc = float('-inf')
            for trail in range(num_trails):
                stats = read_stats_ex3('resnet18', 'cifar10', 1 - pr_rate, trail, False)

                sum_acc += stats.test_acc[((i + 1) * 100) - 1]
                min_acc = min(min_acc, stats.test_acc[((i + 1) * 100) - 1])
                max_acc = max(max_acc, stats.test_acc[((i + 1) * 100) - 1])

            x.append(pr_rate)
            mean_acc.append(sum_acc / num_trails)

            acc_err[0].append((sum_acc / num_trails) - min_acc)
            acc_err[1].append(max_acc - (sum_acc / num_trails))

        plt.errorbar(x, mean_acc, acc_err)
        plt.legend()
