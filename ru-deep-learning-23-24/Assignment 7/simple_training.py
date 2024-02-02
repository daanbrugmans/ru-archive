import torch

import dl_assignment_7_common as common
import lunar_phase

from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from d2l.torch import try_gpu


def load(path: str):
    arch_name = "lenet"
    model = common.create_network(arch_name)
    model.load_state_dict(torch.load(path))
    return model


def train(graph=True):
    arch_name = "lenet"
    dataset_name = "mnist"
    run_name = "1"
    lunar_phase_name = lunar_phase.get_lunar_phase_name()
    model_name = f"{arch_name}-{dataset_name}-{run_name}-{lunar_phase_name}"
    
    device_name = try_gpu()
    net = common.create_network(arch_name)
    data_loaders = common.get_datasets(dataset_name)
    
    common.train(
        net,
        data_loaders,
        optimizer=Adam,
        learning_rate=0.0012,
        loss_fn=CrossEntropyLoss(),
        iterations=1000,
        device=device_name,
        model_file_name=model_name,
        eval_every_n_iterations=100,
        graph=graph
    )
    return net


if __name__ == '__main__':
    train()
