import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self, 
                i_size, 
                num_tasks, 
                num_init_tasks=None,
                regression=[False],
                device='cuda'
                ):
        super().__init__()
        
        assert np.all(np.asarray(i_size) == i_size[0])
        i_size = i_size[0]
        self.device = device
        self.num_tasks = num_tasks
        self.num_init_tasks = num_init_tasks
        self.regression = regression
        self.binary = True
        self.num_outputs = len(regression)
        
        self.components = nn.Linear(i_size, 1)

        self.outputs = nn.Linear(1, self.num_outputs)

        self.to(device)

    def forward(self, X, task_id):
        hidden = self.components(X)
        outputs = self.outputs(hidden)
        return outputs
