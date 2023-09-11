import torch
import torch.nn as nn
from models.base_net_classes import CompositionalNet

class LinearFactored(CompositionalNet):
    def __init__(self, 
                i_size, 
                depth, 
                num_tasks, 
                num_init_tasks=None,
                init_ordering_mode='one_module_per_task',
                regression=[False],
                device='cuda'
                ):
        super().__init__(i_size,
            depth,
            num_tasks,
            num_init_tasks=num_init_tasks,
            init_ordering_mode=init_ordering_mode,
            device=device,
            num_outputs=len(regression))

        i_size = i_size[0]
        self.regression = regression

        self.components = nn.Linear(i_size, self.depth)
        
        self.structure = nn.ParameterList([nn.Parameter(torch.rand(self.depth)) for t in range(self.num_tasks)])

        self.outputs = nn.Linear(1, self.num_outputs)

        self.init_ordering()
        self.to(device)

    def init_ordering(self):
        if self.init_ordering_mode == 'one_module_per_task':
            assert self.num_tasks == self.depth, \
             'Initializing one module per task requires the number of tasks to be the same as the depth'
            for t in range(self.num_tasks):
                self.structure[t].data = torch.zeros(self.depth)
                self.structure[t].data[t] = 1
            self.freeze_structure()
        else:
            raise ValueError('{} is not a valid ordering initialization mode'.format(self.init_ordering_mode))
    
    
    def forward(self, X, task_id):  
        s_t = self.structure[task_id].view(-1, 1)
        hidden = self.components(X).mm(s_t)
        outputs = self.outputs(hidden)
        return outputs


    def freeze_structure(self, freeze=True, task_id=None):
        '''
        Since we are using Adam optimizer, it is important to
        set requires_grad = False for every parameter that is 
        not currently being optimized. Otherwise, even if they
        are untouched by computations, their gradient is all-
        zeros and not None, and Adam counts it as an update.
        '''
        if task_id is None:
            for param in self.structure:
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
        else:
            self.structure[task_id].requires_grad = not freeze
            if freeze:
                self.structure[task_id].grad = None
