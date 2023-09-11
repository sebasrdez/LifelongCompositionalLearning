import torch
import torch.nn as nn
from models.base_net_classes import SoftGatedNet

class MLPSoftGatedLL(SoftGatedNet):
    def __init__(self, 
                i_size, 
                size, 
                depth, 
                num_tasks, 
                num_init_tasks=None,
                regression=[False],
                init_ordering_mode='random',
                device='cuda',
                freeze_encoder=False,
                ):
        super().__init__(i_size,
            depth,
            num_tasks,
            num_init_tasks=num_init_tasks,
            init_ordering_mode=init_ordering_mode,
            device=device,
            num_outputs=len(regression))
        self.size = size
        self.freeze_encoder = freeze_encoder
        self.regression = regression
        
        self.encoder = nn.ModuleList()
        for t in range(self.num_tasks):
            encoder_t = nn.Linear(self.i_size[t], self.size)
            if freeze_encoder:
                for param in encoder_t.parameters():
                    param.requires_grad = False
            self.encoder.append(encoder_t)
        
        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for i in range(self.depth):
            fc = nn.Linear(self.size, self.size)
            self.components.append(fc)

        self.decoder = nn.ModuleList()
        self.binary = True
        for t in range(self.num_tasks):
            decoder_t = nn.Linear(self.size, self.num_outputs)
            self.decoder.append(decoder_t)

        self.structure = nn.ModuleList(nn.ModuleList(nn.Linear(self.size, self.depth) for _ in range(self.depth)) for _ in range(self.num_tasks))
        self.init_ordering()

        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)
  

    def forward(self, X, task_id):
        X = self.encoder[task_id](X)
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            s = self.softmax(self.structure[task_id][k](X))
            for j in range(self.depth):
                fc = self.components[j]
                X_tmp += s[:, j].view(-1, 1) * self.dropout(self.relu(fc(X)))

            X = X_tmp

        return self.decoder[task_id](X)  

