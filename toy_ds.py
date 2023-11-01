import torch
import deepspeed

# Define a simple model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Initialize DeepSpeed engine
model = MyModel()

import debugpy; debugpy.listen(45859); print('Waiting for debugger attach'); debugpy.wait_for_client(); debugpy.breakpoint()
deepspeed_config = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 10,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    }
}

engine, model, _, _ = deepspeed.initialize(model=model, config_params=deepspeed_config)

# Training loop
for step in range(100):
    data = torch.randn(8, 10)
    loss = model(data).mean()
    engine.backward(loss)
    engine.step()
