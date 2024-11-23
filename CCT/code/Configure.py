# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
	"kernel_size":3,
    "num_heads":4,
    "mlp_ratio":2,
    "embedding_dim":256,
    "num_layers": 6,
    "stride": 1,
    "padding": 1,
    "batch_size": 128,
    "workers": 4,
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2470, 0.2435, 0.2616],
    "epochs":300,
    "lr": 0.0005,
    "weight_decay": 3e-2
}

### END CODE HERE