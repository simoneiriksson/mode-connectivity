import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
from torch.distributions import Categorical, MultivariateNormal
from matplotlib import pyplot as plt
from utils import torch_seed

from torchvision.datasets import mnist, CIFAR10
from sklearn.datasets import load_iris, load_wine, load_diabetes
from torchvision.transforms import ToTensor
from torch.utils.data import Subset

def make_loaders(X, y, train_size, batch_size=0):
    indices = torch.randperm(len(X))
    #print(f"{X.shape = }")
    #print(f"{y.shape = }")
    X_ = X[indices]
    y_ = y[indices]
    X_train, X_test = X_[:train_size], X_[train_size:]
    y_train, y_test = y_[:train_size], y_[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    subset_test = Subset(test_dataset, indices=range(len(test_dataset) // 1))
    subset_train = Subset(train_dataset, indices=range(len(train_dataset) // 1))
    if batch_size == 0:
        batch_size = max(len(subset_train), len(subset_test))
    train_loader = DataLoader(subset_train, batch_size=batch_size)
    test_loader = DataLoader(subset_test, batch_size=batch_size)
    return train_loader, test_loader

def gen_polynomial_regression_data(weights = torch.tensor([2.0, 1.0]),
                                   num_train_samples=10, 
                                   num_test_samples=10, 
                                   noise_std=1.0, 
                                   batch_size=0, seed=2):
    def fn(x):
        paws = torch.arange(len(weights), device=x.device).float()
        return weights * x**paws
    N = num_train_samples + num_test_samples
    X = torch.rand(N, 1)*2-1
    y = fn(X) + torch.randn_like(X) * noise_std
    return make_loaders(X, y, num_train_samples, batch_size)    


def gen_linear_regression_data(num_train_samples=10, 
                              num_test_samples=10, 
                              target_sigma=1.0, 
                              batch_size=0, seed=2, weights=None):
    def fn(x):
        if weights is not None:
            return weights[0] + weights[1]*x
        return 2 - 1*x 
    N = num_train_samples + num_test_samples
    with torch_seed(seed):
        X = torch.rand(N, 1)*2-1
        y = fn(X) + torch.randn_like(X) * target_sigma
    return make_loaders(X, y, num_train_samples, batch_size)    

def gen_log_regression_data(num_train_samples=10, 
                              num_test_samples=10, 
                              num_features = 1,
                              num_classes = 2,
                              variance = .1,
                              batch_size=0, seed=2):
    num_samples = num_train_samples + num_test_samples
    with torch_seed(seed):
        means = torch.randn(num_classes, num_features)
        class_weights = torch.nn.functional.softmax(torch.randn(num_classes))
        covariances = torch.eye(num_features).repeat(num_classes, 1, 1)*variance

        # Create a categorical distribution for class selection
        class_dist = Categorical(probs=class_weights)
        # Generate class labels
        y = class_dist.sample((num_samples,))
        x = torch.zeros((num_samples, num_features))
        for k in range(num_classes):
            class_mask = y == k
            num_class_samples = class_mask.sum()
            if num_class_samples > 0:
                mvn = MultivariateNormal(means[k], covariances[k])
                x[class_mask] = mvn.sample((num_class_samples,))
    return make_loaders(x, y, num_train_samples, batch_size)    

# function that takes a model, some observation noise, and generates data
def gen_model_data(model, input_dist, num_train_samples=10, 
                                   num_test_samples=10, noise_std=None, output_dist = None, seed=2, batch_size=0):
    N = num_train_samples + num_test_samples
    if output_dist == None:
        output_dist = lambda x: torch.distributions.Normal(x, noise_std)
    with torch_seed(seed):
        X = input_dist(N)
        modelout = model(X).detach()
        #print(f"{modelout.shape = }")
        #y = modelout + torch.randn_like(modelout) * noise_std
        y = output_dist(modelout).sample()
        #print(f"{y.shape = }")
        #print(f"{X.shape = }")
    return make_loaders(X, y, num_train_samples, batch_size=batch_size)


def get_dataloader_scipy(dataset, datafolder=None, train_share=.9 , batch_size=16, select_features=None):
    if dataset == "iris":
        dataset = load_iris()

    elif dataset == "wine":
        dataset = load_wine()

    elif dataset == "diabetes":
        dataset = load_diabetes()
    if select_features == None:
        X = torch.tensor(dataset.data, dtype=torch.float32)
    else:
        X = torch.tensor(dataset.data, dtype=torch.float32)[:,select_features]
    y = torch.tensor(dataset.target, dtype=torch.int64)
    if hasattr(dataset, "target_names"):
        num_classes = dataset.target_names.shape[0]
    else: num_classes = None
    num_train_samples = int(X.shape[0] * train_share)
    train_loader, test_loader = make_loaders(X, y, num_train_samples, batch_size)
    return train_loader, test_loader, num_classes 