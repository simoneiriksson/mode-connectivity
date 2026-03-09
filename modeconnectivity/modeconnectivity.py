import torch
import torch.nn as nn
import torchmetrics
#import torchmetrics.classification
#import torchmetrics.functional.classification 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from models import MyNet, Lenet5, tiny, MyNet_small, CIFAR10ConvNet
from curve_model import Curve
from train import train
from torchviz import make_dot
import numpy as np
import os
import logging
import sys
from datetime import datetime
from torch.utils.data import Subset
from curve_plots import plot_Curve_losslandscape, affine_subspace, bezier_plot
import torchmetrics
from curve_eval import curve_eval
import argparse
from scheduler import make_scheduler



def curve_fitting(**kargs):
    dataset = kargs.get("dataset", "MNIST")
    batch_size = kargs.get("batchsize", 256)
    seed = kargs.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
    model_name = kargs.get("model", "MyNet")
    #root = "/Users/simondanieleiriksson/My Drive (punkeren@gmail.com)/DTU kurser/Specialkursus Michael"
    root = ".."
    if kargs.get("basefolder") is None:
        base_directory = f"{root}/experiments/curve_experiment_{dataset}_{model_name}"
    else:
        base_directory = kargs.get("basefolder")
    if kargs.get("datafolder") is None:
        datafolder = f"{root}/data"
    else:
        datafolder = kargs.get("datafolder", f"/data")

    createnewfolder = kargs.get("createnewfolder", False)

    if model_name == "MyNet":
        MODEL = MyNet
        model_kargs = {"dropout": kargs.get("model_dropout", 0.5)}
    elif model_name == "MyNet_small":
        MODEL = MyNet_small
        model_kargs = {"dropout": kargs.get("model_dropout", 0.5)}
    elif model_name == "Lenet5":
        MODEL = Lenet5
        model_kargs = {}
    elif model_name == "tiny":
        MODEL = tiny
        model_kargs = {}
    elif model_name == "CIFAR10ConvNet":
        MODEL = CIFAR10ConvNet
        model_kargs = {"dropout": kargs.get("model_dropout", 0.3)}
    else:
        raise ValueError("Model not recognized")

    if dataset == "CIFAR10" and model_name != "CIFAR10ConvNet":
        raise ValueError("For CIFAR10, use model='CIFAR10ConvNet'")
    
    retrain = kargs.get("retrain", True)
    model_lr_start = torch.tensor(kargs.get("model_lr_start", 1e-2))
    model_lr_end = torch.tensor(kargs.get("model_lr_end", 1e-3))
    model_epochs = kargs.get("model_epochs", 25)
    model_optimizer = kargs.get("model_optimizer", "Adam")
    model_scheduler = kargs.get("model_scheduler", "linear")


    retrain_curve = kargs.get("retrain_curve", True)
    curve_lr_start = torch.tensor(kargs.get("curve_lr_start", 1e-1))
    curve_lr_end = torch.tensor(kargs.get("curve_lr_end", 1e-5))
    curve_epochs = kargs.get("curve_epochs", 25)
    curve_optimizer = kargs.get("curve_optimizer", "SGD")
    curve_scheduler = kargs.get("curve_scheduler", "linear")
    
    plot_mesh = kargs.get("plot_mesh", True)
    recalc_mesh = kargs.get("recalc_mesh", True)
    meshpoints = kargs.get("meshpoints", 20)

    plot_bezier = kargs.get("plot_bezier", True)
    bezierpoints = kargs.get("bezierpoints", 20)

    eval_curve = kargs.get("eval_curve", True)
    curve_eval_samplesize = kargs.get("curve_eval_samplesize", 20)


    useGPU = kargs.get("useGPU", True)
    logging_tofile = kargs.get("logging", True)

    if createnewfolder:
        i = 1
        base_directory_i = base_directory
        while os.path.exists(base_directory_i):
            base_directory_i = f"{base_directory}_{i}"
            i += 1
        base_directory = base_directory_i
    
    # create folder if not exists:
    os.makedirs(f"{datafolder}", exist_ok=True)
    os.makedirs(f"{base_directory}/models", exist_ok=True)
    os.makedirs(f"{base_directory}/figures", exist_ok=True)
    os.makedirs(f"{base_directory}/logs", exist_ok=True)
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if logging_tofile:
        handler_file = logging.FileHandler(f"{base_directory}/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log", mode='w') # and log to file
        handler_file.setLevel(logging.DEBUG)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.DEBUG)
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)

    logger.info('Start logging')
    logger.info(f"base_directory is '{base_directory}'")
    logger.info("Parameters:")
    for key, item in kargs.items():
        logger.info(f"{key} = {item}")
    logger.info(f"{plot_mesh=}")

    if dataset == "MNIST":
        train_dataset = datasets.MNIST(root=f'{datafolder}/train', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root=f'{datafolder}/test', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root=f'{datafolder}/train', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root=f'{datafolder}/test', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.CIFAR10(root=f'{datafolder}/train', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=f'{datafolder}/test', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")

    subset_test = Subset(test_dataset, indices=range(len(test_dataset) // 1))
    subset_train = Subset(train_dataset, indices=range(len(train_dataset) // 1))

    test_loader = DataLoader(subset_test, batch_size=batch_size)
    train_loader = DataLoader(subset_train, batch_size=batch_size)
    if useGPU:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else: device = "cpu"

    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
    if retrain:
        logger.info("Begin training of model_start")
        model_start = MODEL(**model_kargs)
        total_iter = model_epochs*train_loader.__len__()
        gamma = ((model_lr_end.log()-model_lr_start.log())/total_iter).exp()
        if model_optimizer == "Adam":
            optimizer_start = torch.optim.Adam(params=model_start.parameters(), lr=model_lr_start.clone())
        elif model_optimizer == "SGD":
            optimizer_start = torch.optim.SGD(params=model_start.parameters(), lr=model_lr_start.clone(), momentum=0.9)
        else: 
            logger.info(f"Wrong model_optimizer: {model_optimizer}")
            AssertionError(f"Wrong model_optimizer: {model_optimizer}")

        if model_scheduler == "exponential":
            scheduler_start = torch.optim.lr_scheduler.ExponentialLR(optimizer_start, gamma=gamma)
        elif model_scheduler == "linear":
            scheduler_start = torch.optim.lr_scheduler.LinearLR(optimizer_start, start_factor=1, end_factor=model_lr_end/model_lr_start, total_iters=total_iter)
        elif model_scheduler == "diy":
            scheduler_start = make_scheduler(optimizer_start, 
                                             train_num_steps=total_iter, 
                                             lr_start_warmup=model_lr_start.clone(), 
                                             lr=model_lr_end.clone(), 
                                             lr_warmup_steps=5*train_loader.__len__(), 
                                             lr_finetune_halftime=total_iter // (5*3), 
                                             lr_finetune_steps=total_iter // 3
            )


        else: 
            logger.info(f"Wrong model_scheduler: {model_scheduler}")
            AssertionError(f"Wrong model_scheduler: {model_scheduler}")
 
        model_start, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots = train(model_start, 
                                                train_loader=train_loader, 
                                                test_loader=test_loader, 
                                                optimizer=optimizer_start, 
                                                scheduler=scheduler_start, 
                                                epochs=model_epochs, loss_fn=loss_fn, device=device, 
                                                logger_info=logger.info,
                                                plot=True, plotpath=f"{base_directory}/start_model/figures",
                                                #plotpath=f"model_start_{MODEL.__name__}_{dataset}",
                                                verbose=True
                                                )

        logger.info("Begin training of model_end")
        model_end = MODEL(**model_kargs)
        if model_optimizer == "Adam":
            optimizer_end = torch.optim.Adam(params=model_end.parameters(), lr=model_lr_start.clone())
        elif model_optimizer == "SGD":
            optimizer_end = torch.optim.SGD(params=model_end.parameters(), lr=model_lr_start.clone(), momentum=0.9)
        else: 
            logger.info(f"Wrong model_optimizer: {model_optimizer}")
            AssertionError(f"Wrong model_optimizer: {model_optimizer}")

        if model_scheduler == "exponential":
            scheduler_end = torch.optim.lr_scheduler.ExponentialLR(optimizer_end, gamma=gamma)
        elif model_scheduler == "linear":
            scheduler_end = torch.optim.lr_scheduler.LinearLR(optimizer_end, start_factor=1, end_factor=model_lr_end/model_lr_start, total_iters=total_iter)
        elif model_scheduler == "diy":
            scheduler_end = make_scheduler(optimizer_end, 
                                             train_num_steps=total_iter, 
                                             lr_start_warmup=model_lr_start.clone(), 
                                             lr=model_lr_end.clone(), 
                                             lr_warmup_steps=5*train_loader.__len__(), 
                                             lr_finetune_halftime=total_iter // (5*3), 
                                             lr_finetune_steps=total_iter // 3
            )
        else: 
            logger.info(f"Wrong model_scheduler: {model_scheduler}")
            AssertionError(f"Wrong model_scheduler: {model_scheduler}")

        model_end, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots = train(model_end, 
                                                train_loader=train_loader, 
                                                test_loader=test_loader, 
                                                optimizer=optimizer_end, 
                                                scheduler=scheduler_end, 
                                                epochs=model_epochs, loss_fn=loss_fn, device=device, logger_info=logger.info,
                                                plot=True, plotpath=f"{base_directory}/end_model/figures", 
                                                #plotname=f"model_end_{MODEL.__name__}_{dataset}"
                                                verbose=True
                                                )

        torch.save(model_start, f"{base_directory}/models/model_start_{MODEL.__name__}_{dataset}.pth")
        torch.save(model_end, f"{base_directory}/models/model_end_{MODEL.__name__}_{dataset}.pth")
        logger.info("finished training of models")
    else:
        model_start = torch.load(f"{base_directory}/models/model_start_{MODEL.__name__}_{dataset}.pth", map_location=torch.device(device), weights_only=False)
        model_end = torch.load(f"{base_directory}/models/model_end_{MODEL.__name__}_{dataset}.pth", map_location=torch.device(device), weights_only=False)

    def curve_fn(param_start, param_end, param_theta, t):
        return param_start * (1-t)**2 + param_end * t**2 + param_theta * 2*t*(1-t)

    if retrain_curve:
        logger.info("Begin training of curve")
        total_iter = curve_epochs*train_loader.__len__()
        curve = Curve(model_start=model_start, model_end=model_end, curve_fn=curve_fn, device=device)
        gamma = ((curve_lr_end.log()-curve_lr_start.log())/(curve_epochs*train_loader.__len__())).exp()

        if curve_optimizer == "Adam":
            optimizer = torch.optim.Adam(params=curve.model_theta.parameters(), lr=curve_lr_start)
        elif curve_optimizer == "SGD":
            optimizer = torch.optim.SGD(params=curve.model_theta.parameters(), lr=curve_lr_start, momentum=0.9)

        if curve_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif curve_scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                                                          end_factor=curve_lr_end/curve_lr_start, 
                                                          total_iters=total_iter)
        elif model_scheduler == "diy":
            scheduler = make_scheduler(optimizer, 
                                             train_num_steps=total_iter, 
                                             lr_start_warmup=curve_lr_start.clone(), 
                                             lr=curve_lr_end.clone(), 
                                             lr_warmup_steps=5*train_loader.__len__(), 
                                             lr_finetune_halftime=total_iter // (5*3), 
                                             lr_finetune_steps=total_iter // 3
            )
                

        curve, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots = train(curve, 
                                                                            train_loader=train_loader, 
                                                                            test_loader=test_loader, 
                                                                            optimizer=optimizer, 
                                                                            scheduler=scheduler, 
                                                                            epochs=curve_epochs,
                                                                            loss_fn=loss_fn, 
                                                                            device=device, 
                                                                            logger_info=logger.info,
                                                                            plot=True, 
                                                                            plotpath=f"{base_directory}/curve_model/figures", 
                                                                            #plotname=f"curvefitting_{MODEL.__name__}_{dataset}", 
                                                                            modeltype="curve", 
                                                                            verbose=True)
        torch.save(curve.model_theta, f"{base_directory}/models/curve.model_theta_{MODEL.__name__}_{dataset}.pth")
        logger.info("finished training of curve")
    else:
        curve = Curve(model_start=model_start, model_end=model_end, curve_fn=curve_fn, device=device)
        curve.model_theta = torch.load(f"{base_directory}/models/curve.model_theta_{MODEL.__name__}_{dataset}.pth", map_location=torch.device(device), weights_only=False)
        
    metrics_dict = {
        #"loss_fn": lambda pred_probs, target: loss_fn(pred_probs.log(), target),
        "Cross Entropy": lambda pred_probs, target: torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)(pred_probs.log(), target),
        #"NegLogLikelihood": lambda pred_probs, target: torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')(pred_probs.log(), target),
        #"Expected Calibration Error": lambda pred_probs, target: torchmetrics.functional.classification.multiclass_calibration_error(preds=pred_probs, target=target, num_classes=10, n_bins=15),
        
        "Expected Calibration Error": torchmetrics.classification.MulticlassCalibrationError(num_classes=10, n_bins=25).to(device),
        "Accuracy": lambda pred_probs, target: torchmetrics.functional.classification.accuracy(preds=pred_probs, target=target, task="multiclass", num_classes=10).to(device),
        "AUROC": lambda pred_probs, target: torchmetrics.functional.classification.auroc(preds=pred_probs, target=target, task="multiclass", num_classes=10).to(device)
    }

    if plot_mesh: 
        fig, ax = plot_Curve_losslandscape(curve, device, f"{base_directory}/figures", test_loader, N_points=meshpoints, loss_fn=loss_fn, recalc_mesh=recalc_mesh, logger_info=logger.info, N_bezierpoints=bezierpoints)
        fig.savefig(f"{base_directory}/figures/loss_landscape.png")
        plt.close()

    if plot_bezier:
        fig, axs, eval_results = bezier_plot(curve, device, test_loader=test_loader, plottype="linear", 
                               N_bezierpoints = bezierpoints,
                               logger_info=logger.info, plot_linear=False, metrics_dict=metrics_dict)
        fig.savefig(f"{base_directory}/figures/metric_along_curve.png")
        plt.close()

        

    #if eval_curve:
    #    curve_eval(curve, samplesize=curve_eval_samplesize, test_loader=test_loader, device=device, logger_info=logger.info, metrics_dict=metrics_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode connectivity')
    parser.add_argument('--basefolder', type=str, help='Folder to store results')
    parser.add_argument('--createnewfolder', default=True, help='Create new folder if basefolder already exists', type=eval, choices=[True, False])
    parser.add_argument('--datafolder', type=str, help='Folder to store data')
    parser.add_argument('--dataset', default = "CIFAR10", type=str, help='Dataset to use', choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument('--batchsize', default = 256, type=int, help='Batch size')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--model', default = "CIFAR10ConvNet",  type=str, help='Model to use (MyNet/Lenet5/tiny/MyNet_small/CIFAR10ConvNet)', choices=["MyNet", "Lenet5", "tiny", "MyNet_small", "CIFAR10ConvNet"])
    parser.add_argument('--retrain', default = True, help='Retrain models if already present in folder', type=eval, choices=[True, False])
    parser.add_argument('--model_lr_start', default = 1e-4, type=float, help='Learning rate start')
    parser.add_argument('--model_lr_end', default = 2e-3, type=float, help='Learning rate end')
    parser.add_argument('--model_epochs', default = 100, type=int, help='Number of epochs')
    parser.add_argument('--model_dropout', default = 0.5, type=float, help='Dropout')
    parser.add_argument('--model_optimizer', default = "Adam", type=str, help='Optimizer to use for model training', choices=["Adam", "SGD"])
    parser.add_argument('--model_scheduler', default = "diy", type=str, help='Optimizer to use for model training', choices=["linear", "exponential", "diy", "none"])
    
    parser.add_argument('--retrain_curve', default = True, help='Retrain curve if already present in folder', type=eval, choices=[True, False])
    parser.add_argument('--curve_lr_start', default = 1e-6, type=float, help='Learning rate start')
    parser.add_argument('--curve_lr_end', default = 5e-2, type=float, help='Learning rate end')
    parser.add_argument('--curve_epochs', default = 200, type=int, help='Number of epochs')
    parser.add_argument('--curve_optimizer', default = "SGD", type=str, help='Optimizer to use for curve training', choices=["Adam", "SGD"])
    parser.add_argument('--curve_scheduler', default = "diy", type=str, help='Optimizer to use for curve training', choices=["linear", "exponential", "diy", "none"])

    parser.add_argument('--plot_mesh', default = True, help='Plot loss landscape', type=eval, choices=[True, False])
    parser.add_argument('--recalc_mesh', default = True, help='Recalculate loss landscape if already present in folder', type=eval, choices=[True, False])
    parser.add_argument('--meshpoints', default = 20, type=int, help='Number of mesh points')

    parser.add_argument('--plot_bezier', default = True, help='Plot bezier curve', type=eval, choices=[True, False])
    parser.add_argument('--bezierpoints', default = 50, type=int, help='Number of bezier points')

    parser.add_argument('--eval_curve', default = False, help='Evaluate curve', type=eval, choices=[True, False])
    parser.add_argument('--curve_eval_samplesize', default = 20, type=int, help='Number of samples for evaluation')
    parser.add_argument('--useGPU', default = True, help='Use GPU if available (cpu/cuda/mps)', type=eval, choices=[True, False])
    parser.add_argument('--logging', default = True, help='log to file', type=eval, choices=[True, False])
    args = parser.parse_args()
    curve_fitting(**vars(args))

    

