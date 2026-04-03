import torch
from matplotlib import pyplot as plt
import numpy as np
import os

def train(model, train_loader=None, test_loader=None, optimizer=None, scheduler=None, epochs=1, 
          loss_fn=None, 
          device="cpu", logger_info=print,
          plot=False, plotpath=None, verbose = False, modeltype="regression", print_every_n_epoch=10):
    """
    Train a model for a fixed number of epochs and optionally plot training diagnostics.
 
    Supports two training modes via the `modeltype` argument:
 
    - "regression" / default: standard training loop. Calls model.train()
      and model.eval() as usual.
    - "curve": mode-connectivity curve training. On each batch, samples a
      random t ~ U(0, 1) and builds a Bézier-interpolated model via
      model.sample_t() and model.sample_model(). Gradients flow only
      through model.model_theta (the trainable midpoint); the start and end
      models are frozen. During the test loop a fresh sample is drawn each batch,
      so the reported test loss reflects the expected loss over the curve.
 
    Training stops early if the learning rate drops below 1e-10.
 
    Args:
        model (nn.Module | Curve): The model to train. Pass a Curve instance
            and set modeltype="curve" for curve training.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test/validation set.
        optimizer (Optimizer): PyTorch optimizer. Must already be configured with
            the correct parameters and initial learning rate.
        scheduler (LRScheduler): Step-level learning rate scheduler (i.e.
            scheduler.step() is called after every batch, not every epoch).
        epochs (int): Maximum number of training epochs.
        loss_fn (callable): Loss function with signature (pred, target) -> scalar.
        device (str): Device string passed to .to(), e.g. "cpu", "cuda",
            or "mps".
        logger_info (callable): Logging function, e.g. logging.info or print.
        plot (bool): If True, saves loss, learning-rate, and accuracy plots to disk.
        plotpath (str | None): Directory in which to save plots. Created if it does
            not exist. Required when plot=True.
        verbose (bool): If True, logs a summary line every print_every_n_epoch
            epochs.
        modeltype (str): "curve" enables Bézier curve training; any other value
            uses the standard training loop.
        print_every_n_epoch (int): How often (in epochs) to emit a log line when
            verbose=True.
 
    Returns:
        model: The trained model (same object as input, moved to device).
        all_train_losses (list[float]): Per-batch training loss values.
        lrs (list[float]): Per-batch learning rate values.
        epoch_train_losses (list[float]): Mean training loss per epoch
            (weighted by batch size).
        test_losses (list[float]): Mean test loss per epoch.
        epoch_train_accuracy (list[float]): Training accuracy per epoch.
        plots (dict[str, tuple[Figure, Axes]]): If plot=True, a dict with keys
            "loss", "learning_rate", and "accuracy", each mapping to
            the corresponding (fig, ax) tuple. Empty dict if plot=False.
    """
    test_losses = []
    epoch_train_losses = []
    all_train_losses = []
    epoch_test_accuracy = []
    epoch_train_accuracy =[]
    epoch_pred_loss = []
    epoch_train_grad = []
    #step_train_grad = []
    lrs = []
    model.to(device)
    batch_number = 0
    train_loss_batch_number =[]
    test_loss_batch_number = []
    
    num_batches_train = len(train_loader)
    total_obs_train = len(train_loader.dataset)
    epoch=0
    while epoch < epochs:
        epoch += 1
        train_loss = 0
        current_correct_num = 0
        for x, y in train_loader:
            this_batch_size = x.shape[0]
            batch_number += 1
            optimizer.zero_grad()
            # For curve model, we need to sample a new point on the curve for each batch, 
            # and set the model to train mode. For regular model, just set to train mode.
            if modeltype == "curve":
                model.sample_t()
                model.sample_model()
                model.model_theta.train()
                model.sampled_model.train()
            else:
                model.train()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_loss = loss_fn(pred, y)
            pred_loss.backward()
            train_loss += pred_loss.item() * this_batch_size / total_obs_train
            pred_class = torch.argmax(pred, dim=-1)
            current_correct_num += (pred_class == y).sum()
            all_train_losses.append(pred_loss.item())
            train_loss_batch_number += [batch_number]

            lrs.append(optimizer.param_groups[0]['lr'].item())
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        train_accuracy = current_correct_num.item() / total_obs_train
        epoch_train_accuracy.append(train_accuracy)
        epoch_train_losses.append(train_loss)
        
        test_loss = 0
        current_correct_num = 0
        total_obs_test = len(test_loader.dataset)
        for i, (test_x, test_y) in enumerate(test_loader):
            this_batch_size = test_x.shape[0]
            if modeltype == "curve":
                model.sample_t()
                model.sample_model()
                model.model_theta.eval()
                model.sampled_model.eval()

            else:
                model.eval()
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            test_pred = model(test_x)
            pred_loss = loss_fn(test_pred, test_y)
            test_loss += pred_loss.item() * this_batch_size / total_obs_test
            pred_class = torch.argmax(test_pred, dim=-1)
            current_correct_num += (pred_class == test_y).sum()
        test_accuracy = current_correct_num.item() / total_obs_test
        epoch_test_accuracy.append(test_accuracy)
        test_losses.append(test_loss)

        test_loss_batch_number += [batch_number]
        if epoch % print_every_n_epoch==0:
            txt = f"epoch = {epoch} \ttrain loss: {epoch_train_losses[-1]:2.5f}, test loss: {test_losses[-1]:2.5f}" + \
                f", test accuracy: {test_accuracy*100:2.2f}, lr: {optimizer.param_groups[0]['lr']:4e}"
            if verbose: logger_info(txt)
        if optimizer.param_groups[0]['lr']<1e-10:
            logger_info("Stopping training because lr is too low")
            break

    plots={}
    if plot:
        os.makedirs(plotpath, exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(train_loss_batch_number, all_train_losses, label="train loss")
        ax.plot(test_loss_batch_number, test_losses, label="test loss")
        ax.plot(test_loss_batch_number, epoch_train_losses, label="train avg over epoch")
        ax.set_xlim(0, batch_number)
        ax.set_ylim(min(all_train_losses), torch.tensor(all_train_losses).quantile(.99).item())
        ax.legend()
        ax.set_title("loss")
        fig.savefig(f"{plotpath}/loss.png")
        plots["loss"] = (fig, ax)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(lrs)
        ax.set_title("learning rate")
        plt.savefig(f"{plotpath}/learning_rate.png")
        plots["learning_rate"] = (fig, ax)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(test_loss_batch_number, epoch_train_accuracy, label="train accuracy")
        ax.plot(test_loss_batch_number, epoch_test_accuracy, label="test accuracy")
        ax.set_xlim(0, batch_number)
        ax.legend()
        ax.set_title("accuracy")
        plt.savefig(f"{plotpath}/accuracy.png")
        plots["accuracy"] = (fig, ax)
        plt.close()

    return model, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots

