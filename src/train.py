import torch
from matplotlib import pyplot as plt
import numpy as np
import os

def train(model, train_loader=None, test_loader=None, optimizer=None, scheduler=None, epochs=1, 
          loss_fn=None, prior_sigma=None, target_sigma=None, sgd_trace=False, sgd_trace_every_n_step=100, num_sgd_trace=10, sgd_trace_lr=None, 
          device="cpu", logger_info=print,
          plot=False, plotpath=None, verbose = False, modeltype="regression"):
    
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
    if sgd_trace:
        sgd_trace_models = []
    epoch=0
    if sgd_trace: sgd_tracing_done = False
    else: sgd_tracing_done = True
    if target_sigma == None:
        target_scale = 1
    else:
        target_scale = 1/ (2*target_sigma**2)
    while (epoch < epochs) or (sgd_tracing_done==False):
        epoch += 1
        train_loss = 0
        current_correct_num = 0
        for x, y in train_loader:
            this_batch_size = x.shape[0]
            batch_number += 1
            optimizer.zero_grad()
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
            scheduler.step()
            #if verbose: logger_info(f"epoch = {epoch}, \tbatch= {batch_number}, train loss: {train_loss:2.5f}, lr: {optimizer.param_groups[0]['lr']:4e}")
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
        txt = f"epoch = {epoch} \ttrain loss: {epoch_train_losses[-1]:2.5f}, test loss: {test_losses[-1]:2.5f}" + \
              f", test accuracy: {test_accuracy*100:2.2f}, lr: {optimizer.param_groups[0]['lr']:4e}"
        if verbose: logger_info(txt)
        if optimizer.param_groups[0]['lr']<1e-10:
            logger_info("Stopping training because lr is too low")
            break
    if plot:
        os.makedirs(plotpath, exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(train_loss_batch_number, all_train_losses, label="train loss")
        ax.plot(test_loss_batch_number, test_losses, label="test loss")
        ax.plot(test_loss_batch_number, epoch_train_losses, label="train avg over epoch")
        ax.set_xlim(0, batch_number)
        ax.set_ylim(min(all_train_losses), torch.tensor(all_train_losses).quantile(.99).item())
        ax.legend()
        fig.savefig(f"{plotpath}/loss.png")
        plt.close()

        # plt.plot(train_loss_batch_number, step_train_grad, label="step gradient")
        # plt.plot(test_loss_batch_number, epoch_train_grad, label="epoch gradient")
        # plt.ylim(0, torch.tensor(step_train_grad).quantile(.99).item())
        # plt.savefig(f"{plotpath}/gradient.png")
        # plt.close()

        plt.plot(lrs)
        plt.savefig(f"{plotpath}/learning_rate.png")
        plt.close()

        plt.plot(test_loss_batch_number, epoch_train_accuracy, label="train accuracy")
        plt.plot(test_loss_batch_number, epoch_test_accuracy, label="test accuracy")
        plt.legend()
        plt.savefig(f"{plotpath}/accuracy.png")
        plt.close()

    if sgd_trace:
        return model, sgd_trace_models, all_train_losses, lrs, epoch_train_losses, test_losses
    else:
        return model, all_train_losses, lrs, epoch_train_losses, test_losses

