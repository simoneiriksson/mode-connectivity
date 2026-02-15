import torch
from matplotlib import pyplot as plt
import numpy as np
import os

def train(model, train_loader=None, test_loader=None, optimizer=None, scheduler=None, epochs=1, 
          loss_fn=None, prior_sigma=None, target_sigma=None, sgd_trace=False, sgd_trace_every_n_step=100, num_sgd_trace=10, sgd_trace_lr=None, 
          device="cpu", logger_info=None,
          plot=False, plotpath=None, verbose = False, modeltype="regression"):
    if logger_info == None: logger_info=print
    test_losses = []
    epoch_train_losses = []
    all_train_losses = []
    epoch_test_accuracy = []
    epoch_train_accuracy =[]
    epoch_pred_loss = []
    epoch_train_grad = []
    step_train_grad = []
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
    sgd_tracing_done = False
    if target_sigma == None:
        target_scale = 1
    else:
        target_scale = 1/ (2*target_sigma**2)
    while (epoch < epochs) or (sgd_tracing_done==False):
        epoch += 1
        train_loss = 0
        accum_pred_loss = 0
        current_correct_num = 0
        epoch_sumsqr_gradient = 0
        for x, y in train_loader:
            batch_number += 1
            optimizer.zero_grad()
            if modeltype == "curve":
                model.sample_t()
                model.sample_model()
                model.model_theta.train()
            else:
                model.train()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if prior_sigma != None:
                num_params = sum(p.numel() for p in model.parameters())
                l2_norm = sum(p.pow(2).sum() for p in model.parameters()) / (2 * prior_sigma**2) 
            else:
                l2_norm = 0
            pred_loss = loss_fn(pred, y) * target_scale
            loss = pred_loss + l2_norm/total_obs_train # Assuming tha twe use a mean-loss
            loss.backward()
            train_loss += loss.item() / num_batches_train
            accum_pred_loss += pred_loss.item() / num_batches_train
            pred_class = torch.argmax(pred, dim=-1)
            current_correct_num += (pred_class == y).sum()
            all_train_losses.append(loss.item())
            train_loss_batch_number += [batch_number]

            sumsqr_gradient = sum([(p.grad**2).sum() for p in model.parameters()]).item()
            step_train_grad.append(sumsqr_gradient)
            epoch_sumsqr_gradient += sumsqr_gradient

            lrs.append(optimizer.param_groups[0]['lr'].item())
            optimizer.step()
            if sgd_trace and epoch > epochs and batch_number % sgd_trace_every_n_step == 0:
                sgd_trace_models.append(torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach())
                logger_info(f"Taking SGD trace number {len(sgd_trace_models)} at: {batch_number = }")
                if len(sgd_trace_models) >= num_sgd_trace:
                    sgd_tracing_done = True
                if sgd_trace_lr:
                    optimizer.param_groups[0]['lr'] = sgd_trace_lr
            elif scheduler:
                scheduler.step()
            #if verbose: logger_info(f"epoch = {epoch}, \tbatch= {batch_number}/{num_batches-1}, train loss: {train_loss:2.5f}, lr: {optimizer.param_groups[0]['lr']:4e}")
        train_accuracy = current_correct_num.item() / total_obs_train
        epoch_train_accuracy.append(train_accuracy)

        epoch_train_losses.append(train_loss)
        epoch_pred_loss.append(accum_pred_loss)

        epoch_train_grad.append(epoch_sumsqr_gradient/num_batches_train)

        test_loss = 0
        current_correct_num = 0
        total_obs_test = len(test_loader.dataset)
        num_batches_test = len(test_loader)
        for i, (test_x, test_y) in enumerate(test_loader):
            if modeltype == "curve":
                model.sample_t()
                model.sample_model()
                model.model_theta.eval()
            else:
                model.eval()
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            test_pred = model(test_x)
            if prior_sigma != None:
                num_params = sum(p.numel() for p in model.parameters())
                l2_norm = sum(p.pow(2).sum() for p in model.parameters()) / (2 * prior_sigma**2) 
            else:
                l2_norm = 0
            pred_loss = loss_fn(test_pred, test_y) * target_scale
            loss = pred_loss + l2_norm/total_obs_test # Assuming tha twe use a mean-loss
            test_loss += loss.item() / num_batches_test
            pred_class = torch.argmax(test_pred, dim=-1)
            current_correct_num += (pred_class == test_y).sum()
        test_accuracy = current_correct_num.item() / total_obs_test
        epoch_test_accuracy.append(test_accuracy)
        test_losses.append(test_loss)
        test_loss_batch_number += [batch_number]
        txt = f"epoch = {epoch} \ttrain loss: {epoch_train_losses[-1]:2.5f}, train prediction loss: {epoch_pred_loss[-1]:2.5f}, regularization loss: {epoch_train_losses[-1]-epoch_pred_loss[-1]:2.5f}, norm of gradient: {epoch_train_grad[-1]:2.5f}, test loss: {test_losses[-1]:2.5f}" + \
              f", test accuracy: {test_accuracy*100:2.2f}, lr: {optimizer.param_groups[0]['lr']:4e}"
        if verbose: logger_info(txt)
        if optimizer.param_groups[0]['lr']<1e-10:
            logger_info("Stopping training because lr is too low")
            break
    if plot:
        fig, ax = plt.subplots()
        ax.plot(train_loss_batch_number, all_train_losses, label="train loss")
        ax.plot(test_loss_batch_number, test_losses, label="test loss")
        ax.plot(test_loss_batch_number, epoch_train_losses, label="train avg over epoch")
        ax.set_xlim(0, batch_number)
        ax.set_ylim(min(all_train_losses), torch.tensor(all_train_losses).quantile(.99).item())
        ax.legend()
        fig.savefig(f"{plotpath}/loss.png")
        plt.close()

        plt.plot(train_loss_batch_number, step_train_grad, label="step gradient")
        plt.plot(test_loss_batch_number, epoch_train_grad, label="epoch gradient")
        plt.ylim(0, torch.tensor(step_train_grad).quantile(.99).item())
        plt.savefig(f"{plotpath}/gradient.png")
        plt.close()

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

