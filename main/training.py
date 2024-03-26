import math
import pandas as pd
import sys
from argparse import Namespace
from typing import Tuple
from unittest import result
from backbones import get_backbone
from models import get_model
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils import get_loss
from utils.loggers import *
from utils.status import ProgressBar
from utils.visualization import vis_acc_mat, vis_curves, get_embeddings, vis_embeddings
from torchmetrics.classification import MulticlassAUROC, F1Score

import ipdb

try:
    import wandb
except ImportError:
    wandb = None


def evaluate(
    model: ContinualModel, 
    dataset: ContinualDataset, 
    i=None,
):
    """
    Evaluates the accuracy of the model for each task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the task-il accuracy for each task
    """
    auc = MulticlassAUROC(num_classes=2)
    status = model.net.training
    model.net.eval()
    accs = []
    aucs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        # if the task id is specified, then only evaluate the model on the i-th task.
        if i is not None and k != i:
            continue
        correct, total = 0.0, 0.0
        for data in (test_loader):
            with torch.no_grad():
                inputs, labels, _ = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs: torch.Tensor = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                auc.update(outputs.softmax(dim=1), labels)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                
        accs.append(correct / total)
        aucs.append(auc.compute().item())
        auc = MulticlassAUROC(num_classes=2)

    model.net.train(status)
    
    # print metrics with 4 decimal points
    accs = [round(a, 4) for a in accs]
    aucs = [round(a, 4) for a in aucs]
    # print("F-1:", f1s)
    return accs, aucs


def train(
    model: ContinualModel, 
    dataset: ContinualDataset, 
    args: Namespace,
    scheduler: object = None,
):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)


    backbone = get_backbone(
        backbone_name=args.backbone, 
        indim=dataset.INDIM, 
        hiddim=args.hiddim, 
        outdim=dataset.N_CLASSES_PER_TASK,
        args=args
    )
    loss_name = get_loss(loss_name=args.loss)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if not args.wandb_name:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)

    # full task performance matrix
    results = [] 
    results_auc = []

    if not args.disable_log:
        logger = Logger(dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    # the random baseline for the forward transfer
    if not args.ignore_other_metrics:
        random_results_class = evaluate(model, dataset)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        print(f'Starting Task {t+1}/{dataset.N_TASKS}')
        # model = get_model(args, backbone, loss_name, dataset.get_transform())
        # model.net.to(model.device)
        # model.current_task = t
        # print("Model has been reset.")
        model.net.train()
        cur_train_loader, _, next_train_loader, _ = dataset.get_data_loaders()

        # the procedure before each task.
        # e.g., store the previous logits in the buffer.
        if hasattr(model, 'begin_task'):
            model.begin_task(cur_train_loader, next_train_loader)

        scheduler = dataset.get_scheduler(model, args)

        # some tricks of increasing the number of epochs 
        real_epochs = get_epochs(model.args.n_epochs, t+1, model.args.epoch_scaling)
        if t == 0:
            real_epochs = 1
        # if t == 0 and args.model in ('multialignsupcon'): 
        #     real_epochs = 100
        for epoch in range(real_epochs):
            # if it's the last task: return None.
            # if t == 0:
            #     break
            try: cur_iter, next_iter = iter(cur_train_loader), iter(next_train_loader)
            except: cur_iter, next_iter = iter(cur_train_loader), None
            # guarantee the current training task is completed exactly 1 epoch.
            for i in range(len(cur_train_loader)): 
                # debug: only try a few steps
                if args.debug_mode and i > 3:
                    break
                
                # use iter().next() to get the next batch of the data
                cur_x, cur_y, cur_idx = next(cur_iter)
                cur_data = cur_x.to(model.device), cur_y.to(model.device), cur_idx

                loss = model.meta_observe(cur_data, None)
                assert not math.isnan(loss)

                progress_bar.prog(i, len(cur_train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        # the procedure after each task.
        # e.g., update the memory bank.
        if hasattr(model, 'end_task'):
            model.end_task(cur_train_loader, next_train_loader)
        
        if hasattr(model, 'log') and not args.nowand:
            model.log(cur_train_loader, wandb)
        
        accs, aucs = evaluate(model, dataset)
        # print("Avg-acc:", round(sum(accs[:t+1]) / len(accs[:t+1]), 4))
        # print("Avg-auc:", round(sum(aucs[:t+1]) / len(aucs[:t+1]), 4))
        results.append(accs)
        results_auc.append(aucs)

        if not args.disable_log:
            acc1 = logger.add_average_i(results=results, i=t, type="acc")
            # acc2 = logger.add_average_iplus1(results=results, i=t)
            auc = logger.add_average_i(results=results_auc, i=t, type="auc")

        if not args.nowand:
            d2={'RESULT_mean_accs': acc1, 'RESULT_mean_auc': auc,
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs)},
                **{f'RESULT_class_auc_{i}': a for i, a in enumerate(aucs)}
                } # on all (not just previous) tasks

            # visualize the embedding after each task.
            if args.visualize:
                dic = get_embeddings(model=model, dataset=dataset, n=t+1)
                d2[f'embeddings (up to domain {t+1})'] = wandb.Image(vis_embeddings(dic))

            wandb.log(d2)

        # checkpointing the backbone model.
        if args.checkpoint: # by default the checkpoints folder is checkpoints
            save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.backbone}/{args.seed}'
            create_if_not_exists(save_folder)
            file_name = f'domain-{t+1}.pt'
            model.save(os.path.join(save_folder, file_name)) # only save the backbone params.

        # step to the next task. (dataset.i += 1)
        dataset.step() 

        print("Training finished.")
        print("Acc matrix: ")
        for j, r in enumerate(results):
            print(f"{j}: {r}")
        print("AUC matrix: ")
        for j, r in enumerate(results_auc):
            print(f"{j}: {r}")

        print("ACC averages: ")
        from utils.metrics import average_i
        for i in range(len(results)):
            print(f"{i}:", round(average_i(results, i), 4))
        print("AUC averages: ")
        for i in range(len(results_auc)):
            print(f"{i}:", round(average_i(results_auc, i), 4))
        print()
        # np.savetxt(f"csvs/acc_matrix_{args.new_pkl.split('/')[1].split('.')[0].split('+')[-1]}.csv", np.round(np.array(results), 4), delimiter=',')
        # np.savetxt(f"csvs/auc_matrix_{args.new_pkl.split('/')[1].split('.')[0].split('+')[-1]}.csv", np.round(np.array(results_auc), 4), delimiter=',')
    # calculate the CL-specific metrics 
    if not args.disable_log:
        logger.add_acc_matrix(results=results)
        logger.add_auc_matrix(results=results_auc)
        logger.add_bwt(results)
        logger.add_forgetting(results)
    if not args.ignore_other_metrics:
        logger.add_fwt(results, random_results_class[0])

    np.savetxt(f"acc_matrix_{args.backbone}_150ep.csv", np.round(np.array(results), 4), delimiter=',')
    np.savetxt(f"auc_matrix_{args.backbone}_150ep.csv", np.round(np.array(results_auc), 4), delimiter=',')
        
    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            # d['acc_matrix'] = wandb.Image(vis_acc_mat(d['acc_matrix']))
            d['wandb_url'] = wandb.run.get_url()
            # if args.visualize:
            #     dic = get_embeddings(model=model, dataset=dataset)
            #     d['embeddings (all domains)'] = wandb.Image(vis_embeddings(dic))
            wandb.log(d)

    if not args.nowand:
        wandb.finish()


def get_epochs(base_epoch, t, scaling='const'):
    if scaling == 'const':
        return base_epoch
    elif scaling == 'linear':
        return math.ceil(t * base_epoch)
    elif scaling == 'sqrt':
        return math.ceil(math.sqrt(t) * base_epoch)


