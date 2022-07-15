import time
import random
import sys
### global imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from args import args
from utils import *
import datasets
import few_shot_eval
import resnet
# import wideresnet
# import resnet12
# import s2m2
# import mlp

if args.ema > 0:
    from torch_ema import ExponentialMovingAverage
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)
if args.wandb:
    import wandb


### global variables that are used by the train function
last_update, criterion = 0, torch.nn.CrossEntropyLoss()

if args.output!='':
    output_file = args.output+str(args.n_shots[0])+'shot_'+args.model+'_'+args.preprocessing+'.txt'



### function to either use criterion based on output and target or criterion_episodic based on features and target
def crit(output, features, target):
    if args.episodic:
        return criterion_episodic(features, target)
    else:
        if args.label_smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes = num_classes, smoothing = args.label_smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        return criterion(output, target)


### main train function
def train(model, train_loader, optimizer, epoch, scheduler, mixup = False, mm = False):
    model.train()
    global last_update
    losses, total = 0., 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)

        # reset gradients
        optimizer.zero_grad()

        if mm: # as in method S2M2R, to be used in combination with rotations
            # if you do not understand what I just wrote, then just ignore this option, it might be better for now
            new_chunks = []
            sizes = torch.chunk(target, len(args.devices))
            for i in range(len(args.devices)):
                new_chunks.append(torch.randperm(sizes[i].shape[0]))
            index_mixup = torch.cat(new_chunks, dim = 0)
            lam = np.random.beta(2, 2)
            output, features = model(data, index_mixup = index_mixup, lam = lam)
            if args.rotations:
                output, _ = output
            loss_mm = lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])
            loss_mm.backward()

        if args.rotations: # generate self-supervised rotations for improved universality of feature vectors
            bs = data.shape[0] // 4
            target_rot = torch.LongTensor(data.shape[0]).to(args.device)
            target_rot[:bs] = 0
            data[bs:] = data[bs:].transpose(3,2).flip(2)
            target_rot[bs:2*bs] = 1
            data[2*bs:] = data[2*bs:].transpose(3,2).flip(2)
            target_rot[2*bs:3*bs] = 2
            data[3*bs:] = data[3*bs:].transpose(3,2).flip(2)
            target_rot[3*bs:] = 3

        if mixup and args.mm: # mixup or manifold_mixup
            index_mixup = torch.randperm(data.shape[0])
            # lam = random.random()       
            lam = 0.5     
            if args.mm:
                output, features = model(data, index_mixup = index_mixup, lam = lam)
            else:
                data_mixed = lam * data + (1 - lam) * data[index_mixup]
                output, features = model(data_mixed)
            if args.rotations:
                output, output_rot = output
                loss = ((lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])) + (lam * crit(output_rot, features, target_rot) + (1 - lam) * crit(output_rot, features, target_rot[index_mixup]))) / 2
            else:
                loss = lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])
        else:
            output, features = model(data)
            if args.rotations:
                output, output_rot = output
                loss = 0.5 * crit(output, features, target) + 0.5 * crit(output_rot, features, target_rot)                
            else:
                loss = crit(output, features, target)

        # backprop loss
        loss.backward()
            
        losses += loss.item() * data.shape[0]
        total += data.shape[0]
        # update parameters
        optimizer.step()
        scheduler.step()
        if args.ema > 0:
            ema.update()

        if few_shot and args.dataset_size > 0:
            length = args.dataset_size // args.batch_size + (1 if args.dataset_size % args.batch_size != 0 else 0)
        else:
            length = len(train_loader)
        # print advances if at least 100ms have passed since last print
        if (batch_idx + 1 == length) or (time.time() - last_update > 0.1) and not args.quiet:
            print("\repoch:{:4d} {:4d} / {:4d} loss: {:.5f} time: {:s} lr: {:.5f} ".format(epoch, 1 + batch_idx, length, losses / total, format_time(time.time() - start_time), float(scheduler.get_last_lr()[0])), end = "")
            last_update = time.time()

        if few_shot and total >= args.dataset_size and args.dataset_size > 0:
            break
            
    if args.wandb:
        wandb.log({"epoch":epoch, "train_loss": losses / total})

    # return train_loss
    return { "train_loss" : losses / total}


# function to train a model using args.epochs epochs
# at each args.milestones, learning rate is multiplied by args.gamma
def train_complete(model, loaders, mixup = False):
    global start_time
    start_time = time.time()

    if few_shot:
        train_loader, train_clean, val_loader, novel_loader = loaders
        for i in range(len(few_shot_meta_data["best_val_acc"])):
            few_shot_meta_data["best_val_acc"][i] = 0
    else:
        train_loader, val_loader, test_loader = loaders

    lr = args.lr

    for epoch in range(args.epochs + args.manifold_mixup):

        if few_shot and args.dataset_size > 0:
            length = args.dataset_size // args.batch_size + (1 if args.dataset_size % args.batch_size != 0 else 0)
        else:
            length = len(train_loader)

        # print('cosine',args.cosine)
        if (args.cosine and epoch % args.milestones[0] == 0) or epoch == 0:
            if lr < 0:
                optimizer = torch.optim.Adam(model.parameters(), lr = -1 * lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
            if args.cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.milestones[0] * length)
                lr = lr * args.gamma
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(np.array(args.milestones) * length), gamma = args.gamma)

        train_stats = train(model, train_loader, optimizer, (epoch + 1), scheduler, mixup = mixup, mm = epoch >= args.epochs)        

        if args.save_model != "" and not few_shot:
            if len(args.devices) == 1:
                torch.save(model.state_dict(), args.save_model)
            else:
                torch.save(model.module.state_dict(), args.save_model)
        
        if (epoch + 1) > args.skip_epochs:
            if args.ema > 0:
                ema.store()
                ema.copy_to()
            res = few_shot_eval.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
            if args.ema > 0:
                ema.restore()
            for i in range(len(args.n_shots)):
                print("val-{:d}: {:.2f}% (history best {:.2f}%) ".format(args.n_shots[i], 100 * res[i][0], 100 * few_shot_meta_data["best_val_acc"][i]), end = '')
                if args.wandb:
                    wandb.log({'epoch':epoch, f'val-{args.n_shots[i]}':res[i][0], f'best-val-{args.n_shots[i]}':few_shot_meta_data["best_val_acc"][i]})
            print()


    if args.epochs + args.manifold_mixup <= args.skip_epochs:
        if args.ema > 0:
            ema.store()
            ema.copy_to()
        res = few_shot_eval.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data)
        if args.ema > 0:
            ema.restore()
    return few_shot_meta_data


### process main arguments
loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
### initialize few-shot meta data
if few_shot:
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    if args.dataset.lower() in ["tieredimagenet", "cubfs"]:
        elements_train, elements_val, elements_novel = elements_per_class
    else:
        elements_val, elements_novel = [elements_per_class] * val_classes, [elements_per_class] * novel_classes
        elements_train = None
    print("Dataset contains",num_classes,"base classes,",val_classes,"val classes")
    print("Generating runs... ", end='')

    val_runs = list(zip(*[few_shot_eval.define_runs(args.n_ways, s, args.n_queries, val_classes, elements_val) for s in args.n_shots]))
    val_run_classes, val_run_indices = val_runs[0], val_runs[1]

    novel_runs = list(zip(*[few_shot_eval.define_runs(args.n_ways, s, args.n_queries, novel_classes, elements_novel) for s in args.n_shots]))
    novel_run_classes, novel_run_indices = novel_runs[0], novel_runs[1]

    print("done.")
    few_shot_meta_data = {
        "elements_train":elements_train,
        "val_run_classes" : val_run_classes,
        "val_run_indices" : val_run_indices,
        "novel_run_classes" : novel_run_classes,
        "novel_run_indices" : novel_run_indices,
        "best_val_acc" : [0] * len(args.n_shots),
        "best_val_acc_ever" : [0] * len(args.n_shots),
        "best_novel_acc" : [0] * len(args.n_shots)
    }


### prepare stats
run_stats = {}
if args.output != "":
    f = open(output_file, "a")
    f.write(str(args))
    f.close()

model = resnet.ResNet18(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
if args.ema > 0:
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
test_stats = train_complete(model, loaders, mixup = args.mixup)
# model.to(args.device)
# test_stats = train_complete(model, loaders, mixup = args.mixup)



