""" Trainer """
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import Config
import utils
from fractal import FractalNet
import torchvision.datasets as dset
import torchvision.transforms as transforms


config = Config()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.name, "tb"))
writer.add_text('config', config.as_markdown(), 0)

# logger
logger = utils.get_logger(os.path.join(config.name, "{}.log".format(config.name)))
logger.info("Run options = {}".format(sys.argv))
config.print_params(logger.info)

# copy scripts
utils.copy_scripts("*.py", config.name)


def main():
    logger.info("Logger is set - training start")

    # set gpu device id
    logger.info("Set GPU device {}".format(config.gpu))
    torch.cuda.set_device(config.gpu)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # dataset class
    if config.data == 'cifar10':
        dset_cls = dset.CIFAR10
        data_shape = (3, 32, 32, 10)
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif config.data == 'cifar100':
        dset_cls = dset.CIFAR100
        data_shape = (3, 32, 32, 100)
        raise NotImplementedError("required: MEAN, STD")
    else:
        raise ValueError(config.data)

    # data transforms
    trn_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


    # get dataset
    train_data = dset_cls(config.data_path, train=True, download=True, transform=trn_transforms)
    valid_data = dset_cls(config.data_path, train=False, download=True, transform=val_transforms)

    # build model
    criterion = nn.CrossEntropyLoss().to(device)
    model = FractalNet(data_shape, config.columns, channels=config.channels,
                       p_local_drop=config.p_local_drop, dropout_probs=config.dropout_probs,
                       global_drop_ratio=config.global_drop_ratio, gap=config.gap,
                       init=config.init, pad_type=config.pad, doubling=config.doubling)
    model = model.to(device)

    # model size
    m_params = utils.param_size(model)
    logger.info("Models:\n{}".format(model))
    logger.info("Model size (# of params) = {:.3f} M".format(m_params))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum)

    # setup data loader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_milestone)

    # evaluate
    #  if config.evaluate:
    #      if os.path.isfile(config.evaluate):
    #          ckpt = torch.load(config.evaluate)
    #          model.load_state_dict(ckpt)

    #      top1 = validate(valid_loader, model, criterion, epoch=0, cur_step=0)

    best_top1 = 0.
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()

        # training
        train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model.state_dict(), config.name, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch+1, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        # gradient clipping
        #nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
