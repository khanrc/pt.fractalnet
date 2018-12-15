""" Test module """
import os
import glob
import pickle
import torch
import torch.nn as nn
import utils
from datasets import get_dataset
from config import TestConfig
from fractal import FractalNet
from collections import OrderedDict


config = TestConfig()
device = torch.device("cuda")
results = OrderedDict()
if os.path.exists("test.p"):
    results = pickle.load(open("test.p", "rb"))


def pprint_res():
    print("--- Results ---")
    for name, (full, deepest) in results.items():
        print("{:50s}: full = {:.4%}, deepest = {:.4%}".format(name, full, deepest))


def test(valid_loader, model, criterion, deepest):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X, deepest=deepest)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

    print("Test: Final Prec@1 {:.4%} Loss {:.4f}".format(top1.avg, losses.avg))

    return top1.avg


if __name__ == "__main__":
    if config.name in results:
        print("{} is already exists: {}".format(config.name, results[config.name]))
        pprint_res()
        print("Test again ...")

    torch.cuda.set_device(0)
    _, valid_data, data_shape = get_dataset(config.data, config.data_path, aug_lv=0)
    criterion = nn.CrossEntropyLoss().to(device)
    model = FractalNet(data_shape, config.columns, config.init_channels, p_ldrop=0.,
                       dropout_probs=[0.]*config.blocks, gdrop_ratio=0., gap=config.gap,
                       pad_type=config.pad, doubling=config.doubling,
                       dropout_pos=config.dropout_pos)
    model = model.to(device)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size,
                                               shuffle=False, num_workers=config.workers,
                                               pin_memory=True)

    path = os.path.join("./runs/", config.name, "best.pth.tar")

    if not os.path.exists(path):
        raise ValueError("The name `{}` is not exists".format(config.name))

    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    print("full model ...")
    full_acc = test(valid_loader, model, criterion, deepest=False)
    print("deepest ...")
    deepest_acc = test(valid_loader, model, criterion, deepest=True)

    results[config.name] = [full_acc, deepest_acc]
    pickle.dump(results, open("test.p", "wb"))

    #print(results)
    pprint_res()
