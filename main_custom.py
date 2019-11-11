# !/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter

from lr_schedulers.flatten_cosanneal import FlatplusAnneal
from models.mxresnet import Mish, SimpleSelfAttention, noop
from optimizers.ranger913a import Ranger
from optimizers.rangerqh import RangerQH
from lossfuns.labelsmoothinloss import LabelSmoothingLoss

#####
# Customize training with the following a bug of tricks for image classification.
# https://medium.com/@lessw/how-we-beat-the-fastai-leaderboard-score-by-19-77-a-cbb2338fab5c
# Author: https://github.com/yumatsuoka
# baseline: Official MNIST example
# Additional features
# - [x] Global Average Pooling.
# - [x] Mish activate function.
# - [x] Self Attention module.
# - [x] Flat plus Cosine Annealing LR Scheduler.
# - [x] Ranger optimizer.
# - [x] Label Smoothing Loss
# - [x] TensorBoard
#####


class Net(nn.Module):
    def __init__(self, sa, gp, mish):
        super(Net, self).__init__()
        if sa:
            print("use Self Attention module")
        if gp:
            print("use global pooling")
        if mish:
            print("use Mish activate function")
            self.act_fn = Mish()
        else:
            self.act_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.sa1 = SimpleSelfAttention(20, ks=1) if sa else noop
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.sa2 = SimpleSelfAttention(50, ks=1) if sa else noop
        if gp:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(50, 50)
            self.fc2 = nn.Linear(50, 10)
        else:
            # original
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.act_fn(self.sa1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.act_fn(self.sa2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    return loss.item()


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss / len(test_loader),
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, 100 * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=["raner", "ranerqh", "sgd"],
        help="choose optimizer from choices",
    )

    parser.add_argument(
        "--sa", action="store_true", help="use self attention module",
    )

    parser.add_argument(
        "--mish", action="store_true", help="use Mish activate function"
    )

    parser.add_argument("--smooth", default=None, help="put float to smooth or sce")

    parser.add_argument("--gp", action="store_true", help="use global pooling")

    parser.add_argument("--fpa", action="store_true", help="use fpa scheduler")

    args = parser.parse_args()

    # Tensorboard
    writer = SummaryWriter()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    model = Net(args.sa, args.gp, args.mish).to(device)

    # choose loss function
    if args.smooth is None:
        print("use CrossEntropy Loss")
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print("use LabelSmoothing Loss")
        criterion = LabelSmoothingLoss(smoothing=float(args.smooth))

    # choose optimizer
    if args.optimizer == "sgd":
        print("use Momentum SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "ranger":
        print("use Ranger optimizer")
        optimizer = Ranger(model.parameters(), lr=args.lr)
    elif args.optimizer == "rangerqh":
        print("use RangerQH optimizer")
        optimizer = RangerQH(model.parameters(), lr=args.lr)

    # choose LR scheduler
    if args.fpa:
        print("use FlatplusAnneal scheduler")
        scheduler = FlatplusAnneal(optimizer, max_iter=args.epochs, step_size=0.7)
    else:
        print("use StepLR scheduler")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            args, model, device, train_loader, optimizer, epoch, criterion
        )
        test_loss, test_acc = test(args, model, device, test_loader, criterion)
        scheduler.step()
        writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalar("acc/test", test_acc, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # tfboard
    images, labels = next(iter(train_loader))
    grid = utils.make_grid(images)
    writer.add_image("images", grid, 0)
    writer.add_graph(model, images)
    writer.close()


if __name__ == "__main__":
    main()
