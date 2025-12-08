from __future__ import print_function

import argparse
import logging
import os

from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    is_distributed = dist.is_available() and dist.is_initialized()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        should_log = batch_idx % args.log_interval == 0
        if is_distributed:
            should_log = should_log and dist.get_rank() == 0
        if should_log:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            logging.info(msg)
            print(msg)


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    is_distributed = dist.is_available() and dist.is_initialized()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = float(correct) / len(test_loader.dataset)
    should_log = True
    if is_distributed:
        should_log = dist.get_rank() == 0
    if should_log:
        msg = "{{metricName: accuracy, metricValue: {:.4f}}};{{metricName: loss, metricValue: {:.4f}}}\n".format(
            accuracy, test_loss)
        logging.info(msg)
        print(msg)
    return accuracy


def train_pytorch():
    """Kubeflow Trainer를 위한 PyTorch DDP 학습 함수"""
    import os

    import torch
    from torch import nn
    import torch.nn.functional as F

    from torchvision import datasets, transforms
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    # [1] Configure CPU/GPU device and distributed backend.
    # Kubeflow Trainer will automatically configure the distributed environment.
    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(
        "Distributed Training with WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}.".format(
            world_size,
            rank,
            local_rank,
        )
    )

    # [2] Define PyTorch CNN Model to be trained.
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # [3] Attach model to the correct device.
    device_obj = torch.device(f"{device}:{local_rank}")
    model = nn.parallel.DistributedDataParallel(Net().to(device_obj))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # [4] Get the Fashion-MNIST dataset on local_rank=0 process.
    if local_rank == 0:
        dataset = datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    dist.barrier()
    dataset = datasets.FashionMNIST(
        "./data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=train_sampler,
    )

    # Test dataset
    test_dataset = datasets.FashionMNIST(
        "./data",
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        sampler=test_sampler,
    )

    # [5] Define the training loop.
    epochs = int(os.getenv("EPOCHS", 10))
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # 셔플링을 위해 epoch 설정
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Attach tensors to the device.
            inputs, labels = inputs.to(device_obj), labels.to(device_obj)

            # Forward pass
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device_obj), target.to(device_obj)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = float(correct) / len(test_loader.dataset)
        if dist.get_rank() == 0:
            print(
                "Test Epoch: {} Accuracy: {:.4f} ({}/{})".format(
                    epoch, accuracy, correct, len(test_loader.dataset)
                )
            )

    # Save model (rank 0만)
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "mnist_cnn.pt")
        print("Model saved to mnist_cnn.pt")

    # Wait for the training to complete and destroy to PyTorch distributed process group.
    dist.barrier()
    if dist.get_rank() == 0:
        print("Training is finished")
    dist.destroy_process_group()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")

    if dist.is_available():
        parser.add_argument("--backend", type=str, help="Distributed backend",
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=args.log_path)

    # Legacy single-process training (for backward compatibility)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data",
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor()
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data",
                              train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor()
                              ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Check if running in Kubeflow Trainer environment
    if os.getenv("LOCAL_RANK") is not None or os.getenv("RANK") is not None:
        train_pytorch()
    else:
        main()
