import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cifar10.vgg import VGG
import nni
from nni.compression.torch import LevelPruner, SlimPruner, FPGMPruner, L1FilterPruner, \
    L2FilterPruner, AGP_Pruner, ActivationMeanRankFilterPruner, ActivationAPoZRankFilterPruner

prune_config = {
    'l1filter': {
        'dataset_name': 'cifar10',
        'model_name': 'vgg16',
        'pruner_class': L1FilterPruner,
        'input_shape': [64, 3, 32, 32],
        'config_list': [{
            'sparsity': 0.1,
            'op_types': ['Conv2d'],
            'op_names': ['excluded_layer_name'],
            'exclude': True
        }]
    },
    'l1filter2': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'pruner_class': L1FilterPruner,
        'input_shape': [64, 1, 28, 28],
        'config_list': [{
            'sparsity': 0.1,
            'op_types': ['Conv2d'],
            'op_names': ['excluded_layer_name'],
            'exclude': True
        }]
    }
}


def get_data_loaders(dataset_name='mnist', batch_size=128):
    assert dataset_name in ['cifar10', 'mnist']

    if dataset_name == 'cifar10':
        ds_class = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.MNIST
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        ds_class = datasets.MNIST
        MEAN, STD = (0.1307,), (0.3081,)

    train_loader = DataLoader(
        ds_class(
            './data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ds_class(
            './data', train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_name='naive'):
    assert model_name in ['naive', 'vgg16', 'vgg19']

    if model_name == 'naive':
        return NaiveModel()
    elif model_name == 'vgg16':
        return VGG(16)
    else:
        return VGG(19)


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('Loss: {}  Accuracy: {})\n'.format(test_loss, acc))
    return acc


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    model_name = prune_config[args.pruner_name]['model_name']
    dataset_name = prune_config[args.pruner_name]['dataset_name']
    train_loader, test_loader = get_data_loaders(dataset_name, args.batch_size)
    model = create_model(model_name).cuda()
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print('loading checkpoint {} ...'.format(args.resume_from))
        model.load_state_dict(torch.load(args.resume_from))
        test(model, device, test_loader)
    else:
        # STEP.1 train from scratch
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        if args.multi_gpu and torch.cuda.device_count():
            model = nn.DataParallel(model)
        print('start training')
        pretrain_model_path = os.path.join(
            args.checkpoints_dir, 'pretrain_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
        for epoch in range(args.pretrain_epochs):
            train(model, device, train_loader, optimizer)
            test(model, device, test_loader)
        torch.save(model.state_dict(), pretrain_model_path)

    # STEP.2 Automatic Prune
    print('start model pruning...')
    # pruner needs to be initialized from a model not wrapped by DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    # STEP.2.1 sensitivity analysis
    outdir = './sens_analysis'
    csv_file_path = os.path.join(outdir, 'sens.csv')
    if os.path.exists(csv_file_path):
        from nni.compression.torch.utils.sensitivity_analysis import load_csv
        sensitivity = load_csv(csv_file_path)
    else:
        from nni.compression.torch.utils.sensitivity_analysis import SensitivityAnalysis
        s_analyzer = SensitivityAnalysis(model=model, val_func=test)
        sensitivity = s_analyzer.analysis(val_args=[model, device, test_loader])
        os.makedirs(outdir, exist_ok=True)
        s_analyzer.export(csv_file_path)

    # STEP.2.2 compress
    from nni.compression.torch.examples import compress
    config_list = prune_config[args.pruner_name]['config_list']
    dummy_input = [torch.randn(prune_config[args.pruner_name]['input_shape']).to("cpu")]
    model = model.to("cpu")
    with torch.onnx.set_training(model, False):
        trace = torch.jit.trace(model, dummy_input)
        torch._C._jit_pass_inline(trace.graph)
    model, fixed_mask = compress(model.to('cpu'), dummy_input, prune_config[args.pruner_name]['pruner_class'],
                                 config_list, ori_metric=0.85, metric_thres=0.10, sensitivity=sensitivity, trace=trace)
    pruned_model_path = os.path.join(args.checkpoints_dir,
                                     'pruned_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
    mask_path = os.path.join(args.checkpoints_dir,
                             'mask_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
    torch.save(fixed_mask, mask_path)
    torch.save(model, pruned_model_path)
    from thop import profile
    macs, params = profile(model, inputs=dummy_input, verbose=False)
    print("MACs: {} G, Params: {} M".format(macs / 1000000000, params / 100000))
    model = model.to(device)

    # STEP.3 Finetune
    print('start finetuning')
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(args.prune_epochs):
        # pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer_finetune)
        test(model, device, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruner_name", type=str, default="l1filter", help="pruner name")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="training epochs before model pruning")
    parser.add_argument("--prune_epochs", type=int, default=10, help="training epochs for model pruning")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="checkpoints directory")
    parser.add_argument("--resume_from", type=str, default=None, help="pretrained model weights")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs for training")

    args = parser.parse_args()
    main(args)
