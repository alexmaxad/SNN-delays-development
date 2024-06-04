from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from config import Config
from snn_delays import SnnDelays
from snn_delays_daleian import SnnDelays_Dale
from snn_daleian import SNN_Dale
import torch
from snn import SNN
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

config = Config()

if config.model_type == 'snn':
    model = SNN(config).to(device)
if config.model_type == 'snn_delays_dale':
    model = SnnDelays_Dale(config).to(device)
if config.model_type == 'snn_dale':
    model = SNN_Dale(config).to(device)
else:
    model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()


print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")


if config.dataset == 'shd':
    train_loader, valid_loader = SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'ssc':
    train_loader, valid_loader, test_loader = SSC_dataloaders(config)
elif config.dataset == 'gsc':
    train_loader, valid_loader, test_loader = GSC_dataloaders(config)
else:
    raise Exception(f'dataset {config.dataset} not implemented')


model.train_model(train_loader, valid_loader, test_loader, device)