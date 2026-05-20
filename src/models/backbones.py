import os
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str, unfrozen_layers:int=None):
        super(FeatureExtractor, self).__init__()
        self.num_ftrs=None
        self.__get_backbone__(backbone)
        self.__unfreeze_layers__(unfrozen_layers)

    def __get_backbone__(self,backbone:str):
        match backbone:
            case "resnet18":
                weights = ResNet18_Weights.IMAGENET1K_V1
                base_model = resnet18(weights=weights)
                self.num_ftrs = base_model.fc.in_features
                self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            case "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V2
                base_model = resnet50(weights=weights)
                self.num_ftrs = base_model.fc.in_features
                self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            case "vitb16":
                weights = ViT_B_16_Weights.IMAGENET1K_V1
                base_model = vit_b_16(weights=weights)
                base_model.heads = nn.Identity()
                self.num_ftrs = base_model.heads.head.in_features
                self.backbone = base_model
            case _:
                print("Não encontrado")
    
    def __unfreeze_layers__(self, n_layers:int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if n_layers == 0:
            return
        if n_layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        indexed = [
            (idx, name, module)
            for idx, (name, module) in enumerate(self.named_modules())
        ]
        convs = [
            (idx, name, module)
            for idx, name, module in indexed
            if isinstance(module, nn.Conv2d)
        ]
        if len(convs) < n_layers:
            raise ValueError("O modelo não contém camadas Conv2d suficientes.")

        conv_idx = convs[-n_layers][0]
        for idx, _, module in indexed:
            if idx >= conv_idx:
                for p in module.parameters(recurse=False):
                    p.requires_grad = True


    def forward(self, x):
        x = self.backbone(x)
        x = x.logits
        return x.view(x.size(0), -1)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "extractor_weight.pt")
        torch.save(self.backbone.state_dict(), weight_path)
    
    def load(self, path):
        weight_path = os.path.join(path, "extractor_weight.pt")
        if not os.path.exists(weight_path):
            return
        self.backbone.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

    def get_trainable_params(self):
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def get_model_size(self):
        param_size = 0
        for param in self.backbone.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.backbone.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024**2