# -*- coding: utf-8 -*-
import os
import torch
from torchvision import datasets, transforms, models

# Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo pré-treinado
resnet50 = models.resnet50(pretrained=True)
resnet50.eval().to(device)

# Transformações para as imagens do ImageNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset e DataLoader
imagenet_validation_dir = '~/DeepLearningExamples/PyTorch/Classification/val'  # Atualize este caminho
imagenet_dataset = datasets.ImageFolder(imagenet_validation_dir, transform=transform)
imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=64, shuffle=False)

# Inferência em lote e exibição de resultados (simplificada)
with torch.no_grad():
    for inputs, labels in imagenet_loader:
        inputs = inputs.to(device)
        outputs = resnet50(inputs)
        # Processar os outputs, e.g., escolher as classes mais prováveis
        # Nota: Este passo depende de como você deseja processar/exibir os resultados
