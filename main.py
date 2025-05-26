import torch
import yaml
import time
import wandb
import os
from dotenv import load_dotenv
from timm import create_model
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from src.model.resnet50 import ResNet50
from src.data.ChromosomeDataset import ChromosomeDataset, load_images
from src.train import training_step, evaluation_step, test_step

if __name__=='__main__':
    with open("configs/run.yaml", "r") as f:
        config = yaml.safe_load(f)
    # read config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    model_name = config['model_name']
    wandb_log = config['wandb_log']
    project_name = config['project_name']
    tags = config['tags']
    project_config = config['project_config']
    train_path = config['train_path']
    test_path = config['test_path']
    val_split = config['val_split']
    workers = config['workers']
    classes = config['classes']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # EfficientNet normalization
    ])

    ims = load_images(train_path)
    fullset = ChromosomeDataset(train_path, ims, transform=transform)
    testset = ChromosomeDataset(test_path, transform=transform)

    train_len = int(len(fullset) * (1 - val_split))
    val_len = len(fullset) - train_len

    generator = torch.Generator().manual_seed(42)
    trainset, valset = random_split(fullset, [train_len, val_len], generator=generator)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

    if model_name == 'resnet50-pretrained':
        model = models.resnet50(pretrained=True)
        
    elif model_name == 'efficientnet':
        model = create_model('efficientnet_b1', pretrained=True)
    
    elif model_name == 'resnet50':
        model = ResNet50(classes, channels=1)
    else:
        model = None
    
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)

    if model_name == 'resnet50-pretrained' or model_name == 'efficientnet':
        for param in model.parameters():
            param.requires_grad = False

    if model_name == 'efficientnet':
        model.classifier = nn.Linear(model.classifier.in_features, classes)
    elif model_name == 'resnet50-pretrained':
        model.fc = nn.Sequential( 
                nn.Linear(model.fc.in_features, 128),
                nn.ReLU(),
                nn.Linear(128, classes),
                nn.Softmax(dim=1) 
                )
        
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    if model_name == 'efficientnet':
        optimizer = optim.Adam(model.classifier.parameters())
    elif model_name == 'resnet50':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    if wandb_log:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        wandb_logger = wandb.init(
            project=project_name,
            name=f"{current_time}",
            tags=tags,
            config=project_config
        )

    for epoch in range(EPOCHS):
        avg_train_loss, avg_train_acc = training_step(model, trainloader, loss_func, optimizer, epoch, EPOCHS, device)
        avg_val_loss, avg_val_acc = evaluation_step(model, valloader, loss_func, epoch, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_acc.append(avg_train_acc)
        val_acc.append(avg_val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Training Accuracy: {avg_train_acc:.2f}%, Validation Accuracy: {avg_val_acc:.2f}%")

    if wandb.run is not None:
        wandb.finish()
    print('Training complete!')
    avg_loss, accuracy, precision, recall, f1 = test_step(model, testloader, loss_func, classes, device)
    print(f"Test Loss {avg_loss}, Test Accurac {accuracy}, Precision {precision}, Recall {recall}, F1 {f1}")
