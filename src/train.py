import torch
from tqdm import tqdm
import wandb
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def training_step(model, dataloader, loss_func, optimizer, epoch, num_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for _, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    if wandb.run is not None:
        wandb.log({
            "train/loss": avg_train_loss,
            "train/acc": accuracy,
            "epoch": epoch
        })
    return avg_train_loss, accuracy

def evaluation_step(model, dataloader, loss_func, epoch, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(), tqdm(dataloader, total=len(dataloader), desc="Validation", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_func(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            pbar.set_postfix(val_loss=val_loss / (batch_idx + 1))

    avg_val_loss = val_loss / len(dataloader)
    accuracy = 100 * correct / total
    if wandb.run is not None:
        wandb.log({
            "val/loss": avg_val_loss,
            "val/acc": accuracy,
            "epoch": epoch
        })
    return avg_val_loss, accuracy
    
def test_step(model, dataloader, loss_func, classes, device):
    model.eval()

    acc_metric = MulticlassAccuracy(num_classes=classes, average='macro').to(device)
    pre_metric = MulticlassPrecision(num_classes=classes, average='macro').to(device)
    rec_metric = MulticlassRecall(num_classes=classes, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=classes, average='macro').to(device)
    loss_metric = MeanMetric().to(device)

    with torch.no_grad(), tqdm(dataloader, desc='Testing', unit='batch') as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = loss_func(output, target)
            preds = torch.argmax(output, dim=1)

            acc_metric.update(preds, target)
            pre_metric.update(preds, target)
            rec_metric.update(preds, target)
            f1_metric.update(preds, target)
            loss_metric.update(preds, target)

            pbar.set_postfix(loss=loss.item())

    avg_loss = loss_metric.compute().item()
    accuracy = acc_metric.compute().item() * 100
    precision = pre_metric.compute().item()
    recall = rec_metric.compute().item()
    f1 = f1_metric.compute().item()

    if wandb.run is not None:
        wandb.log({
            "test/loss": avg_loss,
            "test/acc": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1
        })

    print(f"\nTest Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    return avg_loss, accuracy, precision, recall, f1
