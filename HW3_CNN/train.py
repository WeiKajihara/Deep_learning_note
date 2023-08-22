import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import ResNet
from PicDataset import PicDataset
from tqdm import tqdm

def same_seeds(seed):
    torch.manual_seed(seed)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmard = False
    torch.backends.cudnn.deterministic = True

def main():
    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define parameters
    seed = 6666
    img_size = (224, 224)
    batch_size = 64
    learning_rate = 0.001
    epochs = 50
    classication_num = 11

    # set random seed
    same_seeds(seed)

    # create model
    model = ResNet("ResNet-34", classication_num, True)
    print(model)
    model.load_state_dict(torch.load("hw3_CNN/model/ResNet-34_classifier.pkl"))
    model = model.to(device)

    # define loss function
    loss_func = nn.CrossEntropyLoss().to(device)

    # defing optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # define scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=1, eta_min=learning_rate/10)

    # define summarywriter
    writer = SummaryWriter("hw3_CNN/logs")

    # define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.7, 1.0)), 
        transforms.RandomHorizontalFlip(), # 随机翻转图片
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(180), # 随机旋转图片
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    # load data
    print("Loading train set......")
    train_set = PicDataset("hw3_CNN/hw3_data", train_transform, "train")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    print("Loading valid set......")
    valid_set = PicDataset("hw3_CNN/hw3_data", valid_transform, "valid")
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # training
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        train_total_loss = 0.0
        train_acc = 0.0
        train_acc_num = 0
        samples = 0
        pbar = tqdm(train_loader, ncols=150)
        pbar.set_description(f"Train epoch({epoch + 1}/{epochs})")
        for i, data in enumerate(pbar):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()

            samples += output.size(0)
            train_total_loss += loss.item()
            train_loss = train_total_loss / (i + 1)
            acc = (output.argmax(1) == targets).sum()
            train_acc_num += acc.item()
            train_acc = train_acc_num / samples
            lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix({'lr':lr, 'train acc:':train_acc, 'train loss':train_loss})
        scheduler.step()
        pbar.close()
        writer.add_scalar("Train accuracy", train_acc, epoch)
        writer.add_scalar("Train average loss", train_loss, epoch)

        # valid
        model.eval()
        valid_loss = 0.0
        valid_total_loss = 0.0
        valid_acc = 0.0
        valid_acc_num = 0
        samples = 0
        pbar = tqdm(valid_loader, ncols=150)
        pbar.set_description(f"Valid epoch({epoch + 1}/{epochs})")
        with torch.no_grad():
            for i, data in enumerate(pbar):
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)

                output = model(imgs)
                loss = loss_func(output, targets)

                samples += output.size(0)
                valid_total_loss += loss.item()
                valid_loss = valid_total_loss / (i + 1)
                acc = (output.argmax(1) == targets).sum()
                valid_acc_num += acc.item()
                valid_acc = valid_acc_num / samples

                pbar.set_postfix({'valid acc:':valid_acc, 'valid loss':valid_loss})
            pbar.close()
            writer.add_scalar("Valid accuracy", valid_acc, epoch)
            writer.add_scalar("Valid average loss", valid_loss, epoch)
        
        # save model
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "hw3_CNN/model/ResNet-34_classifier.pkl")
            print(f"Model saved!")
        
    print(f"Best accuracy: {best_acc}, epoch: {best_epoch}")
    with open("hw3_CNN/model/accuracy.txt", 'w', encoding="UTF-8") as f:
        f.write(f"Best accuracy: {best_acc}\n")
        f.flush()

    return

if __name__ == "__main__":
    main()