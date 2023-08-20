import torch
import numpy as np
from torch import nn, optim
from libriDataset import libriDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LibriClassifier
from torch.utils.tensorboard import SummaryWriter

def same_seeds(seed):
    torch.manual_seed(seed)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmard = False
    torch.backends.cudnn.deterministic = True

def main():
    # define parameters
    classification_num = 41
    hidden_layers = 3
    hidden_dim = 1024
    frame_num = 9
    batch_size = 2048
    epochs = 100
    learning_rate = 0.001 
    seed = 0

    same_seeds(seed)

    # define cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = LibriClassifier(input_dim=39 * frame_num, hidden_layers=hidden_layers, hidden_dim=hidden_dim, classify_num=classification_num)
    # model.load_state_dict(torch.load("hw2_classification/model/libriclassifier.pkl"))
    model = model.to(device)
    print("Model created!")
    print(model)

    # define loss function
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)

    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # define scheduler to improve learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=learning_rate/2)

    # create tensorboard writer
    writer = SummaryWriter("hw2_classification/logs")

    # loading data
    print("Loading train set......")
    train_set = libriDataset("hw2_classification/libriphone", "train", frame_num, classification_num)
    print("Loading valid set......")
    valid_set = libriDataset("hw2_classification/libriphone", "valid", frame_num, classification_num)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # training
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        # setting parameters
        train_loss = 0.0
        train_total_loss = 0.0
        train_acc = 0.0
        train_total_acc = 0
        valid_loss = 0.0
        valid_total_loss = 0.0
        valid_acc = 0.0
        valid_total_acc = 0

        # train
        model.train()
        train_step = 0
        samples = 0
        pbar = tqdm(train_loader, ncols=150)
        pbar.set_description(f"Train epoch: {epoch + 1}/{epochs} ")
        for i, data in enumerate(pbar):
            features, targets = data
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(features)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            # print([x.grad for x in optimizer.param_groups[0]['params']])
            # calculate some parameters
            train_step += 1
            samples += output.size(0)
            train_total_loss += loss.item()
            acc = (output.argmax(1) == targets).sum()
            train_total_acc += acc.item()
            train_acc = train_total_acc / samples
            train_loss = train_total_loss / (i + 1)
            lr = optimizer.param_groups[0]["lr"]
            # 更新进度条参数
            pbar.set_postfix({'lr':lr, 'train acc':train_acc, 'loss':train_loss})
        scheduler.step()    # 更新learning rate
        pbar.close()
        
        print(f"Train epoch: {epoch + 1}, total loss: {train_total_loss}")
        # tensorboard note
        writer.add_scalar("Train total loss", train_total_loss, epoch)
        writer.add_scalar("Train average loss", train_loss, epoch)
        writer.add_scalar("Train accuracy", train_acc, epoch)

        # validation
        model.eval()
        valid_step = 0
        samples = 0
        pbar = tqdm(valid_loader, ncols=150)
        pbar.set_description(f"Valid epoch: {epoch + 1}/{epochs} ")
        with torch.no_grad():
            for i, data in enumerate(pbar):
                features, targets = data
                features = features.to(device)
                targets = targets.to(device)

                output = model(features)
                loss = loss_func(output, targets)

                valid_step += 1
                valid_total_loss += loss.item()
                samples += output.size(0)
                acc = (output.argmax(1) == targets).sum()
                valid_total_acc += acc.item()
                valid_acc = valid_total_acc / samples
                valid_loss = valid_total_loss / (i + 1)

                pbar.set_postfix({'valid acc':valid_acc, 'loss':valid_loss})
            
            pbar.close()
            print(f"Valid epoch: {epoch + 1}, total loss: {valid_total_loss}")
        # tensorboard note
        writer.add_scalar("Valid total loss", valid_total_loss, epoch)
        writer.add_scalar("Valid average loss", valid_loss, epoch)
        writer.add_scalar("Valid accuracy", valid_acc, epoch)

        # save model
        if(best_acc < valid_acc):
            best_acc = valid_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "hw2_classification/model/libriclassifier.pkl")
            print("model saved!")

    writer.close()
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best accuracy: {best_acc}")

    return

if __name__ == "__main__":
    main()