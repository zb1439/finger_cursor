from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

import torch.optim as optim
import torchvision.models as models


def preprocess_mobilenet_data(imdata):
    # Resize data. 
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    processed = [preprocess(im) for im in imdata]
    processed = torch.stack(processed)
    return processed

def get_dataloader(X, y):
    dataloader = DataLoader(
        [(X[i], y[i]) for i in range(len(X))], 
        batch_size=16, 
        shuffle=True
    )
    return dataloader



def train(model, dataloader, lr = 1e-4, num_epoch = 30):
    torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    PRINT_BATCH_NUM = len(dataloader)

    best_acc = 0
    best_loss = 1e100000
    
    best_fname = ''

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_loss = 0.0
        correct = 0

        for i, data in tqdm(enumerate(dataloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct += (torch.max(outputs.data, 1)[1] == labels).float().sum().item()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % PRINT_BATCH_NUM == (PRINT_BATCH_NUM - 1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_BATCH_NUM:.3f}')
                running_loss = 0.0

        epoch_acc = correct / len(dataloader.dataset)
        epoch_loss = epoch_loss / len(dataloader.dataset)
        print(f'accuracy: {epoch_acc}')

        if (best_acc < epoch_acc):
            best_loss = epoch_loss
            best_acc = epoch_acc
            best_fname = f'blazehand_acc_{best_acc:.4f}.pt'
            torch.save(model.state_dict(), best_fname)

    return best_acc, best_loss, best_fname


def load_model(model_path, mode='eval'):
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = BlazeHandLandmark().to(gpu)
    model.load_weights(model_path)

    if mode == 'train': model.train()

    return model

def eval_model(model, dataloader, print_graph = False):

    criterion = nn.CrossEntropyLoss()
    
    y_pred = None
    with torch.no_grad():
        correct = 0

        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            outputs = model(inputs)
            pred = torch.max(outputs.data, 1)[1]
            loss = criterion(outputs, labels)

            correct += (pred == labels).float().sum().item()
            running_loss += loss.item()

            if (y_pred is None):
                y_pred = pred.unsqueeze(-1).cpu().detach()
            else:
                y_pred = torch.vstack((y_pred, pred.unsqueeze(-1).cpu().detach()))

            if print_graph:
                for j in range(len(inputs)):
                    plt.imshow(inputs[j].cpu().permute(1, 2, 0))
                    plt.show()
                    print(f"Label: {labels[j]}, {label_names[labels[j]]}")
                    print(f"Pred: {pred[j]}, {label_names[pred[j]]}")

        acc = correct / len(dataloader.dataset)

    print(running_loss)
    print(acc)


if __name__ == '__main__':
    print(cropped_img_data4.shape, y4_i.shape)
    X_train, X_val, y_train, y_val = train_test_split(cropped_img_data4, y4_i, stratify=y4_i, test_size=0.9)

    train_all_img = np.concatenate((cropped_img_data2, cropped_img_data1, cropped_img_data3, X_train), axis=0)
    train_all_y = np.concatenate((y2_i, y1_i, y3_i, y_train), axis=0)

    # train data
    X_train = preprocess_mobilenet_data(train_all_img)
    y_train = train_all_y

    X_val = preprocess_mobilenet_data(X_val)

    print('X_train, y_train, X_val, y_val:', X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    train_dataloader = get_dataloader(X_train, y_train)
    val_dataloader = get_dataloader(X_val, y_val)

    # Load model for training.
    model_path = './blazehand_landmark.pth'
    model = load_model(model_path=model_path, mode='train')

    freeze_blocks = [model.backbone1, model.backbone2, model.backbone3, model.backbone4, model.blaze5]
    for block in freeze_blocks:
        for p in block.parameters():
            p.requires_grad = False

    # Training.
    best_acc, best_loss, best_fname = train(model, train_dataloader, lr=1e-5, num_epoch=2)

    # Eval.
    best_model = load_model(model_path=best_fname)
    eval_model(best_model, val_dataloader)

