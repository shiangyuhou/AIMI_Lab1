import datetime
import csv
from pathlib import Path
from PIL import Image
import os
import shutil
import time
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator
from datetime import datetime
RESULT_DIR = './results'

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, val_acc_list, program_name, result_dir=RESULT_DIR):
    # TODO plot training and testing accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, label='Training Accuracy', marker='o')
    plt.plot(val_acc_list, label='Validation Accuracy', marker='o')

    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(result_dir + '/' + program_name + '/accuracy_curve.png')
    # save the data of both acc
    # so that it can reproduce the plot later
    # np.save(result_dir + f'/{program_name}_train_acc.npy', np.array(train_acc_list))
    np.save(result_dir + '/' + program_name + '/train_acc.npy', np.array(train_acc_list))
    np.save(result_dir + '/' + program_name + '/val_acc.npy', np.array(val_acc_list))



def plot_f1_score(train_f1_score_list, val_f1_score_list, program_name, result_dir=RESULT_DIR):
    # TODO plot testing f1 score curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1_score_list, label='Training F1 Score', marker='o')
    plt.plot(val_f1_score_list, label='Validation F1 Score', marker='o')
    plt.title('Training and Validation F1 Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    # plt.savefig(result_dir + '/' + program_name + '/f1_score_curve.png')

    # np.save(result_dir + '/' + program_name + '/f1_score.npy', np.array(f1_score_list))
    plt.savefig(result_dir + '/' + program_name + '/f1_score_curve.png')

    np.save(result_dir + '/' + program_name + '/train_f1_score.npy', np.array(train_f1_score_list))
    np.save(result_dir + '/' + program_name + '/val_f1_score.npy', np.array(val_f1_score_list))


def plot_confusion_matrix(confusion_matrix, program_name, result_dir=RESULT_DIR):
    # TODO plot confusion matrix of the final test results

    plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Actual Pneumonia', 'Actual Normal'], yticklabels=['Predicted Pneumonia', 'Predicted Normal'])
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Actual Normal', 'Actual Pneumonia'], yticklabels=['Predicted Normal', 'Predicted Pneumonia'])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(result_dir + '/' + program_name + '/confusion_matrix.png')
    np.save(result_dir + '/' + program_name + '/confusion_matrix.npy', np.array(confusion_matrix))

# def train(device, train_loader, model, criterion, optimizer, val_loader):
#     best_acc = 0.0
#     best_f1 = 0.0
#     best_turn = 0
#     best_model_wts = None
#     train_acc_list = []
#     val_acc_list = []
#     train_f1_score_list = []
#     val_f1_score_list = []
#     best_c_matrix = []

#     for epoch in range(1, args.num_epochs+1):
#         model.train()
#         with torch.set_grad_enabled(True):
#             avg_loss = 0.0
#             train_acc = 0.0
#             tp, tn, fp, fn = 0, 0, 0, 0     
#             for _, data in enumerate(tqdm(train_loader)):
#                 inputs, labels = data
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 avg_loss += loss.item()
#                 outputs = torch.max(outputs, 1).indices
#                 sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
#                 tp += sub_tp
#                 tn += sub_tn
#                 fp += sub_fp
#                 fn += sub_fn          

#             avg_loss /= len(train_loader.dataset)
#             train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
#             print(f'Epoch: {epoch}')
#             print(f'↳ Loss: {avg_loss}')
#             print(f'↳ Training Acc.(%): {train_acc:.2f}%')

#         # write validation if you needed
#         # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
#         val_acc, val_f1_score, c_matrix = test(val_loader, model)

#         # save the best model
#         # if val_acc > best_acc:
#         if val_f1_score >= best_f1:
#             best_f1 = val_f1_score
#             best_acc = val_acc
#             best_c_matrix = c_matrix
#             # best_model_wts = model.state_dict()
#             torch.save(model.state_dict(), RESULT_DIR + '/' + program_name + '/' + 'best_model_weights.pt')
#             best_turn = epoch
#             print(f'↳ Best model saved with acc: {best_acc:.2f}%')

#         f1_score = (2*tp) / (2*tp+fp+fn)

#         train_acc_list.append(train_acc)
#         val_acc_list.append(val_acc)
#         train_f1_score_list.append(f1_score)
#         val_f1_score_list.append(val_f1_score)

#     # torch.save(model.state_dict(), RESULT_DIR + '/' + program_name + '/' + 'best_model_weights.pt')
#     # torch.save(model.state_dict(), RESULT_DIR + '/model_weights.pt')
#     # torch.save(best_model_wts, RESULT_DIR + '/' + program_name + '/' + 'best_model_weights.pt')

#     return train_acc_list, train_f1_score_list, best_c_matrix, val_acc_list, val_f1_score_list, best_turn

def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    image_index = 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn
            # output the predict label with the image name
            # for i in range(len(labels)):
            #     print(f'Image: {test_loader.dataset.samples[image_index + i][0]}, True Label: {labels[i].item()}, Predicted Label: {outputs[i].item()}')
            # image_index += len(labels)


        c_matrix = [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

def create_dir(program_name):
    os.makedirs(os.path.join(RESULT_DIR, program_name), exist_ok=True)


def predict(image_paths, model, transform, program_name, batch_size=128):
    class_names = ['NORMAL', 'PNEUMONIA']
    output_path = os.path.join(RESULT_DIR, program_name, 'predictions.csv')
    model.eval()
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Predicted Label', 'Confidence'])
        with torch.set_grad_enabled(False):
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                images = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        images.append(transform(image.convert('RGB')))
                images = torch.stack(images).to(device)
                probabilities = torch.softmax(model(images), dim=1)
                confidence, outputs = torch.max(probabilities, 1)
                for path, output, score in zip(batch_paths, outputs.tolist(), confidence.tolist()):
                    label = class_names[output]
                    print(f'Image: {path}, Predicted Label: {label}, Confidence: {score:.4f}')
                    writer.writerow([str(path), label, score])
    print(f'Predictions saved to: {output_path}')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_result_to_csv(program_name, test_acc, f1_score, best_c_matrix, best_turn, result_dir=RESULT_DIR):
    import csv
    csv_file = os.path.join(result_dir, 'results.csv')
    file_exists = os.path.isfile(csv_file)

    # if file exists, append to it; otherwise, create a new file and write the header
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Program Name', 'Best Turn', 'Test Accuracy', 'F1 Score', 'Confusion Matrix'])
        writer.writerow([program_name, best_turn, test_acc, f1_score, best_c_matrix])

if __name__ == '__main__':

    

    # # record start time
    # start_time = time.time()

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # # for model
    parser.add_argument('--num_classes', type=int, choices=[2], default=2)

    # # for training
    # parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    # parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--wd', type=float, default=0.9)

    # # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # # for data augmentation
    # parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--resize', type=int, default=224)

    # parser.add_argument('--mode', type=int, required=True, help='1: resnet18, 2: resnet50')
    # parser.add_argument('--seed', type=int, required=True, help='remember to set the random seed for reproducibility')
    # parser.add_argument('--flip', type=float, default=0.5, help='probability of horizontal flip')
    # parser.add_argument('--str', type=str, default="none")
    parser.add_argument('--worker', type=int, default=2, help='number of workers for dataloader')
    # parser.add_argument('--aug', type=str, required=True, help='[None, RandomRotation, RandomHorizontalFlip, RandomRotate-Flip, Normalize]')
    # parser.add_argument('--norm', action='store_true', help='whether to apply normalization')

    # if args.aug not in ["None", "RandomRotation", "RandomHorizontalFlip", "RandomRotate-Flip"]:
    #     raise ValueError("Invalid augmentation method. Please choose from [None, RandomRotation, RandomHorizontalFlip, RandomRotate-Flip].")

    # the path of test set 
    # parser.add_argument('--test', type=str, default="none")

    # Evaluate by default; prediction accepts one image, several images, or a folder.
    parser.add_argument('--task', choices=['evaluate', 'predict'], default='evaluate')
    parser.add_argument('--input', nargs='+', help='image paths or folders for prediction')

    args = parser.parse_args()
    if args.batch_size < 1 or args.resize < 1 or args.worker < 0:
        parser.error('batch_size and resize must be positive; worker must be nonnegative')
    if args.task == 'predict' and not args.input:
        parser.error('--input is required for prediction')
    if args.task == 'evaluate' and args.input:
        parser.error('Use --task predict with --input')


    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')
    
    # to make every run reproducible, set the random seed
    # set_seed(args.seed)

    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    # TODO / Change the data augmentation method yourself 

    # transform_train_list = []
    transform_test_list = []
    transform_test_list.append(transforms.Resize((args.resize, args.resize)))
    transform_test_list.append(transforms.ToTensor())
    # transform_train_list.append(transforms.Resize((args.resize, args.resize)))

    # if args.aug == "RandomRotation":
    #     transform_train_list.append(transforms.RandomRotation(args.degree))
    # elif args.aug == "RandomHorizontalFlip":
    #     transform_train_list.append(transforms.RandomHorizontalFlip(p=args.flip))
    # elif args.aug == "RandomRotate-Flip":
    #     transform_train_list.append(transforms.RandomRotation(args.degree))
    #     transform_train_list.append(transforms.RandomHorizontalFlip(p=args.flip))

    # transform_train_list.append(transforms.ToTensor())
    # transform_test_list.append(transforms.ToTensor())
    # if args.norm == True:
    #     transform_test_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    #     transform_train_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
    #                             transform = transforms.Compose(transform_train_list))

    if args.task == 'evaluate':
        test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                                   transform=transforms.Compose(transform_test_list))
        if test_dataset.classes != ['NORMAL', 'PNEUMONIA']:
            parser.error('Test folders must be NORMAL and PNEUMONIA')
    else:
        image_paths = []
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm'}
        for source in args.input:
            path = Path(source)
            if not path.exists():
                parser.error(f'Input does not exist: {path}')
            if path.is_dir():
                image_paths.extend(sorted(p for p in path.rglob('*')
                                          if p.is_file() and p.suffix.lower() in extensions))
            elif path.suffix.lower() in extensions:
                image_paths.append(path)
            else:
                parser.error(f'Unsupported image: {path}')
        if not image_paths:
            parser.error('No supported images found')
    # # split the training dataset into training and validation datasets.
    # split_seed = 42
    # split_generator = torch.Generator().manual_seed(split_seed)
    # val_size = int(0.2 * len(train_dataset))
    # train_size = len(train_dataset) - val_size
    # training_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=split_generator)
    # from torch.utils.data import Subset

    # train_full = ImageFolder(root=os.path.join(args.dataset, 'train'),
    #                         transform=transforms.Compose(transform_train_list))   

    # # val_full = ImageFolder(root=os.path.join(args.dataset, 'train'),
    # val_full = ImageFolder(root=os.path.join(args.dataset, 'val'),
    #                         transform=transforms.Compose(transform_test_list))

    # test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
    #                         transform=transforms.Compose(transform_test_list))

    # # randomly choose 42 as the seed for reproducibility
    # split_seed = 42 
    # generator = torch.Generator().manual_seed(split_seed)

    # indices = torch.randperm(len(train_full),generator=generator).tolist()
    # val_size = int(0.2 * len(train_full))
    # val_indices = indices[:val_size]
    # train_indices = indices[val_size:]

    # training_dataset = Subset(train_full, train_indices)
    # val_dataset = Subset(val_full, val_indices)
    # training_dataset = train_full
    # val_dataset = val_full

    # train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, pin_memory=True, persistent_workers=True)
    if args.task == 'evaluate':
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True, persistent_workers=args.worker > 0)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True, persistent_workers=True)

    # create a directory for the results
    # format RandomRotation_18_2026_deg5_flip0.0_26-10-15_19
    # base_name = os.path.basename(__file__).split('.')[0]
    # program_name = f"{args.aug}_{18 if args.mode == 1 else 50}_se{args.seed}_de{args.degree}_flip{args.flip}_norm{int(args.norm)}_{datetime.now().strftime('%d-%m-%H_%M')}"
    program_name = f"best_weights_{args.task}_{datetime.now().strftime('%m-%d-%H_%M_%S_%f')}"
    create_dir(program_name)

    # print("program_name:", program_name)
    # plot.plot_accuracy([0], [0], program_name)
    # exit(0)
    # TODO / define model 
    # model = ... <- refine by yourself
    model = models.resnet18(weights=None)
    # model = models.resnet18(
    # weights=models.ResNet18_Weights.DEFAULT
    # )

    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)


    model = model.to(device)

    # # define loss function, optimizer
    # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    # criterion = criterion.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # # training
    # train_acc_list, train_f1_score_list, best_c_matrix, val_acc_list, val_f1_score_list, best_turn = train(device, train_loader, model, criterion, optimizer, val_loader)

    # using the best model for testing
    model.load_state_dict(torch.load('best_model_weights_resnet18.pt', map_location=device, weights_only=True))
    if args.task == 'predict':
        predict(image_paths, model, transforms.Compose(transform_test_list), program_name, args.batch_size)
        raise SystemExit(0)

    # testing
    test_acc, f1_score, best_c_matrix = test(test_loader, model)

    
    # plot
    # plot_accuracy(train_acc_list, val_acc_list, program_name)
    # plot_f1_score(train_f1_score_list, val_f1_score_list, program_name)
    plot_confusion_matrix(best_c_matrix, program_name)
    print(f"test_acc: {test_acc:.4f}, f1_score: {f1_score:.4f}, best_c_matrix: {best_c_matrix}")

    # add_result_to_csv(program_name, test_acc, f1_score, best_c_matrix, best_turn)
    # # record end time
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    # print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # print the configuration used for this run
    # for name, value in vars(args).items():
    #     print(f"  {name}: {value}")
    #  ----------------
    # print(f"deg: {args.degree}")
    # print("results:")
    # print(f"  best_turn: {best_turn}")
    # print(f"  test_accuracy: {test_acc:.4f}")
    # print(f"  test_f1_score: {f1_score:.4f}")
    # print(f"  best_confusion_matrix: {best_c_matrix}")
    # print(f"  augmentation: {args.aug}")
    # print(f"transform:{args.str}")
