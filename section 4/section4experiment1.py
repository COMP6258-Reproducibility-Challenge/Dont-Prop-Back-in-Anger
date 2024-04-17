import os.path
import torch.optim
import torchvision as tv
import numpy as np
import time as t
from datetime import datetime
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import InterpolationMode

from wb_data_copy import get_transform_cub
from utils_copy import evaluate, get_y_p
from wbdc import WaterBirdsDatasetCustom
from sklearn.metrics import classification_report
from functools import partial

if __name__ == '__main__':

    # Hyper parameters
    epochs = 50
    momentum = 0.9
    lr = 1e-3
    batchSize = 32
    weightDecay = 1e-2
    lossFunction = torch.nn.CrossEntropyLoss()
    zeroDivision = 0  # Value to use when calculating recall for a class not present in the output, 0 is truthful, 1 inflates results
    resolution = (224, 224)
    # the transform listed in the paper
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 4./3.), interpolation=2),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 4. / 3.), interpolation=InterpolationMode.BILINEAR),  # Equivalent to interpolation=2
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    # the transforms used in their code
    # train_transform = get_transform_cub(target_resolution=resolution, train=True, augment_data=True)
    # test_transform = get_transform_cub(target_resolution=resolution, train=False, augment_data=False)

    # Files and directories
    datasets_root = "..\\..\\data\\sec4_datasets"
    originalDir = os.path.join(datasets_root, "original95")
    fgOnlyDir = os.path.join(datasets_root, "fgOnly")
    dateTime = datetime.now().strftime("_%Y-%m-%d_%H%M%S")
    output_file = os.path.join("results", "OUTPUT" + dateTime + ".txt")
    os.makedirs("results", exist_ok=True)
    params = ("Epochs: %d, Momentum: %f, Learning Rate: %f, Batch Size: %d, Weight Decay: %f, Loss Function: %s, "
              "Resolution: %s, Zero Division: %d \n\n") % (
             epochs, momentum, lr, batchSize, weightDecay, str(lossFunction), str(resolution), zeroDivision)
    params += ("-" * 53) + "\n"

    # Set seed for deterministic outputs
    seed = 3141592654
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Create datasets, WBDC2 is code version, WBDC is paper version
    original95Dataset = WaterBirdsDatasetCustom(originalDir, batch_size=batchSize, train_transform=train_transform, test_transform=test_transform, label="original95")
    original100Dataset = WaterBirdsDatasetCustom(originalDir, batch_size=batchSize, train_transform=train_transform, test_transform=test_transform, label="original100")
    fgOnlyDataset = WaterBirdsDatasetCustom(fgOnlyDir, batch_size=batchSize, train_transform=train_transform, test_transform=test_transform, label="fgOnly")
    # balanced dataset difficult to create (poor reproducibility)
    # balancedDataset = WaterBirdsDatasetCustom(originalDir, transform=transform, batch_size=batchSize, label="balanced")
    datasets = [original95Dataset, original100Dataset, fgOnlyDataset]

    # remove minority groups from original100
    minorityGroups = np.argsort(original100Dataset.train.group_counts.numpy())[:2]
    majorityIds = np.where(~np.isin(original100Dataset.train.group_array, minorityGroups))[0]
    original100Dataset.train.filter(majorityIds)

    # Training
    for dataset in datasets:

        t_0 = t.time()
        print("Now training on %s dataset." % dataset.label)

        # Setup dataloaders
        trainLoader = dataset.train.loader()
        originalTestLoader = original95Dataset.test.loader()
        fgOnlyTestLoader = fgOnlyDataset.test.loader()

        # Setup model
        # model = tv.models.resnet50(pretrained=True)
        model = tv.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Equivalent to above
        d = model.fc.in_features  # Dim of input to fc layer
        model.fc = torch.nn.Linear(d, dataset.train.n_classes)  # Specify new FC layer
        optimiser = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)
        model.cuda()
        model.train()

        # Train loop
        for epoch in range(epochs):

            t0 = t.time()
            runningLoss = 0.0

            for batch in trainLoader:
                # x, y, _, _ = batch
                x, y, _ = batch
                x, y = x.cuda(), y.cuda()
                optimiser.zero_grad()

                logits = model(x)
                loss = lossFunction(logits, y)
                runningLoss += loss

                loss.backward()
                optimiser.step()

            runningLoss /= len(trainLoader)
            t1 = t.time()

            print("Epoch: %d, Training loss: %4.4f, Time taken: %4.2fs."
                  % (epoch + 1, runningLoss, t1 - t0))

        # Model testing
        model.eval()
        get_yp_func = partial(get_y_p, n_places=dataset.train.n_places)  # their stuff
        print("Now testing the model trained on %s dataset." % dataset.label)

        # Test on original data
        print("Testing on %s dataset." % original95Dataset.label)

        yPred, yTrue = [], []
        with torch.no_grad():
            for batch in originalTestLoader:
                # x, y, _, _ = batch
                x, y, _ = batch
                x = x.cuda()

                logits = model(x)

                yPred.append(logits.argmax(1).cpu())
                yTrue.append(y.cpu())

        yPred = np.concatenate(yPred)
        yTrue = np.concatenate(yTrue)
        originalReport = classification_report(yPred, yTrue, zero_division=zeroDivision)
        their_results_orig = evaluate(model, originalTestLoader, get_yp_func)

        # Write predictions to file
        # with open(output_file[:-4] + "-YPRED-ORIG.txt", 'a') as file:
        #     out = str(yPred.tolist()) + "\n"
        #     file.write(out)

        # Test on FG only data
        print("Testing on %s dataset." % fgOnlyDataset.label)

        yPred, yTrue = [], []
        with torch.no_grad():
            for batch in originalTestLoader:
                # x, y, _, _ = batch
                x, y, _ = batch
                x = x.cuda()

                logits = model(x)

                yPred.append(logits.argmax(1).cpu())
                yTrue.append(y.cpu())

        yPred = np.concatenate(yPred)
        yTrue = np.concatenate(yTrue)
        fgOnlyReport = classification_report(yPred, yTrue, zero_division=zeroDivision)
        their_results_fg = evaluate(model, fgOnlyTestLoader, get_yp_func)

        # Write predictions to file
        # with open(output_file[:-4] + "-YPRED-FG.txt", 'a') as file:
        #     out = str(yPred.tolist()) + "\n"
        #     file.write(out)

        # Output results and write to file
        print("Training dataset:", dataset.label)
        print("On original test data:")
        print(originalReport)
        print("\nTheir results report (original): " + str(their_results_orig) + "\n")
        print("On FG only test data:")
        print(fgOnlyReport)
        print("\nTheir results report (FG only): " + str(their_results_fg) + "\n")

        with open(output_file, 'a') as file:
            out = params
            out += "\nTraining dataset: " + dataset.label
            out += "\n\nOn original test data:\n" + originalReport
            out += "\nTheir results report (original): " + str(their_results_orig) + "\n"
            out += "\nOn FG only test data:\n" + fgOnlyReport + "\n"
            out += "\nTheir results report (FG only): " + str(their_results_fg) + "\n\n"
            out += ("-" * 53) + "\n"
            file.write(out)

        params = ""
        t_1 = t.time()

        print("Finished training and testing %s dataset, Time taken: %4.2fs."
              % (dataset.label, t_1 - t_0))
