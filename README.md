# Al-Powered classification of Ovarian cancers Based on Histopathological lmages
Haitham Kussaibi , Elaf Alibrahim, Eman Alamer, Ghada Alhaji, Shrooq Alshehab, Zahraa Shabib, Noor Alsafwani, and Ritesh G. Meneses
MEDRXIV/2024/308520
# METHODOLOGY
## Dataset Preparation and Pre-processing
Sixty-four (20x) whole slide images (WSIs) from the Cancer Imaging Archive and 18 WSIs from KFHU.
## Extract tiles from the WSIs: 
First, using QuPath, pathologists annotated tumor regions of interest (ROIs) on the WSIs, and then tiles of size (224 x 224 pixels) were cropped from those ROIs.
```
/**
 * Script to export image tiles (can be customized in various ways).
 */

// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path (here, relative to project)
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name)
mkdirs(pathOutput)

// Define output resolution in calibrated units (e.g. µm if available)
double requestedPixelSize = 5.0

// Convert output resolution to a downsample factor
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Create an exporter that requests corresponding tiles from the original & labelled image servers
new TileExporter(imageData)
    .downsample(1)   // Define export resolution
    .imageExtension('.tif')   // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(256)            // Define size of each tile, in pixels
    .annotatedTilesOnly(true) // If true, only export tiles if there is a (classified) annotation present
    .overlap(50)              // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)   // Write tiles to the specified directory

print 'Done!'
```
## Pre-processing Techniques
Torchvision normalizing function:

```
(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  
```

## Features extraction:

```
model = models.resnet50(weights='IMAGENET1K_V1')  # Instantiate the ResNet model
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
# Load the saved weights
model.load_state_dict(torch.load('models/trained_resnet50_model.pth', map_location=device))
model = nn.Sequential(*list(model.children())[:-1])
for param in model.parameters():
    param.requires_grad = False
----------------------
with torch.no_grad():
    model.eval()  # Set model to eval mode
    for batch, labels in tqdm.tqdm(dataloader):
        batch = batch.to(device)
        batch_features = model(batch)
        batch_features = batch_features.view(batch_features.size(0), -1)
        wsi_features.append(batch_features.detach().cpu())
        wsi_labels.append(labels)
torch.save({'features': wsi_features, 'labels': wsi_labels}, 'patches_Resnet_features.pth')

```

## Training Process:
```
criterion = nn.CrossEntropyLoss()
# for class weighted
# criterion = ClassWeightedCrossEntropyLoss(class_sizes=class_sizes)
optimizer = torch.optim.ِAdam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
```
### NN-based classifier

```
def normalize_input_data(x, epsilon=1e-5):
    mean = x.mean()
    std = x.std(unbiased=False) + epsilon
    normalized_x = (x - mean) / std
    return normalized_x
class MulticlassClassifier(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128, num_classes=4, dropout_rate=0.5):
        super(MulticlassClassifier, self).__init__()
        self.inst_norm_input = nn.InstanceNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, 64)
        init.xavier_uniform_(self.fc2.weight)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
        init.xavier_uniform_(self.fc3.weight)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = normalize_input_data(x)
        x = self.inst_norm_input(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

### lightGBM

```
# Create a Sequence object for the training data
train_seq = lgb.Dataset(X_train, label=y_train)

# Create a Sequence object for the validation data
val_seq = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

# create a callback function for early stopping
early_stopping_callback = lgb.early_stopping(
    stopping_rounds=10,  # stop training if the validation score doesn't improve for 50 consecutive rounds
    first_metric_only=True,  # only consider the first objective metric for early stopping
    verbose=True  # print messages when early stopping occurs
)

# create a LightGBM model and train it with early stopping
model = lgb.train(
    params,  # model parameters
    train_seq,  # training dataset
    num_boost_round=100,  # maximum number of boosting rounds
    valid_sets=[val_seq],  # validation dataset
    callbacks=[early_stopping_callback],  # list of callback functions
)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
lgb_test = lgb.Dataset(X_test, label=y_test)

# Evaluate LightGBM model
lgb_predictions = model.predict(X_test)

lgb_accuracy = (lgb_predictions.argmax(axis=1) == y_test).mean()
print("LightGBM Accuracy:", lgb_accuracy)
accuracy = accuracy_score(y_test, lgb_predictions.argmax(axis=1))
print('Accuracy:', accuracy)
```
For more information, please see the original study: [10.23750/abm.v95i5.16407](https://doi.org/10.23750/abm.v95i5.16407).

If you wish to reuse any of the codes mentioned above, please ensure to cite the original manuscript accordingly.
```
@article{Kussaibi_Alibrahim_Alamer_Al hajji_Alshehab_Shabib_Alsafwani_Menezes_2024, place={Fidenza, Italy}, title={A robust AI-pipeline for ovarian cancer classification on histopathology images}, volume={95},  url={https://mattioli1885journals.com/index.php/actabiomedica/article/view/16407},  DOI={10.23750/abm.v95i5.16407}, number={5}, journal={Acta Biomedica Atenei Parmensis}, author={Kussaibi, Haitham and Alibrahim , Elaf and Alamer, Eman and Al hajji, Ghadah and Alshehab , Shrooq and Shabib, Zahra and Alsafwani, Noor and Menezes, Ritesh G.}, year={2024}, month={Oct.}, pages={e2024176} }
```
© 2024 anapath.org This code is made available under the Apache-2 License and is available for non-commercial academic purposes.
