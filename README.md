<details>
  
# Al-Powered classification of Ovarian cancers Based on Histopathological lmages
Haitham Kussaibi , Elaf Alibrahim, Eman Alamer, Ghada Alhaji, Shrooq Alshehab, Zahraa Shabib, Noor Alsafwani, and Ritesh G. Meneses
MEDRXIV/2024/308520
# METHODOLOGY
## Dataset Preparation and Pre-processing
Sixty-four (20x) whole slide images (WSIs) from the Cancer Imaging Archive and 18 WSIs from KFHU.
## Extract tiles from the WSIs: 
First, using QuPath, pathologists annotated tumor regions of interest (ROIs) on the WSIs, and then tiles of size (224 x 224 pixels) were cropped from those ROIs. 
## Pre-processing Techniques
Torchvision normalizing function:
<summary>Click to view the code</summary>
```
(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  
```
## Features extraction:
<summary>Click to view the code</summary>
```
ResNet50
```
## Training Process:
### NN-based classifier
<summary>Click to view the code</summary>
```
nn
```
### lightGBM
<summary>Click to view the code</summary>
```
GBM
```
</details>
