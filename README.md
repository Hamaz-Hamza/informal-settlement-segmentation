# Informal Settlement Segmentation

This repository showcases experiments performed to segment informal settlements from Sentinel-2 multi-spectral imagery using both machine learning and deep learning techniques. The dataset and the research paper can be found publicly available here: `https://frontierdevelopmentlab.github.io/informal-settlements/`

## Dataset

The dataset consists of multi-spectral low resolution satellite imagery captured by the Sentinel-2 satellite. The images are of the following areas:

1. Medellín — Colombia
2. Mumbai — India
3. Kibera — Kenya
4. Kianda — Nigeria
5. Makoko — Nigeria
6. Cape Town — South Africa
7. Al Geneina — Sudan
8. El Daein — Sudan
9. "Lower" - Unknown

Each region has its own raster file and informal settlement ground truth provided in `.tif` format. The raster file consists of 10 unnormalized bands of which the first three are blue, green and red bands. The 10-band format follows the conventional Sentinel-2 bands format but with atmospheric bands removed.

## Methodology

The methodology is as follows:
1. Dataset exploration and visualization `(notebook 1)`
2. Dataset preprocessing `(notebook 2)`
3. Making splits (train, test, validation) for pixel wise classification strategy `(notebook 3)`
4. Applying and fine tuning a random forest model `(notebooks 4-6)`
5. Making splits (train, test, validation) for semantic segmentation on image patches `(notebook 7)`
6. Applying and fine tuning a pre-trained convolutional neural network (CNN) model through transfer learning `(notebooks 8-14)`

Each notebooks contains experiments done and the corresponding findings/analysis in a step-by-step walkthrough fashion.

## Final Results

Here are the results obtained at the end of the final machine learning and deep learning sections:

### 1. Machine Learning: Sklearn's Random Forest Classifier
#### Data: 
1. Format: Pixel-wise (all cities combined)
3. Distribution (raw numbers): 9580194 training pixels : 2052899 validation pixels : 2052905 testing pixels
4. Distribution (percentage): 70% training : 15% validation : 15% testing
5. Distribution strategy: Stratified split based on output class

#### Results on test data:
| Class               | IoU (Jaccard) | Precision | Recall | F1-Score |
|--------------------:|--------------:|----------:|-------:|---------:|
| Formal Settlement   | 93.95%        | 97.58%    | 96.20% | 96.88%   |
| Informal Settlement | 50.09%        | 62.04%    | 72.23% | 66.75%   |

<hr>

### 2. Deep Learning: Torchgeo Resnet18 encoder pre-trained on Sentinel-2 imagery, combined with standard UNET decoder.
#### Data:
1. Format: Image patches (all cities combined)
3. Distribution (raw numbers): 3154 training patches : 676 validation patches : 676 testing patches
4. Distribution (percentage): 70% training : 15% validation : 15% testing
5. Distribution strategy: Stratified split based on output class


#### Results on test data:
| Class               | IoU (Jaccard) | Precision | Recall | F1-Score |
|--------------------:|--------------:|----------:|-------:|---------:|
| Formal Settlement   | 94.06%        | 97.47%    | 96.41% | 96.94%   |
| Informal Settlement | 49.94%        | 62.85%    | 70.85% | 66.61%   |


