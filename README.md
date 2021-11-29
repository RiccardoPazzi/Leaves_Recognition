# Plant classification network
### 🎯 The competition (held on Codalab) task was to create a CNN to identify plants from leaves, in the report document a summary of the whole process can be found
### 📈 We had a local training set and a private online test set for evaluation our final mean accuracy was 92.6%
### 🧰 Tools and library used: Python, tensorflow, matplotlib

Our team was composed of three students: Myself, Gianluca Ruberto and Tommaso Brumani. 
We tried many different solutions, which are summarized in the following table (also inside the Report.pdf), the final version we uploaded can be found in the notebook, it uses transfer learning from EfficientNetB7 and fine tuning along with a variety of augmentation techniques and oversampling to compensate for the dataset imbalance.

| Model Name | Description                                                                                                                                       | Score |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| MK II      | Model seen in class                                                                                                                               | 22%   |
| MK X       | Added data augmentation                                                                                                                           | 50%   |
| MK X S     | Added shear to data augmentation and increased picture size from 64x64 grayscale to 128x128 rgb, increased batch size from 8 to 64                | 68%   |
| MK XVI     | Added batch normalization after each convolutional layer                                                                                          | 77%   |
| MK XXII    | Brand new model with a very deep network                                                                                                          | 79%   |
| MK XXIX    | Transfer learning of EfficientNetB7, previous data augmentation is used, input pictures are 256x256 rgb and it is also used the data oversampling | 89%   |
| MK XXIX FT | Fine tuning of EfficientNetB7                                                                                                                     | 92%   |

<img src="https://user-images.githubusercontent.com/62057461/143869086-8d706e10-321a-42c7-aa8e-436bbfeaf242.png" align="center" height="500" width="1000" >

Example of the MK_XVI model which includes Batch Normalization after each convolutional layer and dropout in the dense part. 

In the coolvisuals and augmentation .py files extra visualizations can be found (confusion matrix, visualization of augmentations)
The dataset can be found at: https://data.mendeley.com/datasets/tywbtsjrjv/1
