# Competition: Plant classification
## 🎯 The competition task was to create a CNN to identify plants from leaves, in the report document a summary of the whole process can be found
## 📈 We had a local training set and a private online test set for evaluation our final mean accuracy was 92.6%
## 🧰 Tools and library used: Python, tensorflow, matplotlib

The team was composed of three students: Myself, Gianluca Ruberto and Tommaso Brumani. 
We tried many different solutions, which are summarized in the following table (also inside the Report.pdf), the final version we uploaded can be found in the notebook, it uses transfer learning from EfficientNetB7 and fine tuning along with a variety of augmentation techniques and oversampling to compensate for the dataset imbalance.

![MKVI_struct](https://user-images.githubusercontent.com/62057461/143869086-8d706e10-321a-42c7-aa8e-436bbfeaf242.png)
Example of the MK_XVI model which includes Batch Normalization after each convolutional layer. 
In the coolvisuals and augmentation .py files extra visualizations can be found (confusion matrix, visualization of augmentations)
The dataset can be found at: https://data.mendeley.com/datasets/tywbtsjrjv/1
