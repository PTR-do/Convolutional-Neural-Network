Deep Learning Image Classification and Transfer Learning project.
You can find more information about the experiment conducted for the project in the Power Point presentation.


The datasets used for this project are available at the links:
https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets
https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset


The code is structured as follows:

- dataset_conversion.py: converts .tif images from the UCMerced_LandUse dataset into .jpg format.

- dataset_split.py: splits the UCMerced dataset into Training (70%), Validation (15%), and Test (15%) sets, 
  saving them as optimized tf.data.Dataset objects for high-performance loading.

- CNN_1.py: custom ResNet architecture sesigned to be used for Transfer Learning strategies. 

- CNN_2.py: custom ResNet architecture adapted and trained only for the target dataset.

- First_Strategy_backbone.py: train of CNN_1 on the larger dataset AID. 

- First_Strategy_tuning1.py: fine-tuning of CNN_1 on the target dataset UCMerced_LandUse. 

- First_Strategy_tuning2.py: alternative fine-tuning of CNN_1.

- Second_Strategy.py: baseline training strategy.

- Models_evaluation.py: evaluates CNN_1 and CNN_2 on validation set. 

- Final_Model.py: production-ready inference script. 


Generated Folders (if you run the script):
- ../UCMerced_JPG: contains the converted JPG images.
- ../Dataset_UCMerced: contains the serialized tf.data.Dataset batches (train, val, test).


Generated Weights:
- cnn_1_aid.weights.h5: pre-trained weights of CNN_1 from the AID dataset.
- cnn_1_finetuning1.weights.h5: final weights from the first fine-tuning strategy.
- cnn_1_finetuning2.weights.h5: final weights from the second fine-tuning strategy.
- cnn_2_ucmerced.weights.h5: final weights from the training-from-scratch strategy.
