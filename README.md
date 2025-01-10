This project is about image classification of medical images. The data is from MedMNIST [1] and there is instructions for downloading from their website: https://medmnist.com. The Datasets file might be left empty. However, it is assumed that the datasets, the 28*28 sized breastmnist.npz [2] and bloodmnist.npz [3] should be in the Datasets file to make the project to be able to be properly implemented. The file A and B contain the Python files that include the functions implementing machine learning models for breastMNIST.npz (Task A) and bloodMNIST.npz (Task B) images respectively.

The project can be implemented in the terminal by running the main.py. (i.e. in the terminal, type: python main.py)
In main.py, for Task A, the logistic regression and random forest models are implemented with PCA on of different variance inputs. The Task B part includes the training curves and test result of CNN (convolutional neural network) model. The corresponding figures are generated once running. Examples are shown below:
![TaskA](https://github.com/user-attachments/assets/8f69868c-2b5e-445f-935a-d6067a6d3577)
![TaskB](https://github.com/user-attachments/assets/82ef3c64-85f7-4f9f-91b1-5b71f3fd34a2)

[1] Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

[2] Walid Al-Dhabyani, Mohammed Gomaa, et al., "Dataset of breast ultrasound images," Data in Brief, vol. 28, pp. 104863, 2020.

[3] Andrea Acevedo, Anna Merino, et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems," Data in Brief, vol. 30, pp. 105474, 2020.
