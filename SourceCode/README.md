# Source code for Animal Classification


<p align="justify"> Welcome to the source code repository for the final project in Computer Vision. This project focuses on animal classification from images using basic feature extraction methods such as HOG (Histogram of Oriented Gradients) and various machine learning algorithms (SVM, KNN, RF). </p>

Here is the directory structure of the project:

- `dataset/`: Directory containing datasets used for training, validation and testing [Animals Dataset](https://www.kaggle.com/datasets/antobenedetti/animals)
- `test_image/`: Directory containing some images for demo
- `main.ipynb`: Jupyter notebook used for visualizing data, running machine learning algorithms, demonstrating results and demo.
- `run_knn&rf_NoPCA.py`: Python script to run K-Nearest Neighbors (KNN) and Random Forest (RF) algorithms without using PCA.
- `run_knn&rf_PCA.py`: Python script to run K-Nearest Neighbors (KNN) and Random Forest (RF) algorithms using PCA.
- `run_svm_NoPCA.py`: Python script to run Support Vector Machine (SVM) algorithm without using PCA.
- `run_svm_PCA.py`: Python script to run Support Vector Machine (SVM) algorithm using PCA.
- `results_knn&rf_NoPCA.txt`, `results_knn&rf_PCA.txt`, `results_svm_NoPCA.txt`, `results_svm_PCA.txt`: Text files containing results from corresponding `.py` files.
- `mean_val.py`: Python script to calculate the mean accuracy from result `.txt` files.
- `Statistic.xlsx`: File excel containing experimental results.
