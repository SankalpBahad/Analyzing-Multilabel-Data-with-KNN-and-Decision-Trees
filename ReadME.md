## Statistical Methods in AI: K-Nearest Neighbors and Decision Trees

This repository contains the code for an assignment exploring the application of K-Nearest Neighbors (KNN) and Decision Trees for multilabel classification. 

**Files Description**

1. main.ipynb : contains code for both KNN and Decision Tree Part
2. advertisement.csv : Dataset for Decision Trees Part
3. data.npy : Dataset for KNN part, also used in eval.sh script
4. eval.py : **NON EXECUTABLE** python file, consists of python content in bash file eval.sh
5. eval.sh :  Executable bash file that results in the metrics of test data on the best possible k,encoding and distance metric triplet for ResNet and VIT embeddings

**Project Structure:**

* **`main.ipynb`**: Jupyter Notebook containing the main code for the assignment, including:
    * Exploratory Data Analysis (EDA) for both datasets
    * Implementing KNN from scratch
    * Hyperparameter tuning for KNN
    * Optimization of KNN using vectorization
    * Implementing Decision Tree Classifiers (Powerset and Multioutput)
    * Hyperparameter tuning for Decision Trees
    * Evaluation of model performance with various metrics
* **`eval.sh`**: Bash script for testing unseen data using the trained KNN model.
* **`datasets/`**: Folder containing the datasets used for the assignment:
    * `pictionary.npy`:  Dataset for KNN tasks.
    * `multilabel_dataset.csv`: Dataset for Decision Tree tasks.

**Instructions:**

1. **Install Required Libraries:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```
3. **Explore the code:**
   The notebook provides detailed explanations and code comments. Follow the code step-by-step to understand the implementation of both algorithms and their evaluation.
4. **Test with unseen data:**
   Use the provided `eval.sh` script to test the performance of the best KNN model on unseen data.

**Key Concepts:**

* **K-Nearest Neighbors (KNN):** An intuitive non-parametric algorithm for classification and regression based on similarity between data points.
* **Decision Trees:** A powerful and interpretable algorithm for classification and regression that partitions data based on features.
* **Multilabel Classification:** A scenario where an instance can be assigned multiple labels simultaneously.
* **Hyperparameter Tuning:** The process of finding the optimal values for hyperparameters that control the behavior of a machine learning model.
* **Vectorization:** A technique used to optimize code performance by leveraging vectorized operations in NumPy.

**This assignment provides a hands-on experience in implementing and evaluating two fundamental machine learning algorithms, exploring their performance on multilabel data.** 

**Note:** 

This project is for educational purposes and serves as a starting point for further exploration. Feel free to experiment with the code, explore different hyperparameter values, and apply these techniques to different datasets. 
