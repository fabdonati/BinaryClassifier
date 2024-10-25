import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve, accuracy_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore", category=UserWarning)



# Function to plot exploratory analysis
def exploratory_analysis_plots(training_data):
    # Create 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: Target Class Imbalance
    sns.countplot(ax=axes[0], x='classlabel', data=training_data)
    axes[0].set_title('Target Class Imbalance in Training Dataset')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('Count')

    # Plot 2: Heatmap of Missing Values
    sns.heatmap(training_data.isna(), cbar=False, cmap='viridis', ax=axes[1])
    axes[1].set_title('Heatmap of Missing Values in Training Dataset')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Samples')

    # Plot 3: Distribution of Numerical Columns
    numerical_columns = training_data.select_dtypes(include=['float64']).columns
    for column in numerical_columns:
        sns.histplot(training_data[column], kde=True, ax=axes[2], label=column, element="step", fill=False)
    axes[2].set_title('Distribution of Numerical Columns')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Frequency')
    axes[2].legend(loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def print_dataset_stats(dataset, name):
    """
    Print the head, summary statistics, number of missing values and percentage of missing values for the dataset
    :param dataset: pd.DataFrame: the dataset to analyze
    :param name: str: the name of the dataset
    """
    print(f"Dataset: {name}")
    print("-" * 50)
    print("Head of the dataset:")
    print(dataset.head(), "\n")
    
    print("Summary statistics:")
    print(dataset.describe(), "\n")
    
    print("Number of missing values per column:")
    print(dataset.isna().sum(), "\n")
    
    print("Percentage of missing values per column:")
    print((dataset.isna().sum() / len(dataset)) * 100, "\n")
    print("\n")

def plot_confusion_matrix(ax, y_true, y_pred, title):
    """
    Plot the confusion matrix
    :param ax: object: the axis to plot the matrix
    :param y_true: np.array: the true labels
    :param y_pred: np.array: the predicted labels
    :param title: str: the title of the plot
    """

    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

def plot_precision_recall_curve(ax, y_true, y_scores, title, label):
    """
    Plot the precision-recall curve
    :param ax: object: the axis to plot the curve
    :param y_true: np.array: the true labels
    :param y_scores: np.array: the predicted scores
    :param title: str: the title of the plot
    :param label: str: the label of the curve
    """

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    ax.plot(recall, precision, marker='.', label=f'{label} (AvgPr={avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def plot_results(y_train, y_hat_train, y_vali, y_hat_vali, y_test, y_hat_test, best):
    """
    Plot the confusion matrix and the precision-recall curve for the train, validation and test set
    :param y_train: np.array: the true labels for the train set
    :param y_hat_train: np.array: the predicted labels for the train set
    :param y_vali: np.array: the true labels for the validation set
    :param y_hat_vali: np.array: the predicted labels for the validation set
    :param y_test: np.array: the true labels for the test set
    :param y_hat_test: np.array: the predicted labels for the test set
    :param best: object: the best estimator found by the grid search
    """

    # Create subplots for the 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot Train Confusion Matrix
    plot_confusion_matrix(axes[0, 0], y_train, y_hat_train, 'Train Confusion Matrix')

    # Plot Validation Confusion Matrix
    plot_confusion_matrix(axes[0, 1], y_vali, y_hat_vali, 'Validation Confusion Matrix')

    # Plot Test Confusion Matrix
    plot_confusion_matrix(axes[0, 2], y_test, y_hat_test, 'Test Confusion Matrix')

    # Plot Train Precision-Recall Curve
    y_train_scores = best.predict_proba(x_train)[:, 1]
    plot_precision_recall_curve(axes[1, 0], y_train, y_train_scores, 'Train Precision-Recall Curve', 'Train')

    # Plot Validation Precision-Recall Curve
    y_vali_scores = best.predict_proba(x_vali)[:, 1]
    plot_precision_recall_curve(axes[1, 1], y_vali, y_vali_scores, 'Validation Precision-Recall Curve', 'Validation')

    # Plot Test Precision-Recall Curve
    y_test_scores = best.predict_proba(x_test)[:, 1]
    plot_precision_recall_curve(axes[1, 2], y_test, y_test_scores, 'Test Precision-Recall Curve', 'Test')

    plt.tight_layout()
    plt.show()

def get_classifier(method: str) -> object:
  """
  Get the classifier and the parameters to use for the grid search
  :param method: str: the method to use
  :return: object: the classifier to use
  :return: dict: the parameters to use for the grid search
  :return: str: the label of the classifier
  """

  if method == 'log':
      label = 'Logistic Regression'
      classifier = LogisticRegression(random_state=0)
      params = {
          'C': [0.01, 0.1, 1, 2.5, 5, 10],  # Regularization strength
          'penalty': ['l1', 'l2', 'elasticnet'],  # Penalty types to control complexity
          'solver': ['saga'],  # Solver that supports elasticnet penalty
          'l1_ratio': [0.15, 0.5, 0.85],  # Required for elasticnet penalty
          'class_weight': ['balanced', {0: 1, 1: 10}],  # Handle class imbalance
          'max_iter': [100, 200, 500]  # Limit the number of iterations for convergence
      }
  elif method == 'per': 
      label = 'Perceptron'
      classifier = Perceptron(random_state=0)  
      params = {
          'penalty': [None, 'l2', 'l1', 'elasticnet'],  # Regularization types
          'alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularization strength
          'max_iter': [1000, 2000, 3000],  # Limit the number of iterations
          'class_weight': [None, 'balanced']  # Handle class imbalance
      }
  elif method == 'ada':  
      label = 'AdaBoost'
      classifier = AdaBoostClassifier(random_state=0)
      params = {
          'n_estimators': [50, 100, 200],  # Number of estimators
          'learning_rate': [0.01, 0.1, 1., 10.],  # Learning rate for boosting
          'estimator': [LogisticRegression(), DecisionTreeClassifier(max_depth=15)]
      }
  elif method == 'nb':
      label = 'Naive Bayes'
      classifier = GaussianNB()
      params = {
          'var_smoothing': [1e-9, 1e-8, 1e-7]  # Smoothing parameter
      }  
  elif method == 'svm':
      label = 'Support Vector Machine'
      classifier = SVC(random_state=0, probability=True)  
      params = {
          'C': [0.01, 0.1, 1, 10],  # Regularization strength
          'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types for SVM
          'class_weight': ['balanced', {0: 1, 1: 10}],  # Handle class imbalance
          'gamma': ['scale', 'auto']  # Kernel coefficient
      }
  elif method == 'knn':
      label = 'K-Nearest Neighbors'
      classifier = KNeighborsClassifier()
      params = {
          'n_neighbors': [3, 5, 7, 9, 11, 15],  # Further expand neighbors range
          'weights': ['uniform', 'distance'],  # Keep weight options
          'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],  # Add more distance metrics
          'algorithm': ['auto'],  # Explore KNN algorithm options
      }
  elif method == 'xgb':
      label = 'XGBoost'
      classifier = XGBClassifier(random_state=0)
      params = {
          'n_estimators': [50, 100, 200],  # Number of trees
          'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5, 1.0],  # Learning rate for boosting
          'max_depth': [3, 5, 7, 15],  # Maximum depth of a tree
          'subsample': [0.6, 0.8, 1.0],  # Fraction of samples to use for training
          'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features to use per tree
      }
  elif method == 'rf':
      label = 'Random Forest'
      classifier = RandomForestClassifier(random_state=0)
      params = {
          'n_estimators': [50, 100],  # Increase the number of trees
          'max_depth': [10, 15],  # Limit maximum depth to control tree complexity
          'min_samples_split': [5, 10],  # Decrease minimum samples required to split for finer granularity
          'min_samples_leaf': [5, 10, 15],  # Decrease minimum samples in leaf to allow more detailed splits
          'max_features': ['sqrt', 'log2'],  # Reduce number of features considered at each split
          'class_weight': ['balanced']  # Handle class imbalance
      }
  return classifier, params, label

def clean(data: pd.DataFrame, class_label: str) -> pd.DataFrame:
    """
    Clean the data by replacing the na values and transforming the boolean columns
    :param data: pd.DataFrame: the data to clean
    :param class_label: str: the label of the class
    :return: pd.DataFrame: the cleaned data
    """

    # create a copy of the data to avoid modifying the original one
    data = data.copy(deep=True)

    # replace eventual na values with np.nan
    data.replace(["NA", "N/A", "na", "n/a", "None", "NONE", None], np.nan, inplace=True)
    
    # create imputers, one for numerical using the median and one for binary using the most frequent value
    imputer_num = SimpleImputer(strategy='median')
    imputer_bin = SimpleImputer(strategy='most_frequent')

    # get the columns that can be considered as booleans (drop the nans to avoid being considered as a category) and the numerical ones
    boolean_like = [col for col in data.columns if data[col].dropna().nunique() == 2]
    numeric_like = data.columns.difference(data.select_dtypes(include=['object']).columns)

    # create a label encoder to transform the boolean columns into 0 and 1
    label_encoder = LabelEncoder()
    
    # impute the values and transform the boolean columns
    for c in boolean_like:
        if c not in class_label:
            data[c] = imputer_bin.fit_transform(data[[c]]).flatten()
            data[c] = label_encoder.fit_transform(data[c])

    # impute the values for the numerical columns
    for c in numeric_like:
        data[c] = imputer_num.fit_transform(data[[c]]).flatten()

    # transform the class label into 0 and 1
    data[class_label] = (data[class_label] == "yes.").astype(int)

    return data

def plot_feature_importance(importance_na_df):
    """
    Plot the feature importance and the number of na values for the top 20 features
    :param importance_na_df: pd.DataFrame: the dataframe with the importance and the number of na values
    """

    plt.figure(figsize=(14, 10))
    ax1 = sns.barplot(x=importance_na_df[:20].index, y=importance_na_df[:20]['Importance'], color='b', label='Importance')
    ax2 = ax1.twinx()
    sns.lineplot(x=importance_na_df[:20].index, y=importance_na_df[:20]['NA_Counts'], color='r', marker='o', ax=ax2, label='NA Counts')
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Importance Score', color='b')
    ax2.set_ylabel('Number of NAs', color='r')
    plt.title('Top 20 Feature Importances and NA Counts')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right')
    plt.show()

def prepare_data(train: pd.DataFrame, validation: pd.DataFrame, scale: bool, remove_correlated: bool, check_importance: bool, rebalance: bool, pca=bool) -> pd.DataFrame:
    """
    Prepare the data by cleaning it, encoding the categorical columns, scaling the data, removing correlated features, applying PCA and rebalancing the data
    :param train: pd.DataFrame: the train set
    :param validation: pd.DataFrame: the validation set
    :param scale: bool: whether to scale the data
    :param remove_correlated: bool: whether to remove correlated features
    :param check_importance: bool: whether to check the importance of the features
    :param rebalance: bool: whether to rebalance the data
    :param pca: bool: whether to apply PCA
    :return: pd.DataFrame: the cleaned and prepared data
    """

    # count the number of na values for each column as it will be used to check the importance of the features
    na_counts = train.isna().sum()
    class_label = "classlabel"

    # clean the train and the validation set
    train = clean(train, class_label)
    validation = clean(validation, class_label)
    
    # get the categorical columns and encode them using one hot encoding and drop the first column to avoid multicollinearity
    categorical = train.select_dtypes(include=['object']).columns
    train_ = pd.get_dummies(train, columns=categorical, drop_first=True)
    encoded = train_.columns.difference(train.columns)
    train_[encoded] = train_[encoded].astype(int)

    # repeat the same for the validation set
    categorical = validation.select_dtypes(include=['object']).columns
    validation_ = pd.get_dummies(validation, columns=categorical, drop_first=True)
    encoded = validation_.columns.difference(validation.columns)
    validation_[encoded] = validation_[encoded].astype(int)

    # align the columns of the data and the validation set to avoid having different columns
    train_, validation_ = train_.align(validation_, join='left', axis=1, fill_value=0)

    # check the importance of the features and drop the ones that have a high number of na values and low importance
    # current threshold is set to 40% of the number of rows, and importance threshold is set to 5%
    if check_importance:
        # create a random forest classifier to check the importance of the features on a clean dataset (no nan values)
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(train_[[c for c in train_.columns if c != class_label]], train_[class_label])
        feature_importances = pd.Series(model.feature_importances_, index=train_.columns.drop(class_label))

        # sort the features by importance and create a dataframe with the importance and the number of na values
        feature_importances.sort_values(ascending=False, inplace=True)
        importance_na_df = pd.DataFrame({'Importance': feature_importances, 'NA_Counts': na_counts.reindex(feature_importances.index).values})
        importance_na_df.index.name = 'Feature'
        importance_na_df.sort_values(by='Importance', ascending=False, inplace=True)

        threshold_na = 0.4*len(train_)
        low_importance_threshold = 0.05

        # get the features meeting the criteria and drop them
        plot_feature_importance(importance_na_df) 
        features_to_drop = importance_na_df[(importance_na_df['NA_Counts'] > threshold_na) & (importance_na_df['Importance'] < low_importance_threshold)].index
        train_.drop(columns=features_to_drop, inplace=True)
        validation_.drop(columns=features_to_drop, inplace=True)

    # split the data into features and labels and convert them to numpy arrays of float32 to avoid any issues with the classifiers
    X_train = train_[[c for c in train_.columns if c != class_label]].astype(np.float32).values
    Y_train = train_[[class_label]].astype(np.float32).values
    
    # do the same for the validation set
    X_validation = validation_[[c for c in validation_.columns if c != class_label]].astype(np.float32).values
    Y_validation = validation_[[class_label]].astype(np.float32).values
    
    D = np.concatenate((X_train, Y_train), axis=1)
    T = np.concatenate((X_validation, Y_validation), axis=1)

    # apply a shuffle to the data to prevent any bias    
    np.random.seed(0)
    np.random.shuffle(D)
    
    # in the following I will define my datasets so as to be able to effectively test the model on unseen data (from the 'Validation.csv' dataset)
    # for the train and validation set, I will use the train set from 'Training.csv' using the 80/20 split
    # for the test set, I will use the validation set from 'Validation.csv'
    train_samples = int(D.shape[0] * 0.8)
    x_train, y_train = D[:train_samples, :-1], D[:train_samples, -1]
    x_vali, y_vali = D[train_samples:, :-1], D[train_samples:, -1]
    x_vali, y_vali = D[train_samples:, :-1], D[train_samples:, -1]
    x_test, y_test = T[:, :-1], T[:, -1]

    # apply scaling, as the data is not normalized and would bias the pca
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_vali = scaler.transform(x_vali)
        x_test = scaler.transform(x_test)
    
    # remove correlated features to avoid multicollinearity
    if remove_correlated:
        # calculate the correlation matrix and get the upper triangle without the diagonal
        corr_matrix = np.corrcoef(x_train, rowvar=False)
        upper_triangle = np.triu(corr_matrix, k=1)

        # get the columns that have a correlation higher than 0.75 and drop them
        to_drop = []
        num_cols = upper_triangle.shape[1]
        for i in range(num_cols):
            for j in range(i + 1, num_cols):
                if abs(upper_triangle[i, j]) > 0.75:
                    to_drop.append(j)
        
        # remove duplicates and drop the columns
        to_drop = sorted(set(to_drop))

        # drop the columns from the train, validation and test set
        x_train = np.delete(x_train, to_drop, axis=1)
        x_vali = np.delete(x_vali, to_drop, axis=1)
        x_test = np.delete(x_test, to_drop, axis=1)
    
    # apply PCA to reduce the number of features
    # in this case, I will use 4 components as it corresponds to the elbow of the scree plot
    if pca:
        extractor = PCA(n_components=4)
        x_train = extractor.fit_transform(x_train)
        x_vali = extractor.transform(x_vali)
        x_test = extractor.transform(x_test)

        extractor.fit(x_train)
        plt.plot(range(1, len(extractor.explained_variance_ratio_) + 1), extractor.explained_variance_ratio_, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.show()
    
    # as the data is imbalanced, I will use SMOTE to oversample the minority class
    if rebalance: 
        smote = SMOTE(random_state=0)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        x_vali, y_vali = smote.fit_resample(x_vali, y_vali)
    
    return x_train, y_train, x_vali, y_vali, x_test, y_test
    

if __name__ == "__main__":
    trainset = pd.read_csv("Training.csv", delimiter=";", decimal=",")
    testset = pd.read_csv("Validation.csv", delimiter=';', decimal=",") 

    print_dataset_stats(trainset, "Training Set")

    exploratory_analysis_plots(trainset)

    x_train, y_train, x_vali, y_vali, x_test, y_test = prepare_data(
        trainset,
        testset,
        scale=True,
        remove_correlated=True,
        check_importance=True,
        rebalance=True,
        pca=True
    )

    # get the classifier and the parameters to use from the list:
    # log: Logistic Regression
    # per: Perceptron
    # ada: AdaBoost
    # nb: Naive Bayes
    # svm: Support Vector Machine
    # knn: K-Nearest Neighbors
    # xgb: XGBoost
    # rf: Random Forest
    method = 'log' 
    classifier, params, label = get_classifier(method)

    # perform a grid search to find the best parameters for the classifier
    grid_search = GridSearchCV(estimator=classifier, param_grid=params, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # get the best estimator and the best parameters
    best = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # predict the labels for the train, validation and test set
    y_hat_train = best.predict(x_train)
    train_accuracy = sum(y_hat_train == y_train) / len(y_train)
    print('Train accuracy:', train_accuracy)

    y_hat_vali = best.predict(x_vali)
    vali_accuracy = sum(y_hat_vali == y_vali) / len(y_vali)
    print('Validation accuracy:', vali_accuracy)

    y_hat_test = best.predict(x_test)
    test_accuracy = sum(y_hat_test == y_test) / len(y_test)
    print('Test accuracy:', test_accuracy)

    print(f"\nClassification Report for {label}:\n", classification_report(y_test, y_hat_test))

    # plot the results
    plot_results(y_train, y_hat_train, y_vali, y_hat_vali, y_test, y_hat_test, best)