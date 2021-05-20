# By Sammy Rao
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to merge the combined games and stats data and create the training data
def merge_combined_data():
        
    print('\nReading the CSV files . . .')

    extension = 'csv'

    # Go through all the CSV files in the folder
    CSV_files = [i for i in glob.glob('data/combined_games_and_stats/*.{}'.format(extension))]

    print('Merging the CSV files . . .')

    # Merge all the CSV files 
    merged_CSV_file_df = pd.concat([pd.read_csv(file) for file in CSV_files])

    print('Done!')

    # Save the data as CSV
    merged_CSV_file_df.to_csv('data/training_data/training_data.csv', index = False)

    print('Training data CSV file Saved!')

    # Normalize the training data
    normalize_training_data()  

    # Analyze the training data
    analyze_training_data() 

# Function to normalize the training data
def normalize_training_data():

    print('\nNormalizing training data . . .')
    
    # Read the training data CSV file
    training_data_CSV_df = pd.read_csv('data/training_data/training_data.csv')

    # Create a min max scaler
    scaler = MinMaxScaler(feature_range = (0, 1))
    # Fit and scale the training data for normalization
    normalized_training_data = scaler.fit_transform(training_data_CSV_df.values)
    # Put the normalized training data into a data frame
    normalized_training_data_df = pd.DataFrame(normalized_training_data)
    # Give names to the columns
    normalized_training_data_df.columns = training_data_CSV_df.columns

    print('Done!')

    # Save the data as CSV
    normalized_training_data_df.to_csv('data/training_data/normalized_training_data.csv', index = False)

    print('Normalized training data CSV file Saved!')

    # Save the min max scaler as a pickle file
    joblib.dump(scaler, 'data/models/scaler.pkl')

    print('Scaler PKL file Saved!')              

# Function to analyze the training data
def analyze_training_data():

    print('\nAnalyzing training data . . .')

    # Read the training data CSV file
    training_data_CSV_df = pd.read_csv('data/training_data/normalized_training_data.csv')

    # Drop rows with null or missing values
    training_data_CSV_df = training_data_CSV_df.dropna()

    # Get general info on the number of rows and columns in the training data
    print('\nRows and columns: {}'.format(training_data_CSV_df.shape))

    # Check which columns are independent and which one is dependent
    x_data = training_data_CSV_df.iloc[:, :-1]
    print('\nIndependent columns:')
    print('     ' + str(list(x_data.columns))[1:-1])
    y_data = training_data_CSV_df.iloc[:, -1]
    print('\nDependent column: ' +  '\'' + y_data.name + '\'') 

    # Check the number of occurrences of '0' and '1' in the 'WINNER' column
    num_of_zeros = (training_data_CSV_df['WINNER'] == 0).sum()
    num_of_ones = (training_data_CSV_df['WINNER'] == 1).sum()
    num_of_records = len(training_data_CSV_df)

    print('\nClass balance:')
    print('     Number of occurrences of Class 0: ' + str(num_of_zeros) + ' / ' + str(num_of_records))
    print('     Number of occurrences of Class 1: ' + str(num_of_ones) + ' / ' + str(num_of_records))   

    # Select the first 19 columns and all rows
    x_data = training_data_CSV_df.iloc[:, 0:19]
    # Select the last (20th) column and all rows
    y_data = training_data_CSV_df.iloc[:, -1]

    # Use 'ExtraTreesClassifier' to extract the feature importances 
    feture_importances = ExtraTreesClassifier()
    # Fit and extract the features importances
    fit = feture_importances.fit(x_data, y_data)

    # Plot the graph of the top 10 feature importances for better visualization
    feature_importances_graph = pd.Series(fit.feature_importances_, index = x_data.columns)
    feature_importances_graph.nlargest(10).plot(kind = 'barh', color = 'green')
    plt.title('Top 10 feature importances')
    plt.xlabel('Importances') 
    plt.ylabel('Features')
    # Save the feature importances graph as a PNG file
    plt.savefig('images/top_10_feature_importances.png', bbox_inches = 'tight')
    print('\nFeature importances graph PNG file Saved!')
    plt.show()

    # Use 'SelectKBest' to extract the top 10 best features
    feature_scores = SelectKBest(score_func = chi2, k = 10)
    # Fit and extract the best features
    fit = feature_scores.fit(x_data, y_data)

    # Plot the graph of the top 10 best features for better visualization
    feature_scores_graph = pd.Series(fit.scores_, index = x_data.columns)
    feature_scores_graph.nlargest(10).plot(kind = 'barh', color = 'green')
    plt.title('Top 10 best features')
    plt.xlabel('Scores') 
    plt.ylabel('Features')
    # Save the best features graph as a PNG file
    plt.savefig('images/top_10_best_features.png', bbox_inches = 'tight')
    print('Best features graph PNG file Saved!')
    plt.show()

# Function to train a machine learning model (SVM) with the training data
def train_model():

    print('\nTraining model . . .')
    
    # Read the training data CSV file
    training_data_CSV_df = pd.read_csv('data/training_data/normalized_training_data.csv')

    # Select the first 19 columns and all rows
    x_data = training_data_CSV_df.iloc[:, 0:19]
    # Select the last (20th) column and all rows
    y_data = training_data_CSV_df.iloc[:, -1]

    # Split the data into training data and test data
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.3, random_state = 1)

    # Create the model
    model = SVC(kernel = 'linear', gamma = 'scale', C = 1.0, random_state = 1, probability = True)
    # Fit and train the model with the training data
    model.fit(x_training_data, y_training_data)

    # Make test predictions
    predictions = model.predict(x_test_data)

    print('Done!')

    # Evaluate the performance of the model
    print('\nAccuracy score of the model: {}'.format(mt.accuracy_score(y_test_data, predictions).round(2) * 100) + '%')
    print('\nClassification report of the model:') 
    print(mt.classification_report(y_test_data, predictions))

    # Get the confusion matrix
    confusion_matrix = mt.confusion_matrix(y_test_data, predictions)

    # Plot the confusion matrix
    group_names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    group_percentages = ['{0:.2%}'.format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    labels = [f'{names}\n{percentages}' for names, percentages in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    confusion_matrix = sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot = labels, fmt = '', cmap = 'Blues') 
    plt.title('Confusion matrix of model')    
    plt.show()

    # Save the confusion matrix as a PNG file
    confusion_matrix.figure.savefig('images/confusion_matrix.png')

    print('Confusion matrix PNG file Saved!')

    # Save the trained model as a pickle file
    joblib.dump(model, 'data/models/model.pkl') 

    print('Model PKL file Saved!')      

if __name__ == '__main__':

    # Merge the combined games and stats data
    merge_combined_data()

    # Train a machine learning model
    train_model()