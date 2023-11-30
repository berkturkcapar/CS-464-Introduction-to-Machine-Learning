import pandas as pd
import numpy as np

def createCSVFile(arr, alpha):
    df = pd.DataFrame(arr)
    df.columns = ['Not Spam', 'Spam']
    df.index = ['Not Spam', 'Spam']
    if alpha == -1:
        df.to_csv('bnb.csv', index=True, header=True)
    else:
        df.to_csv(f'mnb{alpha}.csv', index=True, header=True)

def multinomialNaiveBayes(x_train, y_train, x_test, y_test, alpha=0):
    print(f"---------Multinomial Naive Bayes with alpha: {alpha}---------")
    # Define the number of classes and the number of features
    num_classes = 2
    num_features = x_train.shape[1]

    # Compute the prior probabilities of the classes
    priors = np.zeros(num_classes)
    for i in range(num_classes):
        priors[i] = np.sum(y_train == i) / len(y_train)

    # Compute the conditional probabilities of the features given the classes
    conditionals = np.zeros((num_classes, num_features))
    for i in range(num_classes):
        indices = np.where(y_train == i)[0]
        feature_counts = np.sum(x_train[indices], axis=0)
        total_count = np.sum(feature_counts)
        conditionals[i] = (feature_counts + alpha) / (total_count + alpha * num_features)
    
    # Evaluate the model on the test set
    y_pred = np.zeros(len(y_test))
    for i in range(len(y_test)):
        probabilities = np.zeros(num_classes)
        for j in range(num_classes):
            # Compute the log of the conditional probabilities
            # To ignore the warning of log(0), we use np.errstate
            with np.errstate(divide='ignore'):
                logs = np.log(conditionals[j])
            # change all -inf instances in conditionals to -1e12
            logs[np.isneginf(logs)] = -1e12
            probabilities[j] = np.log(priors[j]) + np.sum(logs * x_test[i])
        # If the probabilities are equal, predict '0'
        if (probabilities[0] < probabilities[1]):
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    # Compute the accuracy of the model
    correct_predictions = np.sum(y_pred == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")

    # Compute the confusion matrix
    cm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i][j] = np.sum((y_test == i) & (y_pred == j))
    createCSVFile(cm, alpha)

    # Compute the number of wrong predictions
    wrong_predictions = total_predictions - correct_predictions
    print(f"Number of wrong predictions: {wrong_predictions}\n")

    return accuracy

def bernoulliNaiveBayes(x_train, y_train, x_test, y_test):
    print(f"---------Bernoulli Naive Bayes---------")
    n_train, num_features = x_train.shape
    n_test = x_test.shape[0]

    # Estimate prior probabilities
    p_spam = np.sum(y_train) / n_train
    p_not_spam = 1 - p_spam

    # Estimate likelihood parameters
    count_spam = np.sum(x_train[y_train == 1], axis=0) + 1
    count_not_spam = np.sum(x_train[y_train == 0], axis=0) + 1
    p_word_given_spam = count_spam / (np.sum(count_spam) + num_features)
    p_word_given_not_spam = count_not_spam / (np.sum(count_not_spam) + num_features)

    # Classify test examples
    y_pred = np.zeros(n_test)
    for i in range(n_test):
        log_p_spam_given_x = np.log(p_spam) + np.sum(np.log(p_word_given_spam[x_test[i] == 1])) \
            + np.sum(np.log(1 - p_word_given_spam[x_test[i] == 0]))
        log_p_not_spam_given_x = np.log(p_not_spam) + np.sum(np.log(p_word_given_not_spam[x_test[i] == 1])) \
            + np.sum(np.log(1 - p_word_given_not_spam[x_test[i] == 0]))
        if log_p_spam_given_x > log_p_not_spam_given_x:
            y_pred[i] = 1

    # Compute accuracy and confusion matrix
    correct_predictions = np.sum(y_pred == y_test)
    wrong_predictions = n_test - correct_predictions
    accuracy = correct_predictions / n_test
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_test)):
        cm[y_test[i].astype(int), y_pred[i].astype(int)] += 1

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of wrong predictions: {wrong_predictions}\n")
    createCSVFile(cm, -1)
    return accuracy, cm, wrong_predictions

if __name__ == '__main__':
    # Define the paths to the training and test data
    x_train_path = 'x_train.csv'
    y_train_path = 'y_train.csv'
    x_test_path = 'x_test.csv'
    y_test_path = 'y_test.csv'
    #for i in range(6):
        #multinomialNaiveBayes(x_train_path, y_train_path, x_test_path, y_test_path, i)

    # Load the training and test data into dataframes
    x_train_df = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)
    x_test_df = pd.read_csv(x_test_path)
    y_test_df = pd.read_csv(y_test_path)

    # Convert the dataframes to numpy arrays
    x_train = x_train_df.values
    y_train = y_train_df.values.ravel()
    x_test = x_test_df.values
    y_test = y_test_df.values.ravel()

    # Train the model and evaluate it on the test set (without smoothing)
    multinomialNaiveBayes(x_train, y_train, x_test, y_test)
    # Train the model and evaluate it on the test set (with smoothing)
    multinomialNaiveBayes(x_train, y_train, x_test, y_test, 5)
    # Bernoulli Naive Bayes
    bernoulliNaiveBayes(x_train, y_train, x_test, y_test)