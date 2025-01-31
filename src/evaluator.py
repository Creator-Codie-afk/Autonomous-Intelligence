from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, test_data, test_labels):
  E

    Parameters:
    model: The machine learning model to evaluate.
    test_data (pd.DataFrame): The test data.
    test_labels (pd.Series): The true labels for the test data.

    Returns:
    tuple: Returns (accuracy, report) where accuracy is the accuracy score and report is the classification report.
    """
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

def report_performance(accuracy, report):
   """Report model performance metrics.

    Parameters:
    accuracy (float): The accuracy score of the model.
    report (str): The classification report.
    """
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
