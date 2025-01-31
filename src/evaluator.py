from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, test_data, test_labels):
    """Evaluate a machine learning model using accuracy and classification report."""
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

def report_performance(accuracy, report):
    """Report model performance metrics."""
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
