import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def visualize_lab_values(actual_lab, predicted_lab):
    # Normalize the LAB data for visualization
    scaler_lab = MinMaxScaler()
    actual_lab_norm = scaler_lab.fit_transform(actual_lab)
    predicted_lab_norm = scaler_lab.transform(predicted_lab)

    # Visualizing the actual vs predicted L*a* values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(actual_lab_norm[:, 0], actual_lab_norm[:, 1], c='red', label='Actual L*a*')
    plt.scatter(predicted_lab_norm[:, 0], predicted_lab_norm[:, 1], c='blue', label='Predicted L*a*')
    plt.title('Actual vs Predicted L*a*')
    plt.xlabel('Normalized L*')
    plt.ylabel('Normalized a*')
    plt.legend()

    # Visualizing the actual vs predicted a*b* values
    plt.subplot(1, 2, 2)
    plt.scatter(actual_lab_norm[:, 1], actual_lab_norm[:, 2], c='red', label='Actual a*b*')
    plt.scatter(predicted_lab_norm[:, 1], predicted_lab_norm[:, 2], c='blue', label='Predicted a*b*')
    plt.title('Actual vs Predicted a*b*')
    plt.xlabel('Normalized a*')
    plt.ylabel('Normalized b*')
    plt.legend()

    plt.show()

# Example usage
# visualize_lab_values(output_test_lab, output_pred_lab)
