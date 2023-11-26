from matplotlib import pyplot as plt


def plot_loss(loss_values):
    plt.plot(loss_values, label='Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve over Iterations')
    plt.legend()
    plt.show()
