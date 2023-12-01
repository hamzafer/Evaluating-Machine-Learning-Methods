import colour


def error(output_pred_lab, output_test_lab):
    return colour.delta_E(output_pred_lab, output_test_lab, method='CIE 2000')
    # return np.sqrt(np.sum((output_pred_lab - output_test_lab) ** 2, axis=1)) # CIE 1976
