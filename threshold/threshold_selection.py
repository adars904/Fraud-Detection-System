import numpy as np
from sklearn.metrics import f1_score


def select_best_threshold(best_model, x_val, y_val):
    # ==============================================================================
    # STEP: THRESHOLD SELECTION ON THE VALIDATION SET (x_test IS NOT TOUCHED HERE)
    # ==============================================================================
    # WHAT:
    # RandomForestClassifier.predict() uses a fixed default threshold of 0.5 to
    # turn predicted probabilities into 0/1 predictions. For an imbalanced
    # problem like fraud detection, 0.5 is not necessarily the threshold that
    # gives the best precision/recall trade-off, so we search for a better one
    # using an F1-based sweep.
    #
    # WHY the validation set (not the test set):
    # best_model was fit using only x_train_model/y_train_model, so x_val is
    # genuinely unseen data from the model's point of view - calling
    # best_model.predict_proba(x_val) here is safe. Selecting the threshold on
    # x_val, instead of on x_test, keeps x_test fully reserved for the final,
    # one-time evaluation later in this script.

    # Predicted probability of the positive (fraud) class on the validation set.
    val_probabilities = best_model.predict_proba(x_val)[:, 1]

    # Search a grid of candidate thresholds and keep the one with the best F1
    # score on the validation set. F1 is used because it balances precision and
    # recall, which matters here - optimizing only one of them at the expense
    # of the other is not useful for a fraud-detection system.
    candidate_thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_val_f1 = -1

    for threshold in candidate_thresholds:
        val_preds = (val_probabilities >= threshold).astype(int)
        current_f1 = f1_score(y_val, val_preds)
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_threshold = threshold

    print("Best threshold selected on validation set:", best_threshold)
    print("Validation F1-score at best threshold:", best_val_f1)

    return best_threshold, best_val_f1