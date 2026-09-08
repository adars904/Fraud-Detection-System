from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


def evaluate_model(best_model, best_threshold, x_test, y_test):
    # ==============================================================================
    # STEP: FINAL EVALUATION ON THE UNTOUCHED TEST SET
    # ==============================================================================

    # WHAT:
    # This is the first and only time x_test/y_test are used. Everything
    # before this point (imputation fitting, encoding fitting, feature
    # selection, cross-validation, hyperparameter tuning, and threshold
    # selection) has used only x_train_model or x_val - never x_test.
    #
    # WHY this matters:
    # Because best_model has never seen x_test in any form during fitting, and
    # best_threshold was chosen using only x_val, scoring here gives an
    # unbiased estimate of how the model will perform on genuinely new,
    # unseen transactions - which is what actually matters in a real
    # fraud-detection system.

    # Predicted probability of the positive (fraud) class - needed for
    # threshold-independent metrics like ROC-AUC and PR-AUC, and also used
    # below together with best_threshold to produce hard 0/1 predictions.
    y_pred_proba = best_model.predict_proba(x_test)[:, 1]

    # Hard class predictions (0 = not fraud, 1 = fraud) using best_threshold,
    # which was selected on the validation set above - NOT the default 0.5
    # threshold, and NOT tuned using x_test.
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    # ------------------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------------------
    # WHAT: a 2x2 table of counts:
    #   [[True Negatives,  False Positives],
    #    [False Negatives, True Positives]]
    # WHY it matters for fraud: it separates the two very different types of
    # mistakes a fraud model can make -
    #   False Positives = legitimate transactions flagged as fraud (annoys
    #     customers, creates manual review work),
    #   False Negatives = actual fraud that slipped through undetected (direct
    #     financial loss - usually the more costly mistake in fraud detection).
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------------------
    # Precision, Recall, F1-score (via classification_report)
    # ------------------------------------------------------------------------------
    # Precision (for the fraud class) = of all transactions the model FLAGGED
    # as fraud, what fraction were actually fraud? Low precision means too many
    # legitimate customers get incorrectly flagged.
    #
    # Recall (for the fraud class) = of all the transactions that were ACTUALLY
    # fraud, what fraction did the model catch? This is usually the metric that
    # matters most in fraud detection: missing real fraud (low recall) directly
    # costs money, whereas a false alarm (lower precision) is comparatively
    # cheaper - typically just an extra manual review step.
    #
    # F1-score = the harmonic mean of precision and recall, a single number
    # that balances both concerns when neither can be ignored.
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

    # ------------------------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------------------------
    # WHAT: the probability that the model ranks a randomly chosen fraud
    # transaction higher (more likely to be fraud) than a randomly chosen
    # legitimate transaction, across all possible thresholds.
    # CAUTION: with heavy class imbalance, ROC-AUC can look deceptively high
    # because the false-positive rate is calculated against a very large number
    # of negatives. We report it here for reference, but PR-AUC (below) is the
    # more trustworthy metric for this specific dataset.
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # ------------------------------------------------------------------------------
    # PR-AUC / Average Precision
    # ------------------------------------------------------------------------------
    # WHAT: average_precision_score computes Average Precision (AP) - it
    # summarizes the precision-recall curve as the weighted mean of the
    # precision achieved at each threshold, weighted by the increase in recall
    # from the previous threshold. This is NOT the same computation as taking
    # the trapezoidal area under the precision-recall curve, but AP is the
    # standard PR-AUC-style summary metric used in imbalanced classification
    # (including scikit-learn's own scoring='average_precision' used in
    # GridSearchCV above), so we refer to it here as "PR-AUC / Average
    # Precision" to match that common usage. It is NOT ROC-AUC - see the
    # separate ROC-AUC section above for that metric.
    # WHY it is especially appropriate here: like ROC-AUC, AP is
    # threshold-independent, but unlike ROC-AUC it is not affected by the large
    # number of true negatives, so it gives a much more realistic picture of
    # performance when fraud is rare - which is exactly the situation in this
    # dataset.
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    # ==============================================================================
    # IMPORTANT REMINDER:
    # A high overall accuracy does NOT automatically mean this is a good fraud
    # model. Because fraud is rare, a model could achieve very high accuracy
    # simply by predicting "not fraud" for almost every transaction, while
    # missing nearly all actual fraud (i.e. very poor recall for the fraud
    # class). This is exactly why accuracy was never used as the scoring metric
    # for GridSearchCV, and why precision/recall/F1/PR-AUC are reported above
    # instead of relying on accuracy alone.
    # ==============================================================================  

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }