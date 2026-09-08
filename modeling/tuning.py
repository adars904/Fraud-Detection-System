from sklearn.model_selection import GridSearchCV


def build_grid_search(fraud_pipeline, cv_strategy):
    # ==============================================================================
    # STEP: GRIDSEARCHCV - HYPERPARAMETER TUNING (NOT THE FINAL EVALUATION)
    # ==============================================================================

    # WHAT GridSearchCV DOES:
    # GridSearchCV tries every combination of hyperparameters in param_grid.
    # For EACH combination, it:
    #   1. Splits x_train/y_train into the folds defined by cv_strategy.
    #   2. Fits the WHOLE fraud_pipeline (imputation -> encoding -> feature
    #      selection -> classifier) on the training portion of each fold.
    #   3. Scores the fitted pipeline on the held-out portion of that fold.
    #   4. Averages the score across all folds for that hyperparameter
    #      combination.
    # After trying every combination, it keeps the hyperparameters with the best
    # average cross-validated score.
    #
    # IMPORTANT - GridSearchCV is NOT the final evaluation of the model:
    # The scores GridSearchCV reports (best_score_, cv_results_) come only from
    # x_train, split internally into folds. x_test is never touched during this
    # step. The real, final evaluation happens later, once, on the untouched
    # x_test/y_test.
    #
    # WHY THIS SCORING METRIC (average_precision / PR-AUC):
    # Fraud is rare (a small percentage of all transactions), so:
    #   - Plain accuracy is misleading: a model that always predicts "not
    #     fraud" would already have very high accuracy while being useless.
    #   - ROC-AUC is commonly used, but with extreme class imbalance it can
    #     look overly optimistic, because the false-positive RATE is measured
    #     against a huge number of negatives, making even a mediocre model's
    #     ROC curve look good.
    #   - Average Precision (the area under the Precision-Recall curve, i.e.
    #     PR-AUC) focuses specifically on how well the model ranks and
    #     identifies the rare positive (fraud) class, which is exactly what we
    #     care about here. It is the standard recommended metric for
    #     imbalanced fraud/anomaly-detection problems.
    # We therefore use scoring='average_precision' as the metric GridSearchCV
    # optimizes for, and we will additionally report precision, recall, F1, and
    # ROC-AUC at final evaluation time for a fuller picture.

    # ==============================================================================
    # STEP: HYPERPARAMETER GRID
    # ==============================================================================

    # WHAT each hyperparameter controls:
    #   - classifier__n_estimators: how many decision trees are built in the
    #     forest. More trees generally means a more stable, less noisy
    #     prediction, but takes longer to train.
    #   - classifier__max_depth: how deep each individual tree is allowed to
    #     grow. Shallower trees (e.g. 10) are less likely to overfit noisy
    #     patterns; deeper trees (e.g. 20) can capture more complex interactions
    #     but risk memorizing the training data.
    #   - classifier__min_samples_leaf: the minimum number of samples required
    #     at a leaf node. Larger values force the tree to generalize (each leaf
    #     decision is based on more examples), which helps avoid overfitting to
    #     rare, noisy patterns - useful here because fraud examples are rare
    #     and we don't want the model memorizing individual fraud rows.
    #
    # WHY this search space is reasonable:
    # This grid has 2 x 2 x 2 = 8 hyperparameter combinations. With
    # cv_strategy having 3 folds, that is 8 x 3 = 24 total pipeline fits. Given
    # ~470k training rows and shallow-to-moderate tree depths, this keeps total
    # runtime realistic while still covering the most impactful hyperparameters
    # for a RandomForest on this dataset size. We deliberately avoided adding
    # more hyperparameters (e.g. max_features, min_samples_split) or a wider
    # range of values, since that would multiply the number of fits and could
    # make the search take an unreasonable amount of time on ~590k+ rows.

    param_grid = {
        'classifier__n_estimators': [100,300],
        'classifier__max_depth': [10,20],
        'classifier__min_samples_leaf': [1, 5]
    }

    grid_search = GridSearchCV(
        estimator=fraud_pipeline,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv_strategy,
        n_jobs=1,        # parallelize across folds/hyperparameter combinations
        verbose=2,        # print progress, since this can take a while on ~590k rows
        refit=True        # after finding the best hyperparameters, refit the pipeline on ALL of x_train using them
    )

    return grid_search