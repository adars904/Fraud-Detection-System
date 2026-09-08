def get_best_model(grid_search):
    # ==============================================================================
    # STEP: BEST HYPERPARAMETERS AND BEST CROSS-VALIDATED SCORE
    # ==============================================================================

    # best_params_ : the specific combination of n_estimators / max_depth /
    # min_samples_leaf that produced the highest average cross-validated
    # average_precision score across the 3 folds.
    print("Best hyperparameters found:", grid_search.best_params_)

    # best_score_ : the average_precision score AVERAGED ACROSS THE 3 CV FOLDS
    # for the best hyperparameter combination, computed entirely on
    # x_train_model. This is a cross-validation estimate of performance, NOT
    # the final test score - it tells us how well this configuration is
    # expected to generalize to unseen data, based only on training data. The
    # real, unbiased estimate comes from evaluating on x_test at the very end.
    print("Best cross-validated average_precision score (on training folds only):", grid_search.best_score_)

    # best_estimator_ : the full fraud_pipeline (missing values -> encoding ->
    # feature selection -> classifier), already refit on the ENTIRE
    # x_train_model using the best hyperparameters (because we set refit=True
    # above).
    best_model = grid_search.best_estimator_

    return best_model