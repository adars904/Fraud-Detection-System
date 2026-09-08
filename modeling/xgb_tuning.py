from sklearn.model_selection import GridSearchCV


def build_xgb_grid_search(xgb_pipeline, cv_strategy):
    # ------------------------------------------------------------------------------
    # STEP: Hyperparameter grid - same size/shape as the RandomForest grid
    # ------------------------------------------------------------------------------
    # WHAT each hyperparameter controls:
    #   - classifier__n_estimators: number of boosting rounds (trees added
    #     sequentially). More rounds can improve fit but risks overfitting and
    #     takes longer.
    #   - classifier__max_depth: how deep each individual tree is allowed to
    #     grow. XGBoost trees are typically shallower than RandomForest trees
    #     (3-6 is common) because boosting adds many trees rather than relying
    #     on a few deep ones.
    #   - classifier__min_child_weight: XGBoost's closest analogue to
    #     RandomForest's min_samples_leaf - the minimum sum of instance weight
    #     needed in a child node. Larger values make the model more
    #     conservative (less likely to overfit rare, noisy patterns).
    #
    # This grid intentionally mirrors the RandomForest grid's shape (2 values
    # x 2 values x 2 values = 8 combinations, x 3 CV folds = 24 fits), per the
    # same "same size, most thorough" search used for RandomForest above.
    param_grid_xgb = {
        'classifier__n_estimators': [100,300],
        'classifier__max_depth': [3, 6],
        'classifier__min_child_weight': [1,5]
    }
     
    grid_search_xgb = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid_xgb,
        scoring='average_precision',
        cv=cv_strategy,      # same StratifiedKFold strategy used for RandomForest, for a fair comparison
        n_jobs=1,
        verbose=2,
        refit=True
    )

    return grid_search_xgb