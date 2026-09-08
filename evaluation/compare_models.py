def compare_models(pr_auc, roc_auc, pr_auc_xgb, roc_auc_xgb):
    # ==============================================================================
    #                   MODEL COMPARISON: RandomForest vs XGBoost
    # ==============================================================================
     
    # WHAT:
    # Compare both models on the same untouched x_test using PR-AUC as the
    # primary metric, since this dataset is highly imbalanced and PR-AUC (see
    # the IMPORTANT REMINDER above) is the metric that best reflects
    # performance on the rare fraud class.
    # WHY PR-AUC and not accuracy/ROC-AUC as the deciding metric: consistent
    # with the reasoning already used throughout this script for choosing
    # scoring='average_precision' in GridSearchCV.
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set, same split for both models)")
    print("=" * 60)
    print(f"RandomForest  -> PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"XGBoost       -> PR-AUC: {pr_auc_xgb:.4f} | ROC-AUC: {roc_auc_xgb:.4f}")
     
    if pr_auc_xgb > pr_auc:
        print("\nBest model (by PR-AUC): XGBoost")
        return "xgboost"
    else:
        print("\nBest model (by PR-AUC): RandomForest")
        return "randomforest"