from eda.explore import load_and_merge_data
from cleaning.clean_data import clean_data
from data.split_data import train_test_split_data, train_val_split_data
from features.drop_high_missing import drop_high_missing_cols
from features.function_transformer import check_skew
from features.feature_construction import construct_features
from features.feature_split import split_features
from preprocessing.encoding_transformer import build_encoding_transformer
from features.feature_selection import build_feature_selector
from modeling.pipeline import build_fraud_pipeline
from modeling.cv_strategy import build_cv_strategy
from modeling.tuning import build_grid_search
from modeling.best_model import get_best_model
from threshold.threshold_selection import select_best_threshold
from evaluation.evaluate import evaluate_model
from modeling.xgb_pipeline import build_xgb_pipeline
from modeling.xgb_tuning import build_xgb_grid_search
from evaluation.compare_models import compare_models
from modeling.persistence import save_model


def main():
    # Step 1: EDA / load + merge data
    df1, df2, df3 = load_and_merge_data()

    # Step 2: Data cleaning
    df3 = clean_data(df3)

    # Step 3: Train/test split
    x_train, x_test, y_train, y_test = train_test_split_data(df3)

    # Step 4: Drop high-missing columns
    x_train, x_test = drop_high_missing_cols(x_train, x_test)

    # Step 5: Function transformer (skew check - diagnostic only)
    check_skew(x_train)

    # Step 6: Feature construction
    x_train, x_test = construct_features(x_train, x_test)

    # Step 7: Feature splitting (numerical/categorical, missing-value transformer)
    x_train, x_test, missingvalue_transformer, numerical_features1, low_card_cols, high_card_cols = split_features(x_train, x_test)

    # Step 8: Encoding + feature selection
    encoding_transformer = build_encoding_transformer(low_card_cols, high_card_cols)
    feature_selector = build_feature_selector()

    # Step 9: Complete pipeline (RandomForest)
    fraud_pipeline = build_fraud_pipeline(missingvalue_transformer, encoding_transformer, feature_selector)

    # Step 10: Cross-validation strategy
    cv_strategy = build_cv_strategy()

    # Step 11: Grid search (RandomForest)
    grid_search = build_grid_search(fraud_pipeline, cv_strategy)

    # Step 12: Train/val split for threshold tuning
    x_train_model, x_val, y_train_model, y_val = train_val_split_data(x_train, y_train)

    # Fit RandomForest grid search
    grid_search.fit(x_train_model, y_train_model)

    # Step 13: Best hyperparameters
    best_model = get_best_model(grid_search)

    # Step 14: Threshold selection (RandomForest)
    best_threshold, best_val_f1 = select_best_threshold(best_model, x_val, y_val)

    # Step 15: Final evaluation (RandomForest)
    rf_results = evaluate_model(best_model, best_threshold, x_test, y_test)

    # ================= SECOND MODEL: XGBoost =================

    xgb_pipeline = build_xgb_pipeline(missingvalue_transformer, encoding_transformer, feature_selector, y_train_model)
    grid_search_xgb = build_xgb_grid_search(xgb_pipeline, cv_strategy)
    grid_search_xgb.fit(x_train_model, y_train_model)

    best_model_xgb = get_best_model(grid_search_xgb)
    best_threshold_xgb, best_val_f1_xgb = select_best_threshold(best_model_xgb, x_val, y_val)
    xgb_results = evaluate_model(best_model_xgb, best_threshold_xgb, x_test, y_test)

    # ================= MODEL COMPARISON =================

    compare_models(
        rf_results["pr_auc"], rf_results["roc_auc"],
        xgb_results["pr_auc"], xgb_results["roc_auc"]
    )

    # ================= SAVE BEST MODEL =================

    # Save the best model (XGBoost) for later use in the API
    save_model(best_model_xgb, best_threshold_xgb)


if __name__ == "__main__":
    main()