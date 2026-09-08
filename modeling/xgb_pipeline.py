from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier
from features.feature_construction import UIDFeatureTransformer


def build_xgb_pipeline(missingvalue_transformer, encoding_transformer, feature_selector, y_train_model):
    # ==============================================================================
    #                   SECOND MODEL: XGBoost (RandomForest is unchanged above)
    # ==============================================================================
     
    # WHAT:
    # We train a second, independent pipeline using XGBoost instead of
    # RandomForest, so we can compare the two models on identical
    # train/validation/test splits and pick whichever one actually performs
    # better - without touching or overwriting the RandomForest pipeline,
    # best_model, best_threshold, or any of its results above.
    #
    # WHY XGBoost is worth trying:
    # Gradient-boosted trees build each new tree to correct the errors of the
    # previous ones, rather than averaging many independent trees like
    # RandomForest does. On rare-event tabular problems like fraud detection,
    # this sequential error-correction often produces noticeably better
    # PR-AUC/recall at a given precision level than RandomForest.
    #
    # Requires: pip install xgboost
     
    # ------------------------------------------------------------------------------
    # STEP: Build fresh (unfitted) copies of the shared preprocessing steps
    # ------------------------------------------------------------------------------
    # WHAT: missingvalue_transformer, encoding_transformer, and feature_selector
    # were already fit once inside the RandomForest pipeline above (as part of
    # grid_search.fit()). Reusing those same fitted objects directly inside a
    # second Pipeline would be unsafe - fitting the new XGBoost pipeline would
    # silently overwrite their learned state (fitted medians, encoder
    # categories, selected features), which could corrupt best_model from the
    # RandomForest run if it were ever re-used afterward (e.g. for further
    # analysis) in the same script/session.
    # WHY clone(): sklearn's clone() creates a new, unfitted estimator with the
    # same hyperparameters/configuration - not a fitted copy - so the XGBoost
    # pipeline gets its own independent preproce
    # ssing state, learned freshly
    # inside its own GridSearchCV folds, exactly like the RandomForest pipeline
    # did. UIDFeatureTransformer is likewise instantiated fresh for the same
    # reason (it stores fitted uid_mean_/uid_std_ after fit()).
    missingvalue_transformer_xgb = clone(missingvalue_transformer)
    encoding_transformer_xgb = clone(encoding_transformer)
    feature_selector_xgb = clone(feature_selector)
     
    # ------------------------------------------------------------------------------
    # STEP: Handle class imbalance the XGBoost way
    # ------------------------------------------------------------------------------
    # WHAT: XGBoost has no class_weight='balanced' option like RandomForest.
    # Its equivalent is scale_pos_weight, a single number applied to the
    # positive (fraud) class's gradient during training.
    # WHY computed from y_train_model (not y_train or the full dataset): this
    # keeps the imbalance-handling consistent with "learn only from the
    # training portion" - the same principle already used for UID statistics,
    # imputation, and encoding elsewhere in this script.
    scale_pos_weight_value = (y_train_model == 0).sum() / (y_train_model == 1).sum()
     
    xgb_pipeline = Pipeline(steps=[
     
        # Same UID feature engineering as the RandomForest pipeline, but a
        # fresh instance so its fitted uid_mean_/uid_std_ don't collide with
        # the RandomForest pipeline's own UIDFeatureTransformer instance.
        ('uid_features', UIDFeatureTransformer()),
     
        ('missing_values', missingvalue_transformer_xgb),
     
        ('encoding', encoding_transformer_xgb),
     
        ('feature_selection', feature_selector_xgb),
     
        ('classifier', XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight_value,
            eval_metric='logloss',   # avoids an XGBoost warning; does not change training behaviour
            tree_method='hist',      # much faster on ~470k rows; does not change what is learned, only training speed
            n_jobs=1
        ))
    ])

    return xgb_pipeline