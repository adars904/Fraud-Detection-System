from sklearn.model_selection import StratifiedKFold


def build_cv_strategy():
    # ==============================================================================
    # STEP: CROSS-VALIDATION STRATEGY
    # ==============================================================================

    # WHAT:
    # We use StratifiedKFold instead of plain KFold.
    #
    # WHY:
    # This dataset is highly imbalanced (fraud is a small minority of
    # transactions). Plain KFold splits rows randomly and could, by chance,
    # create folds with very few (or zero) fraud examples, making the score for
    # that fold unreliable. StratifiedKFold preserves the same fraud/non-fraud
    # ratio in every fold, so every fold is a fair, representative sample of the
    # full training set.
    #
    # We use 3 folds (not 5 or 10) specifically to keep computation time
    # reasonable given ~590k total rows (i.e. ~470k rows in x_train) and the
    # fact that each fold fit trains BOTH the feature-selection RandomForest
    # and the final classifier RandomForest.

    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    return cv_strategy