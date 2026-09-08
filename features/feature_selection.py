from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def build_feature_selector():
    # ==============================================================================
    # STEP: FEATURE SELECTION
    # ==============================================================================

    # WHAT:
    # We are going to select a subset of the encoded features using a
    # tree-based feature-importance filter (SelectFromModel wrapped around a
    # RandomForestClassifier).
    #
    # WHY:
    # After missing-value imputation and encoding, our feature space is very
    # wide - hundreds of numerical columns (TransactionAmt, D*, C*, V*, id_*,
    # engineered uid features, etc.) plus one-hot expanded categorical columns.
    # Many of these columns are redundant, near-constant, or simply not useful
    # for separating fraud from non-fraud. Feature selection:
    #   - reduces dimensionality, which reduces training time and memory use
    #     (important given ~590k rows),
    #   - can reduce overfitting caused by noisy/irrelevant columns,
    #   - keeps the features that a tree-based model actually finds useful,
    #     since importance is measured with a tree-based estimator - consistent
    #     with the tree-based models this project is built around.
    #
    # Why THIS method instead of something else:
    #   - Univariate filters (SelectKBest with chi2/mutual_info) look at one
    #     feature at a time and ignore interactions between features, which
    #     matters a lot for fraud (e.g. Amt_to_mean_ratio only makes sense
    #     combined with uid_TransactionAmt_mean).
    #   - Wrapper methods like Recursive Feature Elimination (RFE) refit the
    #     model many times and would be far too slow on ~590k rows with
    #     hundreds of features.
    #   - SelectFromModel with a RandomForest importance threshold is a good
    #     middle ground: it is fit ONCE per pipeline fit, captures feature
    #     interactions (because it's tree-based), and is fast enough to run
    #     inside GridSearchCV on this dataset size.
    #
    # WHERE IN THE PIPELINE (before or after encoding?):
    # Feature selection is applied AFTER missing-value imputation and AFTER
    # categorical encoding, because:
    #   - SelectFromModel measures importance using a numeric feature matrix,
    #     it cannot operate directly on raw strings/categories or on data that
    #     still contains NaNs.
    #   - Once encoding_transformer has produced a fully numeric, fully
    #     imputed table, feature selection can safely rank every column
    #     (original numerical columns, one-hot columns, and ordinal-encoded
    #     columns) on the same footing.
    #
    # HOW WE AVOID DATA LEAKAGE:
    # The feature selector is placed INSIDE the sklearn Pipeline (see the next
    # section), not fit separately beforehand. This means:
    #   - During cross-validation / GridSearchCV, SelectFromModel is re-fit
    #     from scratch on only the training folds of each split.
    #   - It never sees the held-out fold (or the final x_test) while deciding
    #     which features are "important".
    # If we instead fit SelectFromModel once on all of x_train up front and
    # reused the same selected columns across every CV fold, that would leak
    # information from each fold's held-out portion into the feature-selection
    # decision. Keeping it inside the Pipeline avoids this.

    # We use a small, fast RandomForest purely to RANK feature importance.
    # n_jobs is left at 1 here (rather than -1) because this estimator will be
    # refit many times inside GridSearchCV, which is already parallelized with
    # n_jobs=-1 at the GridSearchCV level - parallelizing at both levels at
    # once oversubscribes CPU cores and can actually slow things down.
    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=8,          # shallow trees -> fast to fit, good enough just to rank importance
            class_weight='balanced',  # fraud is rare, so balance the importance ranking too
            random_state=42,
            n_jobs=1
        ),
        threshold='median'  # keep the top ~50% of features by importance, drop the bottom half
    )

    return feature_selector