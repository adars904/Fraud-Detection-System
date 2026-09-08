from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from features.feature_construction import UIDFeatureTransformer


def build_fraud_pipeline(missingvalue_transformer, encoding_transformer, feature_selector):
    # ==============================================================================
    # STEP: COMPLETE THE PREPROCESSING - WHY NO SCALING IS ADDED
    # ==============================================================================
    # We already have missingvalue_transformer and encoding_transformer built
    # above, and they are reused as-is (not recreated) below.
    #
    # We are intentionally NOT adding a scaling step (e.g. StandardScaler or
    # MinMaxScaler) here:
    #   - The models under consideration for this project (Random Forest /
    #     XGBoost / LightGBM / CatBoost) are tree-based. Tree-based models
    #     split on raw feature thresholds (e.g. "TransactionAmt > 120"), so the
    #     absolute scale/units of a feature does not affect how the tree splits
    #     or how importance is measured.
    #   - Scaling only matters for models that rely on distance or gradient
    #     magnitude (e.g. logistic regression, KNN, neural networks, SVM).
    #   - Adding scaling here would only add extra computation with no benefit
    #     to the final model, so it is left out on purpose.


    # ==============================================================================
    # STEP: BUILD THE COMPLETE PIPELINE
    # ==============================================================================

    # WHAT:
    # We combine missingvalue_transformer -> encoding_transformer ->
    # feature_selector -> classifier into a single sklearn Pipeline.
    #
    # WHY put everything in one Pipeline object:
    #   - A Pipeline treats every step as one unit. When we call
    #     pipeline.fit(x_train, y_train), each step's .fit_transform() is
    #     called only on the data it receives; when we call
    #     pipeline.predict(x_test), only .transform() is called on each step
    #     (never .fit()). This is exactly what prevents preprocessing leakage:
    #     imputation medians/most-frequent values, encoding categories, and
    #     feature-importance rankings are all learned ONLY from whatever data
    #     is passed to .fit() - which will always be a training fold, never
    #     x_test and never a held-out CV fold.
    #   - When this Pipeline is handed to GridSearchCV, GridSearchCV
    #     automatically re-fits the ENTIRE pipeline (imputation, encoding,
    #     feature selection, and the model) separately on each training fold,
    #     for every hyperparameter combination. This guarantees no information
    #     from a validation fold ever leaks into preprocessing decisions for
    #     that same fold.
    #
    # ORDER OF TRANSFORMATIONS (matches the diagram you described):
    #   Raw training data
    #        -> missing value handling (missingvalue_transformer)
    #        -> categorical encoding (encoding_transformer)
    #        -> feature selection (feature_selector)
    #        -> machine learning model (classifier)

    fraud_pipeline = Pipeline(steps=[

        # Create UID-based statistical features.
        # Because this is inside the Pipeline, the UID statistics
        # are learned separately inside each CV training fold.
        ('uid_features', UIDFeatureTransformer()),

        ('missing_values', missingvalue_transformer),

        ('encoding', encoding_transformer),

        ('feature_selection', feature_selector),

        ('classifier', RandomForestClassifier(

            random_state=42,

            class_weight='balanced',

            n_jobs=1
        ))
    ])

    return fraud_pipeline