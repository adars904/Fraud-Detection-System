from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_encoding_transformer(low_card_cols, high_card_cols):
    #==============================================================================
    #                      ENCODING CATEGORICAL FEATURES
    # ============================================================================== 
    encoding_transformer=ColumnTransformer(
        transformers=[
            ('low_cardinality', OneHotEncoder(sparse_output=False,handle_unknown='ignore'), low_card_cols),
            ('high_cardinality', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_card_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ) 
    encoding_transformer.set_output(transform="pandas") 

    # ==============================================================================
    # LEAKAGE AUDIT NOTE (per your request - nothing below is changed automatically)
    # ==============================================================================
    # IMPORTANT:
    # Looking at the existing "Missing-value column lists" step above
    # (a_train / a_test / b_train / b_test / a_combined / b_combined), there is
    # a subtle point worth understanding, even though it is NOT the same kind of
    # leakage as computing statistics on x_test.
    #
    # What is happening:
    #   b_test / a_test are built by looking at x_test to see WHICH COLUMN NAMES
    #   contain missing values, and that list is unioned with the train-side list
    #   to decide which columns missingvalue_transformer should impute.
    #
    # Why this is NOT the dangerous kind of leakage:
    #   - No numeric VALUE from x_test (no mean/median/most_frequent statistic)
    #     is used. The actual imputation values are still learned only from
    #     x_train, because missingvalue_transformer.fit() will only ever be
    #     called on x_train (inside the pipeline, during GridSearchCV).
    #   - a_combined/b_combined only decide the STRUCTURE of the transformer
    #     (which column names get an imputer strategy attached), not the
    #     numbers used to fill missing values.
    #
    # Why it is still worth flagging:
    #   - Strictly speaking, touching x_test at all before final evaluation is a
    #     mild deviation from "the test set is only for final evaluation" (see
    #     your Section 8 rule). In a real production setting, a genuinely unseen
    #     column's missingness pattern would not be known in advance.
    #   - In practice here, the risk is very low because IEEE-CIS columns are
    #     fixed/known in advance and this only affects which columns get an
    #     imputer, not any learned numeric value.
    #
    # I am NOT changing this automatically, because you asked me to preserve the
    # original code and only flag potential issues. If you want, the safer
    # alternative is to build a_combined/b_combined from x_train alone, and rely
    # on missingvalue_transformer's remainder='passthrough' + SimpleImputer's
    # ability to also be applied defensively to any column that could contain
    # missing values in production (or add a small "impute-if-present" fallback).
    # Let me know if you'd like me to change this - I will not do it silently.
    # ==============================================================================

    return encoding_transformer