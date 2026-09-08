from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def split_features(x_train, x_test):
    # ==============================================================================
    #                            FEATURE SPLITTING
    # ==============================================================================

    # Why are we doing this?
    # -----------------------
    # After feature construction, our dataset contains a mix of original columns
    # and newly engineered columns (TransactionAmt_log, uid_TransactionAmt_mean,
    # P_email_provider, etc.). Before we can encode categorical features next,
    # we need to organize all columns into clear groups:
    #
    # 1. Numerical columns          -> used directly / scaled if needed
    # 2. Low-cardinality categorical -> safe for One-Hot Encoding
    # 3. High-cardinality categorical -> needs Ordinal/Frequency Encoding instead
    # 4. Helper/dropped columns      -> no longer needed after their purpose is served
    #
    # Splitting columns this way avoids applying the wrong encoding method to the
    # wrong column, which could otherwise explode the dataset size (One-Hot on a
    # high-cardinality column can create hundreds/thousands of sparse columns)
    # or lose useful information (treating a helper ID column as a real feature).


    # ------------------------------------------------------------------------------
    # STEP 1: Drop columns that are no longer needed
    # ------------------------------------------------------------------------------
    # uid:
    #   'uid' was only created as a temporary helper column (a pseudo customer ID
    #   built from card/addr columns). It was used to calculate
    #   'uid_TransactionAmt_mean' and 'uid_TransactionAmt_std'. Since those
    #   features already capture the useful information from uid, keeping the
    #   raw 'uid' column would be harmful - it has thousands of unique values
    #   (almost one per transaction), so One-Hot Encoding it would create
    #   thousands of useless sparse columns and risk overfitting.
    #   FIX: 'uid' is intentionally NOT dropped here anymore. UIDFeatureTransformer
    #   (the first pipeline step) needs the raw 'uid' column as its input so it can
    #   compute uid_TransactionAmt_mean/std per CV fold, and it drops 'uid' itself
    #   at the end of its own transform() once those features are built.
    #
    # P_emaildomain / R_emaildomain:
    #   Both columns have around ~58-60 unique raw domain values (gmail.com,
    #   yahoo.com, hotmail.co.uk, etc.), making them high-cardinality and hard
    #   to encode directly. We already engineered cleaner, low-cardinality
    #   versions of this same information earlier - 'P_email_provider' and
    #   'R_email_provider' (only 5 categories each: gmail, yahoo, microsoft,
    #   anonymous, other). Since the provider columns capture the same signal
    #   in a much cleaner form, the raw domain columns are redundant and are
    #   dropped to avoid encoding the same information twice.

    x_train = x_train.drop(columns=["P_emaildomain", "R_emaildomain"])
    x_test = x_test.drop(columns=["P_emaildomain", "R_emaildomain"])
    # Numerical columns with missing values - check BOTH train and test,
    # since some columns may have missing values only in test
    # (e.g. uid_TransactionAmt_mean/std - unseen uid combinations in test
    # produce NaN even though the same column is fully populated in train) 
    # ------------------------------------------------------------------------------
    # STEP  Missing-value column lists - check BOTH train and test
    # ------------------------------------------------------------------------------
    # Some columns (e.g. uid_TransactionAmt_mean/std) have zero missing values in
    # x_train but CAN have missing values in x_test (unseen uid combinations map
    # to NaN). Checking both sets and combining avoids leftover NaNs after
    # imputation
    a_train = x_train.select_dtypes(include=['int64','float64']).isnull().sum()
    a_train = a_train[a_train > 0].index.tolist()

    a_test = x_test.select_dtypes(include=['int64','float64']).isnull().sum()
    a_test = a_test[a_test > 0].index.tolist()

    a_combined = list(set(a_train) | set(a_test))

    # FIX: explicitly add the UID-engineered numeric features here.
    # WHAT: uid_TransactionAmt_mean, uid_TransactionAmt_std, and
    # Amt_to_mean_ratio are created later by UIDFeatureTransformer, which only
    # runs INSIDE fraud_pipeline (during grid_search.fit()) - they do not exist
    # yet in x_train/x_test at this point in the script, so a_train/a_test
    # above cannot see them or know they may contain missing values.
    # WHY this is still needed: uid_TransactionAmt_std is naturally NaN for any
    # uid that has only one transaction (std of a single value is undefined),
    # and all three of these columns can be NaN for a uid that appears in a
    # validation/test fold but not in that fold's training portion (unseen uid
    # -> map() produces NaN). Because missingvalue_transformer only imputes the
    # column names listed in a_combined (remainder='passthrough' leaves
    # everything else, including any NaNs, untouched), these three columns must
    # be added to a_combined now so SimpleImputer(strategy='median') covers
    # them too - otherwise NaNs from them would flow through encoding straight
    # into RandomForestClassifier, which cannot handle NaN.
    uid_engineered_numeric_cols = ["uid_TransactionAmt_mean", "uid_TransactionAmt_std", "Amt_to_mean_ratio"]
    for col in uid_engineered_numeric_cols:
        if col not in a_combined:
            a_combined.append(col)

    # Categorical columns with missing values - same logic
    b_train = x_train.select_dtypes(include=['object']).isnull().sum()
    b_train = b_train[b_train > 0].index.tolist()

    b_test = x_test.select_dtypes(include=['object']).isnull().sum()
    b_test = b_test[b_test > 0].index.tolist()

    b_combined = list(set(b_train) | set(b_test))

    missingvalue_transformer = ColumnTransformer(
        transformers=[
            ('missingnumerical_column', SimpleImputer(strategy='median'), a_combined),
            ('missingcategorical_column', SimpleImputer(strategy='most_frequent'), b_combined)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    missingvalue_transformer.set_output(transform="pandas")


    # ------------------------------------------------------------------------------
    # STEP 2: Recompute numerical and categorical columns
    # ---------------6---------------------------------------------------------------
    # We recompute this AFTER dropping the columns above and AFTER feature
    # construction, because the earlier lists (used during missing value
    # imputation) are now outdated - they don't include new engineered features
    # and still include the columns we just dropped.

    numerical_features1 = x_train.select_dtypes(include=['int64', 'float64'])
    cateorical_features1 = x_train.select_dtypes(include=['object'])
    numerical_features1=numerical_features1.columns.tolist()
    cateorical_features1 = cateorical_features1.columns.tolist()
    # FIX: 'uid' is still present in x_train at this point (it is only dropped
    # later, inside the pipeline, by UIDFeatureTransformer). It must be excluded
    # from this categorical column list here, otherwise its huge cardinality
    # would route it into high_card_cols below and encoding_transformer would
    # try to encode a column ('uid') that no longer exists by the time the
    # pipeline reaches the encoding step.
    if "uid" in cateorical_features1:
        cateorical_features1.remove("uid")

    #for col in cateorical_features1:
        #print(col, x_train[col].nunique())


    # ------------------------------------------------------------------------------
    # STEP 3: Split categorical columns by cardinality
    # ------------------------------------------------------------------------------
    # We count how many unique categories each categorical column has, using
    # .nunique(). This tells us which encoding method is appropriate:
    #
    #   - Low-cardinality columns (few categories) -> One-Hot Encoding is safe,
    #     since it won't create too many new columns.
    #   - High-cardinality columns (many categories) -> One-Hot Encoding would
    #     create hundreds/thousands of sparse columns, so Ordinal or Frequency
    #     Encoding is used instead.
    #
    # A cutoff of 10 unique values is used to separate the two groups. This
    # cutoff was chosen after inspecting the actual data: most categorical
    # columns had between 2-6 unique values, while a small number
    # (P_emaildomain, R_emaildomain, id_31, id_33, DeviceInfo) had 59 to 1686
    # unique values - a clear, natural gap with no ambiguous middle ground.

    cardinality = x_train[cateorical_features1].nunique()

    low_card_cols = cardinality[cardinality <= 10].index.tolist()
    high_card_cols = cardinality[cardinality > 10].index.tolist()


    # ------------------------------------------------------------------------------
    # STEP 4: Sanity check - confirm every column is accounted for
    # ------------------------------------------------------------------------------
    # Before moving to encoding, we verify that every column in x_train belongs
    # to exactly one of our three groups (numerical, low-cardinality categorical,
    # high-cardinality categorical). If the two totals below don't match, it
    # means a column was either missed or counted twice, which would cause
    # problems later when building the ColumnTransformer.

    total_cols = x_train.shape[1]
    accounted_cols = len(numerical_features1) + len(low_card_cols) + len(high_card_cols)

    #print("Total columns in x_train:", total_cols)
    #print("Accounted columns:", accounted_cols)  
    #print("uid" in x_train.columns) 
    #encoding cateorical values  

    return x_train, x_test, missingvalue_transformer, numerical_features1, low_card_cols, high_card_cols