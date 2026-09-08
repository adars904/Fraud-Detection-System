import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#====================================================================================  
#                FEATURE CONSTRUCTION   
#=====================================================================================  
# ==============================================================================
# FEATURE 1: TransactionAmt_log
# ==============================================================================
# What are we doing?
# ------------------
# We are creating a new feature called 'TransactionAmt_log' from the original
# 'TransactionAmt' column.
#
# Why are we doing this?
# ----------------------
# During EDA, we observed that TransactionAmt is right-skewed.
# Large transaction values are much bigger than normal transactions, which can
# make learning difficult for some machine learning models.
#
# Applying a log transformation:
# - Reduces right skewness.
# - Compresses very large values.
# - Preserves the ordering of transactions.
# - Can improve model performance, especially for linear models.
#
# Why use np.log1p() instead of np.log()?
# ---------------------------------------
# np.log(0) is undefined and returns -inf.
# np.log1p(x) computes log(1 + x), so if TransactionAmt is 0:
#
# np.log1p(0) = log(1) = 0
#
# Therefore, log1p() is safer when zero values may exist.
#
# Data Leakage?
# -------------
# No.
# This transformation uses only the current row's TransactionAmt.
# No information from other rows, the training set, or the test set is used.
#
# Original Feature:
# -----------------
# TransactionAmt
#
# New Feature:
# ------------
# TransactionAmt_log
# ==============================================================================

def transaction_amt(df):
    df = df.copy()

    # Create the log-transformed transaction amount
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    return df


# ==============================================================================
# FEATURE 2 : Extract Decimal Part of Transaction Amount
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# Every transaction amount consists of two parts:
#
# Example:
#
# TransactionAmt = 125.75
#
# Integer Part = 125
# Decimal Part = 75
#
# Instead of ignoring the decimal part, we create it as a separate feature.

# Why can this help?
# ------------------
# In fraud detection, transaction amounts sometimes follow unusual decimal
# patterns.
#
# For example:
#
# 100.00
# 999.99
# 49.99
# 250.50
#
# Fraudulent transactions may contain repeated or uncommon decimal values.
#
# By extracting the decimal part, the model can learn whether certain decimal
# patterns are associated with fraudulent behaviour.

# How is it calculated?
# ---------------------
# Step 1:
# Remove the integer part.
#
# 125.75 -> 0.75
#
# Step 2:
# Multiply by 100.
#
# 0.75 -> 75
#
# This creates a new feature representing only the decimal portion.

# Data Leakage?
# -------------
# No.
#
# The decimal part is calculated only from the current transaction amount.
# No information from other transactions is used.

# New Feature Created
# -------------------
# TransactionAmt_decimal
def TransactionAmt_decimal(df4):
    df4=df4.copy()
    df4["TransactionAmt_decimal"]=((df4["TransactionAmt"]-df4["TransactionAmt"].astype(int))*100).round(2) 
    return df4 


# ==============================================================================
# FEATURE 3 : Create Pseudo User ID (uid)
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# The IEEE-CIS dataset does not contain a real Customer ID or User ID.
#
# Because of this, we cannot directly identify which transactions belong
# to the same customer.

# Solution
# --------
# We create our own identifier called 'uid' by combining several columns
# that together approximately represent a unique customer.
#
# Columns used:
# - card1
# - card2
# - card3
# - card5
# - addr1
# - addr2
#
# Example:
#
# card1 = 1234
# card2 = 150
# card3 = 185
# card5 = 226
# addr1 = 315
# addr2 = 87
#
# uid = "1234_150_185_226_315_87"

# Why is this useful?
# -------------------
# Although uid is not a real customer ID, transactions with the same uid
# are likely to belong to the same user.
#
# This allows us to create powerful user-level features later, such as:
#
# - Average transaction amount per user
# - Number of transactions per user
# - Maximum transaction amount
# - Minimum transaction amount
# - Standard deviation of spending
#
# These features often improve fraud detection performance.

# Is uid a real Customer ID?
# --------------------------
# No.
#
# uid is only a pseudo (approximate) identifier.
#
# Two different users could share the same uid, and the same user might have
# different uids if their card or address information changes.

# Data Leakage?
# -------------
# No.
#
# The uid is created only by combining values from the current row.
# No information from other transactions is used. 
# ==============================================================================
# NOTE: Missing values in uid source columns (card2, card3, card5, addr1, addr2)
# ==============================================================================
# card2, card3, card5, addr1, addr2 contain missing values in this dataset.
#
# When these columns are converted using .astype(str) inside the uid()
# function, any NaN becomes the literal string "nan" instead of throwing
# an error.
#
# Example:
# card1 = 1234, card2 = NaN, card3 = 185, card5 = 226, addr1 = NaN, addr2 = 87
# uid = "1234_nan_185_226_nan_87"
#
# This does NOT cause a crash - the code runs fine.
#
# However, it means multiple different transactions with missing
# card2/card3/card5/addr1/addr2 can end up sharing the same "nan"-containing
# uid, even if they actually belong to different real users.
#
# This is not an error, but it is a data quality tradeoff to be aware of:
# uid-based features (uid_TransactionAmt_mean, uid_TransactionAmt_std,
# Amt_to_mean_ratio) may be slightly less accurate for these rows, since
# they are grouped with other unrelated users who also had missing values
# in the same columns.

# New Feature Created
# -------------------
# uid   
def uid(df5):
    df5=df5.copy()
    df5["uid"]=(df5["card1"].astype(str)+'_'+
              df5["card2"].astype(str)+'_'+
              df5["card3"].astype(str)+'_'+
              df5["card5"].astype(str)+'_'+
              df5["addr1"].astype(str)+'_'+
              df5["addr2"].astype(str)
               ) 
    return df5 


def construct_features(x_train, x_test):
    # Apply the same transformation to both training and testing datasets
    x_train = transaction_amt(x_train)
    x_test = transaction_amt(x_test)

    x_train=TransactionAmt_decimal(x_train)
    x_test=TransactionAmt_decimal(x_test) 

    x_train=uid(x_train)
    x_test=uid(x_test)   

    return x_train, x_test


# ==============================================================================
# FEATURE 4 : Average Transaction Amount per User (uid)
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# Every user has a different spending pattern.
#
# Example:
#
# User A usually spends between $20 and $50.
# User B usually spends between $500 and $1000.
#
# If both users make a $600 transaction:
#
# - For User A, it is unusual.
# - For User B, it is normal.
#
# Looking only at TransactionAmt does not provide enough context.
# Therefore, we calculate each user's average transaction amount.

# What is groupby()?
# ------------------
# groupby() groups together all rows that have the same value.
#
# Example:
#
# uid   TransactionAmt
# A     100
# A     200
# A     300
# B      50
# B      70
#
# After groupby("uid"):
#
# Group A
# --------
# 100
# 200
# 300
#
# Group B
# --------
# 50
# 70
#
# groupby() only creates groups.
# It DOES NOT calculate anything by itself.

# What happens next?
# ------------------
# After creating the groups, we perform an aggregation.
#
# In this feature, we calculate the mean (average)
# TransactionAmt for every uid.
#
# Mean of A = (100 + 200 + 300) / 3 = 200
# Mean of B = (50 + 70) / 2 = 60

# New Feature
# -----------
# uid_TransactionAmt_mean
#
# Every transaction of the same uid receives the same average value.
#
# Example:
#
# uid   TransactionAmt   uid_TransactionAmt_mean
# A     100              200
# A     200              200
# A     300              200
# B      50               60
# B      70               60

# Why is this useful?
# -------------------
# The model can compare a user's current transaction with
# their normal spending behaviour.
#
# This helps identify transactions that are unusually
# large or unusually small for that particular user.

# Data Leakage?
# -------------
# Yes, data leakage can occur if we calculate the averages
# using both x_train and x_test together.
#
# Correct approach:
# 1. Calculate the average only from x_train.
# 2. Apply (map) those averages to both x_train and x_test.
#
# This ensures the model never uses information from the
# test set during training.

# Steps
# -----
# Step 1 : Group transactions by uid.
# Step 2 : Calculate the mean TransactionAmt for each uid.
# Step 3 : Store the result in uid_amt_mean.
# Step 4 : Map the mean back to every transaction. 

# Step 1: Calculate the average TransactionAmt for each uid
class UIDFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        # Calculate UID statistics only from the training data
        # of the current CV fold.
        self.uid_mean_ = X.groupby("uid")["TransactionAmt"].mean()
        self.uid_std_ = X.groupby("uid")["TransactionAmt"].std()

        return self

    def transform(self, X):

        X = X.copy()

        # Apply statistics learned during fit().
        X["uid_TransactionAmt_mean"] = X["uid"].map(self.uid_mean_)
        X["uid_TransactionAmt_std"] = X["uid"].map(self.uid_std_)

        # Calculate transaction amount relative to the UID's
        # average transaction amount.
        mean_amt = X["uid_TransactionAmt_mean"]

        X["Amt_to_mean_ratio"] = np.where(
            mean_amt > 0,
            X["TransactionAmt"] / mean_amt,
            np.nan
        )

        # FIX: drop the raw 'uid' column here (inside the pipeline step) now
        # that it has already been used to build the mean/std/ratio features.
        # 'uid' has thousands of unique values, so leaving it in would make
        # downstream steps (encoding_transformer) treat it as a huge
        # high-cardinality categorical column.
        X = X.drop(columns=["uid"])

        return X
# ==============================================================================
# FEATURE 5 : Standard Deviation of Transaction Amount per User (uid)
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# Different users have different spending behaviours.
#
# Some users spend almost the same amount every time,
# while others spend a wide range of amounts.
#
# The average (mean) tells us the typical transaction amount,
# but it does not tell us how much the spending varies.

# What is Standard Deviation (std)?
# ---------------------------------
# Standard deviation measures the spread (variation) of values
# around the mean.
#
# Low Standard Deviation
# ----------------------
# Transactions:
# 95
# 100
# 105
#
# Mean = 100
# Standard Deviation = Low
#
# This user spends nearly the same amount each time.
#
# High Standard Deviation
# -----------------------
# Transactions:
# 10
# 100
# 190
#
# Mean = 100
# Standard Deviation = High
#
# This user has very different transaction amounts.

# What is groupby() doing?
# ------------------------
# groupby("uid") groups together all transactions that
# belong to the same pseudo user.
#
# Example:
#
# UID A
# -----
# 100
# 120
# 110
#
# UID B
# -----
# 50
# 500
# 900
#
# After grouping, we calculate the standard deviation
# for each uid.

# New Feature
# -----------
# uid_TransactionAmt_std
#
# Every transaction of the same uid receives the same
# standard deviation value.

# Why is this useful?
# -------------------
# The model learns how consistent a user's spending is.
#
# - Low std  -> User usually spends similar amounts.
# - High std -> User's spending varies a lot.
#
# This helps the model understand whether a new transaction
# fits the user's normal spending behaviour.

# Data Leakage?
# -------------
# Yes, data leakage can occur if we calculate the standard
# deviation using both x_train and x_test together.
#
# Correct approach:
# 1. Calculate std only from x_train.
# 2. Map the calculated values to both x_train and x_test.
#
# This prevents information from the test set from
# influencing the training process.

# Steps
# -----
# Step 1 : Group transactions by uid.
# Step 2 : Calculate the standard deviation of TransactionAmt.
# Step 3 : Store the result in uid_amt_std.
# Step 4 : Map the standard deviation back to every transaction.   
# Step 1: Calculate the standard deviation of TransactionAmt for each uid
# Step 2: Map the standard deviation back to every row in x_train
# Step 3: Use the same mapping for x_test
# NOTE: The three comment lines directly above describe the old manual
# groupby/map code that used to sit here. That code has been removed;
# UIDFeatureTransformer.fit()/transform() above now performs the same
# calculation, so it is recomputed separately per CV fold during
# GridSearchCV instead of being learned once on all of x_train.
# ==============================================================================
# FEATURE 6 : Transaction Amount to User Mean Ratio
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# Different users have different spending habits.
#
# Looking only at TransactionAmt is not enough.
#
# Example:
#
# User A usually spends $50.
# A new transaction of $500 is unusual.
#
# User B usually spends $800.
# A transaction of $500 is normal.
#
# Therefore, we compare the current transaction amount
# with the user's average transaction amount.

# Formula
# -------
# Amt_to_mean_ratio =
#
#          Current Transaction Amount
# -----------------------------------------------
#      User's Average Transaction Amount
#
# This tells us how different the current transaction
# is from the user's normal spending behaviour.

# Example
# -------
#
# TransactionAmt = 200
# uid_TransactionAmt_mean = 100
#
# Amt_to_mean_ratio = 200 / 100 = 2.0
#
# The user spent 2 times their normal amount.
#
# --------------------------------------------
#
# TransactionAmt = 50
# uid_TransactionAmt_mean = 100
#
# Amt_to_mean_ratio = 50 / 100 = 0.5
#
# The user spent only half of their normal amount.

# New Feature
# -----------
# Amt_to_mean_ratio
#
# A value:
#
# Around 1.0  -> Normal transaction
# Greater than 1 -> Higher than usual
# Less than 1 -> Lower than usual

# Why is this useful?
# -------------------
# Fraudulent transactions are often much larger or much
# smaller than a user's normal spending pattern.
#
# This feature helps the model identify unusual transactions
# relative to each user's historical behaviour.

# Division by Zero
# ----------------
# If uid_TransactionAmt_mean is 0, division by zero can occur.
#
# A common solution is:
#
# TransactionAmt / (uid_TransactionAmt_mean + 1)
#
# which avoids division-by-zero errors.
#
# If uid_TransactionAmt_mean is missing (NaN),
# the ratio will also become NaN and should be
# handled during preprocessing.

# Steps
# -----
# Step 1 : Take the current TransactionAmt.
# Step 2 : Divide it by uid_TransactionAmt_mean.
# Step 3 : Store the result in Amt_to_mean_ratio.
# Step 4 : Use this feature to compare each transaction
#          with the user's normal spending behaviour.
# NOTE: This used to be calculated manually here with a standalone
# Amt_to_mean_ratio() function applied to the full x_train/x_test (see the
# removed function and calls that lived below this comment). It is now
# calculated instead by UIDFeatureTransformer.transform() above, using only
# the uid_TransactionAmt_mean learned from the current CV fold's training
# data, so it no longer leaks test-fold information.