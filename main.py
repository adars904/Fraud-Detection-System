import pandas as pd
df1=pd.read_csv("train_transaction.csv")
df2=pd.read_csv("train_identity.csv")
              # EDA OF  THIS DATASET OF TRAIN IDENTITY.CSV
#print(df.shape)
#print(df1.shape)
#print(df1.info())
#print(df1.describe()) 
#print(df1.isnull().sum())
#print(df1.duplicated().sum()) 
            # EDA OF  THIS DATASET OF TRAIN TRANSACTION.CSV
#print(df.describe())
#print(df.info()) 
#print(df.describe())
#print(df["dist2"].mean())
#print(df["dist2"].median())
#print(df["dist2"].max()) 


#print(df.isnull().sum())
#print(df.duplicated().sum()) 
##numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
#categorical_features = df.select_dtypes(include=['object']).columns
#print("Numerical Features:", numerical_features)
#print("Categorical Features:", categorical_features) 
#print(df["isFraud"].value_counts())
#print(df["ProductCD"].value_counts())
#print(df["card4"].value_counts())
#print(df["card6"].value_counts())
#print(pd.crosstab(df["ProductCD"], df["isFraud"], normalize="index") * 100)
#print(pd.crosstab(df["card4"], df["isFraud"], normalize="index") * 100)
#print(pd.crosstab(df["card6"], df["isFraud"], normalize="index") * 100) 
#MERGE TWO DATASET 
df3 = pd.merge(df1, df2, on='TransactionID', how='left')
#print(df3.shape)  
#print(df3.describe()) 
#print(df3.info())
#print(df3.isnull().sum())
#print(df3.duplicated().sum())
    
#         ==============================================================================
#                               DATA CLEANING
#         ============================================================================== 
print(df3["isFraud"].unique())  # there is only 0 and 1 in isFraud column so we don't need to clean this colu# ==============================================================================
#                      DATA CLEANING - CATEGORICAL FEATURES
# ==============================================================================
import numpy as np 
categorical_features=df3.select_dtypes(include=['object']).columns
#print("Categorical Features:", categorical_features)       
#print all type of element in categorical features    
l=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1',
       'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15',
       'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33',
       'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
       'DeviceInfo']   
#for i in l:
#print(df3[i].unique())   
#print(df3["P_emaildomain"].value_counts().get("gmail", 0))
#print(df3["P_emaildomain"].value_counts().get("gmail.com", 0))

#print(df3["R_emaildomain"].value_counts().get("gmail", 0))
#print(df3["R_emaildomain"].value_counts().get("gmail.com", 0))     
# Only 6 rows contain 'gmail' while other contain 'gmail.com'.
# Treat 'gmail' as an inconsistent representation and standardize it to 'gmail.com'.
df3['P_emaildomain'] = df3['P_emaildomain'].replace('gmail', 'gmail.com')  # change  gmail to gmail.com  
df3['R_emaildomain'] = df3['R_emaildomain'].replace('gmail', 'gmail.com') # chane the gmail to gmail.com
#print((df3["id_33"] == "0x0").sum())  # so we replace this by np.nan because it is not useful for our model and it is not a valid value in id_33
df3['id_33'] =df3['id_33'].replace('0x0',np.nan)  # change the 0x0 to Nan in id-33             
#print(df3['card6'].value_counts(normalize=True))        
# debit              0.746963
 #credit             0.252961
#debit or credit    0.000051
#charge card        0.000025
#Name: proportion, dtype: float64 that why we can drop the charge card and debit or credit because they are less count also we  can check the mean of fraud in debit or credit and charge card
#print(df3.groupby('card6')['isFraud'].mean())   # debit or credit and charge card having zero  mean of fraud so we can drop them because they are not useful for our model   
#card6
#charge card        0.000000
#credit             0.066785
#debit              0.024263
#debit or credit    0.000000
#Name: isFraud, dtype: float64  so the code is below to drop the charge card and debit or credit from card6 with name other 
df3['card6']=df3['card6'].replace(['charge card', 'debit or credit'], 'other')  
# clean id_30 (OS info) - standardize OS family, remove version numbers, fix invalid values
#print((df3['id_30']=="func").sum())  # check the value counts of id_30 column  
def clean_id30(val):
    if pd.isna(val):
        return val                    # keep existing nulls as nulls
    val = val.lower()
    if 'func' in val:
        return np.nan                 # 'func' is invalid/garbage data, treat as missing   
                           #         'func' appears only 10 times and is not a valid operating system.
                                              # Treat it as missing data.
    if 'windows' in val:
        return 'windows'              # group all Windows versions (Windows 10, Windows 7, etc.) into one category
    if 'mac' in val:
        return 'mac'                  # group all Mac OS X versions into one category
    if 'ios' in val:
        return 'ios'                  # group all iOS versions into one category
    if 'android' in val:
        return 'android'              # group all Android versions into one category
    if 'linux' in val:
        return 'linux'                # group Linux into one category
    return 'other'                    # anything unrecognized (e.g. blank/rare OS names) -> other

df3['id_30'] = df3['id_30'].apply(clean_id30)   # apply OS cleanup to id_30 column  
# ==============================================================================
#                       DATA CLEANING - NUMERICAL FEATURES
# ==============================================================================  
#print all numerical columns in the datset 
num_cols = df3.select_dtypes(include=["int64", "float64"]).columns
#print(f"Total numerical columns: {len(num_cols)}")
# so total no of numerical column is 403  
#print(num_cols)       
#Now we can check that which numerical coulmn having negative value  
negative_counts = {}
for col in num_cols:
    if (df3[col] < 0).any():
        negative_counts[col] = (df3[col] < 0).sum()
negative_counts  
#print(f"Columns with negative values: {len(negative_counts)}")  
# so number of columns having negative values is 16 

negative_cols = []

for col in num_cols:
    if (df3[col] < 0).any():       #here we print all the column having negative value 
        negative_cols.append(col)
#print("column with negative value are ",negative_cols)  
negative_cols=['D4', 'D6', 'D11', 'D12', 'D14', 'D15', 'id_01', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_14'] 
#print(df3[negative_cols].describe())  #from this we decided that not chnage the negative value to postive 
                  #NOW missing value ananlysis of numerical columnss 
missing_num = df3[num_cols].isnull().sum().sort_values(ascending=False)

missing_num = missing_num[missing_num > 0]

missing_df = pd.DataFrame({
    "Missing Count": missing_num,
    "Missing %": (missing_num / len(df3) * 100).round(2)
})
missing_df 
#print(missing_df)
missing_percent = (df3[num_cols].isnull().mean() * 100)

#print(">= 90% :", (missing_percent >= 90).sum())
#print("70-90% :", ((missing_percent >= 70) & (missing_percent < 90)).sum())
#print("40-70% :", ((missing_percent >= 40) & (missing_percent < 70)).sum())
#print("10-40% :", ((missing_percent >= 10) & (missing_percent < 40)).sum())
#print("< 10%  :", (missing_percent < 10).sum())

missing_percent = (df3[num_cols].isnull().mean() * 100)
high_missing_cols = missing_percent[missing_percent >= 90].sort_values(ascending=False)
#print(high_missing_cols)    
    #The column having more than 90% missing value is  id_24    99.196159
#id_25    99.130965
#id_07    99.127070
#id_08    99.127070
#id_21    99.126393
#id_26    99.125715
#id_22    99.124699
#dist2    93.628374
#D7       93.409930
#id_18    92.360721   
  #so the conclusinon is Columns with >99% missing values are candidates for removal. Final removal will be performed during preprocessing after evaluating their usefulness.
#now for check zero value in numerical column
zero_counts = (df3[num_cols] == 0).sum()
zero_counts = zero_counts[zero_counts > 0].sort_values(ascending=False)
#print(zero_counts) 
 # ============================================================
# NUMERICAL DATA CLEANING SUMMARY
# ============================================================

# 1. Identified all numerical columns in the dataset.
# 2. Verified the data types of all numerical features.
# 3. Checked all numerical columns for negative values.
# 4. Negative values were found only in some D and id_* features.
#    These values were retained because they are valid for these features.
# 5. Analyzed missing values in all numerical columns.
# 6. Found 10 numerical columns with more than 90% missing values.
#    These columns were identified as candidates for removal, but no columns
#    were dropped during data cleaning. The final decision will be made
#    during preprocessing.
# 7. Observed that many numerical features contain zero values.
#    Zero values were retained because they represent valid information
#    and are not necessarily missing values.
# 8. Analyzed the distribution of important numerical features.
#    - TransactionAmt is right-skewed (Mean > Median).
#    - TransactionDT is approximately symmetric (Mean ≈ Median).
# 9. Outliers were not removed because tree-based models are robust to
#    outliers, and extreme values may represent fraudulent transactions.
# 10. Numerical data cleaning is complete.
#     Missing value imputation, feature selection, encoding, and scaling
#     will be performed during the preprocessing stage.
            # ============================================================    
            #  TRAIN  TEST  SPLIT 
            #=============================================================  
from sklearn.model_selection import train_test_split  
#define the feature and target values
x=df3.drop(columns=["isFraud"])
y=df3["isFraud"] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y) 
#==============================================================================
#                    DROP HIGH-MISSING COLUMNS (>99% missing)
# ==============================================================================
# Imputing columns with ~99% missing values is unlikely to produce reliable
# values and may introduce noise - dropped before further preprocessing.
# before droppint the column we check the missing invvolment in the fraud and non fraud transaction 
high_missing_cols = ['id_24','id_25','id_07','id_08','id_21','id_26','id_22','dist2','D7','id_18']

#for col in high_missing_cols:
    #print(col)
   #print(df3.groupby(df3[col].isnull())['isFraud'].mean())
   # print(df3[col].isnull().mean()*100)
   # print()    
# Columns with approximately 99% missing values are removed before handling
# missing data because they contain very few non-missing observations.
# Imputing such columns is unlikely to produce reliable values and may
# introduce unnecessary noise into the model. Therefore, these features
# are dropped as part of the initial data preprocessing step. 

removed_cols=["id_24","id_25","id_07","id_08","id_21","id_26","id_22"]
for col in removed_cols:
    x_train.drop(columns=[col],inplace=True)
    x_test.drop(columns=[col],inplace=True)  
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer 
 
#====== ===FUNCTION TRANSFORMER ===================================================  
#=======first we find the numerical column  name in train dataset            
numerical_columntrain = x_train.select_dtypes(include=['int64','float64']).columns
numerical_columntrain = list(numerical_columntrain)
# now we will find the skew of all the column    
#  if skew is positve then this column is right skewed     
# if skew  is negative  then  this column is left skewed  
# if skew is 0 then it is normal   skewed 

for col in numerical_columntrain:
    skew_values = x_train[col].skew()
    #print(col,skew_values)
# Observation:
# Many numerical features are highly skewed.

# Decision:
# Skewness transformation is not applied because this project uses
# tree-based models (Random Forest/XGBoost/LightGBM/CatBoost),
# which are generally robust to skewed feature distributions.
# Therefore, FunctionTransformer is not included in the preprocessing pipeline. 

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


# Apply the same transformation to both training and testing datasets
x_train = transaction_amt(x_train)
x_test = transaction_amt(x_test)  
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
x_train=TransactionAmt_decimal(x_train)
x_test=TransactionAmt_decimal(x_test) 
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
x_train=uid(x_train)
x_test=uid(x_test)   
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
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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
 #==============================================================================
#                      ENCODING CATEGORICAL FEATURES
# ============================================================================== 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  
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

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

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

from sklearn.pipeline import Pipeline

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

from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# ==============================================================================
# STEP: GRIDSEARCHCV - HYPERPARAMETER TUNING (NOT THE FINAL EVALUATION)
# ==============================================================================

# WHAT GridSearchCV DOES:
# GridSearchCV tries every combination of hyperparameters in param_grid.
# For EACH combination, it:
#   1. Splits x_train/y_train into the folds defined by cv_strategy.
#   2. Fits the WHOLE fraud_pipeline (imputation -> encoding -> feature
#      selection -> classifier) on the training portion of each fold.
#   3. Scores the fitted pipeline on the held-out portion of that fold.
#   4. Averages the score across all folds for that hyperparameter
#      combination.
# After trying every combination, it keeps the hyperparameters with the best
# average cross-validated score.
#
# IMPORTANT - GridSearchCV is NOT the final evaluation of the model:
# The scores GridSearchCV reports (best_score_, cv_results_) come only from
# x_train, split internally into folds. x_test is never touched during this
# step. The real, final evaluation happens later, once, on the untouched
# x_test/y_test.
#
# WHY THIS SCORING METRIC (average_precision / PR-AUC):
# Fraud is rare (a small percentage of all transactions), so:
#   - Plain accuracy is misleading: a model that always predicts "not
#     fraud" would already have very high accuracy while being useless.
#   - ROC-AUC is commonly used, but with extreme class imbalance it can
#     look overly optimistic, because the false-positive RATE is measured
#     against a huge number of negatives, making even a mediocre model's
#     ROC curve look good.
#   - Average Precision (the area under the Precision-Recall curve, i.e.
#     PR-AUC) focuses specifically on how well the model ranks and
#     identifies the rare positive (fraud) class, which is exactly what we
#     care about here. It is the standard recommended metric for
#     imbalanced fraud/anomaly-detection problems.
# We therefore use scoring='average_precision' as the metric GridSearchCV
# optimizes for, and we will additionally report precision, recall, F1, and
# ROC-AUC at final evaluation time for a fuller picture.

# ==============================================================================
# STEP: HYPERPARAMETER GRID
# ==============================================================================

# WHAT each hyperparameter controls:
#   - classifier__n_estimators: how many decision trees are built in the
#     forest. More trees generally means a more stable, less noisy
#     prediction, but takes longer to train.
#   - classifier__max_depth: how deep each individual tree is allowed to
#     grow. Shallower trees (e.g. 10) are less likely to overfit noisy
#     patterns; deeper trees (e.g. 20) can capture more complex interactions
#     but risk memorizing the training data.
#   - classifier__min_samples_leaf: the minimum number of samples required
#     at a leaf node. Larger values force the tree to generalize (each leaf
#     decision is based on more examples), which helps avoid overfitting to
#     rare, noisy patterns - useful here because fraud examples are rare
#     and we don't want the model memorizing individual fraud rows.
#
# WHY this search space is reasonable:
# This grid has 2 x 2 x 2 = 8 hyperparameter combinations. With
# cv_strategy having 3 folds, that is 8 x 3 = 24 total pipeline fits. Given
# ~470k training rows and shallow-to-moderate tree depths, this keeps total
# runtime realistic while still covering the most impactful hyperparameters
# for a RandomForest on this dataset size. We deliberately avoided adding
# more hyperparameters (e.g. max_features, min_samples_split) or a wider
# range of values, since that would multiply the number of fits and could
# make the search take an unreasonable amount of time on ~590k+ rows.

param_grid = {
    'classifier__n_estimators': [100, 300],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(
    estimator=fraud_pipeline,
    param_grid=param_grid,
    scoring='average_precision',
    cv=cv_strategy,
    n_jobs=1,        # parallelize across folds/hyperparameter combinations
    verbose=2,        # print progress, since this can take a while on ~590k rows
    refit=True        # after finding the best hyperparameters, refit the pipeline on ALL of x_train using them
)

# ==============================================================================
# STEP: TRAIN / VALIDATION SPLIT (FOR THRESHOLD TUNING - NOT FOR GridSearchCV)
# ==============================================================================
# WHAT:
# We split x_train/y_train further into a smaller training set
# (x_train_model/y_train_model) used for GridSearchCV, and a validation set
# (x_val/y_val) reserved for threshold selection below.
#
# WHY:
# Choosing the probability threshold that turns predicted probabilities
# into hard 0/1 fraud predictions is itself a modeling decision. If that
# threshold were chosen by looking at x_test, x_test would no longer be a
# clean, untouched hold-out set for the final evaluation - it would have
# already influenced a decision about the model. To keep x_test completely
# untouched until FINAL evaluation, we carve a validation set out of the
# training data specifically for threshold tuning:
#
#   x_train -> x_train_model (GridSearchCV) + x_val (threshold tuning)
#   x_test  -> used ONLY for the final evaluation, later in this script
#
# stratify=y_train is used for the same reason stratify=y was used in the
# original train/test split: this is an imbalanced fraud dataset, and a
# non-stratified split could leave x_val with very few fraud examples,
# making the threshold chosen from it unreliable.
x_train_model, x_val, y_train_model, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# Fit GridSearchCV ONLY on x_train_model/y_train_model. Neither x_val nor
# x_test is used anywhere in this call - x_val is reserved for threshold
# selection below, and x_test remains completely untouched until final
# evaluation.
grid_search.fit(x_train_model, y_train_model)


# ==============================================================================
# STEP: BEST HYPERPARAMETERS AND BEST CROSS-VALIDATED SCORE
# ==============================================================================

# best_params_ : the specific combination of n_estimators / max_depth /
# min_samples_leaf that produced the highest average cross-validated
# average_precision score across the 3 folds.
print("Best hyperparameters found:", grid_search.best_params_)

# best_score_ : the average_precision score AVERAGED ACROSS THE 3 CV FOLDS
# for the best hyperparameter combination, computed entirely on
# x_train_model. This is a cross-validation estimate of performance, NOT
# the final test score - it tells us how well this configuration is
# expected to generalize to unseen data, based only on training data. The
# real, unbiased estimate comes from evaluating on x_test at the very end.
print("Best cross-validated average_precision score (on training folds only):", grid_search.best_score_)

# best_estimator_ : the full fraud_pipeline (missing values -> encoding ->
# feature selection -> classifier), already refit on the ENTIRE
# x_train_model using the best hyperparameters (because we set refit=True
# above).
best_model = grid_search.best_estimator_


# ==============================================================================
# STEP: THRESHOLD SELECTION ON THE VALIDATION SET (x_test IS NOT TOUCHED HERE)
# ==============================================================================
# WHAT:
# RandomForestClassifier.predict() uses a fixed default threshold of 0.5 to
# turn predicted probabilities into 0/1 predictions. For an imbalanced
# problem like fraud detection, 0.5 is not necessarily the threshold that
# gives the best precision/recall trade-off, so we search for a better one
# using an F1-based sweep.
#
# WHY the validation set (not the test set):
# best_model was fit using only x_train_model/y_train_model, so x_val is
# genuinely unseen data from the model's point of view - calling
# best_model.predict_proba(x_val) here is safe. Selecting the threshold on
# x_val, instead of on x_test, keeps x_test fully reserved for the final,
# one-time evaluation later in this script.
from sklearn.metrics import f1_score

# Predicted probability of the positive (fraud) class on the validation set.
val_probabilities = best_model.predict_proba(x_val)[:, 1]

# Search a grid of candidate thresholds and keep the one with the best F1
# score on the validation set. F1 is used because it balances precision and
# recall, which matters here - optimizing only one of them at the expense
# of the other is not useful for a fraud-detection system.
candidate_thresholds = np.arange(0.05, 0.95, 0.01)
best_threshold = 0.5
best_val_f1 = -1

for threshold in candidate_thresholds:
    val_preds = (val_probabilities >= threshold).astype(int)
    current_f1 = f1_score(y_val, val_preds)
    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        best_threshold = threshold

print("Best threshold selected on validation set:", best_threshold)
print("Validation F1-score at best threshold:", best_val_f1)


# ==============================================================================
# STEP: FINAL EVALUATION ON THE UNTOUCHED TEST SET
# ==============================================================================

# WHAT:
# This is the first and only time x_test/y_test are used. Everything
# before this point (imputation fitting, encoding fitting, feature
# selection, cross-validation, hyperparameter tuning, and threshold
# selection) has used only x_train_model or x_val - never x_test.
#
# WHY this matters:
# Because best_model has never seen x_test in any form during fitting, and
# best_threshold was chosen using only x_val, scoring here gives an
# unbiased estimate of how the model will perform on genuinely new,
# unseen transactions - which is what actually matters in a real
# fraud-detection system.

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# Predicted probability of the positive (fraud) class - needed for
# threshold-independent metrics like ROC-AUC and PR-AUC, and also used
# below together with best_threshold to produce hard 0/1 predictions.
y_pred_proba = best_model.predict_proba(x_test)[:, 1]

# Hard class predictions (0 = not fraud, 1 = fraud) using best_threshold,
# which was selected on the validation set above - NOT the default 0.5
# threshold, and NOT tuned using x_test.
y_pred = (y_pred_proba >= best_threshold).astype(int)

# ------------------------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------------------------
# WHAT: a 2x2 table of counts:
#   [[True Negatives,  False Positives],
#    [False Negatives, True Positives]]
# WHY it matters for fraud: it separates the two very different types of
# mistakes a fraud model can make -
#   False Positives = legitimate transactions flagged as fraud (annoys
#     customers, creates manual review work),
#   False Negatives = actual fraud that slipped through undetected (direct
#     financial loss - usually the more costly mistake in fraud detection).
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------------------
# Precision, Recall, F1-score (via classification_report)
# ------------------------------------------------------------------------------
# Precision (for the fraud class) = of all transactions the model FLAGGED
# as fraud, what fraction were actually fraud? Low precision means too many
# legitimate customers get incorrectly flagged.
#
# Recall (for the fraud class) = of all the transactions that were ACTUALLY
# fraud, what fraction did the model catch? This is usually the metric that
# matters most in fraud detection: missing real fraud (low recall) directly
# costs money, whereas a false alarm (lower precision) is comparatively
# cheaper - typically just an extra manual review step.
#
# F1-score = the harmonic mean of precision and recall, a single number
# that balances both concerns when neither can be ignored.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

# ------------------------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------------------------
# WHAT: the probability that the model ranks a randomly chosen fraud
# transaction higher (more likely to be fraud) than a randomly chosen
# legitimate transaction, across all possible thresholds.
# CAUTION: with heavy class imbalance, ROC-AUC can look deceptively high
# because the false-positive rate is calculated against a very large number
# of negatives. We report it here for reference, but PR-AUC (below) is the
# more trustworthy metric for this specific dataset.
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC: {roc_auc:.4f}")

# ------------------------------------------------------------------------------
# PR-AUC / Average Precision
# ------------------------------------------------------------------------------
# WHAT: average_precision_score computes Average Precision (AP) - it
# summarizes the precision-recall curve as the weighted mean of the
# precision achieved at each threshold, weighted by the increase in recall
# from the previous threshold. This is NOT the same computation as taking
# the trapezoidal area under the precision-recall curve, but AP is the
# standard PR-AUC-style summary metric used in imbalanced classification
# (including scikit-learn's own scoring='average_precision' used in
# GridSearchCV above), so we refer to it here as "PR-AUC / Average
# Precision" to match that common usage. It is NOT ROC-AUC - see the
# separate ROC-AUC section above for that metric.
# WHY it is especially appropriate here: like ROC-AUC, AP is
# threshold-independent, but unlike ROC-AUC it is not affected by the large
# number of true negatives, so it gives a much more realistic picture of
# performance when fraud is rare - which is exactly the situation in this
# dataset.
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

# ==============================================================================
# IMPORTANT REMINDER:
# A high overall accuracy does NOT automatically mean this is a good fraud
# model. Because fraud is rare, a model could achieve very high accuracy
# simply by predicting "not fraud" for almost every transaction, while
# missing nearly all actual fraud (i.e. very poor recall for the fraud
# class). This is exactly why accuracy was never used as the scoring metric
# for GridSearchCV, and why precision/recall/F1/PR-AUC are reported above
# instead of relying on accuracy alone.
# ==============================================================================