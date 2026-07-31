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
            #=============================================================
            # HANDLING MISSING VALUE   
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
a=x_train.select_dtypes(include=['int64','float64']).isnull().sum()
a=a[a>0].index.tolist() 
b=x_train.select_dtypes(include=['object']).isnull().sum()
b=b[b>0].index.tolist()
#print("numerical columns with missing values:",a) #here we store all the numerical columns having missing value in a list 
#print("categorical columns with missing values:",b) #here we store all the categorical columns having missing value in a list 
#===================================================================================
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer 
missingvalue_transformer = ColumnTransformer(
    transformers=[
    ('missingnumerical_column', SimpleImputer(strategy='median'), a),
    ('missingcategorical_column', SimpleImputer(strategy='most_frequent'), b)
    ]
)  #Here we use the simple imputer to fill the missing value in numerical column with median and in categorical column with most frequent value  

#       FUNCTION TRANSFORMER ===================================================  
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
uid_amt_mean = x_train.groupby("uid")["TransactionAmt"].mean()

# Step 2: Map the average amount back to every row in x_train
x_train["uid_TransactionAmt_mean"] = x_train["uid"].map(uid_amt_mean)

# Step 3: Use the same mapping for x_test
x_test["uid_TransactionAmt_mean"] = x_test["uid"].map(uid_amt_mean)
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
uid_amt_std = x_train.groupby("uid")["TransactionAmt"].std()

# Step 2: Map the standard deviation back to every row in x_train
x_train["uid_TransactionAmt_std"] = x_train["uid"].map(uid_amt_std)

# Step 3: Use the same mapping for x_test
x_test["uid_TransactionAmt_std"] = x_test["uid"].map(uid_amt_std) 
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
def Amt_to_mean_ratio(df6):
    df6=df6.copy()
    df6["Amt_to_mean_ratio"]=(df6["TransactionAmt"]/(df6["uid_TransactionAmt_mean"] +1))
    return df6 
x_train=Amt_to_mean_ratio(x_train)
x_test=Amt_to_mean_ratio(x_test)  
# ==============================================================================
# FEATURE 7 : Email Provider Feature
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# The original email domains have many unique values.
#
# Example:
# gmail.com
# yahoo.com
# hotmail.com
# outlook.com
# live.com
# aol.com
#
# Some of these domains belong to the same company.
#
# Instead of treating every domain as a separate category,
# we group similar domains into one provider family.
#
# This reduces the number of categories and helps the model
# learn general patterns more effectively.

# Example
# -------
#
# Original Domain          New Provider
# -------------------------------------
# gmail.com          --->  gmail
# gmail              --->  gmail
# yahoo.com          --->  yahoo
# hotmail.com        --->  microsoft
# outlook.com        --->  microsoft
# live.com           --->  microsoft
# anonymous.com      --->  anonymous
# aol.com            --->  other

# New Feature
# -----------
# P_email_provider
#
# This feature stores the provider family instead of
# the complete email domain.

# Why is this useful?
# -------------------
# Instead of learning dozens of individual email domains,
# the model learns broader categories such as:
#
# Gmail users
# Yahoo users
# Microsoft users
# Anonymous email users
# Other providers
#
# This reduces feature cardinality while preserving
# meaningful information.

# Data Leakage?
# -------------
# No.
#
# This feature is created independently for each row.
# It does not use statistics from x_train or x_test,
# so there is no risk of data leakage.

# Steps
# -----
# Step 1 : Read the email domain.
# Step 2 : Identify its provider.
# Step 3 : Return the provider name.
# Step 4 : Store the provider in a new feature
#          called P_email_provider.


# ==============================================================================
# CODE
# ==============================================================================

def email_provider(domain):
    if pd.isna(domain):
        return np.nan

    domain = domain.lower()

    if "gmail" in domain:
        return "gmail"

    elif "yahoo" in domain:
        return "yahoo"

    elif "hotmail" in domain or "outlook" in domain or "live.com" in domain:
        return "microsoft"

    elif "anonymous" in domain:
        return "anonymous"

    else:
        return "other"


def add_email_provider(df):
    df = df.copy()
    df["P_email_provider"] = df["P_emaildomain"].apply(email_provider)
    return df


x_train = add_email_provider(x_train)
x_test = add_email_provider(x_test) 
# ==============================================================================
# FEATURE 8 : Recipient Email Provider
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# The R_emaildomain column contains the recipient's email domain.
#
# Similar to P_emaildomain, it has many unique values such as:
#
# gmail.com
# yahoo.com
# hotmail.com
# outlook.com
# live.com
# aol.com
#
# Some of these domains belong to the same company.
# Instead of treating every domain separately,
# we group them into provider families.

# Example
# -------
#
# Original Domain          New Provider
# -------------------------------------
# gmail.com          --->  gmail
# gmail              --->  gmail
# yahoo.com          --->  yahoo
# hotmail.com        --->  microsoft
# outlook.com        --->  microsoft
# live.com           --->  microsoft
# anonymous.com      --->  anonymous
# aol.com            --->  other

# New Feature
# -----------
# R_email_provider
#
# This feature stores the provider family instead of
# the complete recipient email domain.

# Why is this useful?
# -------------------
# Instead of learning dozens of individual recipient
# email domains, the model learns broader categories such as:
#
# - Gmail
# - Yahoo
# - Microsoft
# - Anonymous
# - Other
#
# This reduces the number of unique categories while
# preserving useful information.

# Data Leakage?
# -------------
# No.
#
# This feature is created independently for each row.
# It does not use any statistics from x_train or x_test,
# so there is no risk of data leakage.

# Steps
# -----
# Step 1 : Read the recipient email domain.
# Step 2 : Identify its provider.
# Step 3 : Return the provider name.
# Step 4 : Store the provider in a new feature
#          called R_email_provider. 
def email_provider(domain):
    if pd.isna(domain):
        return np.nan

    domain = domain.lower()

    if "gmail" in domain:
        return "gmail"

    elif "yahoo" in domain:
        return "yahoo"

    elif "hotmail" in domain or "outlook" in domain or "live.com" in domain:
        return "microsoft"

    elif "anonymous" in domain:
        return "anonymous"

    else:
        return "other"


def add_email_provider(df):
    df = df.copy()
    df["R_email_provider"] = df["R_emaildomain"].apply(email_provider)
    return df


x_train = add_email_provider(x_train)
x_test = add_email_provider(x_test)  
# ==============================================================================
# FEATURE 9 : Email Match Feature
# ==============================================================================

# Why are we creating this feature?
# ---------------------------------
# This feature checks whether the purchaser's email domain
# and the recipient's email domain are the same.
#
# If both email domains match, it may indicate a normal transaction.
# If they are different, it may provide an additional fraud signal.
#
# This feature converts a comparison into a simple binary feature.

# Example
# -------
#
# P_emaildomain      R_emaildomain      email_match
# -------------------------------------------------
# gmail.com          gmail.com               1
# yahoo.com          gmail.com               0
# outlook.com        outlook.com             1
# live.com           hotmail.com             0

# New Feature
# -----------
# email_match
#
# Values:
# 1 -> Both email domains are the same.
# 0 -> Email domains are different.

# Why is this useful?
# -------------------
# Fraudulent transactions may use a different recipient
# email domain than the purchaser's email domain.
#
# This feature allows the model to learn whether
# matching or non-matching email domains are associated
# with fraud.

# Data Leakage?
# -------------
# No.
#
# This feature is created independently for each row.
# It does not use any statistics from x_train or x_test.

# Steps
# -----
# Step 1 : Compare P_emaildomain and R_emaildomain.
# Step 2 : If they are equal, return 1.
# Step 3 : Otherwise, return 0.
# Step 4 : Store the result in a new feature
#          called email_match.  
def email_match(df):
    df = df.copy()

    df["email_match"] = (
        df["P_emaildomain"] == df["R_emaildomain"]
    ).astype(int)

    return df

x_train = email_match(x_train)
x_test = email_match(x_test)
