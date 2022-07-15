from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import acquire

def prep_telco():
    df = acquire.get_telco_data()
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype(float)
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)   
    df = pd.concat([df, dummy_df], axis=1)
    return df


def prep_telco_encoded():
    df = acquire.get_telco_data()
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype(float)
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    df['multiple_lines_encoded'] = df.multiple_lines.map({'Yes': 1, 'No': 0, 'No phone service': 2})
    df['online_security_encoded'] = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 2})
    df['online_backup_encoded'] = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 2})
    df['device_protection_encoded'] = df.device_protection.map({'Yes': 1, 'No': 0, 'No internet service': 2})
    df['tech_support_encoded'] = df.tech_support.map({'Yes': 1, 'No': 0, 'No internet service': 2})
    df['streaming_tv_encoded'] = df.streaming_tv.map({'Yes': 1, 'No': 0, 'No internet service': 2})
    df['streaming_movies_encoded'] = df.streaming_movies.map({'Yes': 1, 'No': 0, 'No internet service': 2})
    df['contract_type_encoded'] = df.contract_type.map({'One year': 1, 'Month-to-month': 0, 'Two year': 2})
    df['internet_service_type_encoded'] = df.internet_service_type.map({'Fiber optic': 1, 'DSL': 0, 'None': 2})
    df['payment_type_encoded'] = df.payment_type.map({'Electronic check': 1, 'Mailed check': 0, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    df.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 'internet_service_type', 'payment_type' ], inplace=True)
    return df

# split telco data into train test and validate samples

def split_telco(df):
    telco_train, telco_test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn_encoded )
    telco_train, telco_validate = train_test_split(telco_train, test_size=.3, random_state=123, stratify=telco_train.churn_encoded)
    return telco_train, telco_validate, telco_test