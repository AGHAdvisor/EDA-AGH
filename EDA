import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
st.title("Exploratory Data Analysis (EDA) App")
uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Show data preview
    st.write("### Data Preview")
    st.write(data.head())

    # Step 1: Impute missing values with median for numeric columns
    st.write("### Step 1: Imputing Missing Values with Median")
    for col in data.columns:
        if data[col].isna().sum() > 0:
            if data[col].dtype == 'float64' or data[col].dtype == 'int64':
                median_value = data[col].median()
                data[col].fillna(median_value, inplace=True)
                st.write(f"Imputed missing values in `{col}` with median: {median_value}")
            else:
                st.write(f"Non-numeric column `{col}` has missing values, which will be imputed selectively.")

    # Step 2: Identify numeric, non-numeric, and categorical data
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    nonnumeric_cols = [col for col in data.columns if pd.api.types.is_string_dtype(data[col])]
    categorical_cols = [col for col in nonnumeric_cols if data[col].nunique() < 10]

    st.write("### Step 2: Data Types Summary")
    st.write(f"Numeric Columns: {numeric_cols}")
    st.write(f"Non-Numeric Columns: {nonnumeric_cols}")
    st.write(f"Categorical Columns: {categorical_cols}")

    # Step 3: Convert relevant non-numeric columns to numeric where possible
    st.write("### Step 3: Converting Relevant Non-Numeric Columns to Numeric")
    for col in nonnumeric_cols:
        if "TEMPERATURE" in col or "BLOOD PRESSURE" in col:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)  # Impute with median after conversion
            st.write(f"Converted `{col}` to numeric and imputed missing values with median: {median_value}")

    # Show a preview of the data after conversion
    st.write("### Data After Conversion and Imputation")
    st.write(data.head())

    # Step 4: Normalizing numeric columns
    scaler = MinMaxScaler()
    try:
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        st.write("### Step 4: Data Normalization")
        st.write("Data has been successfully normalized.")
    except ValueError as e:
        st.error(f"Error during normalization: {e}")

    # Step 5: Central Tendency and Dispersion
    st.write("### Step 5: Central Tendency and Dispersion")
    st.write(data.describe())

    # Step 6: Skewness and Kurtosis
    st.write("### Step 6: Skewness and Kurtosis")
    skew_kurt = pd.DataFrame({
        'Skewness': data[numeric_cols].apply(skew),
        'Kurtosis': data[numeric_cols].apply(kurtosis)
    })
    st.write(skew_kurt)

    # Step 7: Distribution Plots to Show Spread and Peakedness
    st.write("### Step 7: Distribution of Each Numeric Feature")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Step 8: Percentiles and Quartiles
    st.write("### Step 8: Percentiles and Quartiles")
    percentiles = data[numeric_cols].quantile([0.25, 0.5, 0.75, 0.95])
    st.write(percentiles)

    # Step 9: Correlation Matrix
    st.write("### Step 9: Correlation Matrix")
    corr_matrix = data[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Step 10: Z-Scores
    st.write("### Step 10: Z-Scores")
    z_scores = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
    st.write(z_scores)

    # Step 11: Inferences and Suggestions
    st.write("### Step 11: Inferences and Suggestions")
    st.write("""
    - **Highly Skewed Data**: Consider applying log or square root transformations to reduce skewness.
    - **Outliers**: Check z-scores to identify and potentially remove outliers.
    - **High Correlation**: Columns with high correlation may be redundant and could be considered for removal.
    - **Normalization Applied**: All numeric columns have been scaled, which can improve model performance.
    """)

else:
    st.write("Please upload an Excel file to begin.")
