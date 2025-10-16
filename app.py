import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ----------------------
# Load dataset
# ----------------------
df = pd.read_csv('data.csv')

# Map diagnosis to 0/1
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df = df.drop(columns=['id'])

# Select only the 17 features
features = [
    'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'symmetry_mean', 'radius_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'fractal_dimension_se', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]
X = df[features]
y = df['diagnosis']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM models
svm_linear = SVC(kernel='linear', probability=True)
svm_linear.fit(X_train_scaled, y_train)

svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_rbf.fit(X_train_scaled, y_train)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")
st.title("🔬 Breast Cancer Tumor Classification")
st.markdown("Predict whether a tumor is **Benign** or **Malignant** using SVM.")

# Sidebar for kernel selection
kernel_choice = st.sidebar.selectbox("Select SVM Kernel", ["Linear", "RBF"])

st.subheader("Enter Tumor Features")

cols = st.columns(3)
user_input = {}

for i, feature in enumerate(features):
    col = cols[i % 3]
    max_val = max(abs(float(X[feature].min())), abs(float(X[feature].max())))
    user_input[feature] = col.number_input(
        label=feature,
        min_value=-max_val,   # negative value correctly passed
        max_value=max_val,    # positive value correctly passed
        value=0.0,            # default neutral value
        step=0.01,
        format="%.4f"
    )



input_df = pd.DataFrame([user_input])

# Handle missing values and scale
input_imputed = imputer.transform(input_df)
input_scaled = scaler.transform(input_imputed)

# Predict button
if st.button("Predict Tumor Type"):
    if kernel_choice == "Linear":
        pred = svm_linear.predict(input_scaled)[0]
    else:
        pred = svm_rbf.predict(input_scaled)[0]
    
    result = "Malignant" if pred == 1 else "Benign"
    
    st.markdown("### Prediction Result:")
    if result == "Malignant":
        st.error(f" The tumor is predicted as: {result}")
    else:
        st.success(f" The tumor is predicted as: {result}")
