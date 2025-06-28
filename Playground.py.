import streamlit as st
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ML Visualizer", layout="centered")
st.title("Interactive ML Model Visualizer")

if "points" not in st.session_state:
    st.session_state.points = []

st.sidebar.header("➕ Add Data Points")

add_mode = st.sidebar.radio("Choose Input Method", ["Manual Entry", "Upload CSV"])

if add_mode == "Manual Entry":
    x_val = st.sidebar.number_input("X value", step=0.1)
    y_val = st.sidebar.number_input("Y value", step=0.1)
    if st.sidebar.button("Add Point"):
        st.session_state.points.append((x_val, y_val))
        st.success(f"Added point ({x_val}, {y_val})")

elif add_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'x' and 'y' columns", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'x' in df.columns and 'y' in df.columns:
            st.session_state.points.extend(list(zip(df['x'], df['y'])))
            st.success(f"Uploaded {len(df)} points successfully!")
        else:
            st.error("CSV must contain 'x' and 'y' columns")

if len(st.session_state.points) < 2:
    st.warning("Please add at least two points to train the model.")
    st.stop()

X = np.array([p[0] for p in st.session_state.points]).reshape(-1, 1)
y = np.array([p[1] for p in st.session_state.points])

split_ratio = st.sidebar.slider("Train/Test Split (%)", 50, 95, 80)
test_size = 1 - split_ratio / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.sidebar.header("🔧 Model Settings")
model_type = st.sidebar.selectbox("Select Model", [
    "Linear Regression",
    "Polynomial Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
    "Support Vector Regression",
    "KNN Regression"
])

x_range = np.linspace(X.min() - 1, X.max() + 1, 200).reshape(-1, 1)
equation = ""
mse = r2 = 0

if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_line = model.predict(x_range)
    equation = f"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}"

elif model_type == "Polynomial Regression":
    poly_degree = st.sidebar.slider("Polynomial Degree", 2, 10, 2)
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)
    y_pred_line = model.predict(poly.transform(x_range))
    equation = " + ".join([f"{model.coef_[i]:.3f}x^{i}" if i != 0 else f"{model.intercept_:.3f}" for i in range(len(model.coef_))])

elif model_type == "Decision Tree Regression":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_line = model.predict(x_range)
    equation = "Decision tree (non-linear fit)"

elif model_type == "Random Forest Regression":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50, 10)
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_line = model.predict(x_range)
    equation = f"Random Forest with {n_estimators} trees"

elif model_type == "Support Vector Regression":
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    model = SVR(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_line = model.predict(x_range)
    equation = f"SVR with {kernel} kernel"

elif model_type == "KNN Regression":
    k = st.sidebar.slider("K Neighbors", 1, 20, 5)
    
    if k > len(X):
        st.error(f"Number of neighbors ({k}) cannot be greater than the number of samples ({len(X)}).")
        st.stop()  
    
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_line = model.predict(x_range)
    equation = f"KNN Regression with K={k}"


st.subheader("📊 Model Evaluation")
if len(X_train) > 1 and len(X_test) > 1:  # Ensure there's enough data for both train and test sets
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
else:
    train_r2 = test_r2 = None
    warnings.warn("Not enough data for R² score calculation.")

st.write(f"**Train MSE:** {mean_squared_error(y_train, y_pred_train):.4f}")
if train_r2 is not None:
    st.write(f"**Train R²:** {train_r2:.4f}")
else:
    st.write("**Train R²:** Not available (insufficient data)")

st.write(f"**Test MSE:** {mean_squared_error(y_test, y_pred_test):.4f}")
if test_r2 is not None:
    st.write(f"**Test R²:** {test_r2:.4f}")
else:
    st.write("**Test R²:** Not available (insufficient data)")

st.write(f"**Equation:** {equation}")
st.write(f"**Train MSE:** {mean_squared_error(y_train, y_pred_train):.4f}")
st.write(f"**Train R²:** {r2_score(y_train, y_pred_train):.4f}")
st.write(f"**Test MSE:** {mean_squared_error(y_test, y_pred_test):.4f}")
st.write(f"**Test R²:** {r2_score(y_test, y_pred_test):.4f}")

fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='blue', label="Train")
ax.scatter(X_test, y_test, color='green', label="Test")
ax.plot(x_range, y_pred_line, color="red", label="Prediction")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("📉 Residual Plot (Test Set)")
residuals = y_test - y_pred_test
fig2, ax2 = plt.subplots()
ax2.scatter(y_pred_test, residuals)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
st.pyplot(fig2)

if st.button("🔄 Clear All Points"):
    st.session_state.points = []
    st.write("Data cleared. You can start adding points again!")
