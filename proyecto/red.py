import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import SGD, Adam # type: ignore
import graphviz

st.set_page_config(page_title="Red Neuronal MLP", layout="wide")

# Cargar datasets
def load_data(problem_type):
    if problem_type == "Regresión":
        df = sns.load_dataset("diamonds")
        df = df.dropna()
        X = df[["carat", "depth", "table"]]
        y = df["price"]
    else:
        df = pd.read_csv("atletas.csv")
        df['Atleta'] = df['Atleta'].map({'Fondista': 1, 'Velocista': 0})
        df = df.dropna()
        X = df[["Edad", "Peso", "Volumen_O2_max"]]
        y = df["Atleta"]
    return X, y

# Sidebar
st.sidebar.title("Configuración del Modelo")
problem_type = st.sidebar.selectbox("Tipo de problema", ["Regresión", "Clasificación"])
optimizer_option = st.sidebar.selectbox("Optimizador", ["adam", "sgd"])
metric_option = st.sidebar.selectbox("Métrica", ["mae", "mse", "accuracy"])
epochs = st.sidebar.slider("Épocas", 10, 100, 50, step=5)
activation_output = "sigmoid" if problem_type == "Clasificación" else "linear"
loss_function = "binary_crossentropy" if problem_type == "Clasificación" else "mse"
activation_hidden = "tanh" if problem_type == "Clasificación" else "relu"

# Datos
X, y = load_data(problem_type)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo MLP
model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation=activation_hidden))
model.add(Dense(8, activation=activation_hidden))
model.add(Dense(1, activation=activation_output))

optimizer = Adam() if optimizer_option == "adam" else SGD()
model.compile(optimizer=optimizer, loss=loss_function, metrics=[metric_option])

# Entrenamiento
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)

# Evaluación
st.title("Red Neuronal Multicapa")
st.write(f"### Tipo de problema: {problem_type}")
st.write("### Arquitectura: Entrada → 16 → 8 → 1")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gráfico de pérdida")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Predicción con tus valores")
    input_vals = {}
    for i, col in enumerate(X.columns):
        input_vals[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_array = np.array(list(input_vals.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0][0]
    if problem_type == "Clasificación":
        st.write("Resultado:", "Fondista" if pred >= 0.5 else "Velocista")
    else:
        st.write("Precio estimado:", round(pred, 2))

with col2:
    st.subheader("Evaluación del Modelo")
    y_pred = model.predict(X_test).flatten()
    if problem_type == "Clasificación":
        y_pred_labels = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred_labels)
        st.write(f"Precisión: {acc:.2f}")
        st.text(classification_report(y_test, y_pred_labels))
    else:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"R²: {r2:.2f}")

st.subheader("Representación de la Red Neuronal")
def draw_network(input_dim, hidden_layers, output_dim, activation_hidden, activation_output):
    dot = graphviz.Digraph()
    # Entrada
    for i in range(input_dim):
        dot.node(f"input_{i}", f"Entrada {i+1}", shape="circle", style="filled", fillcolor="lightblue")

    # Capas ocultas
    for l, n_units in enumerate(hidden_layers):
        for j in range(n_units):
            dot.node(f"hidden_{l}_{j}", f"H{l+1}-{j+1}\n({activation_hidden})", shape="circle", style="filled", fillcolor="lightgreen")

    # Capa de salida
    dot.node("output", f"Salida\n({activation_output})", shape="circle", style="filled", fillcolor="salmon")

    # Conexiones entrada -> primera oculta
    for i in range(input_dim):
        for j in range(hidden_layers[0]):
            dot.edge(f"input_{i}", f"hidden_0_{j}")

    # Conexiones ocultas -> ocultas (si hay más de una capa)
    for l in range(len(hidden_layers) - 1):
        for i in range(hidden_layers[l]):
            for j in range(hidden_layers[l+1]):
                dot.edge(f"hidden_{l}_{i}", f"hidden_{l+1}_{j}")

    # Conexiones última oculta -> salida
    for i in range(hidden_layers[-1]):
        dot.edge(f"hidden_{len(hidden_layers)-1}_{i}", "output")

    return dot

# Dibujar red neuronal
nn_graph = draw_network(
    input_dim=X_train.shape[1],
    hidden_layers=[16, 8],
    output_dim=1,
    activation_hidden=activation_hidden,
    activation_output=activation_output)

st.graphviz_chart(nn_graph)