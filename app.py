import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perceptron import train_perceptron, perceptron_step
from lvq import LVQ

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Perceptron vs LVQ",
    layout="wide"
)

st.title("🎓 Sistem Klasifikasi Kelulusan TOEFL")
st.markdown("---")

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_excel("toefl_ibt_dataset.xlsx")

st.subheader("Dataset TOEFL")

df_display = df.copy()

df_display.insert(
    0,
    "No.",
    range(1, len(df_display) + 1)
)

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True
)
st.markdown("---")

# ==================================================
# PARAMETER MODEL
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    model_choice = st.selectbox(
        "Pilih Model",
        ["Perceptron", "LVQ"]
    )

with col2:
    lr = st.number_input(
        "Learning Rate",
        min_value=0.01,
        value=0.1,
        step=0.01
    )

with col3:
    epochs = st.number_input(
        "Epoch",
        min_value=1,
        value=100,
        step=1
    )

# ==================================================
# TRAINING
# ==================================================

if st.button("🚀 Latih Model"):

    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    X = X / 30

    # ==========================================
    # PERCEPTRON
    # ==========================================

    if model_choice == "Perceptron":

        (
            weights,
            bias,
            epoch_used,
            max_epochs,
            training_time,
            history_loss,
            history_accuracy,
        ) = train_perceptron(
            X,
            y,
            lr=lr,
            epochs=epochs
        )

        st.session_state["model_type"] = "Perceptron"
        st.session_state["weights"] = weights
        st.session_state["bias"] = bias

        st.success("Training Perceptron Berhasil")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Epoch Digunakan", epoch_used)
            st.metric("Waktu Training", f"{training_time:.6f} detik")

        with c2:
            st.metric(
                "Akurasi Training",
                f"{history_accuracy[-1]:.2f}%"
            )
            st.metric(
                "MSE",
                f"{history_loss[-1]:.4f}"
            )

        st.subheader("Bobot Akhir")

        weight_df = pd.DataFrame({
            "Bobot": ["W1", "W2", "W3", "W4"],
            "Value": np.round(weights, 4)
        })

        st.dataframe(
            weight_df,
            use_container_width=True,
            hide_index=True
        )

        st.write("Bias :", bias)
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(12, 5)
        )
        
        # Sumbu X asli Perceptron
        epochs_range = list(range(1, len(history_loss) + 1))
        total_epochs = len(epochs_range)
        
        # Logika merapikan angka sumbu X Perceptron
        if total_epochs <= 10:
            custom_ticks = epochs_range
        else:
            step = 10
            ticks_base = [i for i in range(step, total_epochs, step)]
            custom_ticks = sorted(list(set([1] + ticks_base + [total_epochs])))
        
        # 1. Plot Kurva Error (Sumbu X menggunakan epochs_range)
        ax1.plot(epochs_range, history_loss, color="red", linewidth=2, marker="o") # Ditambah marker bulat agar titik epochnya jelas
        ax1.set_title("Grafik Penurunan Error (MSE) - Perceptron")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE")
        ax1.set_xticks(custom_ticks) # Memaksa angka di bawah hanya menampilkan angka bulat (1,2,3,4,5,6)
        ax1.grid(True)

        # 2. Plot Kurva Akurasi (Sumbu X menggunakan epochs_range)
        ax2.plot(epochs_range, history_accuracy, color="green", linewidth=2, marker="o")
        ax2.set_title("Grafik Peningkatan Akurasi - Perceptron")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Akurasi (%)")
        ax2.set_xticks(custom_ticks) # Memaksa angka di bawah hanya menampilkan angka bulat (1,2,3,4,5,6)
        ax2.grid(True)

        st.pyplot(fig)

    # ==========================================
    # LVQ
    # ==========================================

    else:

        model = LVQ(
            n_inputs=4,
            learning_rate=lr,
            epochs=epochs
        )

        epoch_used, training_time = model.train(
            X,
            y
        )

        st.session_state["model_type"] = "LVQ"
        st.session_state["lvq_model"] = model

        st.success("Training LVQ Berhasil")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Epoch Digunakan", epoch_used)
            st.metric("Waktu Training", f"{training_time:.6f} detik")

        with c2:
            st.metric(
                "Akurasi Training",
                f"{model.accuracy_history[-1]:.2f}%"
            )

            st.metric(
                "Error",
                model.error_history[-1]
            )

        st.subheader("Prototype Akhir")

        st.write("Prototype Class 0")

        prototype0_df = pd.DataFrame({
            "Bobot": ["W1", "W2", "W3", "W4"],
            "Value": np.round(model.weights[0], 4)
        })

        st.dataframe(
            prototype0_df,
            use_container_width=True,
            hide_index=True
        )

        st.write("Prototype Class 1")

        prototype1_df = pd.DataFrame({
            "Bobot": ["W1", "W2", "W3", "W4"],
            "Value": np.round(model.weights[1], 4)
        })

        st.dataframe(
            prototype1_df,
            use_container_width=True,
            hide_index=True
        )

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(12, 5)
        )

       # Sumbu X asli (1 sampai jumlah epoch aktual)
        lvq_epochs_range = list(range(1, len(model.error_history) + 1))
        total_epochs = len(lvq_epochs_range)

        # --------------------------------------------------
        # LOGIKA MERAPIKAN ANGKA SUMBU X (MELOMPAT)
        # --------------------------------------------------
        # Jika epoch sedikit (misal <= 10), tampilkan semua angka secara normal
        if total_epochs <= 10:
            custom_ticks = lvq_epochs_range
        else:
            # Jika epoch banyak (misal 100), buat lompatan kelipatan 10
            step = 10 
            
            # Ambil angka kelipatan 10 (10, 20, 30, dst) yang berada di dalam range
            ticks_base = [i for i in range(step, total_epochs, step)]
            
            # Gabungkan angka 1 di awal, angka basis kelipatan, dan angka epoch terakhir
            custom_ticks = [1] + ticks_base + [total_epochs]
            
            # Hapus duplikasi jika angka terakhir tidak sengaja kembar dengan kelipatan 10
            custom_ticks = sorted(list(set(custom_ticks)))

        # 1. Plot Kurva Error
        ax1.plot(lvq_epochs_range, model.error_history, color="red", marker="o", linewidth=2)
        ax1.set_title("Grafik Penurunan Error (Jumlah Salah Klasifikasi) - LVQ")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Error")
        ax1.set_xticks(custom_ticks)  # Gunakan label lompatan yang sudah dirapikan
        ax1.grid(True)

        # 2. Plot Kurva Akurasi
        ax2.plot(lvq_epochs_range, model.accuracy_history, color="green", marker="o", linewidth=2)
        ax2.set_title("Grafik Peningkatan Akurasi Training - LVQ")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Akurasi (%)")
        ax2.set_xticks(custom_ticks)  # Gunakan label lompatan yang sudah dirapikan
        ax2.grid(True)

        st.pyplot(fig)

st.markdown("---")

# ==================================================
# TESTING
# ==================================================

st.header("🧪 Testing Data Baru")

col1, col2 = st.columns(2)

with col1:
    listening = st.number_input(
        "Listening",
        min_value=0.0,
        max_value=30.0
    )

    reading = st.number_input(
        "Reading",
        min_value=0.0,
        max_value=30.0
    )

with col2:
    speaking = st.number_input(
        "Speaking",
        min_value=0.0,
        max_value=30.0
    )

    writing = st.number_input(
        "Writing",
        min_value=0.0,
        max_value=30.0
    )

if st.button("🔍 Prediksi"):

    if "model_type" not in st.session_state:

        st.warning(
            "Silakan training model terlebih dahulu."
        )

    else:

        x_test = np.array([
            listening,
            reading,
            speaking,
            writing
        ]) / 30

        # ======================================
        # PERCEPTRON
        # ======================================

        if st.session_state["model_type"] == "Perceptron":

            weights = st.session_state["weights"]
            bias = st.session_state["bias"]

            z = np.dot(
                x_test,
                weights
            ) + bias

            prediction = perceptron_step(z)

            st.subheader("Hasil Prediksi")

            st.write("Nilai Z :", z)

            st.subheader("Bobot Akhir")

            weight_df = pd.DataFrame({
                "Bobot": ["W1", "W2", "W3", "W4"],
                "Value": np.round(weights, 4)
            })

            st.dataframe(
                weight_df,
                use_container_width=True,
                hide_index=True
            )

            st.write("Bias :", bias)

            st.write(
                "Output Aktivasi :",
                prediction
            )

            if prediction == 1:
                st.success("LULUS TOEFL")
            else:
                st.error("TIDAK LULUS TOEFL")

        # ======================================
        # LVQ
        # ======================================

        else:

            model = st.session_state["lvq_model"]

            prediction = model.predict(
                x_test
            )

            distances = model.get_distances(
                x_test
            )

            st.subheader("Hasil Prediksi")

            st.write(
                "Prototype Class 0 :"
            )

            prototype0_df = pd.DataFrame({
                "Bobot": ["W1", "W2", "W3", "W4"],
                "Value": np.round(model.weights[0], 4)
            })

            st.dataframe(
                prototype0_df,
                use_container_width=True,
                hide_index=True
            )
            st.write(
                "Prototype Class 1 :"
            )

            prototype1_df = pd.DataFrame({
                "Bobot": ["W1", "W2", "W3", "W4"],
                "Value": np.round(model.weights[1], 4)
            })

            st.dataframe(
                prototype1_df,
                use_container_width=True,
                hide_index=True
            )

            st.write(
                "Distance Prototype 0 :",
                distances[0]
            )

            st.write(
                "Distance Prototype 1 :",
                distances[1]
            )

            st.write(
                "Winner Prototype :",
                prediction
            )

            if prediction == 1:
                st.success("LULUS TOEFL")
            else:
                st.error("TIDAK LULUS TOEFL")