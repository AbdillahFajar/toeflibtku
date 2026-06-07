import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Ditambahkan untuk membaca file Excel
import time

# ==================================================
# LOAD DATASET
# ==================================================

# --- PILIHAN A: MENGGUNAKAN DATASET EXCEL (30 DATA) ---
# Program akan membaca file Excel bernama 'toefl_ibt_dataset.xlsx'
# df = pd.read_excel("toefl_ibt_dataset.xlsx")

# Ambil 4 kolom pertama sebagai input fitur (Listening, Reading, Speaking, Writing)
# X = df.iloc[:, 0:4].values
# ''' Bisa juga menggunakan cara ini:
# X = df[['Listening',
#         'Reading',
#         'Speaking',
#         'Writing']].values
# '''

# Ambil kolom ke-5 atau kolom terakhir sebagai target kelulusan (1 atau 0)
# y = df.iloc[:, 4].values
# Bisa juga menggunakan cara ini: y = df['Target'].values


# --- PILIHAN B: MANUAL DATASET (3 DATA UNTUK HITUNGAN MANUAL BAB III) ---
# Hapus tanda komentar (#) di bawah ini jika ingin kembali menggunakan 3 data manual.
# """
X = np.array([
    [25,20,15,15], # Lulus
    [10,15,10,10], # Tidak Lulus
    [20,20,15,15]  # Lulus
])

y = np.array([
    1,
    0,
    1
])
# """

# NORMALISASI INPUT DATA
X = X / 30

# ==================================================
# ACTIVATION FUNCTION
# ==================================================

# Standar Function Based Code
def perceptron_step(z):
    return 1 if z >= 0 else 0


# ==================================================
# TRAINING LOOP
# ==================================================


def train_perceptron(inputs, targets, lr=0.1, epochs=50):
    print("\n===== PROSES TRAINING PERCEPTRON =====\n")
    weights = np.zeros(inputs.shape[1])
    bias = 0

    history_loss = []
    history_accuracy = []

    start_time = time.time()

    for epoch in range(epochs):

        total_error = 0
        correct_predictions = 0

        for x, target in zip(inputs, targets):

            # ==============================================
            # >>> FASE FORWARD PASS (ALUR MAJU) <<<
            # Melakukan perhitungan kombinasi linear dan fungsi aktivasi
            # ==============================================
            z = np.dot(x, weights) + bias
            y_pred = perceptron_step(z)
            # ==============================================

            error = target - y_pred
            total_error += abs(error)

            if y_pred == target:
                correct_predictions += 1

            # ==============================================
            # >>> FASE BACKWARD PASS (ALUR MUNDUR / UPDATE) <<<
            # Mengalirkan balik nilai error untuk memperbarui bobot dan bias
            # ==============================================
            weights += lr * error * x
            bias += lr * error
            # ==============================================

        mse = total_error / len(inputs)
        history_loss.append(mse)

        accuracy = (correct_predictions / len(inputs)) * 100
        history_accuracy.append(accuracy)

        print(
            f"Epoch {epoch+1:02d} | Error = {total_error} | Akurasi = {accuracy:.2f}% | Bobot = {weights} | Bias = {bias}"
        )

        if total_error == 0:
            end_time = time.time()
            print("\nModel Konvergen Sempurna (Error = 0)!")
            return (
                weights,
                bias,
                epoch + 1,
                epochs,
                end_time - start_time,
                history_loss,
                history_accuracy,
            )

    end_time = time.time()
    print(f"\nTraining Selesai! Model berhenti karena mencapai batas maksimum {epochs} epoch (Error belum 0).")
    return (
        weights,
        bias,
        epochs,
        epochs,
        end_time - start_time,
        history_loss,
        history_accuracy,
    )

if __name__ == "__main__":
    # ==================================================
    # INISIALISASI MODEL & RUN TRAINING (Ubah lr dan epochs sesuai kebutuhan eksperimen di sini!)
    # ==================================================
    weights, bias, epoch_used, max_epochs, training_time, history_loss, history_accuracy = (
        train_perceptron(X, y, lr=0.1, epochs=10)
    )

    # ==================================================
    # HASIL TRAINING & GRAFIK (Sama seperti kode sebelumnya)
    # ==================================================
    print("\n===== HASIL TRAINING =====")
    print("Bobot Akhir             :", weights)
    print("Bias Akhir              :", bias)
    print("Learning Rate           :", 1) # Sesuaikan dengan nilai LR aktual yang Anda pakai
    print("Batas Maksimum Epoch    :", max_epochs)
    print("Jumlah Epoch Digunakan  :", epoch_used)
    print("Waktu Training          :", f"{training_time:.6f} detik")

    # --- TAMBAHAN BARU UNTUK DISPLAY AKURASI & ERROR TERAKHIR ---
    # Mengambil data dari indeks paling akhir [-1] pada list history
    print("Total Error Terakhir    :", int(history_loss[-1] * len(X))) # Mengembalikan ke jumlah total data salah
    print(f"MSE Terakhir            : {history_loss[-1]:.4f}")
    print(f"Akurasi Training Akhir  : {history_accuracy[-1]:.2f}%")
    print("========================================")

    # ==================================================
    # GRAFIK EVALUASI TRAINING
    # ==================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

    plt.tight_layout()
    plt.savefig("grafik_evaluasi_perceptron.png")  
    plt.show()


    # ==========================
    # TESTING DENGAN INPUT DATA BARU
    # ==========================
    print("\n===== TESTING =====")

    listening = float(input("Nilai Listening : "))
    reading = float(input("Nilai Reading   : "))
    speaking = float(input("Nilai Speaking  : "))
    writing = float(input("Nilai Writing   : "))

    # ==========================
    # NORMALISASI INPUT USER
    # ==========================
    x_test = np.array([listening, reading, speaking, writing])
    x_test = x_test / 30

    # ==========================
    # PREDIKSI
    # ==========================
    z = np.dot(x_test, weights) + bias
    prediction = perceptron_step(z)

    # ==========================
    # HASIL TESTING
    # ==========================
    print("\n===== HASIL TESTING =====")
    print("Nilai Net Input (Z)          :", z)
    print("Bobot terakhir yang digunakan:", weights)
    print("Bias terakhir yang digunakan :", bias)
    print("Output Aktivasi              :", prediction)
    print("\n========================================")

    if prediction == 1:
        print("Prediksi : LULUS TOEFL")
    else:
        print("Prediksi : TIDAK LULUS TOEFL")
