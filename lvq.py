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
# X_train = df.iloc[:, 0:4].values
# """ Bisa juga menggunakan cara ini:
# X_train = df[['Listening',
#         'Reading',
#         'Speaking',
#         'Writing']].values
# """

# Ambil kolom ke-5 atau kolom terakhir sebagai target kelulusan (1 atau 0)
# y_train = df.iloc[:, 4].values
# Bisa juga menggunakan cara ini: y_train = df['Target'].values

# --- PILIHAN B: MANUAL DATASET (3 DATA UNTUK HITUNGAN MANUAL BAB III) ---
# Hapus tanda komentar (""") di bawah ini jika ingin kembali menggunakan 3 data manual.
# """
X_train = np.array([
    [25, 20, 15, 15],  # Lulus
    [10, 15, 10, 10],  # Tidak Lulus
    [20, 20, 15, 15],  # Lulus
])
y_train = np.array([1, 0, 1])
# """

# NORMALISASI
X_train = X_train / 30

# ==================================================
# MODEL LEARNING VECTOR QUANTIZATION (LVQ)
# ==================================================

# Class / OOP Based Code
class LVQ:

    def __init__(self, n_inputs, learning_rate=0.1, epochs=50):
        self.n_inputs = n_inputs
        self.lr = learning_rate
        self.epochs = epochs

        # Vektor Prototipe awal diset statis di angka tengah 0.5 agar adil
        self.weights = np.array(
            [
                [0.5, 0.5, 0.5, 0.5],  # Neuron 0 (Tidak Lulus)
                [0.5, 0.5, 0.5, 0.5],  # Neuron 1 (Lulus)
            ]
        )

        self.labels = np.array([0, 1])
        self.error_history = []
        self.accuracy_history = []
        self.weight0_history = []
        self.weight1_history = []

    # Fungsi kompetisi untuk mencari neuron pemenang (winner) berdasarkan jarak terdekat ke input X
    def get_winner(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1) # Ini rumus euclidean distance  (\(d = \sqrt{\sum (w - x)^2}\)).
        winner_idx = np.argmin(distances) #Ini untuk mengambil neuron atau prototype dengan jarak terdekat ke input X berdasarkan jarak euclidean kedua prototype yang sudah dihitung pake rumus distance di atas
        return winner_idx #Kembalikan nilai neuron pemenang yang sudah diambil

    def train(self, X, y):
        print("\n===== PROSES TRAINING LVQ =====\n")
        start_time = time.time()

        # ==================================================
        # PENGATURAN JENIS LEARNING RATE (PILIH SALAH SATU)
        # ==================================================
        # ISI PILIHAN 1: MODE LEARNING RATE DECAY (Mengecil tiap epoch)
        decay_constant = 0.95  
        
        # ISI PILIHAN 2: MODE LEARNING RATE KONSTAN (Aktifkan ini jika ingin statis)
        # decay_constant = 1.0  
        # ==================================================

        for epoch in range(self.epochs):
            total_error = 0
            correct_predictions = 0
            
            # Simpan koordinat bobot sebelum epoch ini dimulai (untuk mendeteksi pergerakan)
            previous_weights = self.weights.copy()

            for x_i, target in zip(X, y):

                # >>> FASE FORWARD PASS (ALUR MAJU) <<<
                winner_idx = self.get_winner(x_i)
                self.last_winner = winner_idx
                winner_weight = self.weights[winner_idx]

                # >>> FASE BACKWARD PASS (ALUR MUNDUR / UPDATE) <<<
                if self.labels[winner_idx] == target:
                    correct_predictions += 1
                    self.weights[winner_idx] += self.lr * (x_i - winner_weight)
                else:
                    total_error += 1
                    self.weights[winner_idx] -= self.lr * (x_i - winner_weight)

            # Simpan riwayat perkembangan model per epoch
            self.error_history.append(total_error)
            accuracy = (correct_predictions / len(X)) * 100
            self.accuracy_history.append(accuracy)

            self.weight0_history.append(self.weights[0].copy())
            self.weight1_history.append(self.weights[1].copy())

            # TAMPILKAN PERKEMBANGAN BOBOT TIAP EPOCH
            print(f"Epoch {epoch+1:02d} | Error = {total_error} | Akurasi = {accuracy:.2f}% | Current LR = {self.lr:.6f}")
            print(f" -> Bobot Prototipe C0 (Tidak Lulus): {self.weights[0]}")
            print(f" -> Bobot Prototipe C1 (Lulus)      : {self.weights[1]}")
            
            # Hitung seberapa jauh bobot bergeser di epoch ini
            weight_shift = np.sum(np.abs(self.weights - previous_weights))
            print(f" -> Pergeseran Bobot pada Epoch Ini : {weight_shift:.8f}")
            print("-" * 65)

            # FAKTOR BERHENTI 1: KONVERGEN SEMPURNA (ERROR = 0)
            if total_error == 0:
                end_time = time.time()
                print("\nModel Konvergen Sempurna (Error = 0)!")
                return (epoch + 1, end_time - start_time)

            # FAKTOR BERHENTI 2: JALUR ALTERNATIF (BOBOT SUDAH STAGNAN / TIDAK BERGERAK)
            # Catatan: Hanya diaktifkan jika menggunakan mode Decay (decay_constant < 1.0)
            if decay_constant < 1.0 and weight_shift < 1e-5 and epoch > 0:
                end_time = time.time()
                print(f"\nModel Konvergen Praktis! Training dihentikan di Epoch {epoch+1} karena bobot sudah optimal.")
                return (epoch + 1, end_time - start_time)

            # PROSES UPDATE LEARNING RATE
            self.lr = decay_constant * self.lr

        end_time = time.time()
        print(f"\nTraining Selesai! Model berhenti karena mencapai batas maksimum {self.epochs} epoch (Error belum 0).")
        return (self.epochs, end_time - start_time)
    
    def predict(self, x):
        winner_idx = self.get_winner(x)
        self.last_winner = winner_idx 
        return self.labels[winner_idx]

    def get_distances(self, x):
        return np.linalg.norm(self.weights - x, axis=1)

if __name__ == "__main__":
    # ==================================================
    # INISIALISASI MODEL & RUN TRAINING (Ubah lr dan epochs sesuai kebutuhan eksperimen di sini)
    # ==================================================
    model = LVQ(n_inputs=4, learning_rate=0.1, epochs=10)
    epoch_used, training_time = model.train(X_train, y_train)

    # ==================================================
    # HASIL TRAINING & GRAFIK
    # ==================================================
    print("\n===== HASIL TRAINING LVQ =====")
    print("Bobot Prototype Akhir C0      :", model.weights[0])
    print("Bobot Prototype Akhir C1      :", model.weights[1])
    print("Learning Rate Awal            :",1)                # Sesuaikan dengan nilai LR awal yang Anda pakai
    print("Learning Rate Terakhir        :", f"{model.lr:.6f}") # Menampilkan LR saat training selesai
    print("Batas Maksimum Epoch          :", model.epochs)
    print("Epoch Digunakan               :", epoch_used)
    print(f"Waktu Training                : {training_time:.6f} detik")
    print(f"Akurasi Training Akhir        : {model.accuracy_history[-1]:.2f}%")
    print(f"Error Akhir                   : {model.error_history[-1]}")
    print("========================================")

    # ==================================================
    # GRAFIK EVALUASI TRAINING LVQ
    # ==================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

    plt.tight_layout()
    plt.savefig("grafik_evaluasi_lvq.png")
    plt.show()



    # ==================================================
    # TESTING REAL TIME
    # ==================================================

    print("\n===== TESTING =====")

    listening = float(input("Listening : "))

    reading = float(input("Reading   : "))

    speaking = float(input("Speaking  : "))

    writing = float(input("Writing   : "))

    # ==========================================
    # NORMALISASI INPUT USER
    # ==========================================

    test_input = np.array([listening, reading, speaking, writing]) / 30

    # ==========================================
    # PREDIKSI
    # ==========================================

    prediction = model.predict(test_input)

    distances = model.get_distances(test_input)

    print("\n===== HASIL PREDIKSI =====")

    print("Jarak ke Prototype 0:", distances[0])
    print("\nJarak ke Prototype 1:", distances[1])
    print("\nWinner Prototype yang Dipakai:", "Prototype",prediction)

    if prediction == 1:

        print("Prediksi : LULUS TOEFL")

    else:

        print("Prediksi : TIDAK LULUS TOEFL")
