import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pywt
import pandas as pd

# Veri setinin bulunduğu klasörün yolu

data_folder = r'C:\Users\Ramazan\PycharmProjects\YSA\deneme'
diagnosis_df = pd.read_excel(r'C:\Users\Ramazan\PycharmProjects\YSA\Labels.xlsx')

# diagnosis_df için Patient ID'lerin ilk 4 karakterini kullanarak yeni bir index oluştur
diagnosis_df['Shortened ID'] = diagnosis_df['Patient ID'].apply(lambda x: x[:4])
diagnosis_df.set_index('Shortened ID', inplace=True)

# Tüm ses dosyalarının isimlerini al
all_files = os.listdir(data_folder)

dominant_frequencies_per_file = []

# Her bir dosya için işlem yap
for file_name in all_files:
    file_path = os.path.join(data_folder, file_name)
    try:
        # Dosyayı oku
        rate, data = wavfile.read(file_path)

        # Empirical Wavelet Transform'ı uygula
        coeffs = pywt.swt(data, 'db2', level=5)

        # Her bir bileşenin domine eden frekansını hesapla
        dominant_frequencies = [np.argmax(np.abs(coeff)) for coeff in coeffs]

        # Dosya isminden 'Patient ID' çıkar (örneğin 'H002_L1.wav' -> 'H002')
        patient_id_shortened = file_name.split('_')[0]
        dominant_frequencies_per_file.append([patient_id_shortened, file_name] + dominant_frequencies)
    except Exception as e:
        print(f"{file_name} okunurken bir hata oluştu: {e}")

# Sonuçları DataFrame'e dönüştür
column_names = ['Shortened ID', 'File Name'] + [f"Level_{i + 1}" for i in range(len(dominant_frequencies))]
df = pd.DataFrame(dominant_frequencies_per_file, columns=column_names)

# Excel dosyası ile birleştir
combined_df = df.join(diagnosis_df, on='Shortened ID')
final_df = combined_df.drop(columns=['Patient ID', 'Shortened ID'])

print(final_df)

# DataFrame'i tamamen göstermek için ayarları değiştir
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


for file_name, frequencies in dominant_frequencies_per_file.items():
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, marker='o', label='Dominant Frequencies')
    plt.title(f"Dominant Frequencies - {file_name}")
    plt.xlabel("Component Level")
    plt.ylabel("Frequency Index")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
