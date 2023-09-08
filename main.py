import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Veri setini yükleme
data = pd.read_csv('C:/Users/LENOVO/Desktop/student_data.csv')
# DataFrame'i inceleyin
print(data.head())
print(data.columns)
data.isnull().values.any()
print(data.dtypes)
# LabelEncoder nesnesini oluşturma
label_encoder = LabelEncoder()

# Her bir kategorik kolonu Label Encoding ile dönüştürme
categorical_columns = ['Mjob', 'Fjob', 'school', 'sex', 'famsize', 'higher', 'internet', 'romantic', 'Pstatus', 'address', 'reason', 'famsup', 'schoolsup', 'guardian', 'paid', 'nursery', 'activities']
for kolon in categorical_columns:
    data[kolon] = label_encoder.fit_transform(data[kolon])

# StandartScaler nesnesini oluşturma
scaler = StandardScaler()

# Tüm kolonları Standartlaştırma işlemine tabi tutma
veri_std = scaler.fit_transform(data)
sutun_sayisi = veri_std.shape[1]

# Standartlaştırılmış verileri yeni bir DataFrame'e dönüştürme
veri_standart_df = pd.DataFrame(data=veri_std, columns=data.columns)

# Bağımsız değişkenleri (X) ve bağımlı değişkeni (y) ayırma
X = veri_standart_df.drop("failures", axis=1)
y = data["failures"]

# Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kullanılacak sınıflandırma modelini seçme (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Modeli eğitme
model.fit(X_train, y_train)
# Modelin test verileri üzerinde değerlendirilmesi
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Eğitilmiş modeli kullanarak tahminler yapma
y_pred = model.predict(X_test)

# Gerçek hedef değerleri tekrar alın
y_true = y_test  # zaten y_test'i kullanıyoruz

# Performans ölçütlerini hesaplama (örneğin, accuracy, precision, recall)
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

# Karmaşıklık matrisini hesaplayın
confusion_mat = confusion_matrix(y_true, y_pred)

# Karmaşıklık matrisini yazdırın
print("Confusion Matrix:")
print(confusion_mat)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Karmaşıklık Matrisi')
plt.colorbar()

classes = ['Başarılı', 'Başarısız']  # Sınıf etiketleri
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('Gerçek Etiketler')
plt.xlabel('Tahmin Edilen Etiketler')
plt.tight_layout()
plt.show()

# Model performans metriklerini görselleştirme
metrikler = ['Accuracy', 'Precision', 'Recall']
degerler = [accuracy, precision, recall]

plt.figure(figsize=(8, 6))
plt.bar(metrikler, degerler, color='skyblue')
plt.title('Model Performans Metrikleri')
plt.ylabel('Değer')
plt.ylim(0, 1)  # Değer aralığını uygun bir şekilde ayarlayabilirsiniz

plt.show()


tahmin1 = model.predict([[ 0.51, 16.50, 0.8, 0.3, 0.9, 2.8, 2.6, 2.2, 2.3, 2.2, 1.8, 1.5, 2.1, 0.6, 0.3, 0.8, 0.6, 0.7, 0.9, 0.1, 0.2, 0.5, 3.1, 3.1, 3.3, 2.1, 2.9, 4.1, 6.2, 10.1, 11.7, 12.1]])

tahmin_str = "test tahmini: " + str(tahmin1[0])
print(tahmin_str)
# Tkinter penceresini oluşturma
root = tk.Tk()
root.title("Veri Tahmin Arayüzü")

# Tahminleme işlemi
def tahminle():
    try:
        # Kullanıcıdan veriyi al
        yeni_veri = {
            "Mjob": mjob_entry.get(),
            "Fjob": fjob_entry.get(),
            "school": school_entry.get(),
            "sex": sex_entry.get(),
            "age": float(age_entry.get()),  # Veriyi float olarak dönüştürün
            "reason": reason_entry.get(),
            "guardian": guardian_entry.get(),
            "address": address_entry.get(),
            "famsize": famsize_entry.get(),
            "Pstatus": Pstatus_entry.get(),
            "Medu": float(Medu_entry.get()),  # Veriyi float olarak dönüştürün
            "Fedu": float(Fedu_entry.get()),  # Veriyi float olarak dönüştürün
            "traveltime": float(traveltime_entry.get()),  # Veriyi float olarak dönüştürün
            "studytime": float(studytime_entry.get()),  # Veriyi float olarak dönüştürün
            "schoolsup": schoolsup_entry.get(),
            "famsup": famsup_entry.get(),
            "paid": paid_entry.get(),
            "activities": activities_entry.get(),
            "nursery": nursery_entry.get(),
            "higher": higher_entry.get(),
            "internet": internet_entry.get(),
            "romantic": romantic_entry.get(),
            "famrel": float(famrel_entry.get()),  # Veriyi float olarak dönüştürün
            "freetime": float(freetime_entry.get()),  # Veriyi float olarak dönüştürün
            "goout": float(goout_entry.get()),  # Veriyi float olarak dönüştürün
            "Dalc": float(Dalc_entry.get()),  # Veriyi float olarak dönüştürün
            "Walc": float(Walc_entry.get()),  # Veriyi float olarak dönüştürün
            "health": float(health_entry.get()),  # Veriyi float olarak dönüştürün
            "absences": float(absences_entry.get()),  # Veriyi float olarak dönüştürün
            "G1": float(G1_entry.get()),  # Veriyi float olarak dönüştürün
            "G2": float(G2_entry.get()),  # Veriyi float olarak dönüştürün
            "G3": float(G3_entry.get()),  # Veriyi float olarak dönüştürün
        }

        # Yeni veriyi işleme
        yeni_veri_df = pd.DataFrame(data=yeni_veri, index=[0])

        # Her bir kategorik kolonu Label Encoding ile dönüştürme
        for kolon in categorical_columns:
            yeni_veri_df[kolon] = label_encoder.transform(yeni_veri_df[kolon])

        # Standartlaştırma işlemi
        yeni_veri_std = scaler.transform(yeni_veri_df)

        # Tahminleme yapma
        tahmin = model.predict(yeni_veri_std)

        # Tahmin sonucunu ekrana yazdır
        sonuc_label.config(text="Tahmin Sonucu: {}".format("Başarısız" if tahmin[0] == 1 else "Başarılı"))

    except Exception as e:
        sonuc_label.config(text="Hata: Veriyi işlerken bir sorun oluştu.")

# Scrollbar ekleyin
scrollbar = ttk.Scrollbar(root, orient="vertical")
scrollbar.pack(side="right", fill="y")

# Canvas ekleyin
canvas = tk.Canvas(root, yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.config(command=canvas.yview)

# İçerik çerçevesi ekleyin
content_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=content_frame, anchor="nw")

# Kullanıcı arayüzü öğelerini içerik çerçevesine ekleyin
mjob_label = ttk.Label(content_frame, text="Mjob:")
mjob_label.grid(row=0, column=0, padx=10, pady=5)
mjob_entry = ttk.Entry(content_frame)
mjob_entry.grid(row=0, column=1, padx=10, pady=5)

fjob_label = ttk.Label(content_frame, text="Fjob:")
fjob_label.grid(row=1, column=0, padx=10, pady=5)
fjob_entry = ttk.Entry(content_frame)
fjob_entry.grid(row=1, column=1, padx=10, pady=5)

school_label = ttk.Label(content_frame, text="school:")
school_label.grid(row=2, column=0, padx=10, pady=5)
school_entry = ttk.Entry(content_frame)
school_entry.grid(row=2, column=1, padx=10, pady=5)

sex_label = ttk.Label(content_frame, text="sex:")
sex_label.grid(row=3, column=0, padx=10, pady=5)
sex_entry = ttk.Entry(content_frame)
sex_entry.grid(row=3, column=1, padx=10, pady=5)

age_label = ttk.Label(content_frame, text="age:")
age_label.grid(row=4, column=0, padx=10, pady=5)
age_entry = ttk.Entry(content_frame)
age_entry.grid(row=4, column=1, padx=10, pady=5)

reason_label = ttk.Label(content_frame, text="reason:")
reason_label.grid(row=5, column=0, padx=10, pady=5)
reason_entry = ttk.Entry(content_frame)
reason_entry.grid(row=5, column=1, padx=10, pady=5)

guardian_label = ttk.Label(content_frame, text="guardian:")
guardian_label.grid(row=6, column=0, padx=10, pady=5)
guardian_entry = ttk.Entry(content_frame)
guardian_entry.grid(row=6, column=1, padx=10, pady=5)

address_label = ttk.Label(content_frame, text="address:")
address_label.grid(row=7, column=0, padx=10, pady=5)
address_entry = ttk.Entry(content_frame)
address_entry.grid(row=7, column=1, padx=10, pady=5)

famsize_label = ttk.Label(content_frame, text="famsize:")
famsize_label.grid(row=8, column=0, padx=10, pady=5)
famsize_entry = ttk.Entry(content_frame)
famsize_entry.grid(row=8, column=1, padx=10, pady=5)

Pstatus_label = ttk.Label(content_frame, text="Pstatus:")
Pstatus_label.grid(row=9, column=0, padx=10, pady=5)
Pstatus_entry = ttk.Entry(content_frame)
Pstatus_entry.grid(row=9, column=1, padx=10, pady=5)

Medu_label = ttk.Label(content_frame, text="Medu:")
Medu_label.grid(row=10, column=0, padx=10, pady=5)
Medu_entry = ttk.Entry(content_frame)
Medu_entry.grid(row=10, column=1, padx=10, pady=5)

Fedu_label = ttk.Label(content_frame, text="Fedu:")
Fedu_label.grid(row=11, column=0, padx=10, pady=5)
Fedu_entry = ttk.Entry(content_frame)
Fedu_entry.grid(row=11, column=1, padx=10, pady=5)

traveltime_label = ttk.Label(content_frame, text="traveltime:")
traveltime_label.grid(row=12, column=0, padx=10, pady=5)
traveltime_entry = ttk.Entry(content_frame)
traveltime_entry.grid(row=12, column=1, padx=10, pady=5)

studytime_label = ttk.Label(content_frame, text="studytime:")
studytime_label.grid(row=13, column=0, padx=10, pady=5)
studytime_entry = ttk.Entry(content_frame)
studytime_entry.grid(row=13, column=1, padx=10, pady=5)

failures_label = ttk.Label(content_frame, text="failures:")
failures_label.grid(row=14, column=0, padx=10, pady=5)
failures_entry = ttk.Entry(content_frame)
failures_entry.grid(row=14, column=1, padx=10, pady=5)

schoolsup_label = ttk.Label(content_frame, text="schoolsup:")
schoolsup_label.grid(row=15, column=0, padx=10, pady=5)
schoolsup_entry = ttk.Entry(content_frame)
schoolsup_entry.grid(row=15, column=1, padx=10, pady=5)

famsup_label = ttk.Label(content_frame, text="famsup:")
famsup_label.grid(row=16, column=0, padx=10, pady=5)
famsup_entry = ttk.Entry(content_frame)
famsup_entry.grid(row=16, column=1, padx=10, pady=5)

paid_label = ttk.Label(content_frame, text="paid:")
paid_label.grid(row=17, column=0, padx=10, pady=5)
paid_entry = ttk.Entry(content_frame)
paid_entry.grid(row=17, column=1, padx=10, pady=5)

activities_label = ttk.Label(content_frame, text="activities:")
activities_label.grid(row=18, column=0, padx=10, pady=5)
activities_entry = ttk.Entry(content_frame)
activities_entry.grid(row=18, column=1, padx=10, pady=5)

nursery_label = ttk.Label(content_frame, text="nursery:")
nursery_label.grid(row=19, column=0, padx=10, pady=5)
nursery_entry = ttk.Entry(content_frame)
nursery_entry.grid(row=19, column=1, padx=10, pady=5)

higher_label = ttk.Label(content_frame, text="higher:")
higher_label.grid(row=20, column=0, padx=10, pady=5)
higher_entry = ttk.Entry(content_frame)
higher_entry.grid(row=20, column=1, padx=10, pady=5)

internet_label = ttk.Label(content_frame, text="internet:")
internet_label.grid(row=21, column=0, padx=10, pady=5)
internet_entry = ttk.Entry(content_frame)
internet_entry.grid(row=21, column=1, padx=10, pady=5)

romantic_label = ttk.Label(content_frame, text="romantic:")
romantic_label.grid(row=22, column=0, padx=10, pady=5)
romantic_entry = ttk.Entry(content_frame)
romantic_entry.grid(row=22, column=1, padx=10, pady=5)

famrel_label = ttk.Label(content_frame, text="famrel:")
famrel_label.grid(row=23, column=0, padx=10, pady=5)
famrel_entry = ttk.Entry(content_frame)
famrel_entry.grid(row=23, column=1, padx=10, pady=5)

freetime_label = ttk.Label(content_frame, text="freetime:")
freetime_label.grid(row=24, column=0, padx=10, pady=5)
freetime_entry = ttk.Entry(content_frame)
freetime_entry.grid(row=24, column=1, padx=10, pady=5)

goout_label = ttk.Label(content_frame, text="goout:")
goout_label.grid(row=25, column=0, padx=10, pady=5)
goout_entry = ttk.Entry(content_frame)
goout_entry.grid(row=25, column=1, padx=10, pady=5)

Dalc_label = ttk.Label(content_frame, text="Dalc:")
Dalc_label.grid(row=26, column=0, padx=10, pady=5)
Dalc_entry = ttk.Entry(content_frame)
Dalc_entry.grid(row=26, column=1, padx=10, pady=5)

Walc_label = ttk.Label(content_frame, text="Walc:")
Walc_label.grid(row=27, column=0, padx=10, pady=5)
Walc_entry = ttk.Entry(content_frame)
Walc_entry.grid(row=27, column=1, padx=10, pady=5)

health_label = ttk.Label(content_frame, text="health:")
health_label.grid(row=28, column=0, padx=10, pady=5)
health_entry = ttk.Entry(content_frame)
health_entry.grid(row=28, column=1, padx=10, pady=5)

absences_label = ttk.Label(content_frame, text="absences:")
absences_label.grid(row=29, column=0, padx=10, pady=5)
absences_entry = ttk.Entry(content_frame)
absences_entry.grid(row=29, column=1, padx=10, pady=5)

G1_label = ttk.Label(content_frame, text="G1:")
G1_label.grid(row=30, column=0, padx=10, pady=5)
G1_entry = ttk.Entry(content_frame)
G1_entry.grid(row=30, column=1, padx=10, pady=5)

G2_label = ttk.Label(content_frame, text="G2:")
G2_label.grid(row=31, column=0, padx=10, pady=5)
G2_entry = ttk.Entry(content_frame)
G2_entry.grid(row=31, column=1, padx=10, pady=5)

G3_label = ttk.Label(content_frame, text="G3:")
G3_label.grid(row=32, column=0, padx=10, pady=5)
G3_entry = ttk.Entry(content_frame)
G3_entry.grid(row=32, column=1, padx=10, pady=5)



# Diğer girdi alanlarını da ekleyin...

tahmin_button = ttk.Button(content_frame, text="Tahminle", command=tahminle)
tahmin_button.grid(row=33, column=0, columnspan=2, padx=10, pady=5, sticky="e")


sonuc_label = ttk.Label(content_frame, text="Tahmin Sonucu: ")
sonuc_label.grid(row=34, column=1,columnspan=2, padx=10, pady=5, sticky="e")

# Canvas'in boyutlarını ayarlayın
content_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Tkinter penceresini çalıştırın
root.mainloop()

