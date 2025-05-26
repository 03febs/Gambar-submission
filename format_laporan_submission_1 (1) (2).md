# Laporan Proyek Machine Learning - Febrie Tsani Sovranita

## Domain Proyek

Penyakit Alzheimer merupakan penyakit dengan kelainan neurodegeneratif yang biasanya sering terjadi pada lansia. Penyakit ini ditandai dengan adanya penurunan kognitif progresif dan kehilangan memori atau demensia. Sampai saat ini, penyebab penyakit Alzheimer belum diketahui secara pasti. namun terdapat beberapa hipotesis yang menyatakan bahwa etiologi dari penyakit tersebut adalah faktor genetik, stress oksidatif, akumulasi Aβ intraseluler dan ekstraseluler, eksitotoksik, peradangan, disfungsi mitokondria, perubahan sitoskeleton, komponen sinapsis, dan hilangnya neuron yang berperan penting dalam mencegah timbulnya suatu penyakit [1].Berdasarkan data Alzheimer's Disease International, menunjukkan bahwa pada tahun 2018, terdapat sekitar 50 juta kasus demensia di seluruh dunia. Angka ini diperkirakan akan meningkat tiga kali lipat hingga tahun 2050, dengan mayoritas kasus terjadi di negara berpenghasilan rendah dan menengah [2].

Gejala  gangguan memori dan penurunan kognitif memiliki dampak negatif pada kualitas  hidup  pasien  serta  beban  sistem Kesehatan [3], maka dari itu, Penelitian dan prediksi dini penyakit Alzheimer sangat penting karena hingga saat ini belum ada obat yang dapat menyembuhkan penyakit ini secara total[4] . Tahap awal diagnosis penyakit Alzheimer dapat meningkatkan efisiensi pengobatan penyakit ini. Diagnosa penyakit alzheimer dapat dibantu dengan pendeteksian dini. Pendekatan matematis pada data mining dapat digunakan untuk menganalisis data dan untuk mendeteksi atau memprediksi penyakit dini alzheimer. Salah satu metode yang digunakan dalam mendeteksi atau memprediksi penyakit dini alzheimer  adalah dengan pendekatan algoritma klasifikasi berbasis machine learning [5] 

Studi sebelumnya [6] menunjukkan bahwa  menunjukkan bahwa berbagai algoritma machine learning seperti Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting, dan Voting Classifier dapat digunakan untuk mengidentifikasi individu yang berisiko tinggi terkena Alzheimer pada tahap awal dengan tingkat akurasi yang baik. Penelitian ini menggunakan data MRI longitudinal dari dataset OASIS dan mengevaluasi kinerja model menggunakan parameter seperti Precision, Recall, Accuracy, dan F1-score. Hasilnya, algoritma Random Forest memberikan hasil terbaik dengan akurasi mencapai 86,92%.

Model prediktif dapat dibuat untuk mengidentifikasi individu yang berisiko sebelum munculnya gejala berat dengan menggunakan data klinis, hasil tes kognitif, dan biomarker dari dataset publik seperti Kaggle. Prediksi yang lebih baik dapat dibuat dengan menggabungkan data dari berbagai sumber, seperti faktor gaya hidup, genetika, dan kondisi kesehatan lainnya. Dengan menggunakan algoritma Decision Tree, Random Forest, SVM, dan XGBoost, penelitian ini mengusulkan sistem otomatisasi untuk analisis prediktor penyakit Alzheimer. Sistem ini tidak hanya mempercepat proses diagnosis dan mengurangi biaya, tetapi juga mengurangi kesalahan manusia untuk membantu tenaga medis membuat keputusan klinis yang lebih baik. Oleh karena itu, diharapkan bahwa penerapan teknik ini dapat membantu mengurangi angka mortalitas akibat Alzheimer melalui diagnosis dan intervensi yang lebih dini dan tepat sasaran.


## Business Understanding

Bagian laporan ini mencakup:
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana kinerja berbagai algoritma klasifikasi (Decision Tree, Random Forest, SVM, XGBoost) dalam memprediksi risiko penyakit Alzheimer?
- Algoritma machine learning mana yang paling efektif untuk memprediksi risiko Alzheimer dengan akurasi tinggi?
- Bagaimana mengembangkan sistem prediksi berbasis machine learning untuk diagnosis awal Alzheimer?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model prediksi yang mampu mengidentifikasi individu berisiko tinggi terkena Alzheimer pada tahap awal.
- Meningkatkan akurasi deteksi dini dengan memanfaatkan data klinis, hasil tes kognitif, dan biomarker dari dataset publik.
- Membandingkan kinerja beberapa algoritma klasifikasi machine learning untuk mendapatkan model terbaik dalam memprediksi penyakit Alzheimer.

### Solution Statement 
Pengembangan model prediksi dini risiko Alzheimer memiliki potensi besar untuk membantu banyak orang, seperti pasien lansia, keluarga atau orang yang mengurus mereka, tenaga medis, dan pembuat kebijakan kesehatan. Dokter dapat merencanakan intervensi non-farmakologis maupun farmakologis lebih awal, seperti terapi kognitif, modifikasi gaya hidup, atau pengelolaan komorbiditas, untuk memperlambat perkembangan penyakit dan meningkatkan kualitas hidup pasien. Mereka dapat melakukan ini dengan prediksi yang akurat

Adapun solusi untuk mencapai tujuan tersebut dapat dirumuskan sebagai berikut:
- Menggunakan dan Membandingkan Beberapa Algoritma Klasifikasi Machine Learning
Mengembangkan model prediksi dengan menggunakan berbagai algoritma klasifikasi, seperti **Decision Tree, Random Forest, Support Vector Machine (SVM)**, dan **XGBoost**. Setiap algoritma akan diuji kinerjanya menggunakan dataset publik dari Kaggle, yang terdiri dari data seperti demografi, riwayat kesehatan, faktor gaya hidup, pengukuran klinis, penilaian kognitif dan fungsional, gejala, diagnosis penyakit Alzheimer, dan informasi lainnya.
**Matriks Evaluasi** yang digunakan yaitu Akurasi, Precision, Recall, F1-score akan digunakan untuk membandingkan performa masing-masing model. Model dengan performa terbaik (misal: akurasi tertinggi) akan dipilih sebagai solusi utama.
- Meningkatkan Akurasi Model dengan Oversampling Menggunakan SMOTE
Proses oversampling menggunakan teknik Synthetic Minority Oversampling Technique (SMOTE) untuk mengatasi ketidakseimbangan data, atau ketidakseimbangan kelas, yang sering terjadi pada data medis. SMOTE melakukan ini dengan mengumpulkan sampel sintetis dari kelas minoritas sehingga proporsi kelas menjadi lebih seimbang.
- Seleksi Fitur dengan Menghapus Variabel yang Tidak Berkontribusi
Dilakukan proses seleksi fitur untuk mengidentifikasi dan menghilangkan variabel yang tidak memberikan kontribusi signifikan terhadap proses pelatihan dan prediksi model. Penghapusan variabel yang tidak relevan ini dapat mengurangi kompleksitas model, mempercepat proses pelatihan, dan berpotensi meningkatkan akurasi model prediksi.


## Data Understanding
Data yang digunakan merupakan dataset public Alzheimer's Disease dari Kaggle. Dataset ini berisi informasi kesehatan yang luas untuk 2.149 pasien, masing-masing diidentifikasi secara unik dengan ID mulai dari 4751 hingga 6900. Dataset ini mencakup rincian demografis, faktor gaya hidup, riwayat kesehatan, pengukuran klinis, penilaian kognitif dan fungsional, gejala, dan diagnosis Penyakit Alzheimer. Data ini sangat ideal bagi para peneliti dan ilmuwan data yang ingin mengeksplorasi faktor-faktor yang terkait dengan Alzheimer, mengembangkan model prediktif, dan melakukan analisis statistik.

Sumber data:
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

### Deskripsi Variabel
Keterangan Dataset:
- `PatientID` : Pengenal unik yang diberikan kepada setiap pasien (4751 hingga 6900).

**Demographic Details**
- `Age` : Usia pasien berkisar antara 60 hingga 90 tahun.  
- `Gender` : Jenis kelamin pasien, di mana 0 menunjukkan Male dan 1 menunjukkan Female.  
- `Ethnicity` : Etnisitas pasien, dikodekan sebagai berikut:
  - 0: Caucasian  
  - 1: African American  
  - 2: Asian  
  - 3: Other  
- `EducationLevel` : Tingkat pendidikan pasien, dikodekan sebagai berikut:
  - 0: None  
  - 1: High School  
  - 2: Bachelor's  
  - 3: Higher  

**Lifestyle Factors**
- `BMI` : Indeks Massa Tubuh pasien, berkisar antara 15 hingga 40.  
- `Smoking` : Status merokok, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `AlcoholConsumption` : Konsumsi alkohol mingguan (dalam satuan unit), berkisar antara 0 hingga 20.  
- `PhysicalActivity` : Aktivitas fisik mingguan (dalam jam), berkisar antara 0 hingga 10.  
- `DietQuality` : Skor kualitas diet, berkisar antara 0 hingga 10.  
- `SleepQuality` : Skor kualitas tidur, berkisar antara 4 hingga 10.  

**Medical History**
- `FamilyHistoryAlzheimers` : Riwayat keluarga dengan penyakit Alzheimer, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `CardiovascularDisease` : Keberadaan penyakit kardiovaskular, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `Diabetes` : Keberadaan diabetes, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `Depression` : Keberadaan depresi, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `HeadInjury` : Riwayat cedera kepala, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `Hypertension` : Keberadaan hipertensi, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  

**Clinical Measurements**
- `SystolicBP` : Tekanan darah sistolik, berkisar antara 90 hingga 180 mmHg.  
- `DiastolicBP` : Tekanan darah diastolik, berkisar antara 60 hingga 120 mmHg.  
- `CholesterolTotal` : Kadar total kolesterol, berkisar antara 150 hingga 300 mg/dL.  
- `CholesterolLDL` : Kadar kolesterol LDL (Low-Density Lipoprotein), berkisar antara 50 hingga 200 mg/dL.  
- `CholesterolHDL` : Kadar kolesterol HDL (High-Density Lipoprotein), berkisar antara 20 hingga 100 mg/dL.  
- `CholesterolTriglycerides` : Kadar trigliserida, berkisar antara 50 hingga 400 mg/dL.  

**Cognitive and Functional Assessments**
- `MMSE` : Skor Mini-Mental State Examination, berkisar antara 0 hingga 30. Skor rendah menunjukkan gangguan kognitif. - `FunctionalAssessment` : Skor penilaian fungsional, berkisar antara 0 hingga 10. Skor rendah menunjukkan gangguan fungsional yang lebih besar.  
- `MemoryComplaints` : Keberadaan keluhan memori, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `BehavioralProblems` : Keberadaan masalah perilaku, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `ADL` : Skor aktivitas kehidupan sehari-hari (Activities of Daily Living), berkisar antara 0 hingga 10. Skor rendah menunjukkan gangguan yang lebih besar.  

**Symptoms**
- `Confusion` : Keberadaan kebingungan, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `Disorientation` : Keberadaan disorientasi, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `PersonalityChanges` : Keberadaan perubahan kepribadian, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `DifficultyCompletingTasks` : Keberadaan kesulitan dalam menyelesaikan tugas, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  
- `Forgetfulness` : Keberadaan pelupa, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  

**Diagnosis Information**
- `Diagnosis` : Status diagnosis penyakit Alzheimer, di mana 0 menunjukkan Tidak (No) dan 1 menunjukkan Ya (Yes).  

**Confidential Information**
- `DoctorInCharge` : Kolom ini berisi informasi rahasia mengenai dokter yang menangani, dengan nilai `"XXXConfid"` untuk semua pasien.

### Info Tipe Data dalam Dataset
Berikut ini jenis tipe data pada setiap kolomnya:
|  #  | Column                     | Dtype   |
|-----|----------------------------|---------|
|  0  | PatientID                  | int64   |
|  1  | Age                        | int64   |
|  2  | Gender                     | int64   |
|  3  | Ethnicity                  | int64   |
|  4  | EducationLevel             | int64   |
|  5  | BMI                        | float64 |
|  6  | Smoking                    | int64   |
|  7  | AlcoholConsumption         | float64 |
|  8  | PhysicalActivity           | float64 |
|  9  | DietQuality                | float64 |
| 10  | SleepQuality               | float64 |
| 11  | FamilyHistoryAlzheimers    | int64   |
| 12  | CardiovascularDisease      | int64   |
| 13  | Diabetes                   | int64   |
| 14  | Depression                 | int64   |
| 15  | HeadInjury                 | int64   |
| 16  | Hypertension               | int64   |
| 17  | SystolicBP                 | int64   |
| 18  | DiastolicBP                | int64   |
| 19  | CholesterolTotal           | float64 |
| 20  | CholesterolLDL             | float64 |
| 21  | CholesterolHDL             | float64 |
| 22  | CholesterolTriglycerides   | float64 |
| 23  | MMSE                       | float64 |
| 24  | FunctionalAssessment       | float64 |
| 25  | MemoryComplaints           | int64   |
| 26  | BehavioralProblems         | int64   |
| 27  | ADL                        | float64 |
| 28  | Confusion                  | int64   |
| 29  | Disorientation             | int64   |
| 30  | PersonalityChanges         | int64   |
| 31  | DifficultyCompletingTasks  | int64   |
| 32  | Forgetfulness              | int64   |
| 33  | Diagnosis                  | int64   |
| 34  | DoctorInCharge             | object  |

### Deskripsi Statistik Untuk Data Numerik
<div style="overflow-x: auto;">
| Stat   | PatientID | Age   | Gender | Ethnicity | EducationLevel | BMI   | Smoking | AlcoholConsumption | PhysicalActivity | DietQuality | FunctionalAssessment | MemoryComplaints | BehavioralProblems | ADL   | Confusion | Disorientation | PersonalityChanges | DifficultyCompletingTasks | Forgetfulness | Diagnosis |
|--------|-----------|-------|--------|-----------|----------------|-------|---------|--------------------|------------------|-------------|----------------------|------------------|--------------------|-------|-----------|----------------|--------------------|----------------------------|----------------|-----------|
| count  | 2149.0    | 2149.0| 2149.0 | 2149.0    | 2149.0         | 2149.0| 2149.0  | 2149.0             | 2149.0           | 2149.0      | 2149.0               | 2149.0           | 2149.0             | 2149.0| 2149.0    | 2149.0         | 2149.0             | 2149.0                     | 2149.0         | 2149.0    |
| mean   | 5825.0    | 74.91 | 0.51   | 0.70      | 1.29           | 27.66 | 0.29    | 10.04              | 4.92             | 4.99        | 5.08                 | 0.21             | 0.16               | 4.98  | 0.21      | 0.16           | 0.15               | 0.16                       | 0.30           | 0.35      |
| std    | 620.51    | 8.99  | 0.50   | 1.00      | 0.90           | 7.22  | 0.45    | 5.76               | 2.86             | 2.91        | 2.89                 | 0.41             | 0.36               | 2.95  | 0.40      | 0.37           | 0.36               | 0.37                       | 0.46           | 0.48      |
| min    | 4751.0    | 60.0  | 0.0    | 0.0       | 0.0            | 15.01 | 0.0     | 0.00               | 0.00             | 0.01        | 0.00                 | 0.00             | 0.00               | 0.00  | 0.00      | 0.00           | 0.00               | 0.00                       | 0.00           | 0.00      |
| 25%    | 5288.0    | 67.0  | 0.0    | 0.0       | 1.0            | 21.61 | 0.0     | 5.14               | 2.57             | 2.46        | 2.57                 | 0.00             | 0.00               | 2.34  | 0.00      | 0.00           | 0.00               | 0.00                       | 0.00           | 0.00      |
| 50%    | 5825.0    | 75.0  | 1.0    | 0.0       | 1.0            | 27.82 | 0.0     | 9.93               | 4.77             | 5.08        | 5.09                 | 0.00             | 0.00               | 5.04  | 0.00      | 0.00           | 0.00               | 0.00                       | 0.00           | 0.00      |
| 75%    | 6362.0    | 83.0  | 1.0    | 1.0       | 2.0            | 33.87 | 1.0     | 15.16              | 7.43             | 7.56        | 7.55                 | 0.00             | 0.00               | 7.58  | 0.00      | 0.00           | 0.00               | 0.00                       | 1.00           | 1.00      |
| max    | 6899.0    | 90.0  | 1.0    | 3.0       | 3.0            | 39.99 | 1.0     | 19.99              | 9.99             | 10.00       | 10.00                | 1.00             | 1.00               | 10.00 | 1.00      | 1.00           | 1.00               | 1.00                       | 1.00           | 1.00      |
</div>

Berdasarkan data distribusi statistik diatas menunjukkan bahwa Dataset ini terdiri dari 2.149 pasien lansia berusia 60–90 tahun dengan komposisi gender yang seimbang. Sebagian besar pasien berasal dari etnis Caucasian dan memiliki pendidikan minimal SMA. Rata-rata BMI mereka adalah 27,6, yang menunjukkan bahwa mereka cenderung kelebihan berat badan. Dengan konsumsi alkohol 10 unit atau lebih per minggu dan kurangnya aktivitas fisik 5 jam per minggu, gaya hidup ini cenderung tidak sehat. Kualitas tidur dan diet pasien juga sedang. Sebagian besar pasien tidak memiliki riwayat penyakit jangka panjang sebelumnya atau kronis. Namun, sekitar 25% pasien memiliki riwayat Alzheimer dalam keluarga.

### Memeriksa Nilai Null
| No. | Nama Kolom                  | Missing Value |
|-----|-----------------------------|----------------|
| 1   | PatientID                   | 0              |
| 2   | Age                         | 0              |
| 3   | Gender                      | 0              |
| 4   | Ethnicity                   | 0              |
| 5   | EducationLevel              | 0              |
| 6   | BMI                         | 0              |
| 7   | Smoking                     | 0              |
| 8   | AlcoholConsumption          | 0              |
| 9   | PhysicalActivity            | 0              |
| 10  | DietQuality                 | 0              |
| 11  | SleepQuality                | 0              |
| 12  | FamilyHistoryAlzheimers     | 0              |
| 13  | CardiovascularDisease       | 0              |
| 14  | Diabetes                    | 0              |
| 15  | Depression                  | 0              |
| 16  | HeadInjury                  | 0              |
| 17  | Hypertension                | 0              |
| 18  | SystolicBP                  | 0              |
| 19  | DiastolicBP                 | 0              |
| 20  | CholesterolTotal            | 0              |
| 21  | CholesterolLDL              | 0              |
| 22  | CholesterolHDL              | 0              |
| 23  | CholesterolTriglycerides    | 0              |
| 24  | MMSE                        | 0              |
| 25  | FunctionalAssessment        | 0              |
| 26  | MemoryComplaints            | 0              |
| 27  | BehavioralProblems          | 0              |
| 28  | ADL                         | 0              |
| 29  | Confusion                   | 0              |
| 30  | Disorientation              | 0              |
| 31  | PersonalityChanges          | 0              |
| 32  | DifficultyCompletingTasks   | 0              |
| 33  | Forgetfulness               | 0              |
| 34  | Diagnosis                   | 0              |
| 35  | DoctorInCharge              | 0              |

**Insight:**
dapat dilihat pada ouput diatas menunjukkan bahwa pada dataset ini tidak terdapat nilai null

### Memeriksa Nilai Duplikat
![Image](https://github.com/user-attachments/assets/5786567d-a529-409a-ae16-836ce59dac2b)
**Insight:**
Berdasarkan output diatas hasilnya menunjukkan tidak ada nilai yang duplikat.

### Menghapus Kolom Yang Tidak Penting
![Image](https://github.com/user-attachments/assets/98968257-42fb-43e1-ada0-131bd07c1081)

**Insight:**
kolom `PatientID` dihapus karena tidak memberikan informasi prediktif karena sifatnya yang unik dan tidan berulang.
kolom `DoctorInCharge` dihapus alasannya tidak memberikan informasi yang berpengaruh terhadap analisis prediktif nantinya, karena isi dari DoctorInCharge hanya XXXConfid.
Jadi total kolom menjadi 33 kolom yang tadinya berjumlah 35 kolom.

### Memeriksa Oulier
![Image](https://github.com/user-attachments/assets/003b2b9d-b186-48cc-9b23-6ebbc99547b3)

Apabila gambar tidak terlalu jelas, maka klik link berikut ini:
https://drive.google.com/file/d/1koUXvCsJUWHpnwCj1atbbbdmeoPT--tS/view?usp=sharing
**Insight:**
Biner/Kategorikal (misalnya: Diabetes, Depression, HeadInjury, CardiovascularDisease) muncul nilai outlier. outlier muncul karena nilai 1 sangat jarang dibandingkan dengan 0. Ini bukan karena nilainya tidak wajar, tetapi karena distribusinya tidak seimbang.
Dalam konteks medis untuk kolom yang mengandung outlier misalnya pada kolom Diabetes, Depression, HeadInjury, CardiovascularDisease tidak dihapus maupun diatasi karena hal tersebut merupakan kondisi nyata dari pasien. Apabila outlier tersebut diatasi atua hapus maka anantinya akan muncul bias pada data.

Berikut ini hasil dari evaluasi 4 model yang direpresentasikan ke dalam bentuk tabel:

### Univariate Analysis
a. Categorical Features
![Image](https://github.com/user-attachments/assets/a62912a3-ed90-4202-a545-752e75f3faa7)
![Image](https://github.com/user-attachments/assets/7c8163fb-1625-41f7-ba3a-4fece948cd1d)
![Image](https://github.com/user-attachments/assets/aa4789a9-d799-4ed2-954d-3773c0b82edc)
![Image](https://github.com/user-attachments/assets/e0dd05a3-b6ab-41dc-977e-40cf42d3cfe1)
![Image](https://github.com/user-attachments/assets/0e04841d-5fbe-4603-8447-e24a902a7aba)
![Image](https://github.com/user-attachments/assets/682372ff-0d44-488c-a747-0dbe96209123)
![Image](https://github.com/user-attachments/assets/0fd5d40d-432f-43a0-83ec-f677e5ea589b)
![Image](https://github.com/user-attachments/assets/dff70609-b0a4-4b62-8181-c60bdfa848d4)
![Image](https://github.com/user-attachments/assets/99c533ab-9f97-4d4f-a606-5f4024efd27e)
![Image](https://github.com/user-attachments/assets/c2aa96ae-ad39-4cf1-b181-4b8ccf9cf7f1)
![Image](https://github.com/user-attachments/assets/17362086-6c2d-4d89-aba6-552c526a541a)
![Image](https://github.com/user-attachments/assets/7948f5ad-7b8e-4c2d-85c6-ee0a6e40ef6f)
![Image](https://github.com/user-attachments/assets/6d9f3664-8903-436b-9014-f4be46b2d211)
![Image](https://github.com/user-attachments/assets/d0eb09e8-a5d8-47bf-b9f8-6061dddaeeeb)
![Image](https://github.com/user-attachments/assets/3db06fb4-577c-461c-a637-00cdf6ecaebb)
![Image](https://github.com/user-attachments/assets/b3306aba-bb3e-408b-913a-af32942f9888)
![Image](https://github.com/user-attachments/assets/fbeebe2b-559a-424d-add6-b5fe7ceef322)

**Insight:**
Gambar diatas merupakan visualisasi univariate analysis pada categorical features memberikan informasi bahwa Sebagian besar populasi sehat secara fisik, tetapi antara 15 dan 30 persen orang mulai mengalami gejala perilaku atau kognitif seperti pelupa, keluhan memori, dan depresi. Dua keluhan paling menonjol, pelupa dan gangguan memori, dapat menjadi sinyal awal gangguan kognitif. Meskipun tidak dominan, komponen risiko seperti depresi, cedera kepala, dan riwayat keluarga Alzheimer masih penting untuk dipelajari lebih lanjut dalam analisis prediktif.

b. Numerical features
![Image](https://github.com/user-attachments/assets/e9079af1-b40b-4d2d-92c2-6c911ae45f46) ![Image](https://github.com/user-attachments/assets/04727168-dfb8-4a67-8b18-9418a292adeb) 
![Image](https://github.com/user-attachments/assets/624a2971-67c0-42e1-848c-ba6bb325fac2) ![Image](https://github.com/user-attachments/assets/44e09e4a-d922-42ef-bb4f-4a5aefa13bb7)
![Image](https://github.com/user-attachments/assets/b092ec34-a880-443d-b1f6-0aac6d50c6b7)![Image](https://github.com/user-attachments/assets/524913f2-d4ab-496b-be12-d53bb3f2ffce)
![Image](https://github.com/user-attachments/assets/3d0532e1-fff1-4d13-b576-ca2a8c429254)![Image](https://github.com/user-attachments/assets/2e12ea1f-ea8c-4889-b489-3bdf60a0d7f8)
![Image](https://github.com/user-attachments/assets/c21f0d24-3613-4923-a4d1-5a6d30dbe526)![Image](https://github.com/user-attachments/assets/6db3c943-92e5-483c-a9cc-22b1b1652234)
![Image](https://github.com/user-attachments/assets/17fadf47-8bc1-4c7d-94dc-46806dee9b45)![Image](https://github.com/user-attachments/assets/c7f52ee3-3001-4d49-b1d2-0f510e6a94f1)
![Image](https://github.com/user-attachments/assets/3915f556-354d-4a10-9720-b5465cd0124e)![Image](https://github.com/user-attachments/assets/eeb2d522-c101-4750-ab83-2cd545c32c76)
![Image](https://github.com/user-attachments/assets/aec25cab-852d-4f74-abf4-927174f5cc6b)

**Insight:**
Gambar diatas merupakan gambar sampel yang diambil dari total 15 gambar. Untuk melihat gambar visualisasi lainnya dapat klik link ini: https://colab.research.google.com/drive/1UFz6Jr9Nb0FkhPnek5N7gQbT9jgbbqFq#scrollTo=G5pIqUwqiqdl 
Dari keseluruhan visualisasi univariate analysis pada numerik features didapat bahwa Distribusi data pada sebagian besar kolom tampak seimbang. Berbeda halnya dengan skor MMSE yang menunjukkan dua puncak distribusi, mengisyaratkan adanya dua klaster populasi yang berbeda.

#### Multivariate Analysis
a. Categorical Feature vs Target
![Image](https://github.com/user-attachments/assets/01e002e0-d297-496f-b04b-aceaed9a9bf5)
Berdasarkan visualisasi diatas didapat bahwa  Ada sejumlah variabel yang kuat berkorelasi dengan kemungkinan diagnosis Alzheimer, menurut hasil visualisasi multivariate antara fitur kategorikal dan variabel target diagnosis. Gejala kognitif seperti memory complaints, confusion, forgetfulness, disorientation, serta perubahan perilaku seperti personality changes dan behavioral problems menunjukkan korelasi yang signifikan dengan penyakit Alzheimer.
Gejala-gejala ini mencerminkan karakteristik klinis utama penyakit ini. Selain itu, tampak bahwa risiko diagnosis sebagian besar dipengaruhi oleh faktor medis seperti riwayat penyakit Alzheimer dalam keluarga, komorbiditas seperti diabetes, hipertensi, dan penyakit kardiovaskular, serta trauma kepala sebelumnya. Secara demografis, kelompok yang lebih rentan adalah perempuan dan kelompok dengan tingkat pendidikan yang lebih rendah. Ini mendukung teori bahwa pendidikan yang lebih tinggi dapat memberikan perlindungan kognitif (kognitif).

Untuk gambar lebih jelasnya dapat klik link berikut ini: https://drive.google.com/file/d/1n-kqyPV1Ow3zPq353MCj5kHNrmOxlLz2/view?usp=sharing

b. Numerical Feature
![Image](https://github.com/user-attachments/assets/d2710400-37b0-4ff5-88c9-019c2a389497)

**Insight:**
Berdasarkan visualisasi diatas menunjukkan bahwa sebagian besar fitur numerik memiliki distribusi unimodal dalam rentang khas masing-masing. Ini termasuk usia (60–90 tahun, puncak 70–75), BMI (20–35), konsumsi alkohol rendah, dan variabel tambahan seperti systolic BP (120–140) dan diastolic BP (70–90). Adanya skewness dan outlier, terutama pada konsumsi alkohol dan triglycerides, ditunjukkan oleh diagonal histogram+KDE. Hanya pasangan logis (Systolic BP vs. Diastolic BP dan CholesterolTotal vs. LDL/Triglycerides) menunjukkan korelasi moderat positif di off-diagonal scatterplots. Pasangan lain tampak acak, menunjukkan korelasi linier yang sangat lemah dan multikollinearitas rendah.
Untuk gambar lebih jelasnya dapat klik link berikut ini: https://drive.google.com/file/d/1QLfQWJPIVyW0N0qf2Jwiz3rvU-Ix7yNe/view?usp=sharing

### Korelasi Antar Fitur Numerikal
![Image](https://github.com/user-attachments/assets/b56ac720-0404-43fc-b519-06312151190f)

apabila gambar tidak terlalu jelas maka klik link ini: https://github.com/03febs/Gambar-submission/issues/8#issue-3089856524 

**Insight:**
Berdasarkan hasil visualisasi diatas menunjukkan bahwa fitur - fitur numerik hampir tidak saling berkorelasi. seperti Tidak ada hubungan linier yang nyata antara fitur numerik, karena semua nilai off-diagonal berada di antara –0,05 dan +0,05.
Contoh nilai tertinggi, meskipun relatif kecil:
•	Age vs SleepQuality ≈ +0.05
•	DietQuality vs SleepQuality ≈ +0.05
•	FunctionalAssessment vs ADL ≈ +0.05
dengan fitur - fitur numerik yang hampir tidak saling berkorelasi, ini menandakan situasi yang ideal untuk banyak algoritma machine learning—fitur independen dan tidak duplikatif

### Distibusi Kelas Variable Target
![Image](https://github.com/user-attachments/assets/34a50729-ae37-41d6-b5bb-6b761b76a84a)

**Insight:**
Berdasarkan Visualisasi diatas variabel target Diagnosis memiliki data yang imbalanced sehingga diperlukan proses oversampling data menggunakan SMOTE

## Data Preparation
Langkah – langkah tahapan data preparation sebagai berikut:

karena semua kolom dataset sudah bersifat numerik maka proses encoding akan dilewati.

### Pemisahan Fitur dan Target
```bash
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
```
Fitur (X) dipisahkan dari target (y), di mana target klasifikasi adalah kolom Diagnosis, yang menunjukkan label klasifikasi Alzheimer (Normal, Mild, Moderate, Severe). Pemisahan ini adalah langkah standar untuk memisahkan input dan output model.

### Mengatasi Kelas Imbalanced menggunakan SMOTE (Synthetic Minority Oversampling Technique)
```bash
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
```
- Catatan data awal memiliki distribusi kelas yang tidak seimbang. Misalnya, jumlah pasien yang terdiagnosis terdeteksi penyakit Alzheimer lebih sedikit dibandingkan dengan jumlah pasien yang tidak terdiagnosis terdeteksi penyakit Alzheimer 
- SMOTE digunakan untuk membuat data sintetis di kelas minoritas sehingga distribusi antar kelas menjadi seimbang.

Adapun alasan mengapa perlu dilakukan proses Oversampling menggunakan SMOTE sebagai berikut:
- Model klasifikasi cenderung bias terhadap kelas mayoritas.
- Penyeimbangan ini membantu model belajar pola dari semua kelas secara adil dan mencegah underfitting pada kelas minoritas.

### Split Data: Data Train dan Data Test
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Dataset dibagi menjadi 70% data train dan 30% data test, kemudian parameter random_state=42 digunakan agar hasil pembagian konsisten jika dijalankan ulang.
Adapun Alasan dilakukan proses ini adalah untuk menguji performa model pada data yang belum pernah dilihat, dan Mencegah overfitting karena evaluasi dilakukan pada data berbeda dari data pelatihan.

### Standarisasi
```bash
# Asumsikan df sudah didefinisikan sebelumnya
# 1. Pisahkan fitur numerik kontinu (exclude 'Diagnosis' dan biner)
numeric_features = [col for col in df.columns
                    if df[col].nunique() > 10 and col != 'Diagnosis']

# 2. Inisialisasi StandardScaler
scaler = StandardScaler()

# 3. Fit dan transform data training, lalu transform data testing (overwrite kolom aslinya)
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 4. (Opsional) Tampilkan preview hasil scaling
print("Hasil standardisasi (5 baris pertama):")
print(X_train[numeric_features].head())
```
- Hanya kolom numerik yang memiliki lebih dari 10 nilai unik dipilih, yang dianggap sebagai variabel kontinu, mencakup atribut seperti usia, BMI, MMSE, dan CholesterolTotal. Fitur biner (0/1) seperti merokok, depresi, dll. tidak distandardisasi karena tidak perlu.
- Mengubah distribusi data numerik menggunakan StandardScaler menjadi rata-rata = 0 dan standar deviasi = 1.
- Fiting hanya dilakukan pada data latihan, kemudian digunakan untuk mengubah data uji.

Adapun alasan melakukan proses standarisasi yaitu hanya fitur kontinu yang perlu distandardisasi, karena model seperti SVM sangat bergantung pada skala data. Selain itu, membantu mempercepat konvergensi dan stabilitas pelatihan model.

## Modeling

### Decision Tree
```bash
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
```
`max_depth=5`: membatasi kedalaman maksimum pohon, mencegah overfitting.
`min_samples_split=10`: node akan dibagi hanya jika memiliki setidaknya 10 sampel.
`random_state=42`: menjamin replikasi hasil.

**Kelebihan**:
- Interpretasi mudah (visual dan aturan keputusan eksplisit).
- Proses pelatihan cepat dan efisien.
- Cocok sebagai baseline model.

**Kekurangan**:
- Rentan overfitting jika tidak dikontrol kedalamannya.
- Algoritma ini kurang stabil ketika ada sedikit perubahan pada data dapat mengubah struktur pohon secara drastis.

### Random Forest
```bash
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
```
- `n_estimators=150`: jumlah pohon dalam ensemble.
- `max_depth=10`: membatasi kedalaman tiap pohon untuk menjaga generalisasi.
- `random_state=42`: untuk hasil replikasi.

**Kelebihan**:
Algorima ini memiliki performa yang lebih stabil dan akurat dibandingkan decision tree tunggal, kemudian tahan terhadap overfitting serta baik dalam menangani missing value dan data tidak seimbang

**Kekurangan**:
Algoritma Random Forest terdiri dari ratusan hingga ribuan pohon keputusan, dengan n_estimators=150, sehingga logika prediksi akhir adalah total dari semua pohon, yang berarti tidak ada jejak keputusan individu. Meskipun setiap pohon dapat dipahami dengan mudah, kombinasi hasil dari banyak pohon membuat alasan akhir prediksi menjadi tidak jelas.Selain itu, kesulitan dalam mengidentifikasi fitur dominan, meskipun Random Forest dapat mengukur kepentingan fitur (feature importance), hubungan non-linear antar fitur dan interaksi kompleks antar pohon sulit dijelaskan secara intuitif. Waktu komputasi relatif lebih tinggi daripada pohon tunggal

### Support Vector Machine (SVM)
```bash
sv_model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
```

- `kernel='linear'`: menggunakan fungsi kernel linier.
- `C=1.0`: parameter regularisasi, semakin kecil maka semakin toleran terhadap kesalahan.
- `gamma='scale'`: otomatis disesuaikan berdasarkan jumlah fitur.

**Kelebihan**:
- Efektif pada data berdimensi tinggi.
- Akurat pada dataset yang relatif kecil.

**Kekurangan**:
- Waktu pelatihan bisa lama pada dataset besar.
- Sensitif terhadap pemilihan parameter dan skala fitur.

### XGBoost 
```bash
xg_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```
- `n_estimators=150`: jumlah iterasi boosting.
- `learning_rate=0.1`: seberapa besar tiap model berkontribusi terhadap prediksi akhir.
- `max_depth=5`: membatasi kedalaman pohon dalam boosting.

**Kelebihan**:
- Akurasi tinggi dan unggul dalam kompetisi data science.
- Dapat menangani data tidak seimbang dan missing values.
- Mampu mengontrol overfitting melalui regularisasi.

**Kekurangan**:
- Waktu pelatihan lebih lama daripada Random Forest.
- Parameter tuning kompleks dan butuh eksperimen.

### Pemilihan Model Terbaik
Setelah seluruh model diuji menggunakan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1-score, hasil menunjukkan bahwa Algoritma XGBoost  secara konsisten memiliki kinerja terbaik di semua metrik utama. Memiliki akurasi tertinggi sebesar 92.45%, ketepatan yang seimbang dan tinggi untuk kedua kelas, dan skor F1 terbaik, yang mengimbangi false positives dan false negatives.

Adapun Alasan Memilih algoritma XGBoost sebagai model terbaik dikarenakan performa Prediksi Tinggi, XGBoost mengungguli semua model lain dalam akurasi dan semua metrik evaluasi, baik berat rata-rata maupun makro.  mendukung teknik regularisasi (L1, L2), XGBoost dapat mempelajari pola data kompleks tanpa overfitting. Tangguh terhadap Data Tidak Seimbang, Sangat fleksibel dan kuat karena kemampuan untuk menangani outlier, nilai yang hilang, dan interaksi fitur nonlinier, XGBoost adalah algoritma yang sangat populer dalam kompetisi data science.

## Evaluation
Model yang dilatih dievaluasi menggunakan Confusion Matrix. Ada berbagai jenis Confusion Matrix, seperti Precision, Recall, dan F1-Score. Confusion Matrix dan Akurasi dapat dibentuk dengan persamaan (1), (2) dan (3).

![Image](https://github.com/user-attachments/assets/f8d64ac2-cc9a-471d-95bf-aa1545eac94a)

Presisi merupakan perbandingan antara jumlah prediksi positif yang tepat (True Positive, TP) dan keseluruhan prediksi positif (True Positive + False Positive, FP). Presisi mengukur seberapa banyak prediksi yang positif dari model yang sebenarnya positif. Recall merupakan perbandingan antara jumlah prediksi positif yang benar (True Positive, TP) dan keseluruhan data yang sesungguhnya positif (True Positive + False Negative, FN). 
Recall mengukur seberapa efektif model dalam mengidentifikasi seluruh contoh positif. F1 Score merupakan rata-rata harmonis antara Presisi dan Recall. Metrik ini memberikan ilustrasi umum mengenai keseimbangan antara presisi dan recall. 

Berikut ini hasil dari evaluasi 4 model yang direpresentasikan ke dalam bentuk tabel:
| Model           | Accuracy | Precision (avg) | Recall (avg) | F1-score (avg) |
|----------------|----------|------------------|---------------|----------------|
| Decision Tree  | 0.8885   | 0.90             | 0.89          | 0.89           |
| Random Forest  | 0.9053   | 0.91             | 0.91          | 0.91           |
| SVM (Linear)   | 0.8273   | 0.83             | 0.83          | 0.83           |
| XGBoost        | **0.9245** | **0.93**         | **0.92**      | **0.92**        |

Adapun visualisasi bar chart yang digunakan untuk membandingkan hasil akurasi dari ke 4 model:
![Image](https://github.com/user-attachments/assets/59825aae-3c1c-45a7-a70a-a7377d8b5b72)

Berdasarkan hasil evaluasi model menggunakan confusion matrix. Berikut adalah kesimpulan mengenai model yang paling baik untuk prediksi penyakit Alzheimer
XGBoost Model: Memiliki akurasi tertinggi sebesar 92.45%, mengungguli semua model lain dengan F1-score untuk kedua kelas mencapai 0,92, sehingga mampu memberikan keseimbangan yang sangat baik antara precision dan recall. Dengan demikian, XGBoost sangat cocok untuk tugas klasifikasi, terutama untuk mendeteksi kedua kelas dengan akurat, termasuk untuk mendeteksi penyakit Alzheimer sejak dini.
Maka dari itu, XGBoost adalah pilihan terbaik untuk klasifikasi diagnosis penyakit Alzheimer karena memiliki kombinasi akurasi tinggi, kestabilan, keseimbangan metrik, dan keunggulan teknis.


## Referensi
[1] J. M. Basak et al., “Bacterial sepsis increases hippocampal fibrillar amyloid plaque load and neuroinflammation in a mouse model of Alzheimer’s disease,” Neurobiol Dis, vol. 152, May 2021, doi: 10.1016/j.nbd.2021.105292.

[2]	P. Scheltens et al., “Alzheimer’s disease,” Apr. 24, 2021, Elsevier B.V. doi: 10.1016/S0140-6736(20)32205-4.

[3]	D. J. Ziegel and E. Indra, “TINJAUAN KOMPARATIF LIMA METODE ANALISIS GERAKAN MATA UNTUK DETEKSI ALZHEIMER,” Jurnal Teknik Informasi dan Komputer (Tekinkom), vol. 7, no. 2, p. 573, Dec. 2024, doi: 10.37600/tekinkom.v7i2.1598.

[4]	N. Rustiana Dewi, A. Desiani, F. Salamah, Y. Andriani, M. Dan Ilmu, and P. Alam, “ALGORITMA K-NEAREST NEIGHBOR (K-NN) DAN SINGLE LAYER PERCEPTRON (SLP) UNTUK KLASIFIKASI PENYAKIT ALZHEIMER,” Jurnal Teknologi Terapan) |, vol. 9, no. 2, 2023, Accessed: May 25, 2025. [Online]. Available: https://jurnal.polindra.ac.id/index.php/jtt/article/view/407

[5]	A. A. Mortara, M. Permatasari, A. Desiani, Y. Andriani, and M. Arhami, “Perbandingan Algoritma C4.5 dan Adaptive Boosting dalam Klasifikasi Penyakit Alzheimer Comparison of C4.5 and Adaptive Boosting Algorithms in Alzheimer’s Disease Classification,” Jurnal Teknologi dan Informasi (JATI), vol. 13, 2023, doi: 10.34010/jati.v13i2.

[6]	C. Kavitha, V. Mani, S. R. Srividhya, O. I. Khalaf, and C. A. Tavera Romero, “Early-Stage Alzheimer’s Disease Prediction Using Machine Learning Models,” Front Public Health, vol. 10, Mar. 2022, doi: 10.3389/fpubh.2022.853294.





**---Ini adalah bagian akhir laporan---**



