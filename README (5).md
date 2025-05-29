
# Laporan Proyek Machine Learning - Febrie Tsani Sovranita




## Domain Proyek

Penyakit Alzheimer merupakan penyakit dengan kelainan neurodegeneratif yang biasanya sering terjadi pada lansia. Penyakit ini ditandai dengan adanya penurunan kognitif progresif dan kehilangan memori atau demensia. Sampai saat ini, penyebab penyakit Alzheimer belum diketahui secara pasti. namun terdapat beberapa hipotesis yang menyatakan bahwa etiologi dari penyakit tersebut adalah faktor genetik, stress oksidatif, akumulasi Aβ intraseluler dan ekstraseluler, eksitotoksik, peradangan, disfungsi mitokondria, perubahan sitoskeleton, komponen sinapsis, dan hilangnya neuron yang berperan penting dalam mencegah timbulnya suatu penyakit [1].Berdasarkan data Alzheimer's Disease International, menunjukkan bahwa pada tahun 2018, terdapat sekitar 50 juta kasus demensia di seluruh dunia. Angka ini diperkirakan akan meningkat tiga kali lipat hingga tahun 2050, dengan mayoritas kasus terjadi di negara berpenghasilan rendah dan menengah [2].

Gejala  gangguan memori dan penurunan kognitif memiliki dampak negatif pada kualitas  hidup  pasien  serta  beban  sistem Kesehatan [3], maka dari itu, Penelitian dan prediksi dini penyakit Alzheimer sangat penting karena hingga saat ini belum ada obat yang dapat menyembuhkan penyakit ini secara total[4] . Tahap awal diagnosis penyakit Alzheimer dapat meningkatkan efisiensi pengobatan penyakit ini. Diagnosa penyakit alzheimer dapat dibantu dengan pendeteksian dini. Pendekatan matematis pada data mining dapat digunakan untuk menganalisis data dan untuk mendeteksi atau memprediksi penyakit dini alzheimer. Salah satu metode yang digunakan dalam mendeteksi atau memprediksi penyakit dini alzheimer  adalah dengan pendekatan algoritma klasifikasi berbasis machine learning [5] 

Studi sebelumnya [6] menunjukkan bahwa  menunjukkan bahwa berbagai algoritma machine learning seperti Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting, dan Voting Classifier dapat digunakan untuk mengidentifikasi individu yang berisiko tinggi terkena Alzheimer pada tahap awal dengan tingkat akurasi yang baik. Penelitian ini menggunakan data MRI longitudinal dari dataset OASIS dan mengevaluasi kinerja model menggunakan parameter seperti Precision, Recall, Accuracy, dan F1-score. Hasilnya, algoritma Random Forest memberikan hasil terbaik dengan akurasi mencapai 86,92%.

Model prediktif dapat dibuat untuk mengidentifikasi individu yang berisiko sebelum munculnya gejala berat dengan menggunakan data klinis, hasil tes kognitif, dan biomarker dari dataset publik seperti Kaggle. Prediksi yang lebih baik dapat dibuat dengan menggabungkan data dari berbagai sumber, seperti faktor gaya hidup, genetika, dan kondisi kesehatan lainnya. Dengan menggunakan algoritma Decision Tree, Random Forest, SVM, dan XGBoost, penelitian ini mengusulkan sistem otomatisasi untuk analisis prediktor penyakit Alzheimer. Sistem ini tidak hanya mempercepat proses diagnosis dan mengurangi biaya, tetapi juga mengurangi kesalahan manusia untuk membantu tenaga medis membuat keputusan klinis yang lebih baik. Oleh karena itu, diharapkan bahwa penerapan teknik ini dapat membantu mengurangi angka mortalitas akibat Alzheimer melalui diagnosis dan intervensi yang lebih dini dan tepat sasaran.
## Bussisnes Understanding

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
- Menggunakan dan Membandingkan Beberapa Algoritma Klasifikasi Machine Learning, Mengembangkan model prediksi dengan menggunakan berbagai algoritma klasifikasi, seperti **Decision Tree, Random Forest, Support Vector Machine (SVM)**, dan **XGBoost**. Setiap algoritma akan diuji kinerjanya menggunakan dataset publik dari Kaggle, yang terdiri dari data seperti demografi, riwayat kesehatan, faktor gaya hidup, pengukuran klinis, penilaian kognitif dan fungsional, gejala, diagnosis penyakit Alzheimer, dan informasi lainnya. **Matriks Evaluasi** yang digunakan yaitu Akurasi, Precision, Recall, F1-score akan digunakan untuk membandingkan performa masing-masing model. Model dengan performa terbaik (misal: akurasi tertinggi) akan dipilih sebagai solusi utama.

- Meningkatkan Akurasi Model dengan Oversampling Menggunakan SMOTE, Proses oversampling menggunakan teknik Synthetic Minority Oversampling Technique (SMOTE) untuk mengatasi ketidakseimbangan data, atau ketidakseimbangan kelas, yang sering terjadi pada data medis. SMOTE melakukan ini dengan mengumpulkan sampel sintetis dari kelas minoritas sehingga proporsi kelas menjadi lebih seimbang.


- Seleksi Fitur dengan Menghapus Variabel yang Tidak Berkontribusi, Dilakukan proses seleksi fitur untuk mengidentifikasi dan menghilangkan variabel yang tidak memberikan kontribusi signifikan terhadap proses pelatihan dan prediksi model. Penghapusan variabel yang tidak relevan ini dapat mengurangi kompleksitas model, mempercepat proses pelatihan, dan berpotensi meningkatkan akurasi model prediksi.



## Data Understanding

### Data Loading
Pada tahap ini dilakukan proses mengambil dan memuat data mentah ke lingkungan pemrosesan agar siap untuk dieksplorasi, dibersihkan, dan diolah lebih lanjut dalam tahapan Machine Learning.

![Image](https://github.com/user-attachments/assets/457ae024-08e1-491b-8284-a4db8a636a29)
| PatientID | Age | Gender | Ethnicity | EducationLevel | BMI       | Smoking | AlcoholConsumption | PhysicalActivity | DietQuality | ... | Forgetfulness | Diagnosis | DoctorInCharge |
|-----------|-----|--------|-----------|----------------|-----------|---------|---------------------|------------------|--------------|-----|----------------|-----------|----------------|
| 4751      | 73  | 0      | 0         | 2              | 22.927749 | 0       | 13.297218           | 6.327112         | 1.347214     | ... | 0              | 0         | XXXConfid      |
| 4752      | 89  | 0      | 0         | 0              | 26.827681 | 0       | 4.542524            | 7.619885         | 0.518767     | ... | 1              | 0         | XXXConfid      |
| 4753      | 73  | 0      | 3         | 1              | 17.795882 | 0       | 19.555085           | 7.844988         | 1.826335     | ... | 0              | 0         | XXXConfid      |
| 4754      | 74  | 1      | 0         | 1              | 33.800817 | 1       | 12.209266           | 8.428001         | 7.435604     | ... | 0              | 0         | XXXConfid      |
| 4755      | 89  | 0      | 0         | 0              | 20.716974 | 0       | 18.454356           | 6.310461         | 0.795498     | ... | 0              | 0         | XXXConfid      |
| ...       | ... | ...    | ...       | ...            | ...       | ...     | ...                 | ...              | ...          | ... | ...            | ...       | ...            |
| 6895      | 61  | 0      | 0         | 1              | 39.121757 | 0       | 1.561126            | 4.049964         | 6.555306     | ... | 0              | 1         | XXXConfid      |
| 6896      | 75  | 0      | 0         | 2              | 17.857903 | 0       | 18.767261           | 1.360667         | 2.904662     | ... | 0              | 1         | XXXConfid      |
| 6897      | 77  | 0      | 0         | 1              | 15.476479 | 0       | 4.594670            | 9.886002         | 8.120025     | ... | 0              | 1         | XXXConfid      |
| 6898      | 78  | 1      | 3         | 1              | 15.299911 | 0       | 8.674505            | 6.354282         | 1.263427     | ... | 1              | 1         | XXXConfid      |
| 6899      | 72  | 0      | 0         | 2              | 33.289738 | 0       | 7.890703            | 6.570993         | 7.941404     | ... | 1              | 0         | XXXConfid      |


Data yang digunakan merupakan dataset public Alzheimer's Disease dari Kaggle. Dataset ini berisi informasi kesehatan yang luas untuk 2.149 pasien, masing-masing diidentifikasi secara unik dengan ID mulai dari 4751 hingga 6900. Dataset ini mencakup rincian demografis, faktor gaya hidup, riwayat kesehatan, pengukuran klinis, penilaian kognitif dan fungsional, gejala, dan diagnosis Penyakit Alzheimer. Data ini sangat ideal bagi para peneliti dan ilmuwan data yang ingin mengeksplorasi faktor-faktor yang terkait dengan Alzheimer, mengembangkan model prediktif, dan melakukan analisis statistik.

Sumber data:
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

### Exploratory Data Analysis (EDA)
### Deskripsi Variabel

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



### Deskripsi Statistik Untuk Data Numerik

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



### Memeriksa Oulier
![Image](https://github.com/user-attachments/assets/7405f04f-9431-4f11-b71d-ba4723b7e51e)

Apabila gambar tidak terlalu jelas, maka klik link berikut ini:
https://drive.google.com/file/d/1fp_9szMFlil997nb5KzPcO-FdrncbrD2/view?usp=sharing
**Insight:**
Biner/Kategorikal (misalnya: Diabetes, Depression, HeadInjury, CardiovascularDisease) muncul nilai outlier. outlier muncul karena nilai 1 sangat jarang dibandingkan dengan 0. Ini bukan karena nilainya tidak wajar, tetapi karena distribusinya tidak seimbang.
Dalam konteks medis untuk kolom yang mengandung outlier misalnya pada kolom Diabetes, Depression, HeadInjury, CardiovascularDisease tidak dihapus maupun diatasi karena hal tersebut merupakan kondisi nyata dari pasien. Apabila outlier tersebut diatasi atua hapus maka anantinya akan muncul bias pada data.

Berikut ini hasil dari evaluasi 4 model yang direpresentasikan ke dalam bentuk tabel:

### Univariate Analysis
```bash
# Membagi fitur menjadi numerikal dan kategorikal
num_features = [col for col in df.columns if df[col].nunique() > 10 and col != 'Diagnosis']
cat_features = [col for col in df.columns if col not in num_features and col != 'Diagnosis']

```
Karena dataset asli dari kagglenya sudah hampir berbentuk numerical maka untuk memudahkan proses EDA dilakukan terlebih dahulu custom label dengan mengubah beberapa variabel numerical menjadi kategorical untuk sementara

a. Categorical Features

| ![](https://github.com/user-attachments/assets/c6cacadb-a3fa-40e4-ac97-e8996942cdc0) | ![](https://github.com/user-attachments/assets/72eb2818-c055-4f7a-bba1-733c2acc9844) | ![](https://github.com/user-attachments/assets/79f141c5-3444-4054-8a61-d928e8877a1c) |
|---|---|---|
| ![](https://github.com/user-attachments/assets/b0026d5c-633b-405c-9fa1-a8a33e0a4dfe) | ![](https://github.com/user-attachments/assets/0b5b786b-2230-4b48-a9ce-c9f3ea1d1f3e) | ![](https://github.com/user-attachments/assets/b0d7de13-bbfd-4ae8-9e0b-455f3408921d) |
| ![](https://github.com/user-attachments/assets/4dcefb87-e416-4e9b-8dae-d170eb1e330c) | ![](https://github.com/user-attachments/assets/9cfef77d-0f05-406f-9da8-132326af72a6) | ![](https://github.com/user-attachments/assets/199d6f00-5bbb-4ac3-bbe3-4273b0ff8c02) |
| ![](https://github.com/user-attachments/assets/6f8806d2-54c9-4514-ab01-df79d2935d75) | ![](https://github.com/user-attachments/assets/26d34ba1-7e85-4890-ab2a-dda86f28cc5e) | ![](https://github.com/user-attachments/assets/b024f1d0-e40c-4db0-9437-6f3295db77b5) |
| ![](https://github.com/user-attachments/assets/9e2afd73-3424-41da-8843-c06099f1d103) | ![](https://github.com/user-attachments/assets/a308b320-712d-4c5c-84d9-13fdaa09c285) | ![](https://github.com/user-attachments/assets/5c439a01-006e-490a-bc75-5253abfc8ebc) |
| ![](https://github.com/user-attachments/assets/d7b6a8ea-e0ca-40e5-94c4-d60f800f1cf7) | ![](https://github.com/user-attachments/assets/17fab614-ba05-4b15-979d-975526cf96b2) | ![](https://github.com/user-attachments/assets/d939b94a-12de-470e-a4bb-41d7d915cce3) |


**Insight:**
Gambar diatas merupakan visualisasi univariate analysis pada categorical features memberikan informasi bahwa Sebagian besar populasi sehat secara fisik, tetapi antara 15 dan 30 persen orang mulai mengalami gejala perilaku atau kognitif seperti pelupa, keluhan memori, dan depresi. Dua keluhan paling menonjol, pelupa dan gangguan memori, dapat menjadi sinyal awal gangguan kognitif. Meskipun tidak dominan, komponen risiko seperti depresi, cedera kepala, dan riwayat keluarga Alzheimer masih penting untuk dipelajari lebih lanjut dalam analisis prediktif.

b. Numerical features
| ![](https://github.com/user-attachments/assets/e9079af1-b40b-4d2d-92c2-6c911ae45f46) | ![](https://github.com/user-attachments/assets/04727168-dfb8-4a67-8b18-9418a292adeb) | ![](https://github.com/user-attachments/assets/624a2971-67c0-42e1-848c-ba6bb325fac2) |
|---|---|---|
| ![](https://github.com/user-attachments/assets/44e09e4a-d922-42ef-bb4f-4a5aefa13bb7) | ![](https://github.com/user-attachments/assets/b092ec34-a880-443d-b1f6-0aac6d50c6b7) | ![](https://github.com/user-attachments/assets/524913f2-d4ab-496b-be12-d53bb3f2ffce) |
| ![](https://github.com/user-attachments/assets/3d0532e1-fff1-4d13-b576-ca2a8c429254) | ![](https://github.com/user-attachments/assets/2e12ea1f-ea8c-4889-b489-3bdf60a0d7f8) | ![](https://github.com/user-attachments/assets/c21f0d24-3613-4923-a4d1-5a6d30dbe526) |
| ![](https://github.com/user-attachments/assets/6db3c943-92e5-483c-a9cc-22b1b1652234) | ![](https://github.com/user-attachments/assets/17fadf47-8bc1-4c7d-94dc-46806dee9b45) | ![](https://github.com/user-attachments/assets/c7f52ee3-3001-4d49-b1d2-0f510e6a94f1) |
| ![](https://github.com/user-attachments/assets/3915f556-354d-4a10-9720-b5465cd0124e) | ![](https://github.com/user-attachments/assets/eeb2d522-c101-4750-ab83-2cd545c32c76) | ![](https://github.com/user-attachments/assets/aec25cab-852d-4f74-abf4-927174f5cc6b) |
| ![](https://github.com/user-attachments/assets/8941a84f-b88a-4d73-beef-88a5d166869a) |   |   |



**Insight:**
Gambar diatas merupakan gambar sampel yang diambil dari total 15 gambar. Untuk melihat gambar visualisasi lainnya dapat klik link ini: https://colab.research.google.com/drive/1UFz6Jr9Nb0FkhPnek5N7gQbT9jgbbqFq#scrollTo=G5pIqUwqiqdl 
Dari keseluruhan visualisasi univariate analysis pada numerik features didapat bahwa Distribusi data pada sebagian besar kolom tampak seimbang. Berbeda halnya dengan skor MMSE yang menunjukkan dua puncak distribusi, mengisyaratkan adanya dua klaster populasi yang berbeda.

#### Multivariate Analysis
a. Categorical Feature vs Target
![Image](https://github.com/user-attachments/assets/df97e5a2-1bc9-42b6-bad0-06cc93954131)
Berdasarkan visualisasi diatas didapat bahwa  Ada sejumlah variabel yang kuat berkorelasi dengan kemungkinan diagnosis Alzheimer, menurut hasil visualisasi multivariate antara fitur kategorikal dan variabel target diagnosis. Gejala kognitif seperti memory complaints, confusion, forgetfulness, disorientation, serta perubahan perilaku seperti personality changes dan behavioral problems menunjukkan korelasi yang signifikan dengan penyakit Alzheimer.
Gejala-gejala ini mencerminkan karakteristik klinis utama penyakit ini. Selain itu, tampak bahwa risiko diagnosis sebagian besar dipengaruhi oleh faktor medis seperti riwayat penyakit Alzheimer dalam keluarga, komorbiditas seperti diabetes, hipertensi, dan penyakit kardiovaskular, serta trauma kepala sebelumnya. Secara demografis, kelompok yang lebih rentan adalah perempuan dan kelompok dengan tingkat pendidikan yang lebih rendah. Ini mendukung teori bahwa pendidikan yang lebih tinggi dapat memberikan perlindungan kognitif (kognitif).

Untuk gambar lebih jelasnya dapat klik link berikut ini: https://drive.google.com/file/d/1EguD4tobG5QKjrAMfMz4E-w8wqFCiaBI/view?usp=sharing

b. Numerical Feature
![Image](https://github.com/03febs/Gambar-submission/blob/main/download%20(76).png?raw=true)

**Insight:**
Berdasarkan visualisasi diatas menunjukkan bahwa sebagian besar fitur numerik memiliki distribusi unimodal dalam rentang khas masing-masing. Ini termasuk usia (60–90 tahun, puncak 70–75), BMI (20–35), konsumsi alkohol rendah, dan variabel tambahan seperti systolic BP (120–140) dan diastolic BP (70–90). Adanya skewness dan outlier, terutama pada konsumsi alkohol dan triglycerides, ditunjukkan oleh diagonal histogram+KDE. Hanya pasangan logis (Systolic BP vs. Diastolic BP dan CholesterolTotal vs. LDL/Triglycerides) menunjukkan korelasi moderat positif di off-diagonal scatterplots. Pasangan lain tampak acak, menunjukkan korelasi linier yang sangat lemah dan multikollinearitas rendah.
Untuk gambar lebih jelasnya dapat klik link berikut ini: https://drive.google.com/file/d/1aZImi2ldNw3pB0ncAQuXmAJPFiz64e9B/view?usp=sharing

### Korelasi Antar Fitur Numerikal
![Image](https://github.com/user-attachments/assets/476fea8e-8873-4a87-9969-14c86d047a84)

apabila gambar tidak terlalu jelas maka klik link ini: https://drive.google.com/file/d/1wwwcTWQlQXIkg4Zik1gVop4VkVBsObYj/view?usp=sharing

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


### Menghapus Kolom Yang Tidak Penting
![Image](https://github.com/user-attachments/assets/98968257-42fb-43e1-ada0-131bd07c1081)

**Insight:**
kolom `PatientID` dihapus karena tidak memberikan informasi prediktif karena sifatnya yang unik dan tidan berulang.
kolom `DoctorInCharge` dihapus alasannya tidak memberikan informasi yang berpengaruh terhadap analisis prediktif nantinya, karena isi dari DoctorInCharge hanya XXXConfid.
Jadi total kolom menjadi 33 kolom yang tadinya berjumlah 35 kolom.

### Pemisahan Fitur dan Target
```bash
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
```
Fitur (X) dipisahkan dari target (y), di mana target klasifikasi adalah kolom Diagnosis, yang menunjukkan label klasifikasi Alzheimer (Normal, Mild, Moderate, Severe). Pemisahan ini adalah langkah standar untuk memisahkan input dan output model.

### Train-Test-Split: Data Train dan Data Test
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Dataset dibagi menjadi 70% data train dan 30% data test, kemudian parameter random_state=42 digunakan agar hasil pembagian konsisten jika dijalankan ulang.
Adapun Alasan dilakukan proses ini adalah untuk menguji performa model pada data yang belum pernah dilihat, dan Mencegah overfitting karena evaluasi dilakukan pada data berbeda dari data pelatihan.

### Mengatasi Kelas Imbalanced menggunakan SMOTE (Synthetic Minority Oversampling Technique)
```bash
# Terapkan SMOTE hanya pada data latih
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```
- Catatan data awal memiliki distribusi kelas yang tidak seimbang. Misalnya, jumlah pasien yang terdiagnosis terdeteksi penyakit Alzheimer lebih sedikit dibandingkan dengan jumlah pasien yang tidak terdiagnosis terdeteksi penyakit Alzheimer 
- SMOTE digunakan untuk membuat data sintetis di kelas minoritas sehingga distribusi antar kelas menjadi seimbang.

Adapun alasan mengapa perlu dilakukan proses Oversampling menggunakan SMOTE sebagai berikut:
- Model klasifikasi cenderung bias terhadap kelas mayoritas.
- Penyeimbangan ini membantu model belajar pola dari semua kelas secara adil dan mencegah underfitting pada kelas minoritas.



### Standarisasi
```bash
# Asumsikan df sudah didefinisikan sebelumnya
# 1. Pisahkan fitur numerik kontinu (exclude 'Diagnosis' dan biner)
numeric_features = [col for col in df.columns
                    if df[col].nunique() > 10 and col != 'Diagnosis']

# 2. Inisialisasi StandardScaler
scaler = StandardScaler()

# 3. Fit dan transform data training, lalu transform data testing (overwrite kolom aslinya)
X_train_resampled[numeric_features] = scaler.fit_transform(X_train_resampled[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 4. (Opsional) Tampilkan preview hasil scaling
print("Hasil standardisasi (5 baris pertama):")
print(X_train_resampled[numeric_features].head())
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
- `max_depth=5`: membatasi kedalaman maksimum pohon, mencegah overfitting.
- `min_samples_split=10`: node akan dibagi hanya jika memiliki setidaknya 10 sampel.
- `random_state=42`: nilai seed untuk memastikan hasil yang dapat direproduksi.

**Kelebihan**:
- Interpretasi mudah (visual dan aturan keputusan eksplisit).
- Proses pelatihan cepat dan efisien.
- Cocok sebagai baseline model.

**Kekurangan**:
- Rentan overfitting jika tidak dikontrol kedalamannya.
- Algoritma ini kurang stabil ketika ada sedikit perubahan pada data dapat mengubah struktur pohon secara drastis.

**Cara Kerja:**

Decision Tree bekerja dengan cara memecah dataset ke dalam subset-subset yang lebih kecil berdasarkan fitur input. Proses pemisahan dilakukan secara rekursif, membentuk struktur seperti pohon dengan node cabang (decision node) dan daun (leaf node).

Pada setiap node, algoritma memilih fitur terbaik untuk membagi data berdasarkan kriteria seperti Gini Impurity atau Information Gain. Model terus melakukan pembagian hingga mencapai kedalaman maksimum (misalnya max_depth=5), atau ketika jumlah minimum sampel dalam sebuah node tidak lagi mencukupi untuk pemisahan (misalnya min_samples_split=10).

Keputusan akhir dibuat berdasarkan mayoritas kelas di daun tempat suatu data berakhir.


### Random Forest
```bash
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
```
- `n_estimators=150`: jumlah pohon dalam ensemble.
- `max_depth=10`: membatasi kedalaman tiap pohon untuk menjaga generalisasi.
- `random_state=42`: nilai seed untuk memastikan hasil yang dapat direproduksi.

**Kelebihan**:

Algorima ini memiliki performa yang lebih stabil dan akurat dibandingkan decision tree tunggal, kemudian tahan terhadap overfitting serta baik dalam menangani missing value dan data tidak seimbang

**Kekurangan**:

Algoritma Random Forest terdiri dari ratusan hingga ribuan pohon keputusan, dengan n_estimators=150, sehingga logika prediksi akhir adalah total dari semua pohon, yang berarti tidak ada jejak keputusan individu. Meskipun setiap pohon dapat dipahami dengan mudah, kombinasi hasil dari banyak pohon membuat alasan akhir prediksi menjadi tidak jelas.Selain itu, kesulitan dalam mengidentifikasi fitur dominan, meskipun Random Forest dapat mengukur kepentingan fitur (feature importance), hubungan non-linear antar fitur dan interaksi kompleks antar pohon sulit dijelaskan secara intuitif. Waktu komputasi relatif lebih tinggi daripada pohon tunggal.

**Cara Kerja**

Random Forest membangun banyak pohon keputusan (decision tree) secara paralel, dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.
Prinsip utama dari Random Forest yaitu:
- Menggunakan teknik bagging (bootstrap aggregating), yaitu membuat subset acak dari data pelatihan dengan pengambilan sampel ulang.
- Setiap pohon dilatih pada subset yang berbeda dan hanya menggunakan sebagian fitur acak pada tiap node, sehingga menghasilkan variasi antar pohon.
- Prediksi akhir ditentukan melalui voting mayoritas (untuk klasifikasi).




### Support Vector Machine (SVM)
```bash
sv_model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
```

- `kernel='linear'`: menggunakan fungsi kernel linier.
- `C=1.0`: parameter regularisasi, semakin kecil maka semakin toleran terhadap kesalahan.
- `gamma='scale'`: otomatis disesuaikan berdasarkan jumlah fitur.
- `random_state=42`: nilai seed untuk memastikan hasil yang dapat direproduksi.

**Kelebihan**:
- Efektif pada data berdimensi tinggi.
- Akurat pada dataset yang relatif kecil.

**Kekurangan**:
- Waktu pelatihan bisa lama pada dataset besar.
- Sensitif terhadap pemilihan parameter dan skala fitur.

**Cara Kerja**

SVM bekerja dengan mencari hyperplane optimal yang memisahkan data dari kelas berbeda dengan margin maksimal. Untuk kasus linier, seperti pada model dengan kernel='linear', algoritma mencari garis (untuk data 2D) atau bidang (untuk data berdimensi lebih tinggi) yang memisahkan kelas-kelas secara optimal.
Fitur penting dari Algoritma SVM yaitu:
- Margin: Jarak antara hyperplane dan titik data terdekat dari masing-masing kelas.
- Support vectors: Titik-titik data yang berada di dekat margin dan menentukan posisi hyperplane.
- Parameter C mengatur trade-off antara margin yang lebar dan kesalahan klasifikasi pada data pelatihan.

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
- `use_label_encoder=False`: menonaktifkan label encoder bawaan XGBoost agar tidak menimbulkan peringatan versi terbaru.
- `eval_metric='logloss'`: metrik evaluasi yang digunakan adalah log loss, cocok untuk klasifikasi biner atau multi-kelas.
- `n_estimators=150`: jumlah total pohon (iterasi boosting) yang akan dibangun.
- `learning_rate=0.1`: kecepatan model dalam belajar; semakin kecil nilainya, semakin hati-hati model memperbaiki kesalahan.
- `max_depth=5`: kedalaman maksimum dari setiap pohon keputusan, digunakan untuk mengontrol kompleksitas model.
- `random_state=42`: nilai seed untuk memastikan hasil yang dapat direproduksi.

**Kelebihan**:
- Akurasi tinggi dan unggul dalam kompetisi data science.
- Dapat menangani data tidak seimbang dan missing values.
- Mampu mengontrol overfitting melalui regularisasi.

**Kekurangan**:
- Waktu pelatihan lebih lama daripada Random Forest.
- Parameter tuning kompleks dan butuh eksperimen.

**Cara Kerja**

XGBoost adalah algoritma boosting yang membangun model secara iteratif. Pada setiap iterasi, XGBoost membangun pohon baru yang berusaha memperbaiki kesalahan dari model sebelumnya, berdasarkan gradient descent pada fungsi loss.
Karakteristik utama Algoritma XGBoost yaitu:
- Menggunakan teknik gradient boosting, yaitu menggabungkan banyak pohon lemah (weak learners) menjadi model kuat.
- Setiap pohon baru dibuat untuk meminimalkan residual error dari prediksi sebelumnya.
- Parameter seperti learning_rate=0.1, n_estimators=150, dan max_depth=5 mengatur seberapa cepat model belajar, jumlah total pohon, dan kedalaman maksimum pohon.


### Pemilihan Model Terbaik
Setelah seluruh model diuji menggunakan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1-score, hasil menunjukkan bahwa Algoritma XGBoost  secara konsisten memiliki kinerja terbaik di semua metrik utama. Memiliki akurasi tertinggi sebesar 91.01%, dan skor F1 terbaik, yang mengimbangi false positives dan false negatives.

Adapun Alasan Memilih algoritma XGBoost sebagai model terbaik dikarenakan performa Prediksi Tinggi, XGBoost mengungguli semua model lain dalam akurasi dan semua metrik evaluasi, baik berat rata-rata maupun makro.  mendukung teknik regularisasi (L1, L2), XGBoost dapat mempelajari pola data kompleks tanpa overfitting. Tangguh terhadap Data Tidak Seimbang, Sangat fleksibel dan kuat karena kemampuan untuk menangani outlier, nilai yang hilang, dan interaksi fitur nonlinier, XGBoost adalah algoritma yang sangat populer dalam kompetisi data science.



## Evaluation
Model yang dilatih dievaluasi menggunakan Confusion Matrix. Ada berbagai jenis Confusion Matrix, seperti Precision, Recall, dan F1-Score. Confusion Matrix dan Akurasi dapat dibentuk dengan persamaan (1), (2) dan (3).

![Image](https://github.com/user-attachments/assets/f8d64ac2-cc9a-471d-95bf-aa1545eac94a)

Presisi merupakan perbandingan antara jumlah prediksi positif yang tepat (True Positive, TP) dan keseluruhan prediksi positif (True Positive + False Positive, FP). Presisi mengukur seberapa banyak prediksi yang positif dari model yang sebenarnya positif. Recall merupakan perbandingan antara jumlah prediksi positif yang benar (True Positive, TP) dan keseluruhan data yang sesungguhnya positif (True Positive + False Negative, FN). 
Recall mengukur seberapa efektif model dalam mengidentifikasi seluruh contoh positif. F1 Score merupakan rata-rata harmonis antara Presisi dan Recall. Metrik ini memberikan ilustrasi umum mengenai keseimbangan antara presisi dan recall. 


### Confusion Matrix

![Image](https://github.com/user-attachments/assets/a5ad7c91-06d2-4d5d-bc2d-c923ee18aae4)
**Insight:**

Dari visualisasi terlihat bahwa model mampu mengklasifikasikan data dengan cukup baik. Sebanyak 369 data dari kelas 0 diprediksi dengan benar (true negative), dan 215 data dari kelas 1 juga diprediksi dengan benar (true positive). Namun, terdapat pula sejumlah kesalahan: 32 data dari kelas 0 salah diprediksi sebagai kelas 1 (false positive), dan 29 data dari kelas 1 salah diprediksi sebagai kelas 0 (false negative).
![Image](https://github.com/user-attachments/assets/03ceb574-d112-4bf9-a293-9f0a2a5c47db)
**Insight:**

Model Random Forest mencatatkan hasil yang mirip dengan XGBoost, yaitu 377 True Negative dan 195 True Positive. Namun, False Negative-nya sedikit lebih tinggi (49), yang berarti model ini sedikit lebih sering gagal mengenali kelas positif dibanding XGBoost, meskipun tetap memiliki performa yang cukup baik secara keseluruhan.
![Image](https://github.com/user-attachments/assets/1fc539ba-8bc0-4ce6-968b-2d9bfb6850d6)
**Insight:**

SVM dengan kernel linear menghasilkan 329 True Negative dan 191 True Positive, dengan False Positive sebesar 72 dan False Negative sebanyak 53. Angka False Positive dan False Negative cukup tinggi dibanding model lain, menandakan bahwa model ini kurang akurat, terutama dalam membedakan kelas positif dan negatif secara tepat.
![Image](https://github.com/user-attachments/assets/0f41fd2e-115b-43bb-87ee-acc508e2ae22)
**Insight:**

Model Random Forest mencatatkan hasil yang mirip dengan XGBoost, yaitu 377 True Negative dan 195 True Positive. Namun, False Negative-nya sedikit lebih tinggi (49), yang berarti model ini sedikit lebih sering gagal mengenali kelas positif dibanding XGBoost, meskipun tetap memiliki performa yang cukup baik secara keseluruhan.

Berikut ini hasil dari evaluasi 4 model yang direpresentasikan ke dalam bentuk tabel:

| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------|--------------|---------------|------------|--------------|
| Decision Tree         | 0.9054       | 0.91          | 0.91       | 0.91         |
| Random Forest         | 0.8868       | 0.89          | 0.89       | 0.89         |
| SVM (Linear Kernel)   | 0.8062       | 0.81          | 0.81       | 0.81         |
| **XGBoost**               | **0.9101**       | **0.91**          | **0.91**       | **0.91**         |

> Catatan: Nilai precision, recall, dan f1-score diambil dari weighted average hasil classification report untuk memastikan konsistensi dengan notebook evaluasi model.



Adapun visualisasi bar chart yang digunakan untuk membandingkan hasil akurasi dari ke 4 model:
![Image](https://github.com/user-attachments/assets/430c1568-1063-4258-b8ee-c3717c9a9dc0)

Berdasarkan hasil evaluasi model menggunakan confusion matrix. Berikut adalah kesimpulan mengenai model yang paling baik untuk prediksi penyakit Alzheimer

Model XGBoost menunjukkan performa terbaik dengan akurasi tertinggi sebesar **91.01%**, serta nilai precision, recall, dan f1-score yang masing-masing mencapai 0.91. Hal ini menunjukkan bahwa model ini mampu mengklasifikasikan kedua kelas dengan baik dan seimbang, baik untuk mendeteksi kasus positif maupun negatif pada diagnosis penyakit Alzheimer.

Meskipun Decision Tree juga menunjukkan performa yang sangat baik dengan akurasi 90.54% dan metrik evaluasi lainnya yang sama kuat (precision, recall, dan f1-score sebesar 0.91), model XGBoost sedikit lebih unggul dari sisi akurasi keseluruhan.

Dengan demikian, XGBoost sangat cocok untuk tugas klasifikasi, terutama untuk mendeteksi kedua kelas dengan akurat, termasuk untuk mendeteksi penyakit Alzheimer sejak dini. Maka dari itu, XGBoost adalah pilihan terbaik untuk klasifikasi diagnosis penyakit Alzheimer karena memiliki kombinasi akurasi tinggi, kestabilan, keseimbangan metrik, dan keunggulan teknis.
## Referensi


[1] J. M. Basak et al., “Bacterial sepsis increases hippocampal fibrillar amyloid plaque load and neuroinflammation in a mouse model of Alzheimer’s disease,” Neurobiol Dis, vol. 152, May 2021, doi: 10.1016/j.nbd.2021.105292.

[2]	P. Scheltens et al., “Alzheimer’s disease,” Apr. 24, 2021, Elsevier B.V. doi: 10.1016/S0140-6736(20)32205-4.

[3]	D. J. Ziegel and E. Indra, “TINJAUAN KOMPARATIF LIMA METODE ANALISIS GERAKAN MATA UNTUK DETEKSI ALZHEIMER,” Jurnal Teknik Informasi dan Komputer (Tekinkom), vol. 7, no. 2, p. 573, Dec. 2024, doi: 10.37600/tekinkom.v7i2.1598.

[4]	N. Rustiana Dewi, A. Desiani, F. Salamah, Y. Andriani, M. Dan Ilmu, and P. Alam, “ALGORITMA K-NEAREST NEIGHBOR (K-NN) DAN SINGLE LAYER PERCEPTRON (SLP) UNTUK KLASIFIKASI PENYAKIT ALZHEIMER,” Jurnal Teknologi Terapan) |, vol. 9, no. 2, 2023, Accessed: May 25, 2025. [Online]. Available: https://jurnal.polindra.ac.id/index.php/jtt/article/view/407

[5]	A. A. Mortara, M. Permatasari, A. Desiani, Y. Andriani, and M. Arhami, “Perbandingan Algoritma C4.5 dan Adaptive Boosting dalam Klasifikasi Penyakit Alzheimer Comparison of C4.5 and Adaptive Boosting Algorithms in Alzheimer’s Disease Classification,” Jurnal Teknologi dan Informasi (JATI), vol. 13, 2023, doi: 10.34010/jati.v13i2.

[6]	C. Kavitha, V. Mani, S. R. Srividhya, O. I. Khalaf, and C. A. Tavera Romero, “Early-Stage Alzheimer’s Disease Prediction Using Machine Learning Models,” Front Public Health, vol. 10, Mar. 2022, doi: 10.3389/fpubh.2022.853294.


**---Ini adalah bagian akhir laporan---**