# Laporan Proyek Machine Learning - Laptop Price Prediction
**Disusun oleh : Muhamad Alan Dharma Saputro Setiawan**
Laporan ini disusun untuk memenuhi submission dicoding proyek pertama predictive analytics. Proyek ini membangun model machine learning yang dapat memprediksi harga laptop berdasarkan fitur-fitur yang terdapat didalamnya

## Domain Proyek
Latar belakang proyek ini adalah untuk menjawab tantangan dalam memprediksi harga laptop secara akurat, terutama dalam konteks pasar elektronik yang dinamis dan permintaan yang berfluktuasi. Seiring dengan perkembangan teknologi dan pergeseran kondisi pasar, kemampuan untuk memprediksi harga laptop secara akurat menjadi sangat penting bagi produsen, peritel, dan konsumen. Dengan memanfaatkan wawasan dari penelitian sebelumnya dan menggunakan teknik pembelajaran mesin yang canggih, proyek ini berupaya mengembangkan model prediksi yang dapat memberikan perkiraan harga laptop yang tepat berdasarkan berbagai fitur dan faktor. Tujuan utamanya adalah untuk memfasilitasi pengambilan keputusan yang tepat bagi para pemangku kepentingan di pasar laptop, memungkinkan mereka untuk menetapkan harga yang kompetitif, mengoptimalkan keuntungan, dan membuat keputusan pembelian yang terinformasi dengan baik.

Produsen dapat mengoptimalkan perencanaan produksi dan manajemen inventaris berdasarkan prediksi harga, memastikan alokasi sumber daya yang efisien dan meminimalkan kelebihan produksi atau kehabisan stok. Selain itu, prediksi harga yang akurat memungkinkan strategi penetapan harga yang kompetitif, memandu produsen untuk menetapkan harga yang memaksimalkan profitabilitas sambil tetap menarik bagi konsumen. Bagi konsumen, prediksi harga yang akurat memberikan transparansi dan kepercayaan diri dalam keputusan pembelian, memungkinkan pilihan yang tepat dan potensi penghematan melalui waktu pembelian yang optimal dan strategi penetapan harga.

Referensi: [Lohakare, S. (2022) Laptop Price Prediction using Machine Learning](https://ijcsmc.com/docs/papers/January2022/V11I1202229.pdf)

## Business Understanding
### Problem Statement
- Pasar laptop sangat luas dan beragam, dengan banyak produsen yang menawarkan berbagai macam produk dengan spesifikasi dan harga yang berbeda. Namun, menentukan harga yang optimal untuk sebuah laptop dapat menjadi tantangan karena faktor-faktor seperti persaingan, kemajuan teknologi, dan preferensi konsumen.
- Konsumen sering kali kesulitan untuk membuat keputusan pembelian yang tepat ketika dihadapkan pada banyak pilihan laptop, masing-masing dengan serangkaian fitur dan harga. Tanpa pemahaman yang jelas tentang bagaimana berbagai fitur berkontribusi pada harga laptop, konsumen mungkin membayar lebih untuk fitur yang tidak mereka butuhkan atau melewatkan fitur-fitur penting karena keterbatasan anggaran.

### Goals
- Mengembangkan model prediktif yang secara akurat memperkirakan harga laptop berdasarkan fitur-fiturnya, termasuk merek, ukuran layar, CPU, GPU, RAM, penyimpanan, sistem operasi, berat, dan lain-lain.
- Memberikan wawasan yang berharga bagi produsen dan konsumen dengan menganalisis hubungan antara berbagai fitur laptop dan dampaknya terhadap harga. Hal ini akan memungkinkan produsen untuk menentukan harga produk mereka secara kompetitif dan membantu konsumen membuat keputusan pembelian yang tepat.
- Model prediktif memungkinkan produsen untuk mengoptimalkan perencanaan produksi dan strategi penetapan harga, memastikan alokasi sumber daya yang efisien dan menetapkan harga yang kompetitif yang memaksimalkan profitabilitas sambil tetap menarik bagi konsumen.
- Model prediktif membantu konsumen membuat keputusan pembelian yang tepat dengan memberikan transparansi ke dalam struktur harga dan proposisi nilai dari berbagai fitur laptop, memungkinkan konsumen mendapatkan laptop yang sesuai dengan kebutuhan dan anggaran mereka.

### Solution Statement
- Memanfaatkan algoritma machine learning seperti Linear Regression, Random Forest, Gradient Boosting, Neural Network, dan Support Vector Machine (SVM) untuk membangun model prediktif untuk estimasi harga laptop.
- Mengevaluasi kinerja setiap model menggunakan metrik evaluasi yang sesuai seperti Mean Squared Error (MSE). Bandingkan performa berbagai model dan pilih model yang memiliki akurasi tertinggi untuk prediksi harga.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah dataset "Laptop Price" yang diperoleh dari Kaggle. Anda dapat mengunduh dataset tersebut dari tautan berikut: [Dataset Harga Laptop](https://www.kaggle.com/datasets/muhammetvarl/laptop-price/data).
### Variables in the Laptop Price Dataset:
- **Company:** Produsen laptop.
- **Product:** Merek dan model laptop.
- **TypeName:** Jenis laptop (Notebook, Ultrabook, Gaming, dll.).
- **Inches:** Ukuran layar laptop.
- **ScreenResolution:** Resolusi layar laptop.
- **Cpu:**  Central Processing Unit (CPU) laptop.
- **Ram:** Random Access Memory (RAM) laptop.
- **Memory:** Memori Hard Disk Drive (HDD) atau Solid State Drive (SSD) laptop.
- **Gpu:** Graphics Processing Unit (GPU) laptop.
- **OpSys:** Sistem Operasi laptop.
- **Weight:** Berat laptop.
- **Price_euros:** Harga laptop dalam Euro.

**Skewness Analysis** :
| Feature       | Skewness    |
|---------------|-------------|
| inches        | -0.438622   |
| ram           | 2.698716    |
| weight        | 1.150804    |
| price         | 1.511147    |
| retina        | 8.496076    |
| ips           | 0.981113    |
| touchscreen   | 1.991028    |
| quad          | 6.531349    |
| cpu_speed     | -0.838246   |
| memory_size   | 1.573719    |
| extra_memory  | 1.825537    |
| width         | 2.210137    |
| height        | 2.117949    |

**Insight** :
1. **Symmetric Distribution:**
   - Fitur seperti "inches" dan "cpu_speed" memiliki nilai kemencengan yang mendekati 0, yang mengindikasikan distribusi yang kurang lebih simetris. Titik data terdistribusi secara merata di sekitar rata-rata.
2. **Right-skewed Distribution (Positive Skewness):**
   - Fitur seperti "ram", "weight", "price", "ips", "touchscreen", "memory_size", "extra_memory", "width", dan "height" menunjukkan kemiringan positif, yang mengindikasikan ekor yang memanjang ke arah nilai yang lebih tinggi. Sebagai contoh, "ram" menunjukkan lebih banyak pengamatan dengan ukuran RAM yang lebih tinggi dibandingkan dengan yang lebih rendah.
3. **Left-skewed Distribution (Negative Skewness):**
   - Hanya "cpu_speed" yang menampilkan kemencengan negatif, yang menyiratkan ekor yang memanjang ke arah nilai yang lebih rendah. Hal ini menunjukkan lebih banyak pengamatan dengan kecepatan CPU yang lebih rendah dibandingkan dengan yang lebih tinggi.
 
**Data Describe** :
|       | inches  | ram   | weight | price | retina | ips   | touchscreen | quad  | cpu_speed | memory_size | extra_memory | width | height |
|-------|---------|-------|--------|-------|--------|-------|-------------|-------|-----------|-------------|--------------|-------|--------|
| count | 1275.00 | 1275.00 | 1275.00 | 1275.00 | 1275.00 | 1275.00 | 1275.00     | 1275.00 | 1275.00     | 1275.00     | 1275.00      | 1275.00 | 1275.00 |
| mean  | 15.02   | 8.44  | 2.04   | 1134.97 | 0.01   | 0.28  | 0.15        | 0.02   | 2.30      | 439.40      | 0.16         | 1900.04 | 1073.90 |
| std   | 1.43    | 5.10  | 0.67   | 700.75  | 0.11   | 0.45  | 0.35        | 0.15   | 0.50      | 355.75      | 0.37         | 493.35  | 283.88  |
| min   | 10.10   | 2.00  | 0.69   | 174.00  | 0.00   | 0.00  | 0.00        | 0.00   | 0.90      | 8.00        | 0.00         | 1366.00 | 768.00  |
| 25%   | 14.00   | 4.00  | 1.50   | 609.00  | 0.00   | 0.00  | 0.00        | 0.00   | 2.00      | 256.00      | 0.00         | 1920.00 | 1080.00 |
| 50%   | 15.60   | 8.00  | 2.04   | 989.00  | 0.00   | 0.00  | 0.00        | 0.00   | 2.50      | 256.00      | 0.00         | 1920.00 | 1080.00 |
| 75%   | 15.60   | 8.00  | 2.31   | 1496.50 | 0.00   | 1.00  | 0.00        | 0.00   | 2.70      | 512.00      | 0.00         | 1920.00 | 1080.00 |
| max   | 18.40   | 64.00 | 4.70   | 6099.00 | 1.00   | 1.00  | 1.00        | 1.00   | 3.60      | 2000.00     | 1.00         | 3840.00 | 2160.00 |

**Insight** :
- Dataset ini berisi informasi tentang 1275 laptop.
- Laptop dalam kumpulan data memiliki harga mulai dari 174 hingga 6099 euro, dengan harga rata-rata sekitar 1135 euro.
- Ukuran layar bervariasi dari 10,1 hingga 18,4 inci, dengan ukuran rata-rata sekitar 15,02 inci.

### Univariate Analysis
**Categorical Features Analysis** :

![company_uni](https://i.ibb.co/sQtKTjV/company-uni.png)
- Dell, Lenovo, dan HP mendominasi pasar, dengan jumlah sekitar dua pertiga dari laptop yang dijadikan sampel.
- Merek-merek lainnya, seperti Asus, Acer, dan MSI, memiliki pangsa pasar yang lebih kecil.
- Merek yang kurang terkenal seperti Razer, Mediacom, dan Vero memiliki kehadiran yang dapat diabaikan dalam kumpulan data.

![type_name](https://i.ibb.co/xjMV8LB/type-name-uni.png)
- Notebook adalah jenis laptop yang paling banyak digunakan, terdiri dari lebih dari setengah dataset.
- Laptop gaming dan ultrabook juga memiliki representasi yang signifikan.
- Workstation dan netbook kurang umum, masing-masing mewakili kurang dari 3% dataset.
 
![opsys](https://i.ibb.co/7jJb7SB/opsys-uni.png)
- Windows 10 adalah sistem operasi yang sangat dominan, dengan prevalensi lebih dari 80%.
- Sebagian kecil laptop tidak memiliki OS yang sudah terinstal atau menggunakan Linux.
- macOS dan Chrome OS memiliki representasi yang terbatas dibandingkan dengan Windows.

![cpu_type](https://i.ibb.co/8zMZb89/cpu-uni.png)
- CPU Intel sangat dominan, mencakup lebih dari 95% laptop yang dijadikan sampel.
- CPU AMD memiliki kehadiran yang jauh lebih kecil, hanya mewakili sekitar 5% dari kumpulan data.

![cpu_name](https://i.ibb.co/1z0Py2w/cpu-name-uni.png)
- Core i7, Core i5, dan Core i3 adalah jenis CPU yang paling umum, secara kolektif mencakup lebih dari 80% dataset.
- Jenis CPU lainnya, seperti Celeron Dual dan Pentium Quad, memiliki bagian yang jauh lebih kecil.
- Ada berbagai macam jenis CPU, tetapi kebanyakan dari mereka memiliki representasi minimal dalam dataset.

![memory_type](https://i.ibb.co/nCDxHMT/memory-uni.png)
- SSD adalah jenis memori yang paling umum, dengan prevalensi lebih dari 65%.
- HDD masih lazim digunakan, tetapi kurang umum dibandingkan dengan SSD.
- Memori flash dan memori hibrida memiliki porsi yang relatif lebih kecil.

![gpu_brand](https://i.ibb.co/R25J0bR/gpu-uni.png)
- GPU Intel adalah yang paling umum, dengan lebih dari separuh laptop memilikinya.
- GPU Nvidia adalah yang paling umum berikutnya, diikuti oleh GPU AMD.
- GPU ARM memiliki kehadiran yang dapat diabaikan dalam kumpulan data.

**Numeric Features Analysis** :

![numeric_uni](https://i.ibb.co/26R83bZ/numeric-univariate.png)
1. Laptop dalam kumpulan data memiliki ukuran layar yang beragam, dengan rata-rata sekitar 15,02 inci.
2. Ukuran RAM sangat bervariasi di antara laptop, dengan rata-rata sekitar 8,44 GB.
3. Berat laptop juga menunjukkan variabilitas, dengan rata-rata sekitar 2,04 kg.
4. Harga laptop menunjukkan variasi yang signifikan, mulai dari 174 euro hingga 6099 euro, dengan harga rata-rata sekitar 1.344,97 euro.
5. Fitur-fitur lain seperti layar retina, layar IPS, layar sentuh, prosesor quad-core, kecepatan CPU, ukuran memori, dan memori ekstra mengikuti pola variabilitas dan distribusi yang serupa.

### Multivariate Analysis
**Categorical Features Analysis** :

![company_multivariate](https://i.ibb.co/S5qj6q7/company-multi.png)
- Dell, Lenovo, dan HP adalah produsen laptop utama dengan harga rata-rata.
- Razer dan LG memiliki harga yang lebih tinggi, mungkin untuk produk premium.
- Microsoft, Xiaomi, dan Google menawarkan opsi yang ramah anggaran.

![type_name](https://i.ibb.co/g3Rghwk/type-name-multi.png)
- Laptop workstation memiliki harga tertinggi.
- Laptop gaming mengikuti, mencerminkan fitur premium mereka.
- Netbook adalah yang paling terjangkau.

![opsys](https://i.ibb.co/svgyPFS/opsys-name.png)
- Laptop macOS dan Windows 7 lebih mahal.
- Laptop Chrome OS dan Android lebih murah.
- Tidak ada laptop OS yang ramah anggaran.

![cpu_name](https://i.ibb.co/PWHYLXZ/cpu-multi.png)
- CPU Intel berkorelasi dengan harga yang lebih tinggi.
- CPU AMD biasanya ditemukan dalam opsi yang lebih ramah anggaran.

![cpu_type](https://i.ibb.co/02g5Yz5/cpu-type-multi.png)
- CPU Core i7 memiliki harga tertinggi, diikuti oleh i5 dan i3.
- CPU tingkat pemula lebih terjangkau.
- CPU khusus seperti Core M memiliki harga yang lebih tinggi.

![type_memory](https://i.ibb.co/8zCQWDJ/memory-multi.png)
- Laptop yang dilengkapi SSD lebih mahal.
- Laptop memori hibrida memiliki harga yang cukup terjangkau.

![gpu_brand](https://i.ibb.co/F86rtTg/gpu-multi.png)
- GPU Nvidia diasosiasikan dengan laptop dengan harga yang lebih tinggi.
- GPU AMD lebih murah, sedangkan Intel dan ARM adalah pilihan anggaran.

**Numeric Features Analysis** :

![korelasi-numerik](https://i.ibb.co/yVqWSxM/EDA-gambar-1.png)
Harga memiliki korelasi positif tertinggi dengan RAM (0,740), diikuti oleh lebar (0,552) dan tinggi (0,548). Fitur lain seperti kecepatan CPU (0,429), memori ekstra (0,306), dan IPS (0,251) juga menunjukkan korelasi positif dengan harga, meskipun tidak sekuat RAM, lebar, dan tinggi. Sebaliknya, ukuran memori (-0,122) memiliki korelasi negatif dengan harga, yang mengindikasikan sedikit penurunan harga seiring dengan meningkatnya ukuran memori. Berdasarkan wawasan ini, masuk akal untuk menghapus fitur dengan koefisien korelasi rendah seperti '_product_', '_cpu_type_', '_quad_', '_retina_', dan '_inches_' untuk merampingkan model dan meningkatkan kinerjanya.

## Data Preparation
- **Transforming Data**:
Tentukan fungsi seperti `extract_features_resolution`, `extract_cpu_features`, dan `extract_features_memory` untuk mengekstrak fitur yang relevan dari kolom yang ada. Gunakan fungsi-fungsi ini untuk membuat kolom baru berdasarkan fitur yang diekstrak, seperti 'retina', 'ips', 'touchscreen', 'quad', 'resolution', 'cpu_name', 'cpu_type', 'cpu_speed', 'memory_size', 'type_memory', 'extra_memory', dan 'gpu_brand'.

- **One-Hot Encoding**: 
Teknik ini mengubah variabel kategorikal menjadi format biner, membuat variabel dummy untuk setiap kategori. Hal ini difasilitasi oleh `OneHotEncoder` dari `sklearn.preprocessing`, membuat data kategorikal kompatibel dengan algoritme pembelajaran mesin.

- **Feature Selection**:
Dengan menganalisis korelasi antara fitur dan variabel target (harga), hanya fitur dengan korelasi yang signifikan yang dipertahankan. Fitur dengan koefisien korelasi di atas 0,1 atau di bawah -0,1 dengan variabel target akan dipertimbangkan, mengurangi dimensi dan fokus pada prediktor yang paling berpengaruh.

- **Data Splitting**:
Dataset dibagi menjadi set pelatihan dan pengujian menggunakan `train_test_split` dari `sklearn.model_selection`. Langkah ini memastikan model dilatih pada sebagian data dan dievaluasi pada data yang tidak terlihat, sehingga membantu mengukur kinerjanya pada pengamatan baru.

- **Feature Scaling**:
Standarisasi diterapkan pada fitur untuk memastikan fitur tersebut memiliki nilai rata-rata 0 dan deviasi standar 1. Hal ini sangat penting untuk model seperti jaringan syaraf, di mana fitur dengan skala yang berbeda dapat memengaruhi pengoptimalan. `StandardScaler` dari `sklearn.preprocessing` memfasilitasi transformasi ini.

## Modeling
Pada tahap pemodelan, lima algoritma machine learning diterapkan: Linear Regression, Random Forest Regression, Gradient Boosting Regression, Neural Network, dan Support Vector Machine (SVM). 
- **Linear Regression**: Model ini menghitung hubungan antara variabel independen dan variabel target secara linear. Model ini sederhana dan mudah diinterpretasikan, namun dapat berkinerja buruk jika hubungan antara fitur dan target tidak linier. Regresi linier dipilih karena kesederhanaan dan kemampuannya untuk diinterpretasikan, yang sejalan dengan tujuan proyek untuk mengembangkan model prediksi harga laptop. Proyek ini menggunakan `sklearn.linear_model.LinearRegression` untuk pembuatan model. Karena regresi linier tidak melibatkan hiperparameter, pengaturan default digunakan untuk pelatihan model.
- **Random Forest Regression**:  Metode ensemble ini menggabungkan beberapa pohon keputusan untuk membuat prediksi. Algoritma random forest adalah teknik dalam pembelajaran mesin dengan metode ansambel. Algoritma ini beroperasi dengan membangun banyak pohon keputusan pada waktu pelatihan. Proyek ini menggunakan `sklearn.ensemble.RandomForestRegressor` dengan menyertakan X_train dan y_train dalam membangun model. Parameter yang digunakan dalam proyek ini adalah:
  - `n_estimator` = 100 (default)
  - `max_depth` = Tidak ada (default)
  - `random_state` = Tidak ada (default)
- **Gradient Boosting Regression**: Metode ansambel ini membangun pohon secara berurutan, masing-masing mengoreksi kesalahan yang dibuat oleh pohon sebelumnya. Regresi gradient boosting dipilih karena keefektifannya dalam menangkap pola-pola yang kompleks dan kemampuannya untuk memperbaiki kelemahan dari pohon keputusan individual. Proyek ini menggunakan `sklearn.ensemble.GradientBoostingRegressor` untuk pembuatan model. Parameter yang digunakan dalam proyek ini adalah:
  - `n_estimator` = 100 (default)
   - `max_depth` = 3 (default)
  - `learning_rate` = 0.1 (default)
- **Neural Network**: Model pembelajaran mendalam ini terdiri dari lapisan-lapisan neuron yang saling berhubungan.Jaringan saraf dengan arsitektur model sekuensial yang terdiri dari dua lapisan tersembunyi digunakan karena kemampuannya untuk mempelajari pola dan hubungan yang kompleks dalam data. Proyek ini menggunakan `tensorflow.keras.Sequential` untuk pembuatan model. Parameter yang digunakan dalam proyek ini adalah:
  - Hidden layer neurons = Masing-masing 64 dan 32 neuron
  - Activation function = ReLU (Rectified Linear Unit)
  - Optimization algorithm = Adam optimizer
  - Loss function = Mean Squared Error (MSE)
- **Support Vector Machine (SVM)**: Model ini menemukan hyperplane yang paling baik memisahkan kelas-kelas dalam ruang fitur.Regresi Support Vector Machine (SVM) dengan kernel linier dipilih karena efektivitasnya dalam ruang dimensi tinggi dan keserbagunaannya karena fungsi kernel yang berbeda yang tersedia. Proyek ini menggunakan `sklearn.svm.SVR` untuk pembuatan model. Parameter yang digunakan dalam proyek ini adalah:
  - `kernel` = 'linear'
  - `C` = 1.0 (default)
  - `gamma` = 'scale' (default)

## Evaluation
Metrik evaluasi proyek ini mencakup Mean Squared Error (MSE) untuk set data pelatihan dan pengujian.

| Model            | Train MSE | Test MSE  |
|------------------|-----------|-----------|
| LinearRegression | 109.63    | 121.21    |
| RandomForest     | 14.58     | 71.95     |
| GradientBoosting | 49.87     | 71.04     |
| NeuralNetwork    | 83.49     | 99.82     |
| SVM              | 137.87    | 152.63    |

**MSE**: Mengukur rata-rata perbedaan kuadrat antara nilai aktual dan nilai prediksi. Semakin rendah nilai yang dihasilkan menunjukkan kinerja yang lebih baik.

![evaluate](https://i.ibb.co/7jkQHhs/evaluate.png)
Hasil:
- **Linear Regression**: Menunjukkan MSE pelatihan terendah tetapi MSE pengujian lebih tinggi, menunjukkan potensi overfitting.
- **Random Forest Regression**: Menunjukkan MSE pengujian yang jauh lebih rendah, menunjukkan generalisasi yang lebih baik.
- **Gradient Boosting Regression**: Menunjukkan MSE pelatihan yang lebih rendah daripada Random Forest tetapi MSE pengujian sedikit lebih tinggi.
- **Neural Network**: Memiliki MSE pelatihan yang lebih rendah tetapi MSE pengujian yang lebih tinggi, menunjukkan overfitting.
- **Support Vector Machine (SVM)**: Menunjukkan MSE tertinggi untuk set data pelatihan dan pengujian.

**Model Random Forest Regression** memiliki kinerja terbaik dengan MSE pengujian terendah, yang menunjukkan generalisasi yang lebih baik. Namun, analisis dan optimasi lebih lanjut diperlukan untuk semua model untuk mengurangi potensi overfitting.

| y_true | prediksi_LinearRegression | prediksi_RandomForest | prediksi_GradienBoosting | prediksi_NeuralNetwork | prediksi_SVM |
|--------|---------------------------|-----------------------|--------------------------|-------------------------|--------------|
| 449.0  | 638.5                     | 428.6                 | 504.4                    | 647.900024              | 651.2        |


