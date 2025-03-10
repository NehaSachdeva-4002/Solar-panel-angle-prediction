﻿# 🌞 Solar Panel Angle Optimization
 ![image](https://github.com/user-attachments/assets/c2ba177f-dd1f-44e9-acc6-4418da8bef3c)
![image](https://github.com/user-attachments/assets/0ede309d-b156-43b0-b4b2-3382d510ab03)


## 📌 Project Overview
The **Solar Panel Angle Optimization** project utilizes **AI-driven models** to determine the optimal tilt angle for solar panels based on environmental conditions. The system leverages **machine learning models** trained on synthetic and real-world datasets to improve energy efficiency.

## 🚀 Features
- **Predicts optimal solar panel angles** based on weather conditions and location.
- **Uses machine learning models** for real-time tilt angle recommendations.
- **Processes weather & solar radiation data** such as temperature, humidity, GHI, DNI, and DHI.
- **Flask-based web interface** for easy user interaction.
- **Supports integration with real-world weather data APIs.**

## 🛠️ Tech Stack
- **Programming Languages:** Python, JavaScript, HTML/CSS
- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn, TensorFlow
- **Solar Calculations:** Pysolar
- **Data Processing:** Pandas, NumPy

## 📂 Project Structure
```
├── backend.py            # Flask backend for handling requests
│   main.py               # Data preprocessing & feature extraction + ML model for angle prediction
│
├── templates/
│   │   ├── index.html        # Web interface for user inputs
│
├── solar_dataset.csv     # Dataset for training/testing 
│
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
└── .gitignore                # Ignore unnecessary files
```

## 🔧 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/solar-panel-optimization.git
cd solar-panel-optimization
```

### 2️⃣ Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Flask Application
```bash
python backend/app.py
```
Visit `http://127.0.0.1:5000/` in your browser.

## 📊 Usage
1. Enter your **latitude, longitude, date, and time**.
2. The system will fetch weather conditions and predict the optimal tilt angle.
3. The output will display the **recommended tilt angle** with real-time visualization.

## 🧠 Machine Learning Model
The model uses:
- **Feature extraction** (Solar Zenith Angle, Azimuth, Weather Data)
- **Normalization** (MinMax Scaling)
- **Regression models** for tilt angle prediction

## 🤝 Contributing
1. **Fork** the repository.
2. Create a **new branch** (`git checkout -b feature-branch`).
3. **Commit** your changes (`git commit -m "Added new feature"`).
4. **Push** to the branch (`git push origin feature-branch`).
5. Open a **Pull Request** for review.

## 📜 License
This project is licensed under the **MIT License**.

## 🌟 Acknowledgments
- **Pysolar** for solar angle calculations.
- **Scikit-learn & TensorFlow** for ML modeling.
- **Flask** for backend development.

🔗 **GitHub Repository:** [https://github.com/NehaSachdeva-4002/Solar-panel-angle-prediction.git]

