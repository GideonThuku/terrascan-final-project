# 🛰️ TerraScan AI: AI-Powered Land Health Analysis

**LIVE DEMO:** [https://terrascanafritech.streamlit.app/](https://terrascan-final-project.streamlit.app/)  

A project by **Gideon Thuku**.

---

## 📖 About The Project

TerraScan is an advanced, user-friendly web application designed to analyze the health of land and vegetation using satellite imagery and Artificial Intelligence.  

It helps farmers, environmentalists, researchers, and land managers quickly assess land degradation, monitor crop health, and make informed decisions for sustainable land management.  

This project goes beyond simple data processing. It integrates two distinct AI models to provide a truly smart analysis:

- **Computer Vision (CV) Model** (TensorFlow/MobileNetV2) to classify the land use type (e.g., Forest, Valley, Agricultural Land) from the true-color satellite image.  
- **Machine Learning (ML) Model** (Scikit-learn/RandomForest) that acts as a "smart recommendation engine," providing custom advice based on both the land's health and its classified type.  

The core of the health analysis is the **Normalized Difference Vegetation Index (NDVI)**, a trusted scientific method to measure vegetation greenness and vitality. By combining NDVI with AI, TerraScan provides a clear, actionable picture of land health.

---

## ✨ Key Features

- **Interactive Map Selection:** Draw a custom Area of Interest (AOI) directly on an interactive map.  
- **Dynamic Mock Data:** Simulates real-world analysis by generating unique satellite data based on the location you select.  
- **AI-Powered Land Classification (CV):** Automatically identifies the most likely land use type (e.g., Forest, Meadow) using a computer vision model.  
- **Advanced Vegetation Analysis (NDVI):** Calculates the NDVI for detailed vegetation health statistics.  
- **Smart Recommendation Engine (ML):** Generates actionable advice using a machine learning model trained on land health and land type.  
- **Rich Visualizations:**  
  - Health Map showing NDVI patterns  
  - True Color Image from satellite data  
- **Data Export:** Download a full, comprehensive analysis report in CSV format, including all AI insights.

---

## ⚙️ How It Works

1. **Select Area:** Draw a polygon or rectangle on the Folium map.  
2. **Set Parameters:** Adjust the NDVI threshold slider for analysis sensitivity.  
3. **Fetch Data:** The app fetches dynamic mock satellite data (RGB and NDVI arrays) based on the selected area's coordinates.  
4. **Run AI Land Classifier (CV):** The RGB "true color" image is fed into a pre-trained MobileNetV2 model to classify the land use type.  
5. **Process NDVI:** The NDVI array is analyzed to calculate the percentage of healthy vs. degraded land based on the user's threshold.  
6. **Run AI Recommender (ML):** The degradation percentage and the land use type are fed into a custom-trained RandomForestClassifier model.  
7. **Display Results:** The dashboard updates with all results: health scores, maps, the CV land type, and the ML-driven recommendation.

---

## 💻 Technology Stack

- **Language:** Python  
- **Framework:** Streamlit  
- **AI / Machine Learning:**  
  - TensorFlow (Keras) — for the Computer Vision (CV) model (MobileNetV2)  
  - Scikit-learn — for training and running the ML Recommendation Engine (RandomForestClassifier)  
- **Data Processing:** Pandas, NumPy, Joblib  
- **Mapping:** Folium, streamlit-folium  
- **Visualization:** Pillow, Matplotlib  

---

## 🚀 Getting Started (Running Locally)

### Prerequisites

- Python 3.8+  
- pip  
- A (free) Planet API key is optional; the app will use mock data if no key is found.

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/terrascan-ai.git
cd terrascan-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Add Planet API key
mkdir .streamlit
echo 'PLANET_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"' > .streamlit/secrets.toml

# --- CRITICAL FIRST STEP ---
# You must train the ML model at least once to create the model file
python train_recommender.py

# Run the app
streamlit run app.py
