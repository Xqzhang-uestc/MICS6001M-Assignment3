# MICS6001M-Lab3

## 📌 Authors  
**Xianqing Zhang**, **Wenjuan Zhou**  
---

## 🌐 GitHub Repository  

**Access the project**:  
```bash
git clone https://github.com/Xqzhang-uestc/MICS6001M-Assignment3.git
cd MICS6001M-Assignment3
```

**Support**:  
⭐ Star the repo if you find this useful!  

---

## 🗂 Project Structure  
### Core Scripts  
| File | Purpose |  
|------|---------|  
| `task1-*.py` | Model training with different configurations |  
| `task2_EDA.py` | Exploratory data analysis and statistical tests |  
| `task2_[MLP\|RNN\|LSTM].py` | Model-specific training scripts |  
| `task3-prediction.py` | Prediction generation and metric evaluation |  
| `task3-visualization.py` | Time series visualization tools |  

### Utilities  
- `reshape.py`: Convert time series → supervised learning format  
- `split.py`: Train/test splitting  
- `model_structure.py`: Custom model architectures  
- `statistical_test.py`: ADF/Ljung-Box tests  

### Outputs  
- `*_results.txt`: Performance metrics (ARMA/Random Walk/White Noise)  
- `pics/`: Saved visualizations  

---

## 🔍 Key Features  
### 📊 Data Preparation  
- **Reshaping**: Sliding window conversion via `reshape.py`  
- **Splitting**: Configurable train/test ratios in `split.py`  

### 🔎 Exploratory Analysis  
- **Statistical Tests**: Stationarity (ADF) and white noise (Ljung-Box) verification  
- **Visualization**:  
  - ACF/PACF plots  
  - Lag analysis  
  - Noise distribution profiling  

### 🤖 Model Training  
| Model | Script | Highlights |  
|-------|--------|------------|  
| MLP | `task2_MLP.py` | Supports ReLU/Tanh/Sigmoid activations |  
| RNN | `task2_RNN.py` | Basic recurrent architecture |  
| LSTM | `task2_LSTM.py` | Long-term dependency capture |  

### 📈 Performance Evaluation  
- **Metrics**: MSE, MAE, MAPE  
- **Noise Sensitivity**: Comparative analysis under varying noise levels  
- **Visual Reports**: Prediction vs. ground truth plots  

---

## 🏆 Key Findings  
1. **Architecture Comparison**  
   - LSTM outperformed on sequential data (avg. 15% lower MSE vs. RNN)  
   - MLP showed faster training but higher variance  

2. **Activation Functions**  
   ```python  
   # Performance ranking (avg MSE):  
   ReLU > Tanh > Sigmoid  # For high-noise datasets  
   Tanh ≈ Sigmoid > ReLU  # For smooth periodic series  
   ```  

3. **Noise Impact**  
   ![Noise vs Accuracy](pics/noise_impact.png)  
   *Prediction degradation follows logarithmic trend with added noise*

---

## 🛠 Installation  
**Requirements**: Python 3.11+  
```bash
pip install -r requirements.txt  # Includes:  
# numpy, matplotlib, pandas, tensorflow,  
# statsmodels, pmdarima, sklearn
```

---

## 🚀 Quick Start  
1. **Prepare Data**  
   ```bash  
   python reshape.py  
   python split.py  
   ```  

2. **Run Analysis**  
   ```bash  
   python task2_EDA.py 
   ```  

3. **Train Models**  
   ```bash  
   python task2_LSTM.py
   ```  

4. **Generate Reports**  
   ```bash  
   python task3-prediction.py 
   python task3-visualization.py  
   ```  

---

## 📌 Notes  
- Custom architectures can be defined in `model_structure.py`  
- Raw results are timestamped in `results/` subdirectory  
- For extended visualization options, see `draw.py` documentation  
