# **Real-Time Noise Suppression Using AI**  

## **Overview**  
This project develops a **deep learning-based noise cancellation system** for **industrial environments**. Using **MFCC, Wavelet Transforms, and CNNs**, the model filters out unwanted noise in real time, improving signal clarity for **acoustic monitoring systems**. The optimized model is deployed on **Raspberry Pi with OpenVINO**, ensuring **low-latency inference** for on-device processing.  

## **Features**  
- **Noise Feature Extraction:** Uses **MFCC (Mel-Frequency Cepstral Coefficients)** and **Wavelet Transforms** to extract critical sound patterns.  
- **Deep Learning-Based Filtering:** Trained **CNN model** to distinguish and suppress industrial noise.  
- **Edge AI Deployment:** Optimized for **Raspberry Pi & OpenVINO**, enabling real-time inference with minimal latency.  

## **Tech Stack**  
- **Signal Processing:** Librosa, PyWavelets, SciPy  
- **Deep Learning:** TensorFlow, Keras, CNNs  
- **Edge Deployment:** OpenVINO, TensorFlow Lite, Raspberry Pi  
- **Visualization & Testing:** Matplotlib, Seaborn  

## **Dataset**  
- Industrial noise recordings from **factories and machinery environments**.  
- Labeled dataset with **clean and noisy audio samples** for supervised training.  

## **Installation & Usage**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/sourajyoti593/Real-Time-Noise-Suppression-Using-AI.git
   cd noise-suppression-ai
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Train the model:  
   ```bash
   python train.py
   ```  
4. Deploy on Raspberry Pi with OpenVINO:  
   ```bash
   python deploy.py
   ```  

## **Results**  
- Achieved **90% noise reduction** in real-time acoustic monitoring.  
- **<10ms latency** for inference on **Raspberry Pi using OpenVINO**.  

## **Future Improvements**  
- Extend to **multi-channel noise suppression** for enhanced accuracy.  
- Deploy on **Jetson Nano for high-performance edge inference**.  

---
