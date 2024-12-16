# American Sign Language Detection 🤟

This project leverages Machine Learning and Computer Vision to detect signs in American Sign Language (ASL) that represent English alphabets. The application features a user-friendly interface for real-time sign detection and translation into English letters, aiming to bridge communication gaps and promote inclusivity.

---

## 🎯 Problem Statement  
Many individuals face communication barriers due to a lack of understanding of sign language. This project addresses this challenge by creating a tool to detect and translate ASL gestures into English alphabets, fostering better communication.  

---

## 💡 Solution  
Our solution includes:  
- **Real-Time Sign Detection**: Uses a camera feed to capture and analyze ASL signs.  
- **Machine Learning Model**: A Random Forest Classifier trained on ASL signs with an impressive 95% accuracy.  
- **Interactive User Interface**: Converts detected signs into English letters for easy comprehension.  

---

## 🛠️ Tools & Technologies  
- **Programming Language**: Python  
- **Libraries**: OpenCV, NumPy, Pandas, Scikit-learn, Tkinter (for UI)  
- **Machine Learning**: Random Forest Classifier  

---

## 🚀 Features  
1. Real-time video stream processing for sign detection.  
2. Accurate translation of ASL signs into English alphabets.  
3. Simple, intuitive user interface for seamless interaction.  

---

## 🧠 How It Works  
1. **Data Collection**: Collected a dataset of ASL gestures representing English alphabets.  
2. **Model Training**:  
   - Trained a Random Forest Classifier on the dataset.  
   - Achieved a 95% accuracy on test data.  
3. **Real-Time Detection**:  
   - Used OpenCV to capture and preprocess the video stream.  
   - Integrated the trained model to detect and classify signs in real time.  
4. **User Interface**:  
   - Developed a UI to display the detected alphabets to users in a clear and interactive manner.  

---

## 🔧 Installation and Setup  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- Required libraries: Install using `pip install -r requirements.txt`  

### Steps  
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/asl-detection.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   python main.py