
# Next Word Prediction Using [LSTM]()  

This project leverages **Long Short-Term Memory (LSTM)** networks to predict the next word in a given sequence of words using Shakespeare's *Hamlet* as the dataset. By training on this rich and complex text, the model can learn to generate contextually accurate predictions. This is a sequence prediction problem where the goal is:  

- **To predict the most likely next word in a given sequence.**  
## Dataset Information  
The dataset used in this project consists of the text from Shakespeare's *Hamlet*. The dataset is tokenized and processed to create sequences suitable for training the LSTM model.  

## Technology Used  
### Programming Language  
- Python  

### Libraries   
- TensorFlow  
- Keras  
- NumPy  
- Pandas  
- Streamlit  
- Matplotlib
- scikit-learn
  
## Model Architecture  
The LSTM model is structured as follows:  
- **Input Layer:** An embedding layer that converts words into dense vector representations.  
- **Hidden Layers:**  
  - LSTM Layer 1: 150 neurons, return_sequences set to True.  
  - Dropout Layer: Dropout rate of 0.2 to prevent overfitting.  
  - LSTM Layer 2: 100 neurons.  
- **Output Layer:** Dense layer with softmax activation for predicting the probability distribution of the next word.  
- **Optimizer:** Adam optimizer for minimizing the loss function.  
- **Loss Function:** Categorical Crossentropy for multi-class classification.  
- **Metrics:** Accuracy.  

## Preprocessing Steps  
1. **Data Cleaning:**  
   Removed irrelevant characters and normalized text to lowercase.  
2. **Tokenization:**  
   Tokenized the text data into sequences of words.  
3. **Padding:**  
   Padded sequences to ensure uniform input lengths.  
4. **Splitting the Dataset:**  
   Split the data into 80% training and 20% testing sets.  


## Getting Started  

To run this project, execute the following commands in the terminal:

- **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```  
- **Create a virtual environment:**  
   ```bash
   conda create -p venv python==3.11 -y  
   ```  
- **Activate the virtual environment:**  
   ```bash
   conda activate venv/  
   ```  
- **Install dependencies:**  
   ```bash
   pip install -r requirements.txt  
   ```  
- **Run the Streamlit Application:**  
   ```bash
   streamlit run app.py  
   ```

## Learn More About LSTM Networks  
- **[Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)**  


## [Screenshot of the Streamlit Web App]() 
![lstm](https://github.com/user-attachments/assets/a3e90009-0404-4467-b30f-cbedfc7406dd)
