# Music Classification Using Machine Learning

## Project Overview
This project classifies classical music compositions by six renowned composers using machine learning techniques. By extracting features from MIDI files, such as pitch histograms, average note durations, and key signatures, we trained various machine learning models to predict the composer of a given composition.

## Features Extracted
- **Pitch Histograms**: Frequency of pitch occurrences across 12 semitones.
- **Average Note Duration**: Captures the typical length of notes in a composition.
- **Key Signatures**: Analyzes the tonic and mode of each piece.
- **Number of Chords**: Indicates harmonic complexity.
- **Melodic Intervals**: Identifies pitch changes to understand stylistic differences.

## Machine Learning Models Used
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Logistic Regression
- k-Nearest Neighbors (k-NN)
- Voting Ensemble
- Stacked Ensemble

## Results
The models achieved varying levels of accuracy. The best-performing models were:
- **Random Forest**: 92.19% accuracy
- **Stacked Ensemble**: 92.05% accuracy

Other models like Gradient Boosting and Voting Ensemble also performed well, with accuracy rates above 88%.

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/arjun210/Music-Classification-Using-Machine-Learning.git
   cd Music-Classification-Using-Machine-Learning
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the feature extraction script:
   ```bash
   python feature_extraction.py
   ```
4. Train and evaluate the models:
   ```bash
   python model_training.py
   ```

## Dependencies
- **Python 3.8+**: The core programming language used.
- **music21**: For feature extraction from MIDI files.
- **scikit-learn**: For implementing machine learning models.
- **matplotlib**: For visualizing results.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.

## Project Structure

- `Music-Classification-Using-Machine-Learning/`
  - `dataset/` - MIDI files and processed datasets
  - `feature_extraction.py` - Code for extracting features from MIDI files
  - `model_training.py` - Code for training and evaluating models
  - `report.pdf` - Detailed project report
  - `presentation.pptx` - PowerPoint presentation
  - `README.md` - Project documentation
  - `requirements.txt` - List of required Python libraries




## Challenges Faced
- **Feature Extraction**: This step was the most time-consuming, taking around 4–5 hours due to various challenges like parsing errors and inconsistencies in MIDI file formats. Multiple iterations were needed to ensure accurate feature extraction.
- **Model Training and Tuning**: Hyperparameter tuning, especially for MLP and SVM models, was computationally expensive and required significant experimentation.

## Future Work
- **Dataset Expansion**: Incorporate more composers and compositions to improve model generalizability.
- **Deep Learning Models**: Explore the use of CNNs and RNNs for improved performance in music classification tasks.
- **Additional Features**: Include more complex features like tempo variations, dynamic intensity, and polyphonic texture analysis to enhance model accuracy.

## References
- music21 Documentation: https://web.mit.edu/music21  
- Scikit-learn Documentation: https://scikit-learn.org  
- [Stanford Project Report](https://cs230.stanford.edu/projects_fall_2018/reports/12441334.pdf)
- Lebar, Justin & Chang, Gary & Yu, David. (2012). Classifying Musical Scores by Composer: A machine learning approach.
- Shi, Sander. (2018). GitHub Repository. [Midi Classification Tutorial](https://github.com/sandershihacker/midi-classification-tutorial).
- Cataltepe, Zehra, et al. "Music genre classification using MIDI and audio features." *EURASIP Journal on Advances in Signal Processing* 2007, no. 1 (2007): 036409.
- Kalingeri, Vasanth, and Srikanth Grandhe. "Music generation with deep learning." arXiv preprint arXiv:1612.04928 (2016).
- Dorsey, Brannon. (2017). GitHub Repository. [Midi-Rnn](https://github.com/brannondorsey/midi-rnn/blob/master/train.py).

   
