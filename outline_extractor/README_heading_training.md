# Heading Classifier Training (Offline, Hackathon-Ready)

## Overview
This script (`train_heading_classifier.py`) trains a lightweight RandomForest model to classify PDF text spans as headings (H1/H2/H3) or not, using features extracted from PyMuPDF. The model is saved as `heading_classifier.pkl` for use in the main extractor.

## How to Use

1. **Add Sample PDFs**
   - Place a few representative PDFs (with real headings) in the `outline_extractor/` directory as `sample1.pdf`, `sample2.pdf`, etc.
   - For best results, use PDFs with clear heading structure (e.g., arXiv, PubMed, or your own labeled data).

2. **Run the Training Script**
   ```sh
   python train_heading_classifier.py
   ```
   - This will extract features, train a RandomForest, print a classification report, and save the model as `heading_classifier.pkl`.
   - It will also save a sample CSV of features/labels for reproducibility.

3. **Use the Model in the Extractor**
   - The main extractor will automatically use `heading_classifier.pkl` if present for heading detection.

## Customization
- You can expand the dataset by adding more PDFs and/or labeling headings more precisely.
- For hackathon use, keep the model and all data local (no online calls).
- The model is well under 200MB and works fully offline.

## Files
- `train_heading_classifier.py`: Training script
- `heading_classifier.pkl`: Trained model (used by extractor)
- `sample_heading_features.csv`: Sample features/labels for reproducibility
- `sample1.pdf`, `sample2.pdf`, ...: Example PDFs for training

## Tips
- For best results, use a diverse set of PDFs and label headings accurately.
- You can use any scikit-learn classifier (RandomForest, SVM, etc.) as long as the model is saved as `heading_classifier.pkl`.
- The extractor will fallback to rules/regex if the model is missing.

---

**This setup ensures your PDF outline extractor is robust, offline, and hackathon-ready!** 