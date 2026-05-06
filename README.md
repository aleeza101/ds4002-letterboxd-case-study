# ds4002-letterboxd-case-study
# Sentiment vs. Star Ratings on Letterboxd
### DS 4002 – Case Study | Snow Stormers (Group 8) | Spring 2026

## Overview

This repository contains all materials for a case study examining whether sentiment extracted from Letterboxd movie review text aligns with the numeric star ratings assigned by the same reviewers. We apply a pretrained HuggingFace RoBERTa transformer model to 4,755 reviews from the Letterboxd Top 250 and evaluate performance overall and across review subgroups.

**Research Question:** To what extent does the sentiment extracted from a movie review's text align with the numeric star rating provided by the same reviewer?

## Repository Structure

ds4002-letterboxd-case-study/
│
├── README.md
├── data/
│   └── letterboxd_top250_reviews_clean.csv   # cleaned review-level dataset
├── src/
│   ├── hugging_face_capstone.py              # HuggingFace model inference
│   └── sentiment_analysis.py                 # VADER sentiment helper
├── notebooks/
│   └── Project#1_M13_Analysis.ipynb          # full analysis notebook
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_by_length.csv
│   ├── accuracy_by_language.csv
│   ├── accuracy_by_emoji.csv
│   └── accuracy_by_caps_ratio.csv
└── docs/
├── hook_document.pdf
└── rubric.pdf

## Software and Requirements

Python 3.x and Jupyter Notebook or VS Code are required. Install all dependencies by running pip install pandas numpy matplotlib scikit-learn transformers torch langdetect in your terminal.

## Data

The dataset is a cleaned, review-level version of the Letterboxd Top 250 dataset sourced from Kaggle. Each row represents one review and includes the review text, star rating, and engineered text features. The dataset contains 4,755 reviews across 250 films. Key columns include review_text, star_rating, review_word_len, emoji_flag, caps_ratio, and language.

## How to Reproduce Results

Clone the repository by running git clone https://github.com/aleeza101/ds4002-letterboxd-case-study.git in your terminal, then navigate into the folder with cd ds4002-letterboxd-case-study. Install dependencies with pip install -r requirements.txt. Open notebooks/Project#1_M13_Analysis.ipynb and run all cells in order — this will load and explore the cleaned dataset, run the HuggingFace sentiment model on all reviews, map star ratings to sentiment categories (1–2 = Negative, 3 = Neutral, 4–5 = Positive), compute accuracy, macro F1, and the confusion matrix, and evaluate model accuracy across subgroups including language, review length, emoji usage, and capitalization. To run model inference on its own, execute python src/hugging_face_capstone.py from the command line.

## Key Results

Overall accuracy was 36.6% with a macro F1-score of 0.223. The dominant class — positive reviews — made up 93.1% of the dataset, which is the primary driver of the model's low performance. Feature-level analysis showed that language, review length, and capitalization ratio all influence predictive accuracy, with English mid-length reviews performing most consistently.

## References

Hutto, C. J. and Gilbert, E. "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." ICWSM, 2014. https://ojs.aaai.org/index.php/ICWSM/article/view/14550

Cardiff NLP. "Twitter-RoBERTa-base for Sentiment Analysis." HuggingFace Model Hub. https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

## Authors

Aleeza Sadiq, Austin Blackburn, Jacob VanBenschoten — University of Virginia, School of Data Science
