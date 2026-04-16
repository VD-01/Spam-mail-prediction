import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    base_dir =r"C:\Minor\archive"
    data_path =os.path.join(base_dir, "email_text.csv")
    origin_path =os.path.join(base_dir, "email_origin.csv")
    models_dir =os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Load dataset
    print("Loading data...")
    try:
        df_text = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    try:
        print("Loading origin data, this might take a minute...")
        df_origin = pd.read_csv(origin_path)
        df_origin = df_origin.rename(columns={'origin': 'text'})
        df = pd.concat([df_text, df_origin], ignore_index=True)
        print("Datasets combined.")
    except Exception as e:
        print(f"Warning: Could not load origin dataset: {e}. Proceeding with email_text only.")
        df = df_text

    print(f"Dataset shape: {df.shape}")
    print("Preparing data...")
    df = df.dropna(subset=['text', 'label']).copy()
    
    df['text'] = df['text'].astype(str)
    
    X = df['text']
    y = df['label'] # 1 is Spam, 0 is Not Spam

    #Split (80/20)
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #TF-IDF Vectorization
    print("Vectorizing text using TF-IDF...")
    # Limiting features to 10,000 for efficiency and memory management, also adding common english stop words removal.
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', lowercase=True)
    X_train_idf = vectorizer.fit_transform(X_train)
    X_test_idf = vectorizer.transform(X_test)

    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_idf, y_train)

    #Model Evaluation
    print("Evaluating Model...")
    predictions = model.predict(X_test_idf)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\n--- Results ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Not Spam (0)", "Spam (1)"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Not Spam (0)", "Spam (1)"], 
                yticklabels=["Not Spam (0)", "Spam (1)"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(models_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")

    print(f"\nSaving model and vectorizer to {models_dir}...")
    with open(os.path.join(models_dir, 'spam_classifier_nb.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Done!")

if __name__ == "__main__":
    main()
