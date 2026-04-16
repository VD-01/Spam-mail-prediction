import os
import sys
import pickle

def main():
    base_dir = r"C:\Minor\archive"
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, 'spam_classifier_nb.pkl')
    vec_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("Error: Saved models not found. Please run 'python train.py' first.")
        sys.exit(1)

    print("Loading model and vectorizer...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    if len(sys.argv) > 1:
        messages = [" ".join(sys.argv[1:])]
    else:
        print("\nNo message provided. Running default tests...")
        messages =[
            "hi i've just updated from the gulus and i checked out the latest patch notes. Everything seems to be working fine on our end.",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
            "Please review the attached quarterly financial report and let me know if you have any questions before our 10 AM meeting.",
            "Mega authentic v i a g r a discount price! Click the link below to claim your prescription instantly.",
            "Hey team, just a quick reminder about the upcoming deadline for the Q3 project deliverables. Let's make sure we hit our targets!"
        ]

    print("\nPredictions:\n")
    for msg in messages:
        msg_idf = vectorizer.transform([msg])
        
        prediction = model.predict(msg_idf)[0]
        probs = model.predict_proba(msg_idf)[0]
        
        label ="Spam" if prediction ==1 else "Not Spam"
        confidence = probs[1] if prediction ==1 else probs[0]
        
        print(f"Message:'{msg}'")
        print(f"Prediction: [{label}] (Confidence: {confidence*100:.1f}%)\n")

if __name__ == "__main__":
    main()
