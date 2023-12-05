from django.shortcuts import render


def index(request):
    #Extractive_Summarisation

    def preprocess(sentences):
        import spacy as sp
        nlp = sp.load("en_core_web_sm")
        preprocessed_sentences = []

        for sentence in sentences:
            doc = nlp(sentence)
            extracted_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            preprocessed_sentences.append(" ".join(extracted_words))
        return preprocessed_sentences

    def generate_summary(text, no_of_sentences=5):
        if len(text.strip()) == 0:
            return "No Input to Summarize...!"

        # Loading Libraries
        import nltk
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re
        try:
            # Cleaning Data
            text = re.sub(r'\[[^\]]*\]', ' ', text)
            text = re.sub(r' +', ' ', text)

            # Original Sentences
            nltk.download('punkt')
            original_sentences = nltk.sent_tokenize(text)

            if len(original_sentences) < 10:
                max_len = len(original_sentences)
                min_len = 1
                interval = max_len // 4
                intervals = list(range(min_len, max_len, interval))
                if no_of_sentences == 4:
                    no_of_sentences = intervals[1]
                elif no_of_sentences == 7:
                    no_of_sentences = intervals[2]
                elif no_of_sentences == 10:
                    no_of_sentences = intervals[-1]

            # preprocess

            preprocessed_sentences = preprocess(original_sentences)

            # Vectorizer
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(preprocessed_sentences)

            # Calculate sentence scores
            sent_scores = matrix.toarray().sum(axis=1)

            # Rank the scores
            ranked_scores = (-sent_scores).argsort()

            top_score_indices = sorted(ranked_scores[:no_of_sentences])

            # Select the top sentences
            final_sentences = [original_sentences[i] for i in top_score_indices]

            # Join the top sentences to form the summary
            return " ".join(final_sentences)
        except Exception as e:
            return f"Unable to Summarize...!"

    if request.method == 'POST':
        text = request.POST.get('text_input', 'default')
        text_len = request.POST.get('text_len')
        summary = generate_summary(text, int(text_len))
        return render(request, 'Home.html', {'user_text': text, 'summary_text': summary})
    else:
        # Render the initial form
        return render(request, 'Home.html')


def about(request):
    return render(request, 'About.html')

