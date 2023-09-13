import os
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from textblob import TextBlob
from collections import Counter
from langdetect import detect
import pandas as pd
import langid
import re
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from gensim import corpora, models
import seaborn as sns

import argparse
# YouTube API key
api_key = "AIzaSyCaTiHDbu9UsVKVpHSxsmg80_0u0viaM9E"

# YouTube API client
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

def extract_video_id(video_link):
    # Regular expressions to match both standard and shortened formats
    standard_pattern = r'(?:\?v=|&v=|/videos/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|v\/|youtube.com\/watch\?v=)([a-zA-Z0-9_-]{11})'
    shortened_pattern = r'youtu.be\/([a-zA-Z0-9_-]{11})'

    # Attempt to match the video ID in the link
    match_standard = re.search(standard_pattern, video_link)
    match_shortened = re.search(shortened_pattern,video_link)

    if match_standard:
        return match_standard.group(1)
    elif match_shortened:
        return match_shortened.group(1)
    else:
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform YouTube video analysis')
    parser.add_argument('video_link', type=str, help='YouTube video link')
    args = parser.parse_args()
    return args.video_link

# video_id = "Fu1Xjdnbba8"  #actual video ID
video_link = parse_arguments()
video_id=extract_video_id(video_link)

try:
    video_info = youtube.videos().list(
        part="status",
        id=video_id
    ).execute()

    if "items" in video_info and video_info["items"]:
        privacy_status = video_info["items"][0]["status"]["privacyStatus"]

        if privacy_status == "public":
            print("The video is public and accessible.")
        elif privacy_status == "private":
            print("The video is private and not accessible.")
        elif privacy_status == "unlisted":
            print("The video is unlisted and accessible with a direct link.")
        else:
            print("The video may have age restrictions or other privacy settings.")
    else:
        print("Video details not found. Please check the video ID.")
except HttpError as e:
    error_message = e.content.decode("utf-8")
    print(f"An error occurred: {error_message}")

# Get video comments
non_text_pattern = r'[^\x00-\x7F]+'
def remove_non_text(comment):
    return re.sub(non_text_pattern, '', comment)
comments = []
sentiments = []
polarities = []
languages=[]
common_keyword=[]
comment_timestamps = []
top_level_comments = []
try:
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100 
    ).execute()
    if "items" not in results:
        print("No comments found for this video.")
    while results:
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = remove_non_text(comment["textDisplay"])
            timestamp = comment["publishedAt"]
            if len(comment_text) >= 5:
                comments.append(comment_text)
                comment_timestamps.append(timestamp)
        if "nextPageToken" in results:
            next_page_token = results["nextPageToken"]
            # Use the nextPageToken to fetch the next page of comments
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=next_page_token
            ).execute()
        else:
            break

except HttpError as e:
    error_message = e.content.decode("utf-8")
    print(f"An error occurred: {error_message}")

if not comments:
    print("Comment Not Found")
else:
# Topic Modeling (LDA)
    nlp = spacy.load("en_core_web_sm")
    comments_text = ' '.join(comments)
    doc = nlp(comments_text)
    tokens = [token.text for token in doc if token.is_alpha]
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print("Topic:", topic)

    topic_labels, topic_words = zip(*topics)

    # Define colors for the bars
    topic_colors = ['blue', 'green', 'red', 'purple', 'orange']  # You can define your own colors

    # Create a bar chart for each topic
    plt.figure(figsize=(12,6))
    for i, (label, words) in enumerate(zip(topic_labels, topic_words)):
        word_list = words.split()[:5]  # Display the top 5 words for each topic
        plt.barh(label, 1, color=topic_colors[i], label=f"{label}: {', '.join(word_list)}")

    plt.xlabel("Word Probability")
    plt.ylabel("Topic")
    plt.title("Topic Analysis")
    plt.gca().invert_yaxis()  # Invert the y-axis for readability

    # Add a legend for colors
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), title="Topics")
    plt.show()


    # Named Entity Recognition (NER)
    entities = []
    for comment in comments:
        doc = nlp(comment)
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
    entity_counts = Counter(entities)
    print("Top Named Entities:", entity_counts.most_common(10))

    # Text Classification (Example: Spam Detection)
    spam_keywords = ["click", "buy", "free", "win", "discount"]
    spam_count = 0
    for comment in comments:
        if any(keyword in comment.lower() for keyword in spam_keywords):
            spam_count += 1
    print("Spam Comments:", spam_count)

    # Word Cloud
    comment_text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, stopwords=set(STOPWORDS)).generate(comment_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


    emotion_lexicon = {}
    lexicon_file_path = "C:/Users/Admin/Desktop/post_analysis/post_analysis/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt" 
    with open(lexicon_file_path, "r",encoding="utf-8") as lexicon_file:
        for line in lexicon_file:
            parts = line.strip().split('\t')[:3]
            if len(parts) >= 3:
                word, emotion, value = parts
                if word not in emotion_lexicon:
                    emotion_lexicon[word] = []
                emotion_lexicon[word].append(emotion)
            else:
                print(f"Ignored line: {line.strip()}")

    # Function to analyze emotions in a comment
    def analyze_emotions(comment):
        tokens = comment.split()  # Tokenize the comment (you can use a more advanced tokenizer if needed)
        emotion_counts = Counter()

        for token in tokens:
            if token in emotion_lexicon:
                emotions = emotion_lexicon[token]
                emotion_counts.update(emotions)

        return emotion_counts

    # Analyze emotions for each comment
    comment_emotions = [analyze_emotions(comment) for comment in comments]

    # Print the emotion analysis for the first comment
    print("Emotion Analysis for the First Comment:")
    print(comment_emotions[0])

    # Aggregate emotion counts across all comments
    total_emotion_counts = Counter()
    for emotion_count in comment_emotions:
        total_emotion_counts.update(emotion_count)

    emotion_colors = {
        "anger": "red",
        "fear": "orange",
        "sadness": "yellow",
        "joy": "green",
        "surprise": "purple",
    }

    # Plot the emotion analysis
    if total_emotion_counts:
        emotions, counts = zip(*total_emotion_counts.items())
        colors = [emotion_colors.get(emotion, "gray") for emotion in emotions]
        plt.figure(figsize=(10, 6))
        plt.bar(emotions, counts, color=colors)
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.title("Emotion Analysis Based on Comments")
        plt.xticks(rotation=45)
        legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=emotion) for emotion, color in emotion_colors.items()]
        plt.legend(handles=legend_labels, title="Emotion Categories")
        plt.show()
    else:
        print("No emotions detected in the comments.")


    comments_df = pd.DataFrame({"Timestamp": comment_timestamps, "Comment": comments})
    comments_df["Timestamp"] = pd.to_datetime(comments_df["Timestamp"])
    comments_df = comments_df.sort_values(by="Timestamp")
    comments_df = comments_df.reset_index(drop=True)

    # Group comments by day and count the number of comments per day
    daily_comment_counts = comments_df.resample('D', on='Timestamp').count()

    # Plot the daily comment frequency
    plt.figure(figsize=(12, 6))
    plt.plot(daily_comment_counts.index, daily_comment_counts["Comment"])
    plt.xlabel("Date")
    plt.ylabel("Comment Count")
    plt.title("Comment Frequency Over Time")
    plt.show()

    comments_df["Sentiment"] = [TextBlob(comment).sentiment.polarity for comment in comments_df["Comment"]]

    # Group comments by day and calculate the average sentiment score per day
    daily_sentiment = comments_df.resample('D', on='Timestamp')["Sentiment"].mean()

    # Plot the daily sentiment scores
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment.index, daily_sentiment)
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.title("Sentiment Analysis Over Time")
    plt.show()



    positive_count = 0
    neutral_count = 0
    negative_count = 0
    keyword_counts = Counter()
    language_keyword_counts = {} 

    for comment in comments:
    
        lang, _ = langid.classify(comment)
        languages.append(lang)

        analysis = TextBlob(comment)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            sentiment = "Positive"
            positive_count += 1
        elif polarity == 0:
            sentiment = "Neutral"
            neutral_count+=1
        else:
            sentiment = "Negative"
            negative_count+=1

        sentiments.append(sentiment)
        polarities.append(polarity)

        print(f"Comment: {comment}")
        print(f"Language: {lang}")
        print(f"Sentiment: {sentiment}")
        print(f"Polarity: {polarity}\n")

        words = analysis.words
        keyword_counts.update(words)
        most_common_keyword, _ = keyword_counts.most_common(1)[0]
        common_keyword.append(most_common_keyword)

        if lang not in language_keyword_counts:
            language_keyword_counts[lang] = Counter()
        language_keyword_counts[lang].update(words)


    total_comments = len(comments)

    positive_percentage = round((positive_count / total_comments) * 100, 2)
    neutral_percentage = round((neutral_count / total_comments) * 100,2)
    negative_percentage = round((negative_count / total_comments) * 100,2)

    print(f"Positive Comments: {positive_count} ({positive_percentage:}%)")
    print(f"Neutral Comments: {neutral_count} ({neutral_percentage:}%)")
    print(f"Negative Comments: {negative_count} ({negative_percentage:}%)")

    most_common_keyword, most_common_count = keyword_counts.most_common(1)[0]

    print(f"The most common keyword is '{most_common_keyword}' with {most_common_count} occurrences.")

    # for lang, lang_counts in language_keyword_counts.items():
    #     most_common_lang_keyword, most_common_lang_count = lang_counts.most_common(1)[0]
    #     print(f"The most common keyword in {lang} is '{most_common_lang_keyword}' with {most_common_lang_count} occurrences.")
        
        

    positive_percentage_str = f"{positive_percentage}%"
    neutral_percentage_str = f"{neutral_percentage}%"
    negative_percentage_str = f"{negative_percentage}%"

    result_data = {
        'Comment': comments,
        'Language': languages,
        'Sentiment': sentiments,
        'Polarity': polarities,
        'Most Common Keyword':common_keyword,
        'Positive':positive_percentage_str,
        'Neutral':neutral_percentage_str,
        'Negative':negative_percentage_str,
        
    }

    df = pd.DataFrame(result_data)


    # # Sentiment Analysis Visualization
    # sns.set(style="whitegrid")
    # sentiment_counts = Counter(sentiments)
    # sentiment_df = pd.DataFrame(sentiment_counts.items(), columns=["Sentiment", "Count"])
    # if not sentiment_df.empty:
    #     plt.figure(figsize=(8, 5))
    #     sns.barplot(x="Sentiment", y="Count", data=sentiment_df)
    #     plt.title("Sentiment Analysis")
    #     plt.show()
    # else:
    #     print("No sentiment data to plot.")

    sentiment_counts = df['Sentiment'].value_counts()
    colors = ['blue', 'green', 'red'] 
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Sentiment Distribution')
    plt.axis('equal') 
    plt.show()

    # Group the DataFrame by language and find the most common keyword in each group
    most_common_keywords = df.groupby('Language')['Most Common Keyword'].first()
    # Count the number of occurrences for each most common keyword
    keyword_counts = df['Most Common Keyword'].value_counts()
    # Define colors for the bars
    color_palette = plt.colormaps.get_cmap('tab20')
    # Create a horizontal bar chart
    plt.figure(figsize=(12, 8))

    for i, (lang, keyword) in enumerate(most_common_keywords.items()):
        color = color_palette(i)
        count = keyword_counts[keyword]
        
        plt.barh(lang, count, color=color, label=f"{keyword} ({count})")

    plt.title('Most Common Keyword in Each Language')
    plt.xlabel('Keyword Count')
    plt.legend()
    plt.gca().invert_yaxis() 
    plt.show()
    df.to_csv('sentiment_results.csv', index=False)








