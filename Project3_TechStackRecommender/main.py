import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# SETUP: Load the Job Role Database
# ==========================================
df = pd.read_csv('raw_skills.csv')

# ==========================================
# STEP 1: INGESTION (Bypassing Cold Start)
# ==========================================
# We use an onboarding survey to force ingestion and capture the user state.
print("--- TECH STACK RECOMMENDER ---")
print("Enter at least 3 skills to map your career profile (e.g., 'python blender deep learning').")
user_input = input("Your Skills: ")

# ==========================================
# STEP 2 & 3: VECTOR MAPPING & SCORING
# ==========================================
# We combine the user input with the dataset to ensure a shared vocabulary space.
all_text_data = df['Skills'].tolist()
all_text_data.append(user_input)

# Apply TF-IDF Weighting to penalize generic terms and reward specific ones.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text_data)

# Separate the item vectors (job roles) from the user vector (last item in the matrix).
job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

# Calculate Cosine Similarity: The mathematical angle between the user and job vectors.
# This renders the output invariant to vector magnitude.
similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

# Add the calculated scores back to our dataframe
df['Match_Score'] = similarity_scores

# ==========================================
# STEP 4: SORTING & FILTERING
# ==========================================
# Sort the results in descending order based on the cosine similarity scores.
sorted_df = df.sort_values(by='Match_Score', ascending=False)

# Truncate the output to generate the Top-N list (Top 3) to prevent choice overload.
top_3_roles = sorted_df.head(3)

# ==========================================
# OUTPUT: The Top-N List
# ==========================================
print("\n--- YOUR TOP CAREER MATCHES ---")
for index, row in top_3_roles.iterrows():
    # Convert the decimal score to a percentage for a commercial-grade UI experience.
    match_percentage = round(row['Match_Score'] * 100, 1)
    print(f"Role: {row['Role']} | Confidence Match: {match_percentage}%")
    print(f"Required Skills: {row['Skills']}\n")
