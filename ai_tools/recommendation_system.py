import time
import asyncio # Added asyncio
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

# --- Use updated DB functions ---
import database.database as db
# --- Use updated AI tools (now async) ---
from ai_tools.analyze_system import extract_tags_and_genres # Now async
# We need a way to summarize for mode 1 & 3, use the clients from analyze_system
from ai_tools.analyze_system import mixtral_client, mixtral_model, summarize_client, summarize_model
import re # For cleaning summaries
import pandas as pd

# --- Model Initialization ---
# Similarity model remains synchronous
try:
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SentenceTransformer model loaded.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    similarity_model = None

# --- Helper Functions ---

def format_books(book_list: List[Dict]) -> str:
    """
    Formats a list of book dictionaries into a readable string.
    """
    if not book_list:
        return "К сожалению, не удалось подобрать рекомендации."

    formatted_books = []
    for i, book in enumerate(book_list):
        title = book.get("Title", "Без названия")
        # Assuming authors might be string like 'Author One, Author Two'
        authors_raw = book.get("Authors", "Автор неизвестен")
        # Simple cleaning for author string if needed
        authors = authors_raw.replace("By ", "").strip() if isinstance(authors_raw, str) else "Автор неизвестен"
        formatted_books.append(f"{i+1}. *{title}*\n   👤 _{authors}_") # Using Markdown

    return "\n\n".join(formatted_books)

# Translation removed - not used in user_handlers workflow

def compute_similarity(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two texts using SentenceTransformer.
    Returns 0.0 if the model isn't loaded or texts are invalid.
    """
    if not similarity_model:
        print("Similarity model not loaded.")
        return 0.0
    if not isinstance(text1, str) or not isinstance(text2, str) or not text1 or not text2:
         # print("Invalid input for similarity computation.")
         return 0.0 # Avoid errors with empty strings

    try:
        # Ensure texts are not excessively long (can cause memory issues)
        max_len = 10000 # Limit text length for embedding
        text1 = text1[:max_len]
        text2 = text2[:max_len]

        embeddings1 = similarity_model.encode([text1], convert_to_tensor=True)
        embeddings2 = similarity_model.encode([text2], convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarity.item()
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0

# --- Preference Extraction Functions (Async Adapted) ---

async def get_preferences_from_input(user_input: str) -> Dict[str, List[str]]:
    """
    Extracts tags and genres from user's text input using AI model.
    """
    # Translation removed
    # Call the adapted async function
    extracted_data = await extract_tags_and_genres(user_input)
    print(f"Preferences from input '{user_input[:50]}...': {extracted_data}")
    return extracted_data

async def summarize_for_prefs(text: str, target_words: int = 150) -> str:
    """
    Helper to summarize text specifically for preference extraction.
    Uses the Summarization client (DeepSeek).
    """
    if not summarize_client or not summarize_model:
         print("Summarization client not available for summarize_for_prefs.")
         # Return a chunk of the original text as fallback
         return text[:target_words * 10]

    # Simple summarization prompt
    prompt = (
         f"Сделай очень краткое содержание следующего текста на русском языке (примерно {target_words} слов), "
         f"сфокусировавшись на ключевых темах, стиле и атмосфере. "
         "Это нужно для подбора похожих книг. "
         "Ответ должен быть только сжатым текстом."
    )
    messages = [
         {"role": "system", "content": prompt},
         {"role": "user", "content": text}
    ]
    try:
         completion = await summarize_client.chat_completion(
             model=summarize_model,
             messages=messages,
             max_tokens=int(target_words * 3),
             temperature=0.5
         )
         summary = completion.choices[0].message.content.strip()
         summary = re.sub(r'^(В этом фрагменте|В данном тексте|Этот сегмент о)\s*:?\s*', '', summary, flags=re.IGNORECASE).strip()
         return summary
    except Exception as e:
         print(f"Error in summarize_for_prefs: {e}")
         return text[:target_words * 10] # Fallback

async def get_preferences_from_book_content(book_content: str) -> Dict[str, List[str]]:
    """
    Summarizes book content and extracts tags/genres from the summary.
    """
    if not book_content:
        return {"tags": [], "genres": []}

    # Summarize the book content first (using helper)
    # Limit input length to avoid excessive summarization time/cost
    max_content_len = 20000 # Approx 20k chars
    compressed_text = await summarize_for_prefs(book_content[:max_content_len], target_words=200) # Summarize to ~200 words

    # Extract preferences from the summary
    extracted_data = await extract_tags_and_genres(compressed_text)
    print(f"Preferences from book content summary: {extracted_data}")
    return extracted_data


def combine_preferences(pref_input: Dict[str, List[str]], pref_history: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Combines tags and genres from two preference dictionaries.
    """
    # Ensure keys exist and contain lists
    tags_input = pref_input.get("tags", []) if isinstance(pref_input.get("tags"), list) else []
    genres_input = pref_input.get("genres", []) if isinstance(pref_input.get("genres"), list) else []
    tags_history = pref_history.get("tags", []) if isinstance(pref_history.get("tags"), list) else []
    genres_history = pref_history.get("genres", []) if isinstance(pref_history.get("genres"), list) else []

    # Combine all into a single list for filtering convenience
    combined_all = list(set(tags_input + genres_input + tags_history + genres_history))
    # Return combined list, maybe under a single key? Let's keep the dict structure.
    # Return unique tags and genres separately
    combined_tags = list(set(tags_input + tags_history))
    combined_genres = list(set(genres_input + genres_history))
    return {"tags": combined_tags, "genres": combined_genres}

def filter_books_by_prefs(dataset, preferences: Dict[str, List[str]], max_candidates=50) -> 'pd.DataFrame':
    """
    Filters the dataset based on combined tags and genres.
    Handles potential string vs list 'category' column.
    """
    prefs_list = list(set(preferences.get("tags", []) + preferences.get("genres", [])))
    if not prefs_list or dataset.empty:
        # If no preferences or empty dataset, return a sample of the dataset or empty
        return dataset.sample(min(len(dataset), max_candidates)) if not dataset.empty else dataset

    # Check the type of the 'category' column
    if not dataset.empty and isinstance(dataset.iloc[0]['category'], str):
        # If it's a string (e.g., "['Fiction', 'Sci-Fi']"), try to evaluate it or use string matching
        try:
            # Attempt to evaluate the string representation of the list
            dataset['category_list'] = dataset['category'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
            filtered_books = dataset[dataset["category_list"].apply(lambda x: isinstance(x, list) and any(p.lower() in item.lower() for item in x for p in prefs_list))].copy()
            filtered_books.drop(columns=['category_list'], inplace=True, errors='ignore')
        except:
            # Fallback to simple string contains matching if eval fails
             pattern = '|'.join(re.escape(p) for p in prefs_list)
             filtered_books = dataset[dataset["category"].str.contains(pattern, case=False, na=False)].copy()

    elif not dataset.empty and isinstance(dataset.iloc[0]['category'], list):
        # If it's already a list
        filtered_books = dataset[dataset["category"].apply(lambda x: isinstance(x, list) and any(p.lower() in item.lower() for item in x for p in prefs_list))].copy()
    else:
        # Unknown format or empty dataset
        print("Warning: 'category' column format not recognized or dataset empty. Returning random sample.")
        return dataset.sample(min(len(dataset), max_candidates)) if not dataset.empty else dataset

    print(f"Filtered down to {len(filtered_books)} candidates based on preferences.")
    # Return a sample of the filtered books
    return filtered_books.sample(min(len(filtered_books), max_candidates), random_state=1) # Added random_state for consistency

# --- Search Modes (Async Adapted) ---

async def search_books_1_mode(book_content: str, dataset: 'pd.DataFrame') -> List[Dict]:
    """Mode 1: Recommendations based on a given book's content."""
    start_time = time.time()
    print("Recommendation Mode 1: Based on book content")
    # Extract preferences directly from the provided book content
    prefs_book = await get_preferences_from_book_content(book_content)
    if not prefs_book.get("tags") and not prefs_book.get("genres"):
         print("Warning: Could not extract preferences from book content.")
         # Fallback: maybe recommend based purely on description similarity? Or return empty?
         # Let's try filtering with empty prefs (will return random sample)

    # Filter candidates based on extracted preferences
    candidate_books = filter_books_by_prefs(dataset, prefs_book)
    if candidate_books.empty:
        return []

    # Summarize the input book content again for similarity comparison (shorter summary)
    summary_for_similarity = await summarize_for_prefs(book_content, target_words=100)

    # Compute similarity between the input book's summary and candidate descriptions
    candidate_books["similarity"] = candidate_books["description"].apply(
        lambda desc: compute_similarity(summary_for_similarity, desc)
    )

    # Select top N based on similarity
    top_books = candidate_books.sort_values(by="similarity", ascending=False).head(5)
    recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")

    # Debug print
    for _, book in top_books.iterrows():
        print(f"  - {book['Title']} ({book['Authors']}), Similarity: {book['similarity']:.4f}")
    print(f"Mode 1 execution time: {time.time() - start_time:.4f} seconds")
    return recommendations

async def search_books_2_mode(user_input: str, dataset: 'pd.DataFrame') -> List[Dict]:
    """Mode 2: Recommendations based on user's text description."""
    start_time = time.time()
    print("Recommendation Mode 2: Based on user input")
    # Extract preferences from user input
    prefs_input = await get_preferences_from_input(user_input)
    if not prefs_input.get("tags") and not prefs_input.get("genres"):
         print("Warning: Could not extract preferences from user input.")

    # Filter candidates
    candidate_books = filter_books_by_prefs(dataset, prefs_input)
    if candidate_books.empty:
        return []

    # Compute similarity between the user input and candidate descriptions
    candidate_books["similarity"] = candidate_books["description"].apply(
        lambda desc: compute_similarity(user_input, desc)
    )

    # Select top N
    top_books = candidate_books.sort_values(by="similarity", ascending=False).head(5)
    recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")

    # Debug print
    for _, book in top_books.iterrows():
        print(f"  - {book['Title']} ({book['Authors']}), Similarity: {book['similarity']:.4f}")
    print(f"Mode 2 execution time: {time.time() - start_time:.4f} seconds")
    return recommendations

async def search_books_3_mode(user_input: str, book_content: str, dataset: 'pd.DataFrame') -> List[Dict]:
    """Mode 3: Recommendations based on user input AND a book's content/style."""
    start_time = time.time()
    print("Recommendation Mode 3: Based on user input + book content")
    # Extract preferences from both sources
    prefs_input = await get_preferences_from_input(user_input)
    prefs_book = await get_preferences_from_book_content(book_content)

    # Combine preferences
    prefs_combined = combine_preferences(prefs_input, prefs_book)
    if not prefs_combined.get("tags") and not prefs_combined.get("genres"):
         print("Warning: Could not extract combined preferences.")

    # Filter candidates
    candidate_books = filter_books_by_prefs(dataset, prefs_combined)
    if candidate_books.empty:
        return []

    # Compute similarity based on USER INPUT primarily, as style is already factored into filtering
    candidate_books["similarity"] = candidate_books["description"].apply(
        lambda desc: compute_similarity(user_input, desc)
    )

    # Select top N
    top_books = candidate_books.sort_values(by="similarity", ascending=False).head(5)
    recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")

    # Debug print
    for _, book in top_books.iterrows():
        print(f"  - {book['Title']} ({book['Authors']}), Similarity: {book['similarity']:.4f}")
    print(f"Mode 3 execution time: {time.time() - start_time:.4f} seconds")
    return recommendations


# --- Main Recommendation Function (Async Adapted) ---

async def get_book_recommendations(
    user_prefs: str,              # User text input (mode 2, 3)
    book_content: str,            # Full text of selected book (mode 1, 3)
    mode: int,                    # 1, 2, or 3
    user_id: int,                 # For history
    exclude_previous: bool,       # Flag to exclude history
    dataset: 'pd.DataFrame'       # Pre-loaded dataset from user_handlers
) -> str:                         # Returns formatted string
    """
    Generates book recommendations based on the specified mode.
    Handles exclusion of previously recommended books.
    """
    print(f"Generating recommendations: mode={mode}, exclude={exclude_previous}, user={user_id}")
    loop = asyncio.get_running_loop()
    recommendations = []
    processed_dataset = dataset.copy() # Work on a copy

    # --- Exclusion Logic ---
    if exclude_previous:
        try:
            previously_recommended_titles = await loop.run_in_executor(
                None, db.load_recommendation_history, user_id
            )
            if previously_recommended_titles:
                print(f"Excluding {len(previously_recommended_titles)} previously recommended titles.")
                processed_dataset = processed_dataset[
                    ~processed_dataset['Title'].astype(str).str.lower().isin([t.lower() for t in previously_recommended_titles])
                ]
                if processed_dataset.empty and not dataset.empty:
                     print("Warning: Dataset became empty after excluding previous recommendations. Using original dataset.")
                     processed_dataset = dataset.copy() # Fallback if exclusion removes everything
        except Exception as e:
            print(f"Error loading or applying recommendation history for user {user_id}: {e}")
            # Continue without exclusion on error

    if processed_dataset.empty:
        print("Dataset is empty, cannot generate recommendations.")
        return "К сожалению, база данных книг пуста или не загружена."

    # --- Mode Logic ---
    try:
        if mode == 1:
            if not book_content: return "Ошибка: Для режима 1 нужен текст книги-примера."
            recommendations = await search_books_1_mode(book_content, processed_dataset)
        elif mode == 2:
            if not user_prefs: return "Ошибка: Для режима 2 нужно описание желаемой книги."
            recommendations = await search_books_2_mode(user_prefs, processed_dataset)
        elif mode == 3:
            if not user_prefs or not book_content: return "Ошибка: Для режима 3 нужно и описание, и текст книги-примера."
            recommendations = await search_books_3_mode(user_prefs, book_content, processed_dataset)
        else:
            return f"Ошибка: Неизвестный режим рекомендаций ({mode})."

        # --- Save History ---
        if recommendations:
            try:
                titles_to_save = [book['Title'] for book in recommendations if 'Title' in book]
                if titles_to_save:
                    await loop.run_in_executor(None, db.save_recommendation_history, user_id, titles_to_save)
            except Exception as e:
                print(f"Error saving recommendation history for user {user_id}: {e}")

    except Exception as e:
        print(f"Error during recommendation generation (mode {mode}, user {user_id}): {e}")
        # Optionally log traceback: import traceback; traceback.print_exc()
        return "Произошла ошибка при подборе рекомендаций."

    # --- Format and Return ---
    return format_books(recommendations)