import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
from database.database import load_recommendation_history, save_recommendation_history

# Импорт функций анализа из analyze_system и сжатия из summarize_system
from ai_tools.analyze_system import extract_tags_and_genres
from ai_tools.summarize_system import compress_text

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def format_books(book_list):
    """
    Форматирует список книг в читаемый вид.
    
    :param book_list: список книг, где каждая книга представлена словарем с ключами 'title' и 'authors'
    :return: строка с форматированным списком книг
    """
    formatted_books = []
    for book in book_list:
        title = book.get("Title", "Без названия")
        authors = book.get("Authors", "Без автора").replace("By ", "").strip()
        formatted_books.append(f"📖 {title}\n   Автор: {authors}\n")
    
    return "\n".join(formatted_books)

# ------------------------------
# Функция для перевода текста с русского на английский
def translate_to_english(text: str) -> str:
    translator = Translator()
    return translator.translate(text, dest='en').text

# ------------------------------
# Функция для вычисления косинусного сходства между двумя текстами
def compute_similarity(text1: str, text2: str) -> float:
    """
    Вычислить косинусное сходство между двумя текстами с использованием эмбеддингов.
    """
    embeddings1 = similarity_model.encode([text1], convert_to_tensor=True)
    embeddings2 = similarity_model.encode([text2], convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_similarity.item()


# ------------------------------
# Функция для извлечения предпочтений из пользовательского ввода
def get_preferences_from_input(user_input: str) -> Dict[str, List[str]]:
    """
    Получить теги и жанры из пользовательского ввода.
    Перевод осуществляем, если требуется, затем анализируем текст.
    """
    # Переводим ввод (если база на английском, можно оставить теги на английском)
    translated_input = translate_to_english(user_input)
    extracted_data = extract_tags_and_genres(translated_input)
    print(extracted_data)
    return extracted_data


# ------------------------------
# Функция для извлечения предпочтений из выбранной книги
def get_preferences_from_history(selected_history: str) -> Dict[str, List[str]]:
    """
    Сжать описание выбранной книги из истории и извлечь теги и жанры.
    """
    # Сжимаем описание книги
    compressed_text = compress_text(selected_history)
    all_tags, all_genres = [], []

    # Извлекаем теги и жанры из сжатого текста
    extracted_data = extract_tags_and_genres(compressed_text)
    print(extracted_data)
    all_tags.extend(extracted_data.get('tags', []))
    all_genres.extend(extracted_data.get('genres', []))

    # Убираем дубликаты
    return {"tags": list(set(all_tags)), "genres": list(set(all_genres))}, compressed_text

# ------------------------------
# Функция для объединения предпочтений (если учтены и ввод, и история)
def combine_preferences(pref_input: Dict[str, List[str]], pref_history: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Объединить теги и жанры, полученные из пользовательского ввода и истории.
    """
    combined_tags = list(set(pref_input.get("tags", []) + pref_history.get("tags", [])))
    combined_genres = list(set(pref_input.get("genres", []) + pref_history.get("genres", [])))
    return {"tags": combined_tags, "genres": combined_genres}

def compute_similarity(text1: str, text2: str) -> float:
    """
    Вычислить косинусное сходство между двумя текстами с использованием эмбеддингов.
    """
    embeddings1 = similarity_model.encode([text1], convert_to_tensor=True)
    embeddings2 = similarity_model.encode([text2], convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_similarity.item()


def combine_preferences(pref_input, pref_history):
    """
    Объединить теги и жанры, полученные из пользовательского ввода и истории.
    """
    combined_tags = list(set(pref_input.get("tags", []) + pref_history.get("tags", [])))
    combined_genres = list(set(pref_input.get("genres", []) + pref_history.get("genres", [])))
    return {"tags": combined_tags, "genres": combined_genres}

def filter_books_by_tags(dataset, preferences, max_candidates=40):
    """Фильтрует книги по тегам и оставляет не более max_candidates книг."""
    filtered_books = dataset[dataset["category"].apply(lambda x: any(pref in x for pref in preferences))].copy()
    return filtered_books.sample(min(len(filtered_books), max_candidates))  # Оставляем не более 20 книг

def search_books_1_mode(selected_history, dataset):
    start_time = time.time()
    """Сравнивает сжатый текст книги с 20 отфильтрованными книгами по тегам и выбирает топ-5."""
    pref_history, compressed_text = get_preferences_from_history(selected_history)
    preferences = pref_history["tags"] + pref_history["genres"]
    
    filtered_books = filter_books_by_tags(dataset, preferences)

    filtered_books["similarity"] = filtered_books["Description"].apply(lambda x: compute_similarity(compressed_text, x))

    top_books = filtered_books.sort_values(by="similarity", ascending=False).head(5)
    recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")
    for _, book in top_books.iterrows():
        print(f"{book['Title']} - {book['Authors']}, Similarity: {book['similarity']}")
    # recommendations_str = "; ".join(f"{book['Title']} - {book['Authors']}" for book in recommendations)
    print(f"Execution time: {time.time() - start_time:.4f} seconds")
    return recommendations

def search_books_2_mode(user_input, dataset):
    """Сравнивает пользовательский ввод с 20 отфильтрованными книгами по тегам и выбирает топ-5."""
    pref_input = get_preferences_from_input(user_input)
    preferences = pref_input["tags"] + pref_input["genres"]

    filtered_books = filter_books_by_tags(dataset, preferences)

    filtered_books["similarity"] = filtered_books["Description"].apply(lambda x: compute_similarity(user_input, x))
        
    top_books = filtered_books.sort_values(by="similarity", ascending=False).head(5)
    recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")
    for _, book in top_books.iterrows():
        print(f"{book['Title']} - {book['Authors']}, Similarity: {book['similarity']}")

    return recommendations

def search_books_3_mode(user_input, selected_history, dataset):
    """Объединяет предпочтения из ввода и истории, фильтрует 20 книг по тегам и выбирает топ-5."""
    pref_input = get_preferences_from_input(user_input)
    pref_history = get_preferences_from_history(selected_history)
    pref_combine = combine_preferences(pref_input, pref_history)
    preferences = pref_combine["tags"] + pref_combine["genres"]

    filtered_books = filter_books_by_tags(dataset, preferences)

    filtered_books["similarity"] = filtered_books["Description"].apply(lambda x: compute_similarity(user_input, x))

    top_books = filtered_books.sort_values(by="similarity", ascending=False).head(5)
    recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")
    for _, book in top_books.iterrows():
        print(f"{book['Title']} - {book['Authors']}, Similarity: {book['similarity']}")
    return recommendations


# def search_books_1_mode(selected_history, dataset):
#     pref_history = get_preferences_from_history(selected_history)
#     preferences = pref_history["tags"] + pref_history["genres"]
    
#     # Фильтруем книги, оставляя только те, которые содержат хотя бы одно совпадение с предпочтениями
#     filtered_books = dataset[dataset["category"].apply(lambda x: any(pref in x for pref in preferences))].copy()
#     print(filtered_books)
    
#     # Подсчет количества совпадений
#     filtered_books["match_count"] = filtered_books["category"].apply(lambda x: sum(pref in x for pref in preferences))
    
#     # Сортировка по количеству совпадений в убывающем порядке
#     top_books = filtered_books.sort_values(by="match_count", ascending=False).head(5)
    
#     # Формируем список словарей с книгами
#     recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")
#     print(recommendations)

#     # Объединяем книги в строку "Название - Автор" с разделителем "; "
#     recommendations_str = "; ".join(f"{book['Title']} - {book['Authors']}" for book in recommendations)
#     print(recommendations_str)

#     return recommendations,recommendations_str
# def search_books_2_mode(user_input, dataset):
#     pref_input = get_preferences_from_input(user_input)
    
#     # Объединяем теги и жанры в один список
#     preferences = pref_input["tags"] + pref_input["genres"]
    
#     # Фильтруем книги, оставляя только те, которые содержат хотя бы одно совпадение с предпочтениями
#     filtered_books = dataset[dataset["category"].apply(lambda x: any(pref in x for pref in preferences))].copy()
#     print(filtered_books)
    
#     # Подсчет количества совпадений
#     filtered_books["match_count"] = filtered_books["category"].apply(lambda x: sum(pref in x for pref in preferences))
    
#     # Сортировка по количеству совпадений в убывающем порядке
#     top_books = filtered_books.sort_values(by="match_count", ascending=False).head(5)
    
#     # Формируем список словарей с книгами
#     recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")

#     # Объединяем книги в строку "Название - Автор" с разделителем "; "
#     recommendations_str = "; ".join(f"{book['Title']} - {book['Authors']}" for book in recommendations)

#     return recommendations, recommendations_str


# def search_books_3_mode(user_input, selected_history, dataset):
#     """
#     Поиск книг в базе с учетом категории и вычисления косинусного сходства между описаниями книг и текстами ввода/истории.
#     """
#     # Получаем предпочтения из ввода и истории
#     pref_input = get_preferences_from_input(user_input)
#     pref_history = get_preferences_from_history(selected_history)

#     # Объединяем предпочтения
#     pref_combine = combine_preferences(pref_input, pref_history)
#     preferences = pref_combine["tags"] + pref_combine["genres"]
#     # Фильтруем книги, оставляя только те, которые содержат хотя бы одно совпадение с предпочтениями
#     filtered_books = dataset[dataset["category"].apply(lambda x: any(pref in x for pref in preferences))].copy()
#     print(filtered_books)
    
#     # Подсчет количества совпадений
#     filtered_books["match_count"] = filtered_books["category"].apply(lambda x: sum(pref in x for pref in preferences))
    
#     # Сортировка по количеству совпадений в убывающем порядке
#     top_books = filtered_books.sort_values(by="match_count", ascending=False).head(5)
    
#     # Формируем список словарей с книгами
#     recommendations = top_books[["Title", "Authors"]].to_dict(orient="records")

#     # Объединяем книги в строку "Название - Автор" с разделителем "; "
#     recommendations_str = "; ".join(f"{book['Title']} - {book['Authors']}" for book in recommendations)

#     return recommendations, recommendations_str


# ------------------------------
# Функция для формирования рекомендаций по предпочтениям пользователя
def get_book_recommendations(user_input, selected_book, mode, user_id, history_exclude_option, dataset):
    """
    Формирует рекомендации по трём режимам:
      1. Учитывать и пользовательский ввод, и выбранные книги из истории.
      2. Учитывать только пользовательский ввод.
      3. Учитывать только выбранные книги из истории.
    Опционально исключает ранее рекомендованные книги.
    """

    recommendations = []
    if history_exclude_option:
        previously_recommended = load_recommendation_history(user_id)
        dataset = dataset[~dataset['Title'].isin(previously_recommended)]

    if mode == 1:
        # Режим 1: объединение пользовательского ввода и истории
        recommendations = search_books_1_mode(selected_book, dataset)

    elif mode == 2:
        # Режим 2: учитывать только пользовательский ввод
        recommendations = search_books_2_mode(user_input, dataset)

    elif mode == 3:
        # Режим 3: учитывать только книги из истории
        recommendations= search_books_3_mode(user_input, selected_book, dataset)

    # Сохраняем историю рекомендованных книг
    save_recommendation_history(user_id, [book['Title'] for book in recommendations])
   
    return format_books(recommendations)
