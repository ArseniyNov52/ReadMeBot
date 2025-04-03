import asyncio
import re
from typing import List, Dict
from huggingface_hub import AsyncInferenceClient # Use Async client

# --- Database Import ---
import database.database as db # Use updated db functions

# --- Client Initialization ---
# Client for Mixtral (as used in original analyze_system)
# NOTE: Ensure this API key and provider are correct for Mixtral usage
try:
    mixtral_client = AsyncInferenceClient(
        provider="hf-inference", # Or "together", etc. depending on where you run Mixtral
        api_key="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" # Key from original analyze_system
    )
    mixtral_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print("Mixtral client initialized.")
except Exception as e:
    print(f"Error initializing Mixtral client: {e}")
    mixtral_client = None
    mixtral_model = None

# Client for DeepSeek (as used in updated summarize_system - needed for update_book_content summary)
# NOTE: Ensure this API key and provider are correct for DeepSeek usage
try:
    # Assuming summarize_system's client setup is desired for summarization tasks here too
    # If you want to use Mixtral for everything, adjust accordingly.
    summarize_api_keys = ['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'] # Key from summarize_system
    summarize_api_key_index = 0
    summarize_client = AsyncInferenceClient(
        provider="together", # Provider from summarize_system
        api_key=summarize_api_keys[summarize_api_key_index]
    )
    summarize_model = "deepseek-ai/DeepSeek-V3" # Model from summarize_system
    print("Summarization (DeepSeek) client initialized.")
except Exception as e:
    print(f"Error initializing Summarization (DeepSeek) client: {e}")
    summarize_client = None
    summarize_model = None
# ------------------------------
# Renamed and adapted from determine_text_type
async def extract_text_parameters(text_fragment: str) -> str:
    """
    Analyzes a text fragment to extract key parameters characteristic of its type
    (e.g., characters, plot, setting for literary text).
    Uses the Mixtral client.
    """
    if not mixtral_client or not mixtral_model:
        print("Mixtral client not available for extract_text_parameters.")
        # Return generic parameters as fallback
        return "основные события, ключевые идеи, главные герои"

    PROMPT_TEMPLATE = '''Analyze the given text and determine its type (e.g., scientific, literary, journalistic, technical, philosophical, etc.). Based on the identified type, extract only the key parameters that characterize this type of text.

        Output the result strictly as a comma-separated list of parameters in russian without any additional text.

        Examples of key parameters for different text types:
        - **Literary**: персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды
        - **Scientific**: основные термины, гипотезы, методы, доказательства, выводы
        - **Journalistic**: ключевые события, участники, место, время, аргументы
        - **Technical**: предмет описания, термины, инструкции, алгоритмы, примеры
        - **Philosophical**: основные идеи, аргументы, философские термины, парадоксы, концепции

        You need to answer just the list of parameters, nothing else.

        Examples of output:
        персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды
        предмет описания, термины, инструкции
        аргументы, философские термины, парадоксы, концепции

        Text:
        "{text}"
        '''
    messages = [
        {
            "role": "user",
            "content": (
                PROMPT_TEMPLATE.format(text=text_fragment)
            )
        }
    ]
    try:
        completion = await mixtral_client.chat_completion( # Use async client
            model=mixtral_model,
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        text_params = completion.choices[0].message.content.strip()
        # Basic validation: check if it looks like a comma-separated list
        if not text_params or len(text_params.split(',')) < 2:
             print(f"Warning: extract_text_parameters returned potentially invalid format: {text_params}. Using fallback.")
             return "основные события, ключевые идеи, главные герои" # Fallback
        return text_params
    except Exception as e:
        print(f"Error in extract_text_parameters: {e}")
        return "основные события, ключевые идеи, главные герои" # Fallback


# ------------------------------
# Adapted function
async def extract_tags_and_genres(text: str) -> Dict[str, List[str]]:
    """
    Extract tags using an AI model and determine genres heuristically.
    Uses the Mixtral client for tag extraction.
    """
    tags = []
    if mixtral_client and mixtral_model:
        try:
            # Извлечение тегов using Mixtral
            response_tags = await mixtral_client.chat_completion( # Use async client
                model=mixtral_model,
                messages=[{
                    "role": "user",
                    "content": (
                            "Extract keyword tags from the following text. "
                            "Write only tags separated by commas. One or two words per tag maximum. Use Nominative case. Write in Russian. "
                            "The first letter of each tag must be capitalized. Do not add any explanation before or after the list. "
                            "Example: Фантастика, Космос, Приключения, Драконы"
                            "Text: " + text
                    )
                }],
                max_tokens=150,
                temperature=0.2
            )
            raw_tags = response_tags.choices[0].message.content
            # Clean up the tags
            tags = [tag.strip().capitalize() for tag in raw_tags.split(",") if tag.strip() and len(tag.strip()) > 1]
            # Further cleanup: remove potential leading/trailing quotes or unwanted characters
            tags = [re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s-]', '', tag).strip() for tag in tags]
            tags = [tag for tag in tags if tag] # Remove empty strings after cleaning
            print(f"Extracted tags: {tags}")
        except Exception as e:
            print(f"Error extracting tags with Mixtral: {e}")
            tags = [] # Fallback to empty list
    else:
        print("Mixtral client not available for tag extraction.")


    # Эвристическое определение жанров (same as before)
    possible_genres = {
        "Учебный": ["образование", "учебный", "курс", "пособие"],
        "Научпоп": ["научно-популярный", "наука", "популярный", "исследование", "открытие"],
        "Научный": ["исследование", "теория", "статья", "диссертация", "монография"],
        "Художественный": ["роман", "рассказ", "повесть", "сюжет", "персонажи", "проза"],
        "Технический": ["инструкция", "руководство", "документация", "спецификация"],
        "Философский": ["философия", "идеи", "концепции", "эссе", "трактат"],
        "Журналистика": ["статья", "репортаж", "интервью", "новости"]
    }
    genres = []
    text_lower = text.lower()
    for genre, keywords in possible_genres.items():
        if any(keyword in text_lower for keyword in keywords):
            genres.append(genre)

    # If no specific genre found, default to "Художественный" if tags were extracted, else empty
    if not genres and tags:
         genres.append("Художественный")
    elif not genres and not tags:
         # If text is very short or uninformative, genre might be impossible
         pass

    # Combine tags and genres, avoid duplicates
    final_tags = list(set(tags + genres)) # Treat genres also as tags for filtering? Or keep separate? Let's combine for simplicity.
    # Return in the original dict format if needed, or adjust as required by recommendation system
    # Keeping original format for compatibility:
    return {"tags": tags, "genres": genres}


# ------------------------------
# Adapted function
async def ask_question(question: str, book_context: str, chat_history: list, answer_size: str = "medium") -> str:
    """
    Answers a user's question based on book context and chat history.
    Adjusts answer length via prompt instructions. Uses the Mixtral client.
    """
    if not mixtral_client or not mixtral_model:
        print("Mixtral client not available for ask_question.")
        return "Извините, не могу сейчас ответить на вопрос, сервис недоступен."

    # Determine length instruction for the prompt
    if answer_size == "short":
        length_instruction = "Ответь очень кратко, буквально одно-два предложения (примерно 30-50 слов)."
    elif answer_size == "long":
        length_instruction = "Ответь развернуто и подробно, приведи примеры или цитаты, если уместно (примерно 150-250 слов)."
    else: # Medium or default
        length_instruction = "Ответь по существу, не слишком коротко, но и не слишком длинно (примерно 70-100 слов)."

    # Construct messages list
    system_prompt = (
        "Ты - ИИ-ассистент, специализирующийся на анализе содержания книг. "
        "Ты отвечаешь на вопросы пользователя по предоставленному контексту книги и истории предыдущего диалога. "
        "Анализируй стиль, символизм, темы произведения, если это релевантно вопросу. "
        f"{length_instruction} "
        "Отвечай только на русском языке."
        "Не выдумывай информацию, которой нет в контексте."
        "Если ответ не может быть дан на основе контекста, так и скажи."
    )
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (ensure it's in the correct format)
    formatted_history = []
    if isinstance(chat_history, list):
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Limit history length to avoid exceeding token limits
                if len(formatted_history) < 10: # Keep last 5 Q/A pairs max
                    formatted_history.append({"role": msg["role"], "content": msg["content"]})
            else:
                print(f"Warning: Skipping invalid chat history item: {msg}")
    messages.extend(formatted_history)

    # Add current context and question
    messages.append({"role": "user", "content": f"Контекст книги:\n```\n{book_context}\n```\n\nВопрос: {question}"})

    try:
        completion = await mixtral_client.chat_completion( # Use async client
            model=mixtral_model,
            messages=messages,
            max_tokens=500, # Adjust based on expected 'long' answer size + buffer
            temperature=0.6
        )
        final_answer = completion.choices[0].message.content.strip()
        return final_answer
    except Exception as e:
        print(f"Error in ask_question: {e}")
        return "Извините, произошла ошибка при обработке вашего вопроса."


# ------------------------------
# Adapted function - This updates the AI context for the *original* book reading
async def update_book_content(user_id: int, book_id: int, page_to_update_until: int, last_update_page: int, current_context: str | None) -> str | None:
    """
    Generates a summary of the newly read portion (original book) and appends it to the context.
    This is used for the AI chat context when reading the non-summarized book.
    Uses the Summarization (DeepSeek) client.

    Args:
        user_id: User ID.
        book_id: Book ID.
        page_to_update_until: The page number the user has just reached.
        last_update_page: The last page number for which context was updated.
        current_context: The existing AI context summary from the database.

    Returns:
        The updated context string (old context + new summary), or the original context if no update needed/possible.
        Returns None on major errors (like failing to get text).
    """
    print(f"Запуск update_book_content: user={user_id}, book={book_id}, update_to={page_to_update_until}, last_update={last_update_page}")

    if not summarize_client or not summarize_model:
        print("Summarization client not available for update_book_content.")
        return current_context # Return existing context if summarizer fails

    if page_to_update_until <= last_update_page:
        print("No new pages to update context for.")
        return current_context # No new pages read

    loop = asyncio.get_running_loop()

    # 1. Get full text and total pages
    full_text = await loop.run_in_executor(None, db.get_book_full_text, user_id, book_id)
    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)
    selected_book = next((book for book in user_books if book['book_id'] == book_id), None)

    if not full_text or full_text.startswith("Ошибка:") or not selected_book:
        print(f"Error: Cannot get full text or book info for book_id {book_id}.")
        # Decide whether to return None or current_context. Returning current prevents losing old context.
        return current_context

    total_pages = selected_book.get('total_pages')
    if total_pages is None or total_pages <= 0:
        print(f"Warning: Invalid total_pages ({total_pages}) for book {book_id}. Cannot reliably estimate text chunk.")
        # Can't proceed with estimation, return existing context
        return current_context

    # Ensure pages are within bounds
    last_update_page = max(0, last_update_page)
    page_to_update_until = min(page_to_update_until, total_pages) # Cap at total pages
    if page_to_update_until <= last_update_page: # Check again after capping
         return current_context

    # 2. Estimate the character slice for the new pages
    # This is a rough approximation!
    text_len = len(full_text)
    start_char = int((last_update_page / total_pages) * text_len)
    end_char = int((page_to_update_until / total_pages) * text_len)
    # Add a small buffer around the estimated slice to increase chances of capturing full sentences
    buffer = 500 # Characters buffer
    start_char = max(0, start_char - buffer)
    end_char = min(text_len, end_char + buffer)

    # Ensure start is before end
    if start_char >= end_char:
         print(f"Warning: Calculated empty or invalid text slice [{start_char}:{end_char}] for pages {last_update_page}-{page_to_update_until}. Skipping context update.")
         return current_context

    new_text_chunk = full_text[start_char:end_char]
    print(f"Estimated text chunk size for pages {last_update_page}-{page_to_update_until}: {len(new_text_chunk)} chars")

    if not new_text_chunk.strip():
        print("Estimated text chunk is empty. Skipping context update.")
        return current_context

    # 3. Summarize the new chunk
    # Determine target size - make it proportional to the number of pages read
    num_new_pages = page_to_update_until - last_update_page
    # Target roughly 15-25 words per page added?
    target_summary_words = max(50, num_new_pages * 20)
    # Limit max summary size per chunk
    target_summary_words = min(target_summary_words, 500) # Max 500 words per update chunk

    summary_prompt = (
        f"Ты - ИИ, который помогает составить краткое содержание прочитанного фрагмента книги. "
        f"Тебе дан фрагмент текста, соответствующий страницам с {last_update_page + 1} по {page_to_update_until}. "
        f"Сделай краткое изложение этого фрагмента на русском языке объемом примерно {target_summary_words} слов. "
        "Сосредоточься на ключевых событиях, действиях персонажей, основной идее. "
        "Не начинай с фраз 'В этом фрагменте', 'Этот текст о'. Просто изложи суть."
        "Избегай прямых цитат. Пересказывай своими словами."
        "Ответ должен быть только сжатым текстом, без дополнительных фраз."
    )
    messages = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": new_text_chunk}
    ]

    try:
        completion = await summarize_client.chat_completion( # Use async client
            model=summarize_model,
            messages=messages,
            max_tokens=int(target_summary_words * 3), # Generous token buffer
            temperature=0.5
        )
        new_summary = completion.choices[0].message.content.strip()
        # Post-processing similar to summarize_segment
        new_summary = re.sub(r'^(В этом фрагменте|В данном тексте|Этот сегмент о)\s*:?\s*', '', new_summary, flags=re.IGNORECASE).strip()
        print(f"Generated summary for new chunk, length: {len(new_summary)} chars")

        if not new_summary:
            print("Warning: Generated summary was empty.")
            return current_context # Return old context if summary failed

        # 4. Combine with existing context
        if current_context and current_context.strip():
            # Add a separator if old context exists
            updated_context = current_context + "\n\n[Далее:]\n" + new_summary
        else:
            updated_context = new_summary

        # Limit total context size to avoid issues (e.g., max 30k chars)
        max_total_context = 30000
        if len(updated_context) > max_total_context:
            print(f"Warning: Total context exceeds {max_total_context} chars. Truncating...")
            # Simple truncation from the beginning
            updated_context = updated_context[-max_total_context:]
            # Try to find a sentence start after truncation
            first_sentence_break = re.search(r'[.!?]\s+', updated_context)
            if first_sentence_break:
                 updated_context = updated_context[first_sentence_break.end():]

        return updated_context

    except Exception as e:
        print(f"Error summarizing text chunk in update_book_content: {e}")
        return current_context # Return existing context on error