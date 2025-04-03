import asyncio
import re
from huggingface_hub import AsyncInferenceClient

# --- Локальные импорты ---
# Добавляем импорты из database
import database.database as db

# Список API-ключей (если используется)
api_keys = ['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']
api_key_index = 0
client = AsyncInferenceClient(
    provider="together",
    api_key=api_keys[api_key_index]
)


async def generate_instructions(text_fragment: str) -> str:
    """Генерирует инструкции (ключевые параметры) для саммаризации на основе фрагмента текста."""
    messages = [
        {
            "role": "user",
            "content": (
                f'''Analyze the given text fragment and determine its type (e.g., scientific, literary, journalistic, technical, philosophical, etc.). Based on the identified type, extract only the key parameters that characterize this type of text.

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

                Text Fragment:
                "{text_fragment}"
                '''
            )
        }
    ]
    try:
        # Модель для определения типа/параметров
        completion = await client.chat_completion(
            messages=messages,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1", # Или другая подходящая модель
            max_tokens=100,
            temperature=0.1 # Низкая температура для точности
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка при генерации инструкций для саммаризации: {e}")
        # Возвращаем общие параметры в случае ошибки
        return "основные события, ключевые идеи, главные герои"

def segment_text(text: str, num_segments: int) -> list[str]:
    """Делит текст на примерно равное количество сегментов по предложениям."""
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    if not sentences:
        return []

    total_sentences = len(sentences)
    # Не делить на больше сегментов, чем есть предложений
    actual_num_segments = max(1, min(num_segments, total_sentences))
    segment_size = (total_sentences + actual_num_segments - 1) // actual_num_segments # Округление вверх

    segments = []
    for i in range(actual_num_segments):
        start_index = i * segment_size
        end_index = min((i + 1) * segment_size, total_sentences)
        segment_sentences = sentences[start_index:end_index]
        if segment_sentences:
            # Добавляем точку в конце, если ее нет
            segment_text = '. '.join(segment_sentences)
            if not segment_text.endswith(('.', '!', '?')):
                 segment_text += '.'
            segments.append(segment_text)

    return segments


async def summarize_segment(segment: str, target_words: int, instructions: str) -> str:
    """Сжимает один сегмент текста."""
    system_prompt = (
        f"Сгенерируй краткое содержание (summary) на русском языке объемом примерно {target_words} слов. "
        f"Уделяй особое внимание следующим аспектам: {instructions}. "
        "Это фрагмент из большего текста. Не начинай с фраз 'В этом тексте', 'Этот фрагмент о'. "
        "Просто передай суть сегмента как связный абзац."
        "Избегай прямых цитат. Пересказывай своими словами."
        "Ответ должен быть только сжатым текстом, без дополнительных фраз."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": segment}
    ]

    try:
        # Модель для саммаризации
        completion = await client.chat_completion(
            messages=messages,
            model="deepseek-ai/DeepSeek-V3", # Или другая модель, например, более новая Mixtral или другая
            max_tokens=int(target_words * 2.5), # Даем запас токенов (примерно 1 слово ~ 1.5-2 токена)
            temperature=0.5 # Средняя температура для баланса между точностью и креативностью
        )
        summary = completion.choices[0].message.content.strip()
        # Постобработка: убираем возможные начальные/конечные фразы, которые модель могла добавить
        summary = re.sub(r'^(В этом фрагменте|В данном тексте|Этот сегмент о)\s*:?\s*', '', summary, flags=re.IGNORECASE)
        return summary
    except Exception as e:
        print(f"Ошибка при саммаризации сегмента: {e}")
        # В случае ошибки возвращаем пустую строку или исходный сегмент (сокращенный)
        return segment[:target_words*10] # Возвращаем начало сегмента как запасной вариант


# --- Переработанная основная функция ---
async def summarize_text_and_update_db(user_id: int, book_id: int, overall_target_chars: int):
    """
    Получает полный текст книги из БД, сжимает его до target_chars
    и сохраняет результат в book_context в таблице reading_state.
    """
    print(f"Начало сжатия для user_id={user_id}, book_id={book_id}, target_chars={overall_target_chars}")

    # 1. Получаем полный текст и общее кол-во страниц из БД
    loop = asyncio.get_running_loop()
    full_text = await loop.run_in_executor(None, db.get_book_full_text, user_id, book_id)
    # Получаем total_pages (нужно для update_ai_context)
    # Проще всего получить его из информации о книге
    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)
    selected_book = next((book for book in user_books if book['book_id'] == book_id), None)
    total_pages = selected_book.get('total_pages') if selected_book else 0 # 0 если книга не найдена

    if not full_text or full_text.startswith("Ошибка"):
        print(f"Ошибка: Не удалось получить текст книги {book_id} для сжатия.")
        return # Не можем продолжить без текста

    if total_pages is None or total_pages <= 0:
         print(f"Предупреждение: Не удалось определить кол-во страниц для книги {book_id}. update_page будет {total_pages}")


    # 2. Определение инструкций для сжатия (на основе фрагмента)
    mid_point = len(full_text) // 2
    fragment = full_text[max(mid_point - 700, 0): min(mid_point + 700, len(full_text))]
    instructions = await generate_instructions(fragment)
    print(f"Инструкции для сжатия: {instructions}")

    # 3. Сегментация текста
    # Рассчитываем количество сегментов и целевой размер сегмента в словах
    # Примерно 5 символов на слово
    overall_target_words = max(100, overall_target_chars // 5)
    num_segments = max(5, overall_target_words // 150) # Делим на сегменты по ~150 слов
    segments = segment_text(full_text, num_segments)
    if not segments:
         print(f"Ошибка: Не удалось разделить текст книги {book_id} на сегменты.")
         return

    print(f"Текст разделен на {len(segments)} сегментов.")
    target_words_per_segment = (overall_target_words + len(segments) - 1) // len(segments) # Округление вверх
    print(f"Целевой размер сегмента: ~{target_words_per_segment} слов")


    # 4. Асинхронная саммаризация сегментов
    tasks = [summarize_segment(seg, target_words_per_segment, instructions) for seg in segments]
    summaries = await asyncio.gather(*tasks)

    # 5. Объединение результатов и финальная проверка длины (опционально)
    final_summary = ' '.join(filter(None, summaries)).strip() # Убираем пустые результаты, если были ошибки
    # Можно добавить еще один проход сжатия, если результат сильно длиннее цели, но пока оставим так.
    print(f"Финальное сжатие готово, длина: {len(final_summary)} символов.")


    # 6. Сохранение результата в БД
    # Сохраняем сжатый текст в book_context и помечаем, что обновили до последней страницы
    update_page_value = total_pages if total_pages is not None else 1 # Используем 1 если total_pages неизвестно
    success = await loop.run_in_executor(
        None, db.update_ai_context, user_id, book_id, update_page_value, final_summary
    )

    if success:
        print(f"Сжатый текст для книги {book_id} (user {user_id}) сохранен в БД. update_page={update_page_value}")
    else:
        print(f"Ошибка: Не удалось сохранить сжатый текст для книги {book_id} в БД.")

    # Функция больше не возвращает текст, так как он сохранен в БД