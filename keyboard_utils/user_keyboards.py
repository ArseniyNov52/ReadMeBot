# Убран hashlib, так как больше не нужен
from aiogram.types import (InlineKeyboardMarkup, InlineKeyboardButton)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from lexicon.lexicon import LEXICON # Убедитесь, что LEXICON содержит нужные тексты кнопок

# ------------------------------
# Основные кнопки
button_preferences = InlineKeyboardButton(text="✨ Рекомендации ✨", callback_data="preferences") # Текст изменен для ясности
button_read = InlineKeyboardButton(text="📚 Читать мои книги 📖", callback_data="read") # Текст изменен для ясности
button_upload = InlineKeyboardButton(text="⬆️ Загрузить книгу 📥", callback_data="upload")
start_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [button_preferences],
    [button_read],
    [button_upload]
])

# ------------------------------
# Кнопки выбора режима учета ранее рекомендованных книг (Inline)
# Reply-клавиатуры менее удобны в FSM, переделаем на Inline
button_exclude_prev_yes = InlineKeyboardButton(text="Да, исключать", callback_data="exclude_yes")
button_exclude_prev_no = InlineKeyboardButton(text="Нет, не исключать", callback_data="exclude_no")
exclude_prev_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [button_exclude_prev_yes],
    [button_exclude_prev_no]
])
# ReplyKeyboardRemove больше не нужен для этой логики

# ------------------------------
# Кнопки выбора режима рекомендаций
button_mode_1 = InlineKeyboardButton(text="Похожие на мою книгу", callback_data="mode_1") # Текст изменен
button_mode_2 = InlineKeyboardButton(text="По моему описанию", callback_data="mode_2") # Текст изменен
button_mode_3 = InlineKeyboardButton(text="По описанию + стиль моей книги", callback_data="mode_3") # Текст изменен
recommendation_mode_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [button_mode_1],
    [button_mode_2],
    [button_mode_3]
])

# ------------------------------
# Клавиатура для выбора длины ответа от AI
button_short_answer = InlineKeyboardButton(text="Краткий", callback_data="answer_short")
button_medium_answer = InlineKeyboardButton(text="Средний", callback_data="answer_medium")
button_detailed_answer = InlineKeyboardButton(text="Развернутый", callback_data="answer_long") # Изменено на long для консистентности
answer_length_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [button_short_answer, button_medium_answer, button_detailed_answer]
])

# ------------------------------
# Кнопки управления чтением/сжатием
# Используем одну кнопку "Назад/Отмена" с разными callback_data
cancel_reading_button = InlineKeyboardButton(text="◀️ Завершить чтение", callback_data="cancel_reading")
cancel_compress_button = InlineKeyboardButton(text="Пропустить сжатие ▶️", callback_data="cancel_compress")
cancel_compress_keyboard = InlineKeyboardMarkup(inline_keyboard=[[cancel_compress_button]]) # Только кнопка пропуска

# ------------------------------
# Клавиатура для чата с AI
ai_leave_button = InlineKeyboardButton(text="◀️ Вернуться к чтению", callback_data="leave_ai_chat") # Текст изменен
ai_keyboard = InlineKeyboardMarkup(inline_keyboard=[[ai_leave_button]])

# ------------------------------
# Функция создания пагинации для чтения книги (переработана)

# --- УДАЛЕНО: Клавиатура create_compressed_view_keyboard больше не нужна для чтения ---
# def create_compressed_view_keyboard(ai_chat_callback: str = 'chat_with_ai_compressed', cancel_callback: str = 'cancel_reading') -> InlineKeyboardMarkup:
#     buttons = [
#         [InlineKeyboardButton(text="💬 Чат с ИИ", callback_data=ai_chat_callback)],
#         [InlineKeyboardButton(text="📕 Завершить чтение", callback_data=cancel_callback)]
#     ]
#     return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_pagination_keyboard(
    prev_callback: str | None,
    page_text: str,
    next_callback: str | None,
    ai_chat_callback: str | None,
    cancel_callback: str | None
) -> InlineKeyboardMarkup:
    """
    Создает клавиатуру пагинации.
    Кнопки добавляются, только если соответствующий callback_data не None.
    """
    kb_builder = InlineKeyboardBuilder()
    row_buttons = []
    if prev_callback:
        row_buttons.append(InlineKeyboardButton(text=LEXICON.get('backward', '⬅️ Назад'), callback_data=prev_callback))

    # Кнопка с номером страницы (некликабельная)
    row_buttons.append(InlineKeyboardButton(text=page_text, callback_data='ignore_page')) # callback_data для игнорирования нажатия

    if next_callback:
        row_buttons.append(InlineKeyboardButton(text=LEXICON.get('forward', 'Вперед ➡️'), callback_data=next_callback))

    if row_buttons:
         kb_builder.row(*row_buttons)

    second_row = []
    if ai_chat_callback:
         # Используем единый callback для чата, логика внутри хендлера разберется
         second_row.append(InlineKeyboardButton(text=LEXICON.get('chat_with_ai', '💬 Чат с ИИ'), callback_data='chat_with_ai')) # Изменено на общий 'chat_with_ai'
    if cancel_callback:
        second_row.append(InlineKeyboardButton(text=LEXICON.get('cancel_reading', '⏹️ Закончить'), callback_data=cancel_callback))

    if second_row:
        kb_builder.row(*second_row)

    return kb_builder.as_markup()

# ------------------------------
# Функции создания клавиатур выбора книги для рекомендаций (переработаны)
# Принимают список словарей книг из БД

def create_book_selection_keyboard(books: list[dict], callback_prefix: str) -> InlineKeyboardMarkup | None:
    """
    Создает клавиатуру для выбора книги из списка.

    :param books: Список словарей, каждый содержит 'book_id' и 'book_name'.
    :param callback_prefix: Префикс для callback_data (напр., 'mode1_' или 'mode3_').
    :return: Объект InlineKeyboardMarkup или None, если список книг пуст.
    """
    if not books:
        return None

    builder = InlineKeyboardBuilder()
    for book in books:
        book_id = book.get('book_id')
        book_name = book.get('book_name', 'Без названия')
        if book_id:
            # Ограничиваем длину текста кнопки, чтобы избежать ошибок Telegram
            display_name = book_name[:50] + '...' if len(book_name) > 50 else book_name
            builder.row(InlineKeyboardButton(
                text=display_name,
                callback_data=f"{callback_prefix}{book_id}"
            ))
    # Можно добавить кнопку "Отмена" или "Назад"
    # builder.row(InlineKeyboardButton(text="Отмена", callback_data="cancel_recommendation_selection"))
    return builder.as_markup()

def create_mode1_history_keyboard(books: list[dict]) -> InlineKeyboardMarkup | None:
    """Создает клавиатуру для выбора книги для Режима 1 рекомендаций."""
    return create_book_selection_keyboard(books, "mode1_")

def create_mode3_history_keyboard(books: list[dict]) -> InlineKeyboardMarkup | None:
    """Создает клавиатуру для выбора книги для Режима 3 рекомендаций."""
    return create_book_selection_keyboard(books, "mode3_")