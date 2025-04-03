import os
import asyncio # <-- Импорт asyncio
import math # <-- Импорт math для расчета страниц
from aiogram import Router, F, Bot
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from psycopg2.extras import RealDictCursor # Используется во вспомогательной функции

# --- Локальные импорты ---
from config.config import Config, load_config
from lexicon.lexicon import LEXICON
from keyboard_utils.user_keyboards import ( # Импортируем обновленные клавиатуры
    start_keyboard,
    exclude_prev_keyboard,
    answer_length_keyboard,
    recommendation_mode_keyboard,
    create_pagination_keyboard, # Используем всегда эту клавиатуру для чтения
    cancel_compress_keyboard,
    create_mode1_history_keyboard,
    create_mode3_history_keyboard,
    # --- УДАЛЕНО: create_compressed_view_keyboard ---
)
# --- Импорт функций БД ---
import database.database as db

# --- Импорты AI ---
# Предполагаем, что эти функции адаптированы и вызываются корректно
from ai_tools.summarize_system import summarize_text_and_update_db # Используем новую функцию
from ai_tools.analyze_system import ask_question, update_book_content
from ai_tools.recommendation_system import get_book_recommendations
# Загрузка датасета для рекомендаций (если он нужен)
import pandas as pd
try:
    # Укажите правильный путь к вашему датасету
    recommendation_dataset = pd.read_csv('ai_tools/books_db.csv')
    # Здесь может потребоваться предобработка датасета (например, конвертация строк в списки)
    # recommendation_dataset['category'] = recommendation_dataset['category'].apply(eval) # Пример
    print("Датасет для рекомендаций загружен.")
except FileNotFoundError:
    print("Ошибка: Файл датасета для рекомендаций не найден!")
    recommendation_dataset = pd.DataFrame() # Пустой датафрейм, чтобы избежать ошибок
except Exception as e:
    print(f"Ошибка при загрузке или обработке датасета: {e}")
    recommendation_dataset = pd.DataFrame()

# --- Конфигурация и константы ---
config: Config = load_config() # Путь к .env можно опустить, если он стандартный
BOOKS_DIRECTORY = db.USER_BOOKS_BASE_DIR
COMPRESSED_PAGE_SIZE = 1000 # <-- НОВОЕ: Количество символов на "страницу" сжатого текста

# --- Определения состояний FSM ---
class UploadBookState(StatesGroup):
    waiting_for_book = State()

class ReadBookState(StatesGroup):
    reading = State()

class InputPrefsState(StatesGroup):
    waiting_for_input = State() # Для режима 2
    waiting_for_input_mode3 = State() # Для режима 3 (текст)
    waiting_for_exclude_option = State() # Выбор исключения
    waiting_for_mode1_book = State() # Выбор книги для режима 1
    waiting_for_mode3_book = State() # Выбор книги для режима 3

class CompressBookState(StatesGroup):
    awaiting_daily_read_pages = State()
    awaiting_days_to_finish = State()

class ChattingWithModelState(StatesGroup):
    awaiting_message_for_model = State() # Ожидание вопроса пользователя
    awaiting_answer_size = State()

# --- Инициализация роутера ---
router = Router()

# --- Вспомогательная функция для отображения контента (ПЕРЕРАБОТАНА) ---
async def display_book_content(message_or_callback_query: Message | CallbackQuery, state: FSMContext, user_id: int, book_id: int):
    """
    Отображает контент книги постранично (оригинальной или сжатой).
    Всегда использует пагинацию.
    """
    loop = asyncio.get_running_loop()
    data = await state.get_data()
    current_page = data.get('page', 0)
    total_pages = data.get('total_pages') # Это total_pages для *текущего* вида (оригинал/сжатый)
    has_compressed_context = data.get('has_compressed_context', False)
    book_name = data.get('book_name', 'Книга')
    # --- НОВОЕ: Получаем сжатый контент из state, если он есть ---
    compressed_content = data.get('compressed_book_context') if has_compressed_context else None

    content = None
    keyboard = None
    error_message = None # Для сообщений об ошибках или конце книги

    if current_page < 0: current_page = 0 # Защита

    # --- Логика получения контента страницы ---
    if has_compressed_context:
        if compressed_content:
            start_index = current_page * COMPRESSED_PAGE_SIZE
            end_index = start_index + COMPRESSED_PAGE_SIZE
            page_content_raw = compressed_content[start_index:end_index]

            if not page_content_raw and start_index >= len(compressed_content):
                error_message = "Вы достигли конца сжатой версии книги."
                # Корректируем страницу на последнюю, если вышли за пределы
                if total_pages is not None and total_pages > 0:
                    current_page = total_pages - 1
                    await state.update_data(page=current_page)
                else:
                    current_page = 0 # Если страниц нет, остаемся на 0
                    await state.update_data(page=current_page)

            content = page_content_raw if page_content_raw else error_message or "(Пустая страница)"
            content = f"**{book_name} (Сжатая версия)**\n\n{content}" # Добавляем заголовок
        else:
            # Сценарий, когда флаг есть, а контента нет (ошибка)
            error_message = "Ошибка: Сжатый контент не найден, хотя ожидался."
            content = error_message
            # Сбрасываем флаг и пытаемся показать оригинал? Или просто ошибка? Пока ошибка.
            await state.update_data(has_compressed_context=False, compressed_book_context=None)
            # Перезагрузить состояние? Пока оставим так
            total_pages = data.get('original_total_pages') # Возвращаемся к оригинальным страницам
            await state.update_data(total_pages=total_pages) # Обновляем total_pages в state

    else: # Обычный режим (чтение из файла)
        content_result = await loop.run_in_executor(None, db.get_book_page_content, user_id, book_id, current_page)

        if content_result is None or content_result.startswith("Ошибка") or content_result == "Вы достигли конца книги.":
            error_message = content_result if content_result else "Не удалось загрузить страницу книги."
            # Корректируем страницу, если вышли за пределы
            if content_result == "Вы достигли конца книги." and total_pages is not None and total_pages > 0:
                 current_page = total_pages - 1
                 await state.update_data(page=current_page)

            if content_result is None or content_result.startswith("Ошибка"):
                 await state.clear() # Сбрасываем состояние при ошибке загрузки
                 keyboard = start_keyboard # Возврат в меню
                 content = error_message
                 # Отправляем сообщение об ошибке и выходим
                 await message_or_callback_query.answer(content, reply_markup=keyboard)
                 return
            else: # Конец книги
                 content = error_message # "Вы достигли конца книги."
        else:
             content = content_result

    # --- Формирование клавиатуры пагинации ---
    tp_display = total_pages if total_pages is not None else '?'
    page_display = f'{current_page + 1}/{tp_display}' if total_pages is not None else f'Стр. {current_page + 1}'

    # Определяем доступность кнопок Вперед/Назад
    can_go_back = current_page > 0
    can_go_forward = total_pages is None or (current_page < total_pages - 1)

    # Не показываем кнопку "Вперед", если есть сообщение об ошибке/конце книги
    if error_message:
        can_go_forward = False

    keyboard = create_pagination_keyboard(
         prev_callback='backward' if can_go_back else None,
         page_text=page_display,
         next_callback='forward' if can_go_forward else None,
         ai_chat_callback='chat_with_ai', # Всегда одна кнопка чата
         cancel_callback='cancel_reading'
    )
    await state.set_state(ReadBookState.reading) # Устанавливаем/подтверждаем состояние чтения

    # --- Отправка или редактирование сообщения ---
    target_message = message_or_callback_query.message if isinstance(message_or_callback_query, CallbackQuery) else message_or_callback_query
    try:
        # Пытаемся редактировать существующее сообщение
        await target_message.edit_text(content, reply_markup=keyboard, parse_mode="Markdown")
    except Exception:
         # Если не вышло (старое сообщение или первый вызов) - отправляем новое
         # Если исходное сообщение было от CallbackQuery, отвечаем на него
         if isinstance(message_or_callback_query, CallbackQuery):
             await message_or_callback_query.message.answer(content, reply_markup=keyboard, parse_mode="Markdown")
         else: # Если исходное было Message
             await message_or_callback_query.answer(content, reply_markup=keyboard, parse_mode="Markdown")


# --- Хендлеры ---

@router.message(CommandStart())
async def process_start_command(message: Message, state: FSMContext):
    user_id = message.from_user.id
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, db.clear_all_book_sessions, user_id)
    await message.answer(LEXICON["/start"], reply_markup=start_keyboard)
    await state.clear()

@router.callback_query(F.data == "upload")
async def process_upload_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Пожалуйста, отправьте файл книги в формате PDF или TXT.")
    await state.set_state(UploadBookState.waiting_for_book)
    await callback.answer()

@router.message(UploadBookState.waiting_for_book, F.document)
async def process_book_upload(message: Message, state: FSMContext, bot: Bot):
    # (код загрузки файла остался без изменений)
    file = message.document
    assert file is not None

    user_id = message.from_user.id
    file_name = file.file_name
    loop = asyncio.get_running_loop()

    file_ext = file_name.split('.')[-1].lower()
    if file_ext not in ['pdf', 'txt']:
        if not file.mime_type or file.mime_type not in ["application/pdf", "text/plain"]:
            await message.answer("Формат файла не поддерживается. Пожалуйста, отправьте файл в формате PDF или TXT.")
            return

    user_books_dir = os.path.join(BOOKS_DIRECTORY, str(user_id))
    os.makedirs(user_books_dir, exist_ok=True)
    file_path = os.path.join(user_books_dir, file_name)

    if os.path.exists(file_path):
         await message.answer(f"Книга с именем '{file_name}' уже существует. Загрузка отменена.")
         await state.clear()
         return

    status_msg = None
    try:
        status_msg = await message.answer("⏳ Загружаю файл...")
        await bot.download(file=file, destination=file_path)
        print(f"Файл скачан: {file_path}")

        await status_msg.edit_text("⏳ Обрабатываю книгу и добавляю в библиотеку...")
        added_book_info = await loop.run_in_executor(None, db.add_book_to_db, user_id, file_name, file_path)

        if added_book_info:
            tp_str = f" ({added_book_info.get('total_pages', '?')} стр.)" if added_book_info.get('total_pages') is not None else ""
            await status_msg.edit_text(f"✅ Книга '{file_name}'{tp_str} успешно добавлена.", reply_markup=start_keyboard)
            await state.clear()
        else:
            await status_msg.edit_text(f"❌ Не удалось добавить информацию о книге '{file_name}' в базу данных.")
            if os.path.exists(file_path):
                 try: os.remove(file_path)
                 except OSError as e: print(f"Ошибка при удалении файла {file_path}: {e}")
                 else: print(f"Удален файл из-за ошибки добавления в БД: {file_path}")
            await state.clear()

    except Exception as e:
        print(f"Ошибка при скачивании или обработке файла (user: {user_id}, file: {file_name}): {e}")
        if status_msg:
            try:
                await status_msg.edit_text("❌ Произошла ошибка при загрузке файла.")
            except Exception as edit_err:
                 print(f"Не удалось изменить статусное сообщение: {edit_err}")
                 await message.answer("❌ Произошла ошибка при загрузке файла.")
        else:
            await message.answer("❌ Произошла ошибка при загрузке файла.")

        if os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError as e: print(f"Ошибка при удалении файла {file_path} после исключения: {e}")
        await state.clear()


@router.callback_query(F.data == "read")
async def process_read_callback(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)

    if user_books:
        buttons = []
        for book in user_books:
            callback_data = f"read_book_{book['book_id']}"
            tp_info = f" ({book.get('total_pages', '?')} стр.)" if book.get('total_pages') is not None else ""
            display_name = book['book_name'][:50] + '...' if len(book['book_name']) > 50 else book['book_name']
            buttons.append([InlineKeyboardButton(text=f"{display_name}{tp_info}", callback_data=callback_data)])

        if buttons:
             buttons.append([InlineKeyboardButton(text="◀️ Назад в меню", callback_data="back_to_start")])
             keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
             await callback.message.edit_text("Выберите книгу для чтения:", reply_markup=keyboard)
        else:
             await callback.message.edit_text("Не удалось сформировать список книг.", reply_markup=start_keyboard)
             await state.clear()
    else:
        await callback.message.edit_text("Ваш список книг пуст. Сначала загрузите книгу.", reply_markup=start_keyboard)
        await state.clear()
    await callback.answer()

@router.callback_query(F.data == "back_to_start")
async def process_back_to_start(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, db.clear_all_book_sessions, user_id)
    await callback.message.edit_text(LEXICON["/start"], reply_markup=start_keyboard)
    await state.clear()
    await callback.answer()

# process_book_selection - Изменения для загрузки данных
@router.callback_query(F.data.startswith("read_book_"))
async def process_book_selection(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    try:
        book_id = int(callback.data.split("_")[2])
    except (IndexError, ValueError):
        await callback.message.edit_text("Ошибка: Некорректные данные книги.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer()
        return

    # 1. Получаем основную информацию о книге
    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)
    selected_book = next((book for book in user_books if book['book_id'] == book_id), None)

    if not selected_book:
        await callback.message.edit_text("Не удалось найти выбранную книгу.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer()
        return

    book_name = selected_book['book_name']
    original_total_pages = selected_book.get('total_pages') # Оригинальное кол-во страниц

    # 2. Получаем состояние чтения (включая сжатый контент, если есть)
    reading_state_details = await loop.run_in_executor(None, db.get_reading_state_details, user_id, book_id)
    current_page = 0
    is_opened = False
    has_compressed_context = False
    book_context_from_db = None # Контекст из БД (может быть сжатым или последним обновлением оригинала)
    compressed_total_pages = None # <-- НОВОЕ: Рассчитанное кол-во страниц для сжатой версии

    if reading_state_details:
        current_page = reading_state_details.get('current_page', 0)
        is_opened = reading_state_details.get('is_opened', False)
        book_context_from_db = reading_state_details.get('book_context') # Получаем book_context из БД

        # Определяем, является ли контекст из БД сжатым
        # Простой эвристический способ: если is_opened=True и context есть,
        # и он не обновлялся постранично (update_page >= total_pages или близко к нему),
        # то скорее всего это результат полного сжатия.
        # Более надежно - добавить флаг is_compressed в БД, но пока так.
        db_update_page = reading_state_details.get('update_page', 0)
        # Проверяем, был ли контекст результатом полного сжатия
        # (update_page соответствует общему числу страниц)
        # ИЛИ (если total_pages не определилось) update_page > 1
        # И контекст существует
        is_fully_summarized = book_context_from_db and original_total_pages and db_update_page >= original_total_pages
        # Доп. проверка если total_pages = None (или 0), но update_page > 1 (значит было сжатие)
        is_summarized_no_pages = book_context_from_db and not original_total_pages and db_update_page > 1

        if is_fully_summarized or is_summarized_no_pages:
             has_compressed_context = True
             # Рассчитываем страницы для сжатой версии
             compressed_total_pages = math.ceil(len(book_context_from_db) / COMPRESSED_PAGE_SIZE)
             if compressed_total_pages == 0 and len(book_context_from_db) > 0:
                 compressed_total_pages = 1
             # Важно: Если книга была сжата, но пользователь листал оригинал,
             # то current_page из БД может быть большим. При переключении на сжатую
             # версию, его нужно будет сбросить или скорректировать.
             # Пока берем current_page из БД как есть. Коррекция будет при отображении.
             print(f"Обнаружен сжатый контекст для книги {book_id} (user {user_id}).")
        else:
             # Контекст есть, но это не полное сжатие (результат постраничного обновления)
             # Оставляем has_compressed_context = False
             print(f"Обнаружен контекст для ИИ (не сжатый) для книги {book_id}.")


    # 3. Устанавливаем сессию и сбрасываем другие
    await loop.run_in_executor(None, db.clear_all_book_sessions, user_id)
    await loop.run_in_executor(None, db.set_book_session_status, user_id, book_id, True)

    # 4. Обновляем FSM state
    # Определяем, какие total_pages использовать для отображения
    view_total_pages = compressed_total_pages if has_compressed_context else original_total_pages
    # Определяем, какой контекст записать в FSM
    # Записываем ВСЕГДА контекст из БД (он либо сжатый, либо последний обновленный оригинал)
    fsm_context = book_context_from_db

    await state.update_data(
        book_id=book_id,
        book_name=book_name,
        original_total_pages=original_total_pages, # Сохраняем оригинал
        compressed_total_pages=compressed_total_pages, # Сохраняем сжатое (может быть None)
        total_pages=view_total_pages, # Страницы для текущего отображения
        page=current_page,
        has_compressed_context=has_compressed_context,
        compressed_book_context=fsm_context # Сохраняем ЛЮБОЙ контекст из БД в FSM
    )

    # 5. Логика показа: спрашивать о сжатии или сразу показывать
    if not is_opened: # Книга открывается впервые
        await callback.message.edit_text(
            f"Вы открываете книгу \"{book_name}\" впервые.\n\n"
            "Хотите сжать её с помощью ИИ под ваш темп чтения? "
            "(Это может занять несколько минут).\n\n"
            "Если да, укажите, сколько страниц в день вы готовы читать:",
            reply_markup=cancel_compress_keyboard
        )
        await state.set_state(CompressBookState.awaiting_daily_read_pages)
    else: # Книга уже открывалась (неважно, сжата или нет)
        # display_book_content сам разберется, что показывать (оригинал или сжатую постранично)
        await display_book_content(callback, state, user_id, book_id) # Используем callback для редактирования

    await callback.answer()


# --- Обработка сжатия ---
@router.message(CompressBookState.awaiting_daily_read_pages)
async def handle_daily_read_pages(message: Message, state: FSMContext):
    try:
        daily_pages = int(message.text)
        if daily_pages <= 0:
             await message.answer("Пожалуйста, введите положительное число страниц.")
             return
        await message.answer(f"Отлично, {daily_pages} стр./день.\nСколько дней у вас есть на прочтение книги?",
                             reply_markup=cancel_compress_keyboard)
        await state.set_state(CompressBookState.awaiting_days_to_finish)
        await state.update_data(daily_pages=daily_pages)
    except ValueError:
        await message.answer("Пожалуйста, введите число страниц (целое).")

# handle_days_to_finish - Изменения для расчета страниц и обновления FSM
@router.message(CompressBookState.awaiting_days_to_finish)
async def handle_days_to_finish(message: Message, state: FSMContext):
    try:
        days_to_finish = int(message.text)
        if days_to_finish <= 0:
             await message.answer("Пожалуйста, введите положительное число дней.")
             return

        user_id = message.from_user.id
        loop = asyncio.get_running_loop()
        data = await state.get_data()
        book_id = data.get("book_id")
        daily_pages = data.get("daily_pages")
        original_total_pages = data.get("original_total_pages") # Оригинальные страницы

        if not all([book_id, daily_pages]):
             await message.answer("Произошла ошибка, не хватает данных для сжатия. Попробуйте снова.")
             await state.clear(); return

        # Устанавливаем флаг is_opened = True в БД
        await loop.run_in_executor(None, db.set_book_opened_status, user_id, book_id, True)

        await message.answer(f"Принято: {daily_pages} стр/день за {days_to_finish} дней.")
        processing_msg = await message.answer("⏳ Производится сжатие текста с помощью ИИ... Это может занять несколько минут.")

        # Рассчитываем целевое кол-во символов (приблизительно)
        # Используем оригинальные страницы для расчета цели, если они есть
        avg_chars_per_page = 1800 # Среднее число символов на страницу (можно настроить)
        if original_total_pages and original_total_pages > 0 :
            # Если есть оригинальные страницы, расчет точнее
             avg_chars_per_page_calc = await loop.run_in_executor(None, db.get_book_full_text, user_id, book_id)
             if avg_chars_per_page_calc and not avg_chars_per_page_calc.startswith("Ошибка"):
                  avg_chars_per_page = max(500, len(avg_chars_per_page_calc) // original_total_pages) # Минимум 500 на всякий случай
        target_chars = daily_pages * days_to_finish * avg_chars_per_page
        print(f"Запуск summarize_text_and_update_db для book_id={book_id}, user_id={user_id}, target_chars={target_chars}")

        summarize_success = False
        new_compressed_context = None
        try:
            # Вызываем функцию сжатия. Она обновляет БД сама.
            # Нам нужно будет прочитать результат из БД.
            # Функция возвращает void или bool (успех/неудача), а не текст.
            await summarize_text_and_update_db(user_id, book_id, target_chars)

            # После выполнения читаем результат из БД
            state_details = await loop.run_in_executor(None, db.get_reading_state_details, user_id, book_id)
            if state_details and state_details.get('book_context'):
                # Проверяем, что update_page соответствует полному сжатию
                db_update_page = state_details.get('update_page', 0)
                is_fully_summarized = original_total_pages and db_update_page >= original_total_pages
                is_summarized_no_pages = not original_total_pages and db_update_page > 1

                if is_fully_summarized or is_summarized_no_pages:
                    new_compressed_context = state_details['book_context']
                    await processing_msg.edit_text("✅ Сжатие завершено! Обновляю данные...")
                    summarize_success = True
                else:
                    # Сжатие могло завершиться, но update_page не обновился до конца (ошибка?)
                    await processing_msg.edit_text("⚠️ Сжатие завершилось, но результат может быть неполным. Открываю оригинал.")
                    summarize_success = False
            else:
                 await processing_msg.edit_text("❌ Сжатие завершилось, но не удалось получить сжатый текст из БД. Открываю оригинал.")
                 summarize_success = False

        except Exception as e:
            print(f"Ошибка при вызове или обработке результата summarize_text_and_update_db: {e}")
            try: await processing_msg.edit_text("❌ Произошла ошибка во время сжатия текста. Открываю книгу без сжатия.")
            except: await message.answer("❌ Произошла ошибка во время сжатия текста. Открываю книгу без сжатия.")
            summarize_success = False

        # --- Обновление FSM и показ контента ---
        if summarize_success and new_compressed_context:
            # Рассчитываем страницы для сжатой версии
            new_compressed_total_pages = math.ceil(len(new_compressed_context) / COMPRESSED_PAGE_SIZE)
            if new_compressed_total_pages == 0 and len(new_compressed_context) > 0:
                 new_compressed_total_pages = 1

            # Обновляем FSM state для сжатого режима
            await state.update_data(
                has_compressed_context=True,
                compressed_book_context=new_compressed_context,
                compressed_total_pages=new_compressed_total_pages,
                total_pages=new_compressed_total_pages, # Теперь показываем страницы сжатой версии
                page=0 # Начинаем читать сжатую версию с начала
            )
            # --- ВАЖНО: Обновить current_page = 0 в БД для сжатой версии ---
            await loop.run_in_executor(None, db.update_reading_state, user_id, book_id, 0)
            print(f"Установлена страница 0 для сжатой книги user {user_id}, book {book_id}")

        else: # Если сжатие не удалось или отменено
            # Возвращаемся к оригинальным данным в FSM
            await state.update_data(
                has_compressed_context=False,
                compressed_book_context=None, # Убираем старый контекст, если он был
                compressed_total_pages=None,
                total_pages=original_total_pages, # Показываем страницы оригинала
                # page остается тем, что было до попытки сжатия (если книга была открыта ранее)
                # или 0, если открывалась впервые.
                # Надо явно установить страницу на 0, если переходим к оригиналу после неудачного сжатия
                page = 0 # Сбрасываем страницу на начало оригинала
            )
            # Обновляем страницу в БД на 0 для оригинала
            await loop.run_in_executor(None, db.update_reading_state, user_id, book_id, 0)
            print(f"Сжатие не удалось. Возврат к оригиналу, стр. 0. user {user_id}, book {book_id}")


        # Показываем контент (сжатый постранично или оригинальный постранично)
        await display_book_content(message, state, user_id, book_id)

    except ValueError:
        await message.answer("Пожалуйста, введите число дней (целое).")
    except KeyError as e:
         print(f"Ошибка: Отсутствует ключ в FSM state при сжатии: {e}")
         await message.answer("Произошла внутренняя ошибка. Попробуйте начать заново.")
         await state.clear()

# Кнопка "Пропустить сжатие"
@router.callback_query(F.data == "cancel_compress", CompressBookState.awaiting_daily_read_pages)
@router.callback_query(F.data == "cancel_compress", CompressBookState.awaiting_days_to_finish)
async def cancel_compress(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    data = await state.get_data()
    book_id = data.get("book_id")
    original_total_pages = data.get("original_total_pages") # Получаем оригинальные стр

    if not book_id:
        await callback.message.edit_text("Ошибка: Не удалось определить книгу.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer()
        return

    # Устанавливаем флаг opened в БД
    await loop.run_in_executor(None, db.set_book_opened_status, user_id, book_id, True)

    # Обновляем FSM для НЕсжатого режима
    await state.update_data(
        has_compressed_context=False,
        compressed_book_context=None,
        compressed_total_pages=None,
        total_pages=original_total_pages, # Явно ставим оригинальные страницы
        page=0 # Начинаем с 0 страницы оригинала
    )
    # Обновляем страницу в БД
    await loop.run_in_executor(None, db.update_reading_state, user_id, book_id, 0)

    await callback.message.edit_text("Хорошо, открываю книгу без сжатия...") # Редактируем старое сообщение
    await display_book_content(callback, state, user_id, book_id) # Показываем оригинал постранично
    await callback.answer()

# Кнопка "Завершить чтение"
@router.callback_query(F.data == "cancel_reading", ReadBookState.reading)
@router.callback_query(F.data == "cancel_reading", ChattingWithModelState.awaiting_message_for_model) # Из чата тоже
async def cancel_reading(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    data = await state.get_data()
    book_id = data.get("book_id")
    current_page = data.get("page", 0) # Получаем текущую страницу для сохранения

    if book_id is not None and current_page is not None:
        # Сохраняем текущую страницу перед выходом
        # ВАЖНО: Не сохраняем страницу, если книга была сжата, т.к. page относится к "сжатым страницам"
        # и не соответствует нумерации оригинальной книги. При следующем открытии
        # оригинальной книги начнется с 0. Если открывается сжатая, начнется с 0 сжатой.
        # Можно усложнить и сохранять отдельно, но пока так проще.
        # Поэтому сохраняем, только если НЕ has_compressed_context
        if not data.get('has_compressed_context', False):
            await loop.run_in_executor(None, db.update_reading_state, user_id, book_id, current_page)
            print(f"Сохранена страница {current_page} для оригинала user {user_id}, book {book_id}")
        else:
            # Для сжатой книги сбрасываем страницу в БД на 0 при выходе,
            # чтобы при следующем открытии (оригинала или сжатой) начать с начала
            await loop.run_in_executor(None, db.update_reading_state, user_id, book_id, 0)
            print(f"Выход из сжатой книги. Сброшена страница на 0 в БД для user {user_id}, book {book_id}")

        # Сбрасываем флаг сессии
        await loop.run_in_executor(None, db.set_book_session_status, user_id, book_id, False)

    await callback.message.edit_text(LEXICON["/start"], reply_markup=start_keyboard)
    await state.clear()
    await callback.answer()

# --- Пагинация ---

# --- ИЗМЕНЕННЫЙ ХЕНДЛЕР ПАГИНАЦИИ ВПЕРЕД ---
@router.callback_query(ReadBookState.reading, F.data == 'forward')
async def process_forward_press(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    data = await state.get_data()
    book_id = data.get('book_id')
    current_page = data.get('page')
    total_pages = data.get('total_pages') # total_pages для текущего вида (оригинал/сжатый)
    has_compressed_context = data.get('has_compressed_context', False)

    if book_id is None or current_page is None or total_pages is None:
        await callback.message.edit_text("Произошла ошибка состояния чтения.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer()
        return

    next_page = current_page + 1

    # --- Проверка достижения конца книги ---
    if next_page >= total_pages:
        await callback.answer("Вы уже на последней странице.")
        # --- ВАЖНО: Даже если мы на последней странице, проверим, не нужно ли обновить контекст ИИ для ОРИГИНАЛА ---
        if not has_compressed_context:
            # Получаем последнее состояние AI из БД, чтобы знать update_page
            reading_state_ai = await loop.run_in_executor(None, db.get_reading_state_details, user_id, book_id)
            last_update_page = 0
            current_fsm_context = data.get('compressed_book_context') # Текущий контекст в FSM
            db_book_context = None # Контекст из БД

            if reading_state_ai:
                 last_update_page = reading_state_ai.get('update_page', 0)
                 db_book_context = reading_state_ai.get('book_context')

            # Если update_page меньше последней страницы оригинала, нужно обновить
            original_total_pages = data.get('original_total_pages')
            if original_total_pages is not None and original_total_pages > last_update_page:
                 page_to_update_until = original_total_pages # Обновляем до конца книги

                 print(f"[Last Page Check] Обновление AI контекста ОРИГИНАЛА до стр. {page_to_update_until} (user {user_id}, book {book_id})")
                 try:
                      # Получаем текущий контекст из FSM или БД для передачи в update_book_content
                      context_to_pass = current_fsm_context if current_fsm_context else db_book_context

                      new_book_context = await update_book_content(
                          user_id, book_id, page_to_update_until, last_update_page, context_to_pass
                      )
                      # new_book_context может быть None при ошибке или равен context_to_pass, если ничего не добавилось
                      if new_book_context is not None:
                          # Сохраняем в БД ВСЕГДА, чтобы обновить update_page
                          await loop.run_in_executor(
                              None, db.update_ai_context, user_id, book_id, page_to_update_until, new_book_context
                          )
                          # Обновляем FSM ТОЛЬКО если контекст реально ИЗМЕНИЛСЯ
                          if new_book_context != context_to_pass:
                              await state.update_data(compressed_book_context=new_book_context)
                              print(f"[Last Page Check] AI контекст (для чата) обновлен в FSM до стр. {page_to_update_until}.")
                          else:
                              print(f"[Last Page Check] Контекст AI не изменился или update_book_content вернул старый, FSM не обновлен.")
                      else:
                          # Если update_book_content вернул None (ошибка), запишем в БД None и обновим update_page
                          await loop.run_in_executor(
                              None, db.update_ai_context, user_id, book_id, page_to_update_until, None
                          )
                          # Также очистим FSM
                          await state.update_data(compressed_book_context=None)
                          print(f"[Last Page Check] Функция update_book_content вернула None, контекст AI в БД и FSM очищен, update_page={page_to_update_until}.")
                 except Exception as e:
                      print(f"[Last Page Check] Ошибка при вызове или сохранении update_book_content: {e}")
            else:
                print("[Last Page Check] Контекст ИИ уже обновлен до конца оригинала.")
        # Выход, т.к. дальше листать некуда
        return

    # --- Логика обновления AI контекста при пагинации ОРИГИНАЛЬНОЙ книги (НЕ на последней странице) ---
    if not has_compressed_context:
        original_total_pages = data.get('original_total_pages')
        # Проверяем, нужно ли обновить контекст (каждые N страниц ИЛИ на предпоследней)
        if original_total_pages is not None: # Только если знаем общее число страниц оригинала
            SHOULD_UPDATE_AI = (next_page % 30 == 0) or (next_page == original_total_pages - 1)
            if SHOULD_UPDATE_AI:
                reading_state_ai = await loop.run_in_executor(None, db.get_reading_state_details, user_id, book_id)
                last_update_page = 0
                current_fsm_context = data.get('compressed_book_context') # Текущий контекст в FSM
                db_book_context = None # Контекст из БД

                if reading_state_ai:
                     last_update_page = reading_state_ai.get('update_page', 0)
                     db_book_context = reading_state_ai.get('book_context')

                # Обновляем включительно до *следующей* страницы (той, на которую перешли)
                # Так как next_page - это индекс (0..N-1), то page_to_update_until = next_page + 1
                page_to_update_until = next_page + 1
                if page_to_update_until > last_update_page:
                     print(f"Обновление AI контекста для ОРИГИНАЛА на стр. {page_to_update_until} (user {user_id}, book {book_id})")
                     try:
                          context_to_pass = current_fsm_context if current_fsm_context else db_book_context
                          new_book_context = await update_book_content(
                              user_id, book_id, page_to_update_until, last_update_page, context_to_pass
                          )
                          if new_book_context is not None:
                               await loop.run_in_executor(
                                   None, db.update_ai_context, user_id, book_id, page_to_update_until, new_book_context
                               )
                               if new_book_context != context_to_pass:
                                   await state.update_data(compressed_book_context=new_book_context)
                                   print(f"AI контекст (для чата) обновлен в FSM до стр. {page_to_update_until}.")
                               else:
                                   print(f"Контекст AI не изменился или update_book_content вернул старый, FSM не обновлен.")
                          else:
                               await loop.run_in_executor(
                                   None, db.update_ai_context, user_id, book_id, page_to_update_until, None
                               )
                               await state.update_data(compressed_book_context=None)
                               print(f"Функция update_book_content вернула None, контекст AI в БД и FSM очищен, update_page={page_to_update_until}.")
                     except Exception as e:
                          print(f"Ошибка при вызове или сохранении update_book_content: {e}")


    # Обновляем ТОЛЬКО номер страницы в FSM
    await state.update_data(page=next_page)
    # Отображаем новую страницу
    await display_book_content(callback, state, user_id, book_id)
    await callback.answer() # Подтверждаем нажатие кнопки


@router.callback_query(ReadBookState.reading, F.data == 'backward')
async def process_backward_press(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    # loop = asyncio.get_running_loop() # Не нужен для простого декремента
    data = await state.get_data()
    book_id = data.get('book_id')
    current_page = data.get('page')
    # total_pages не нужен для движения назад

    if book_id is None or current_page is None:
        await callback.message.edit_text("Произошла ошибка состояния чтения.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer()
        return

    if current_page <= 0:
        await callback.answer("Вы уже на первой странице.")
        return

    prev_page = current_page - 1
    # Обновляем ТОЛЬКО номер страницы в FSM
    await state.update_data(page=prev_page)
    await display_book_content(callback, state, user_id, book_id) # Обновляем сообщение
    await callback.answer()


# --- Чат с AI ---

# Точка входа в чат
@router.callback_query(ReadBookState.reading, F.data == 'chat_with_ai')
async def process_ai_chat_press(callback: CallbackQuery, state: FSMContext):
    user_data = await state.get_data()
    book_id = user_data.get("book_id")
    book_name = user_data.get("book_name")
    if not book_id or not book_name:
        await callback.message.edit_text("Ошибка: Не удалось определить книгу для чата.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer(); return

    # --- Дополнительная проверка и обновление контекста перед входом в чат ---
    # Особенно актуально, если пользователь долистал до конца и нажал чат
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    data = await state.get_data()
    has_compressed_context = data.get('has_compressed_context', False)

    if not has_compressed_context:
         # Проверяем, обновлен ли контекст до текущей страницы (или до конца, если на последней)
         current_page = data.get('page', 0)
         original_total_pages = data.get('original_total_pages')
         target_update_page = current_page + 1 # Должен быть обновлен до СЛЕДУЮЩЕЙ страницы
         if original_total_pages and current_page == original_total_pages -1:
             target_update_page = original_total_pages # На последней странице - до конца

         reading_state_ai = await loop.run_in_executor(None, db.get_reading_state_details, user_id, book_id)
         last_update_page = 0
         if reading_state_ai:
              last_update_page = reading_state_ai.get('update_page', 0)

         if target_update_page > last_update_page:
              print(f"[Chat Entry Check] Контекст устарел (target={target_update_page}, last={last_update_page}). Обновляю...")
              status_msg = await callback.message.answer("⏳ Обновляю контекст для ИИ...") # Временное сообщение
              try:
                   current_fsm_context = data.get('compressed_book_context')
                   db_book_context = reading_state_ai.get('book_context') if reading_state_ai else None
                   context_to_pass = current_fsm_context if current_fsm_context else db_book_context

                   new_book_context = await update_book_content(
                       user_id, book_id, target_update_page, last_update_page, context_to_pass
                   )
                   if new_book_context is not None:
                        await loop.run_in_executor(
                            None, db.update_ai_context, user_id, book_id, target_update_page, new_book_context
                        )
                        if new_book_context != context_to_pass:
                            await state.update_data(compressed_book_context=new_book_context)
                            print(f"[Chat Entry Check] AI контекст обновлен в FSM до стр. {target_update_page}.")
                   else:
                        await loop.run_in_executor(
                            None, db.update_ai_context, user_id, book_id, target_update_page, None
                        )
                        await state.update_data(compressed_book_context=None)
                        print(f"[Chat Entry Check] update_book_content вернула None, контекст очищен.")
                   await status_msg.delete() # Удаляем временное сообщение
              except Exception as e:
                   print(f"[Chat Entry Check] Ошибка обновления контекста: {e}")
                   await status_msg.edit_text("⚠️ Не удалось обновить контекст ИИ.")
                   # Не прерываем вход в чат, используем что есть

    # --- Переход в состояние чата ---
    await callback.message.edit_text(
        f"💬 Вхожу в режим чата с ИИ по книге \"{book_name[:60]}...\".\n"
        "Задайте свой вопрос:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Вернуться к книге", callback_data="leave_ai_chat")]
        ])
    )
    await state.set_state(ChattingWithModelState.awaiting_message_for_model)
    await callback.answer()


# handle_user_question_input - без изменений
@router.message(ChattingWithModelState.awaiting_message_for_model, F.text)
async def handle_user_question_input(message: Message, state: FSMContext):
    await state.update_data(question=message.text)
    await message.answer("Выберите желаемый размер ответа:", reply_markup=answer_length_keyboard)
    await state.set_state(ChattingWithModelState.awaiting_answer_size)

# --- ИЗМЕНЕННЫЙ ХЕНДЛЕР ОБРАБОТКИ ОТВЕТА AI ---
@router.callback_query(ChattingWithModelState.awaiting_answer_size, F.data.startswith("answer_"))
async def handle_user_question_size(callback: CallbackQuery, state: FSMContext):
    answer_size = callback.data.split("_")[1] # 'short', 'medium', 'long'
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    data = await state.get_data()
    book_id = data.get("book_id")
    question = data.get("question")
    fsm_book_context = data.get("compressed_book_context") # Контекст из FSM

    if not book_id or not question:
        await callback.message.edit_text("Ошибка: Отсутствует информация о книге или вопросе.",
                                         reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                                             [InlineKeyboardButton(text="◀️ Вернуться к книге", callback_data="leave_ai_chat")]
                                         ]))
        await state.set_state(ChattingWithModelState.awaiting_message_for_model)
        await callback.answer(); return

    await callback.message.edit_text("⏳ Думаю над вашим вопросом...")

    book_context_for_ai = None
    chat_history = []
    db_book_context = None # Контекст из БД

    try:
        # --- УЛУЧШЕННАЯ ЗАГРУЗКА КОНТЕКСТА И ИСТОРИИ ---
        reading_state_details = await loop.run_in_executor(None, db.get_reading_state_details, user_id, book_id)

        if reading_state_details:
             chat_history = reading_state_details.get('chat_history') if reading_state_details.get('chat_history') else []
             db_book_context = reading_state_details.get('book_context')
             # db_update_page = reading_state_details.get('update_page', 0) # update_page здесь не так важен

             # Выбираем самый полный контекст: из БД или из FSM
             # Отдаем предпочтение контексту из БД, так как он может быть только что обновлен
             # перед входом в чат или при листании до конца. FSM обновляется с задержкой.
             if db_book_context:
                  book_context_for_ai = db_book_context
                  # Если контекст из БД отличается от FSM, обновим FSM
                  if book_context_for_ai != fsm_book_context:
                       await state.update_data(compressed_book_context=book_context_for_ai)
                       print(f"[Chat Handler] Обновлен FSM контекст из БД (длина {len(book_context_for_ai)}).")
             else:
                  # Если в БД контекста нет (очень странно, но возможно), берем из FSM
                  book_context_for_ai = fsm_book_context
        else:
             # Если деталей из БД нет, используем то, что есть в FSM
             book_context_for_ai = fsm_book_context

        # --- Проверка на пустой контекст ---
        if not book_context_for_ai:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Контекст для AI (book_context_for_ai) не найден ни в FSM, ни в БД для user {user_id}, book {book_id}.")
            book_context_for_ai = "Контекст книги недоступен." # или ""

        # --- Вызов AI для ответа ---
        answer = await ask_question(
            question,
            book_context_for_ai, # Передаем лучший найденный контекст
            chat_history,
            answer_size
        )

        # --- Сохранение чата в БД ---
        await loop.run_in_executor(None, db.add_chat_message, user_id, book_id, 'user', question)
        await loop.run_in_executor(None, db.add_chat_message, user_id, book_id, 'assistant', answer)

        # --- Отображение ответа ---
        await callback.message.edit_text(answer, reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Вернуться к книге", callback_data="leave_ai_chat")]
        ]))

    except Exception as e:
        print(f"Ошибка при вызове ask_question или сохранении чата (user: {user_id}, book: {book_id}): {e}")
        # --- Отображение ошибки ---
        await callback.message.edit_text("Произошла ошибка при обработке вашего вопроса.", reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Вернуться к книге", callback_data="leave_ai_chat")]
        ]))

    # Возвращаемся в состояние ожидания следующего вопроса
    await state.set_state(ChattingWithModelState.awaiting_message_for_model)
    await callback.answer()


# Кнопка "Вернуться к книге" из чата AI
@router.callback_query(ChattingWithModelState.awaiting_message_for_model, F.data == "leave_ai_chat")
async def process_ai_leave_press(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    data = await state.get_data()
    book_id = data.get("book_id")

    if not book_id:
        await callback.message.edit_text("Не удалось определить книгу. Возврат в главное меню.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer(); return

    # Просто вызываем display_book_content, он покажет нужную страницу (сжатую или нет)
    # display_book_content сам установит правильное состояние ReadBookState.reading
    await display_book_content(callback, state, user_id, book_id)
    await callback.answer()


# --- Рекомендации (код без изменений) ---
# Нажатие кнопки "Рекомендации"
@router.callback_query(F.data == "preferences")
async def process_preferences_press(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Исключать ранее рекомендованные книги из новых подборок?",
                                     reply_markup=exclude_prev_keyboard)
    await state.set_state(InputPrefsState.waiting_for_exclude_option)
    await callback.answer()

# Выбор опции исключения (Да/Нет)
@router.callback_query(InputPrefsState.waiting_for_exclude_option, F.data.in_(['exclude_yes', 'exclude_no']))
async def process_exclude_option_callback(callback: CallbackQuery, state: FSMContext):
    exclude_previous = (callback.data == 'exclude_yes')
    await state.update_data(exclude_previous=exclude_previous)
    await callback.message.edit_text("Выберите режим подбора рекомендаций:",
                                     reply_markup=recommendation_mode_keyboard)
    await callback.answer() # Убрали переход в след. состояние, он будет при выборе режима

# --- Режим 1 (на основе книги) ---
@router.callback_query(F.data == "mode_1") # Убрали состояние из фильтра
async def process_mode1_press(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)
    if not user_books:
         await callback.message.edit_text("У вас нет загруженных книг для этого режима.", reply_markup=start_keyboard)
         await state.clear(); await callback.answer(); return

    keyboard = create_mode1_history_keyboard(user_books)
    if not keyboard:
        await callback.message.edit_text("Не удалось создать список книг.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer(); return

    await callback.message.edit_text("Выберите книгу, похожие на которую вы бы хотели найти:", reply_markup=keyboard)
    await state.set_state(InputPrefsState.waiting_for_mode1_book)
    await callback.answer()

# Выбор книги для Режима 1
@router.callback_query(InputPrefsState.waiting_for_mode1_book, F.data.startswith("mode1_"))
async def process_mode1_recommendation(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    try:
        book_id = int(callback.data.split("_")[1])
    except (IndexError, ValueError):
        await callback.message.answer("Ошибка: Некорректные данные книги."); await state.clear(); await callback.answer(); return

    data = await state.get_data()
    exclude_previous_recs = data.get('exclude_previous', False)
    await callback.message.edit_text("⏳ Подбираю рекомендации на основе выбранной книги...")

    book_text = await loop.run_in_executor(None, db.get_book_full_text, user_id, book_id)

    if book_text is None or book_text.startswith("Ошибка"):
        await callback.message.edit_text(book_text or "Не удалось получить текст книги-основы.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer(); return

    try:
        recommendations_str = await get_book_recommendations(
            user_prefs="",
            book_content=book_text,
            mode=1,
            user_id=user_id,
            exclude_previous=exclude_previous_recs,
            dataset=recommendation_dataset
        )
        if recommendations_str and not recommendations_str.startswith("К сожалению") and not recommendations_str.startswith("Ошибка"):
            response_text = "✅ Рекомендую похожие книги:\n\n" + recommendations_str
            await callback.message.edit_text(response_text, reply_markup=start_keyboard, parse_mode="Markdown")
        else:
            await callback.message.edit_text(recommendations_str or "К сожалению, не удалось подобрать рекомендации.", reply_markup=start_keyboard)
    except Exception as e:
        print(f"Ошибка при получении рекомендаций (Режим 1, user: {user_id}, book: {book_id}): {e}")
        await callback.message.edit_text("❌ Ошибка при подборе рекомендаций.", reply_markup=start_keyboard)

    await state.clear(); await callback.answer()

# --- Режим 2 (на основе описания) ---
@router.callback_query(F.data == "mode_2") # Убрали состояние из фильтра
async def process_mode2_press(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Опишите книгу или жанр, который вы ищете (например, 'легкая научная фантастика о контакте' или 'детектив в стиле Агаты Кристи'):")
    await state.set_state(InputPrefsState.waiting_for_input)
    await callback.answer()

# Получение текстового описания для Режима 2
@router.message(InputPrefsState.waiting_for_input, F.text)
async def process_mode2_recommendation(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_prefs_text = message.text
    data = await state.get_data()
    exclude_previous_recs = data.get('exclude_previous', False)
    processing_msg = await message.answer("⏳ Ищу книги по вашему описанию...")

    try:
        recommendations_str = await get_book_recommendations(
            user_prefs=user_prefs_text,
            book_content="",
            mode=2,
            user_id=user_id,
            exclude_previous=exclude_previous_recs,
            dataset=recommendation_dataset
        )
        await processing_msg.delete()

        if recommendations_str and not recommendations_str.startswith("К сожалению") and not recommendations_str.startswith("Ошибка"):
            response_text = "✅ Нашел для вас следующие книги:\n\n" + recommendations_str
            await message.answer(response_text, reply_markup=start_keyboard, parse_mode="Markdown")
        else:
            await message.answer(recommendations_str or "К сожалению, не удалось подобрать рекомендации.", reply_markup=start_keyboard)
    except Exception as e:
        try: await processing_msg.delete()
        except: pass
        print(f"Ошибка при получении рекомендаций (Режим 2, user: {user_id}): {e}")
        await message.answer("❌ Ошибка при подборе рекомендаций.", reply_markup=start_keyboard)

    await state.clear()

# --- Режим 3 (описание + стиль книги) ---
@router.callback_query(F.data == "mode_3") # Убрали состояние из фильтра
async def process_mode3_press(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Сначала опишите желаемую книгу/жанр:")
    await state.set_state(InputPrefsState.waiting_for_input_mode3)
    await callback.answer()

# Получение текстового описания для Режима 3
@router.message(InputPrefsState.waiting_for_input_mode3, F.text)
async def process_mode3_history_selection(message: Message, state: FSMContext):
    user_id = message.from_user.id
    loop = asyncio.get_running_loop()
    user_prefs_text = message.text
    await state.update_data(input_text=user_prefs_text)

    user_books = await loop.run_in_executor(None, db.get_user_books, user_id)
    if not user_books:
         await message.answer("У вас нет загруженных книг для выбора стиля.", reply_markup=start_keyboard)
         await state.clear(); return

    keyboard = create_mode3_history_keyboard(user_books)
    if not keyboard:
         await message.answer("Не удалось создать список книг для выбора стиля.", reply_markup=start_keyboard)
         await state.clear(); return

    await message.answer("Теперь выберите книгу, стиль которой нужно учесть:", reply_markup=keyboard)
    await state.set_state(InputPrefsState.waiting_for_mode3_book)

# Выбор книги для стиля в Режиме 3
@router.callback_query(InputPrefsState.waiting_for_mode3_book, F.data.startswith("mode3_"))
async def process_mode3_recommendation(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    loop = asyncio.get_running_loop()
    try:
        style_book_id = int(callback.data.split("_")[1])
    except (IndexError, ValueError):
        await callback.message.edit_text("Ошибка: Некорректные данные книги.", reply_markup=start_keyboard); await state.clear(); await callback.answer(); return

    data = await state.get_data()
    user_prefs_text = data.get("input_text")
    exclude_previous_recs = data.get('exclude_previous', False)

    if not user_prefs_text:
         await callback.message.edit_text("Ошибка: Отсутствует описание желаемой книги.", reply_markup=start_keyboard); await state.clear(); await callback.answer(); return

    await callback.message.edit_text("⏳ Подбираю рекомендации с учетом описания и стиля...")

    style_book_text = await loop.run_in_executor(None, db.get_book_full_text, user_id, style_book_id)

    if style_book_text is None or style_book_text.startswith("Ошибка"):
        await callback.message.edit_text(style_book_text or "Не удалось получить текст книги для анализа стиля.", reply_markup=start_keyboard)
        await state.clear(); await callback.answer(); return

    try:
        recommendations_str = await get_book_recommendations(
            user_prefs=user_prefs_text,
            book_content=style_book_text,
            mode=3,
            user_id=user_id,
            exclude_previous=exclude_previous_recs,
            dataset=recommendation_dataset
        )
        if recommendations_str and not recommendations_str.startswith("К сожалению") and not recommendations_str.startswith("Ошибка"):
            response_text = "✅ Вот что я подобрал с учетом ваших пожеланий и стиля:\n\n" + recommendations_str
            await callback.message.edit_text(response_text, reply_markup=start_keyboard, parse_mode="Markdown")
        else:
            await callback.message.edit_text(recommendations_str or "К сожалению, не удалось подобрать рекомендации.", reply_markup=start_keyboard)
    except Exception as e:
        print(f"Ошибка при получении рекомендаций (Режим 3, user: {user_id}, style_book: {style_book_id}): {e}")
        await callback.message.edit_text("❌ Ошибка при подборе рекомендаций.", reply_markup=start_keyboard)

    await state.clear(); await callback.answer()


# --- Обработчик неизвестных команд/текста в глобальном состоянии ---
@router.message(F.text)
async def process_unknown_message(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Извините, я не понимаю эту команду или текст. Используйте кнопки меню.")

# Обработчик неизвестных колбеков (проверяем список известных)
# Обновленный список известных callback prefixes/data
KNOWN_CALLBACKS = (
    'read_book_', 'mode1_', 'mode3_', 'answer_', 'exclude_',
    'cancel_', 'leave_ai_chat', 'back_to_start', 'ignore_page',
    'upload', 'read', 'preferences', 'mode_1', 'mode_2', 'mode_3',
    'chat_with_ai', 'forward', 'backward', 'cancel_compress',
    'cancel_reading' # Добавим на всякий случай, хотя он должен быть покрыт cancel_
)

@router.callback_query(~F.data.startswith(KNOWN_CALLBACKS))
async def process_unknown_callback(callback: CallbackQuery):
    # Дополнительная проверка для полных совпадений (не startswith)
    if callback.data not in KNOWN_CALLBACKS:
        print(f"Неизвестный callback: {callback.data} от user {callback.from_user.id}")
        try:
            # Отвечаем на колбек, чтобы убрать "часики"
            await callback.answer("Неизвестное действие или устаревшая кнопка.", show_alert=True)
            # Можно также отредактировать сообщение, если нужно
            # await callback.message.edit_text("Произошла ошибка или кнопка устарела. Попробуйте вернуться в главное меню.", reply_markup=start_keyboard)
        except Exception as e:
            print(f"Ошибка при ответе на неизвестный callback: {e}")