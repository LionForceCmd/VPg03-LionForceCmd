# Базовый агент LangChain + свой tool

Агент по примеру из урока VPg03: системный промпт, tools, модель, структурированный ответ (`dataclass`), память (`InMemorySaver` + `thread_id`), контекст пользователя (`user_id`).

**Свой tool:** `summarize_expenses(file_name)` читает CSV с личными расходами и возвращает общую сумму, суммы по категориям и самую большую трату. Второй tool `get_user_budget` берёт месячный лимит из контекста (аналог `get_user_location` из урока).

## Запуск
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # вписать OPENAI_API_KEY (и OPENAI_BASE_URL для proxyapi)
python main.py
```
Проверка без ключа (фейковая модель): `python test_offline.py`

## Права

Все права защищены. Любое использование этого проекта, в том числе коммерческое или некоммерческое, запрещено.
