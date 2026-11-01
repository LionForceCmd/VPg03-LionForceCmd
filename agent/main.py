"""Базовый агент LangChain по примеру из урока VPg03 + собственный tool.

Из урока: системный промпт, tools, модель, структурированный ответ (dataclass),
память (InMemorySaver + thread_id), runtime-контекст (user_id).
Свой tool: summarize_expenses - сводка личных расходов из CSV по категориям.
"""
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """Ты помощник по личным финансам. Если вопрос про расходы из
файла - обязательно вызови summarize_expenses и опирайся только на его
результат, ничего не выдумывай. Если пользователь говорит "мой бюджет" или
"мой лимит" - сначала узнай его через get_user_budget. Отвечай серьёзно и кратко."""

SAMPLES = Path(__file__).parent / "samples"

# Контекст пользователя (аналог get_user_location из урока): месячный лимит, руб.
BUDGETS = {"1": 20000, "2": 35000}


@dataclass
class Context:
    user_id: str


@tool
def get_user_budget(runtime: ToolRuntime[Context]) -> str:
    """Возвращает месячный лимит расходов пользователя по user_id из контекста."""
    limit = BUDGETS.get(runtime.context.user_id)
    return f"{limit} руб. в месяц" if limit else "лимит не задан"


@tool
def summarize_expenses(file_name: str) -> str:
    """Читает CSV с расходами (колонки date,category,amount,comment) из папки
    samples и возвращает общую сумму, суммы по категориям (по убыванию) и
    самую большую трату.

    Args:
        file_name: имя файла в папке samples, например "expenses.csv".
    """
    path = SAMPLES / Path(file_name).name  # .name - защита от выхода из папки
    if path.suffix.lower() != ".csv" or not path.is_file():
        return f"Файл {file_name} не найден. Доступно: " + ", ".join(
            p.name for p in SAMPLES.glob("*.csv")
        )
    by_cat = defaultdict(float)
    biggest = None
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            amount = float(row["amount"])
            by_cat[row["category"]] += amount
            if biggest is None or amount > biggest[0]:
                biggest = (amount, row["category"], row["comment"])
    total = sum(by_cat.values())
    cats = ", ".join(f"{c}: {s:.2f}" for c, s in
                     sorted(by_cat.items(), key=lambda x: -x[1]))
    return (f"{path.name}: всего {total:.2f} руб.; по категориям - {cats}; "
            f"самая большая трата - {biggest[0]:.2f} ({biggest[1]}, {biggest[2]})")


@dataclass
class ResponseFormat:
    """Структурированный ответ агента."""
    answer: str
    key_numbers: list[str] | None = None  # ключевые цифры, если были


def build_agent(model=None):
    if model is None:
        model = init_chat_model(
            "openai:gpt-4o-mini",
            temperature=0,
            max_tokens=1500,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_budget, summarize_expenses],
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=InMemorySaver(),  # память диалога
    )


def ask(agent, text, thread_id, user_id="1"):
    cfg = {"configurable": {"thread_id": thread_id}}
    res = agent.invoke({"messages": [{"role": "user", "content": text}]},
                       config=cfg, context=Context(user_id=user_id))
    return res["structured_response"]


if __name__ == "__main__":
    agent = build_agent()
    print(ask(agent, "Сколько я потратил за октябрь в expenses.csv и на что больше всего?", "t1"))
    # тот же thread_id -> агент помнит предыдущий вопрос
    print(ask(agent, "Укладываюсь ли я в мой лимит?", "t1"))
