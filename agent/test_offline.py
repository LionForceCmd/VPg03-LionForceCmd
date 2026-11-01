"""Офлайн-проверка без API-ключа: фейковая модель вызывает tools, затем отдаёт ResponseFormat."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from main import build_agent, ask


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


script = iter([
    call("summarize_expenses", {"file_name": "expenses.csv"}, "c1"),
    call("ResponseFormat", {"answer": "Потрачено 18309.80 руб., больше всего на продукты",
                            "key_numbers": ["18309.80", "Продукты"]}, "c2"),
    call("get_user_budget", {}, "c3"),
    call("ResponseFormat", {"answer": "Лимит 20000, укладываетесь"}, "c4"),
])
agent = build_agent(Fake(messages=script))
print(ask(agent, "Сколько я потратил за октябрь?", "t1"))
print(ask(agent, "Укладываюсь ли я в мой лимит?", "t1", user_id="1"))
state = agent.get_state({"configurable": {"thread_id": "t1"}})
print("сообщений в памяти треда:", len(state.values["messages"]))
for m in state.values["messages"]:
    if m.type == "tool":
        print("TOOL:", m.name, "->", m.content)
