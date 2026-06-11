import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import json
import io
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3-0324:novita"
MAX_STEPS = 10

SYSTEM_PROMPT = """Ты — агент для анализа данных. Датасет уже загружен и доступен как переменная `df`.

Ты работаешь в цикле: думаешь -> вызываешь инструмент -> видишь результат -> думаешь дальше.
Используй инструменты последовательно, чтобы провести полноценный анализ. В процессе анализа обязательно хотя бы раз вызывай инструмент run_python

Доступные инструменты:
- run_python: выполнить Python-код. Переменные df, pd, np, plt доступны автоматически.
  Текстовые результаты выводи через print(). Графики строй через plt — они будут показаны автоматически.
  Пример графика: fig, ax = plt.subplots(); ax.hist(df['col']); plt.tight_layout()
  НЕ нужно делать return, plt.show() или st.pyplot() — это происходит автоматически.
- finish: завершить анализ и вернуть итоговый отчёт пользователю.

Правила:
- Всегда начинай с изучения данных (df.info(), df.describe(), пропуски).
- Затем проводи анализ по инструкции пользователя.
- Каждый вызов run_python — одна логическая задача (либо вычисление, либо один график).
- Если в ответе инструмента написано "График показан" — он успешно отображён, не повторяй.
- В конце вызови finish с подробным текстовым отчётом на русском языке.
- В run_python запрещены: os, subprocess, open(), socket, eval, exec, __import__.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Выполняет Python-код и возвращает stdout + графики. "
                "Переменные df (DataFrame), pd, np, plt доступны. "
                "Текст выводи через print(). Графики строй через plt — показываются автоматически, return не нужен."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python-код для выполнения"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Завершает анализ и показывает итоговый отчёт пользователю.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {
                        "type": "string",
                        "description": "Итоговый отчёт на русском языке с выводами и инсайтами"
                    }
                },
                "required": ["report"]
            }
        }
    }
]

FORBIDDEN = [
    'import os', 'import subprocess', 'import socket', 'import shutil',
    'import pathlib', '__import__', 'open(', 'sys.exit', 'os.', 'subprocess.',
    'socket.', 'popen',
    'ignore my', 'disregard', 'forget your instructions',
    'забудь инструкции', 'игнорируй', 'новая роль', 'притворись',
]

def is_safe(code: str) -> tuple[bool, str]:
    lower = code.lower()
    for token in FORBIDDEN:
        if token.lower() in lower:
            return False, f"Запрещённый паттерн: `{token}`"
    return True, ""

def execute_python(code: str, df: pd.DataFrame) -> dict:
    safe, reason = is_safe(code)
    if not safe:
        return {"stdout": "", "error": reason, "figures": []}

    stdout_buf = io.StringIO()
    result = {"stdout": "", "error": None, "figures": []}
    figs_before = set(plt.get_fignums())

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "print": lambda *a, **kw: stdout_buf.write(" ".join(str(x) for x in a) + "\n"),
    }

    try:
        exec(compile(code, "<agent>", "exec"), namespace)
        result["stdout"] = stdout_buf.getvalue()

        figs_after = set(plt.get_fignums())
        new_fig_nums = figs_after - figs_before
        result["figures"] = [plt.figure(n) for n in sorted(new_fig_nums)]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["stdout"] = stdout_buf.getvalue()

    return result

def call_llm(messages: list, tools: list) -> dict:
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"model": MODEL, "messages": messages, "tools": tools, "tool_choice": "auto"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]

def run_agent(df: pd.DataFrame, user_instruction: str, history: list):
    messages = history + [{"role": "user", "content": user_instruction}]
    steps = []
    for step_num in range(MAX_STEPS):
        assistant_msg = call_llm(messages, TOOLS)
        messages.append(assistant_msg)

        if not assistant_msg.get("tool_calls"):
            text = assistant_msg.get("content", "")
            if text:
                steps.append({"type": "text", "content": text})
            break

        for tool_call in assistant_msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = json.loads(tool_call["function"]["arguments"])
            call_id = tool_call["id"]

            if fn_name == "finish":
                report = fn_args.get("report", "")
                steps.append({"type": "report", "content": report})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "Отчёт показан пользователю"
                })
                return steps, messages

            if fn_name == "run_python":
                code = fn_args.get("code", "")
                steps.append({"type": "code", "content": code})

                exec_result = execute_python(code, df)

                tool_output_parts = []
                if exec_result["stdout"]:
                    tool_output_parts.append(exec_result["stdout"])
                    steps.append({"type": "result", "content": exec_result["stdout"]})
                if exec_result["error"]:
                    tool_output_parts.append(f"ОШИБКА: {exec_result['error']}")
                    steps.append({"type": "error", "content": exec_result["error"]})
                for fig in exec_result["figures"]:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    steps.append({"type": "figure", "content": buf})
                    tool_output_parts.append("График построен и показан пользователю")
                plt.close("all")

                tool_output = "\n".join(tool_output_parts) or "Код выполнен (вывода нет)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_output
                })

    return steps, messages

for key, default in [("df", None), ("history", []), ("sessions", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("<h1 style='text-align:center'>Агент для анализа данных</h1>", unsafe_allow_html=True)

if st.session_state["df"] is None:
    _, col, _ = st.columns([1, 3, 1])
    with col:
        uploaded = st.file_uploader("Файл (CSV / Excel / JSON)", type=["csv", "xlsx", "json"])
        instruction = st.text_area(
            "Инструкция для агента", height=120,
            placeholder="Например: проанализируй выбросы и покажи распределения числовых колонок"
        )

        if st.button("Запустить анализ", use_container_width=True):
            if uploaded is None:
                st.warning("Сначала загрузите файл")
            elif not instruction.strip():
                st.warning("Напишите инструкцию")
            else:
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                df = {"csv": pd.read_csv, "xlsx": pd.read_excel, "json": pd.read_json}[ext](uploaded)
                st.session_state["df"] = df

                col_info = ", ".join(f"{c} ({t})" for c, t in df.dtypes.items())
                first_msg = (
                    f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов\n"
                    f"Колонки: {col_info}.\n"
                    f"Первые 3 строки:\n{df.head(3).to_string()}\n\n"
                    f"Задача: {instruction.strip()}"
                )

                history = [{"role": "system", "content": SYSTEM_PROMPT}]

                with st.spinner("Агент работает…"):
                    steps, history = run_agent(df, first_msg, history)

                st.session_state["history"] = history
                st.session_state["sessions"].append({
                    "label": instruction.strip()[:60],
                    "steps": steps,
                })
                st.rerun()

else:
    df = st.session_state["df"]

    for session in st.session_state["sessions"]:
        st.markdown(f"### {session['label']}")

        for step in session["steps"]:
            if step["type"] == "code":
                with st.expander("Агент вызвал инструмент: run_python", expanded=False):
                    st.code(step["content"], language="python")

            elif step["type"] == "result":
                with st.expander("Результат выполнения", expanded=True):
                    st.text(step["content"])

            elif step["type"] == "figure":
                st.image(step["content"])

            elif step["type"] == "error":
                st.error(f"Ошибка при выполнении: {step['content']}")

            elif step["type"] == "report":
                st.success("Итоговый отчёт агента")
                st.markdown(step["content"])

            elif step["type"] == "text":
                st.info(step["content"])

        st.divider()

    with st.sidebar:
        st.markdown("### Новый запрос")
        st.caption(f"{df.shape[0]} строк x {df.shape[1]} столбцов")

        follow_up = st.text_area(
            "Инструкция", height=150,
            placeholder="Например: построй тепловую карту корреляций"
        )

        if st.button("Отправить", use_container_width=True):
            if follow_up.strip():
                with st.spinner("Агент думает…"):
                    steps, history = run_agent(df, follow_up, st.session_state["history"])
                st.session_state["history"] = history
                st.session_state["sessions"].append({
                    "label": follow_up.strip()[:60],
                    "steps": steps,
                })
                st.rerun()

        st.divider()

        if st.button("Начать заново", use_container_width=True):
            st.session_state["df"] = None
            st.session_state["history"] = []
            st.session_state["sessions"] = []
            st.rerun()