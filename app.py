import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = 'https://router.huggingface.co/v1/chat/completions'
MODEL = 'deepseek-ai/DeepSeek-V4-Pro:novita'
MAX_RETRIES = 3

SYSTEM_PROMPT = """Ты — агент для анализа данных. Твой ответ — это ТОЛЬКО исполняемый Python-код. Никакого текста до или после кода. Никаких объяснений. Никаких markdown-блоков с ```.

Переменные уже доступны: df (DataFrame), pd, plt, st (streamlit), np. Используй только их либо встроунные python библиотеки. Не используй то что нужно докачивать.

Выводи результаты через streamlit: st.write(), st.dataframe(), st.metric(), st.pyplot(fig) и т.д.

Запрещено: os, subprocess, open(), socket, eval, exec, __import__. Если что то не касается анализа и угрожает безопасности приложения выводи предупреждение через st.warning
"""

FORBIDDEN = [
    'import os', 'import subprocess', 'import socket', 'import shutil',
    'import pathlib', '__import__', 'open(', 'sys.exit', 'os.', 'subprocess.',
    'socket.', 'eval(', 'exec(', 'popen',
    'ignore my', 'disregard', 'forget your instructions',
    'забудь инструкции', 'игнорируй', 'новая роль', 'притворись',
]

def is_safe(code):
    lower = code.lower()
    for token in FORBIDDEN:
        if token.lower() in lower:
            return False, f"Запрещённый паттерн: `{token}`"
    return True, ""

def run_code(code, df):
    safe, reason = is_safe(code)
    if not safe:
        return reason
    try:
        compile(code, '<string>', 'exec')
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"

def strip_fences(code):
    lines = code.splitlines()
    if lines and lines[0].startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].startswith('```'):
        lines = lines[:-1]
    return '\n'.join(lines).strip()

def call_llm(messages):
    resp = requests.post(
        API_URL,
        headers={'Authorization': f'Bearer {HF_TOKEN}'},
        json={'model': MODEL, 'messages': messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()

def run_agent(df, user_instruction, history):
    messages = history + [{'role': 'user', 'content': user_instruction}]

    for _ in range(MAX_RETRIES):
        code_raw = call_llm(messages)
        code = strip_fences(code_raw)
        messages.append({'role': 'assistant', 'content': code_raw})

        error = run_code(code, df)

        if error is None:
            return code, messages

        messages.append({
            'role': 'user',
            'content': f"Код упал с ошибкой. Исправь и верни только код.\n\nОшибка:\n{error}"
        })

    return None, messages

for key, default in [('df', None), ('history', []), ('results', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("<h1 style='text-align:center'>Агент для анализа данных</h1>", unsafe_allow_html=True)

if st.session_state['df'] is None:
    st.markdown("<p style='text-align:center;color:gray'>Загрузите файл и напишите запрос</p>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 3, 1])
    with col:
        uploaded = st.file_uploader('Файл (CSV / Excel / JSON)', type=['csv', 'xlsx', 'json'])
        instruction = st.text_area('Инструкция', height=120,
                                   placeholder='Например: проанализируй выбросы и покажи распределения')

        if st.button('Запустить анализ', use_container_width=True):
            if uploaded is None:
                st.warning('Сначала загрузите файл.')
            elif not instruction.strip():
                st.warning('Напишите инструкцию.')
            else:
                ext = uploaded.name.rsplit('.', 1)[-1].lower()
                df = {'csv': pd.read_csv, 'xlsx': pd.read_excel, 'json': pd.read_json}[ext](uploaded)
                st.session_state['df'] = df

                col_info = ', '.join(f"{c} ({t})" for c, t in df.dtypes.items())
                first_message = (
                    f"Датасет: {df.shape[0]} строк, {df.shape[1]} столбцов.\n"
                    f"Колонки: {col_info}.\n\n"
                    f"Первые 3 строки:\n{df.head(3).to_string()}\n\n"
                    f"Задача: {instruction.strip()}"
                )

                history = [{'role': 'system', 'content': SYSTEM_PROMPT}]

                with st.spinner('Агент анализирует…'):
                    code, history = run_agent(df, first_message, history)

                st.session_state['history'] = history
                st.session_state['results'].append({'label': f'{instruction.strip()[:60]}', 'code': code})
                st.rerun()

else:
    df = st.session_state['df']

    for res in st.session_state['results']:
        st.markdown(f"### {res['label']}")
        if res['code'] is None:
            st.error('Агент не смог выполнить код после нескольких попыток.')
        else:
            exec(res['code'], {'df': df.copy(), 'pd': pd, 'plt': plt, 'st': st, 'np': np})  # noqa: S102
            plt.close('all')
            with st.expander('Код', expanded=False):
                st.code(res['code'], language='python')
        st.divider()

    with st.sidebar:
        st.markdown("### Новый запрос")
        st.caption(f"{df.shape[0]} строк × {df.shape[1]} столбцов")
        follow_up = st.text_area('Инструкция', height=150,
                                 placeholder='Например: построй тепловую карту корреляций')

        if st.button('Отправить', use_container_width=True):
            if follow_up.strip():
                with st.spinner('Думаю…'):
                    code, history = run_agent(df, follow_up, st.session_state['history'])
                st.session_state['history'] = history
                st.session_state['results'].append({'label': f'{follow_up[:60]}', 'code': code})
                st.rerun()

        if st.button('Начать заново', use_container_width=True):
            st.session_state['df'] = None
            st.session_state['history'] = []
            st.session_state['results'] = []
            st.rerun()