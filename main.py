import pathlib
import time
import os

import pandas as pd
import streamlit as st

from PIL import Image
from dotenv import load_dotenv, find_dotenv

from data_processor import MarketingCompanyPredictorPreprocessor

load_dotenv(find_dotenv())


def process_main_page():
    show_main_page()
    process_side_bar_inputs()

def show_main_page():
    image = Image.open('data/picture.png')
    icon = Image.open('data/icon.png')
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Binary classification project",
        page_icon=icon
    )
    st.title('Классификация клиентов банка')
    st.subheader('Определяем, кто из клиентов положительно отреагирует на предложение банка, а кто – нет.')
    st.image(image)
    st.write('Один из способов повысить эффективность взаимодействия банка с клиентами — '
             'отправлять предложение о новой услуге не всем клиентам, а только некоторым, '
             'которые выбираются по принципу наибольшей склонности к отклику на это предложение.'
             ' В данном проекте реализован линейный классификтор, обученный на банковских данных.')




def tabs(prediction_probas):
    tab1, tab2, tab3 = st.tabs(['Аналитика',
                                'Подробнее о модели',
                                'Что сделано'])
    with tab1:
        st.write('**Возраст клиентов**')
        st.image(Image.open('data/age_plot.png'))
        st.write('Самый распространённый возраст - от 30 до 40 лет')
        st.divider()

        st.write('**Пол клиентов**')
        st.image(Image.open('data/gender_plot.png'))
        st.write('* Мужчин 65%', '\n'
                 '* Женщин 35%')
        st.divider()

        st.write('**Семейное положение**')
        st.image(Image.open('data/martial_plot.png'))
        st.write('Клиентов состоящих в браке больше всех (~ 62%)')
        st.divider()

        st.write('**Количество детей**')
        st.image(Image.open('data/child_plot.png'))
        st.divider()

        st.write('**Уровень образования**')
        st.image(Image.open('data/ed_plot.png'))
        st.divider()

        st.write('**Персональный доход**')
        st.image(Image.open('data/income_plot.png'))
        st.write('Распредление похоже на логнормальное')
        st.divider()
        st.write('**Корреляционная матрица**')
        st.image(Image.open('data/corr_heatmap.png'))

    with tab2:
        st.write("**Вероятность предсказания**")
        st.write(prediction_probas)
        data = {'Class': [0, 1],
                'Precision': [0.93, 0.15],
                'Recall': [0.46, 0.75],
                'F1-Score': [0.61, 0.26],
                'Support': [4089, 543]}

        prediction_probas = pd.DataFrame(data)
        st.write('**classification_report**')
        st.write(prediction_probas)
        st.write('**Устарновленный порог**')
        st.write("* 0.13")
        st.write("**Качество модели**")
        st.write("* Accuracy: 0.639",'\n'
                 "* Precision: 0.174",'\n'
                 "* Recall: 0.556",'\n'
                 "* F1 Score: 0.265",'\n'
                 "* ROC-AUC: 0.648")


    with tab3:
        with open('report.txt', 'r') as file:
            review = file.read()
        st.write(review)


def sidebar_input_features():
    sex = st.sidebar.radio("Пол", ("Мужской", "Женский"))

    age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
                            step=1)

    education = st.sidebar.selectbox("Образование", (
        'Среднее специальное', 'Среднее', 'Высшее', 'Неполное среднее',
        'Неоконченное высшее', 'Два и более высших образования'))

    martial_status = st.sidebar.selectbox("Семейное положение", (
        'Разведен(а)', 'Состою в браке', 'Вдовец/Вдова',
        'Не состоял в браке', 'Гражданский брак'))

    children = st.sidebar.slider("Количество детей", min_value=0, max_value=10, value=0, step=1)

    dependants = st.sidebar.slider("Количество иждивенцев клиента", min_value=0, max_value=10, value=0, step=1)

    work_status = st.sidebar.radio("Работает ли клиент", ("Да", "Нет"))

    pens_status = st.sidebar.radio("На пенсии ли клиент", ("Да", "Нет"))

    regadress = st.sidebar.selectbox("Область регистрации клиента", (
        'Читинская область', 'Новгородская область', 'Орловская область',
        'Самарская область', 'Кемеровская область', 'Приморский край',
        'Липецкая область', 'Усть-Ордынский Бурятский АО',
        'Калининградская область', 'Вологодская область',
        'Ярославская область', 'Краснодарский край', 'Воронежская область',
        'Амурская область', 'Костромская область', 'Челябинская область',
        'Оренбургская область', 'Тюменская область', 'Мурманская область',
        'Томская область', 'Красноярский край', 'Новосибирская область',
        'Татарстан', 'Адыгея', 'Бурятия', 'Тульская область', 'Якутия',
        'Пермская область', 'Саратовская область', 'Ставропольский край',
        'Калужская область', 'Пензенская область', 'Белгородская область',
        'Ульяновская область', 'Алтайский край', 'Московская область',
        'Волгоградская область', 'Тамбовская область',
        'Ростовская область', 'Коми', 'Свердловская область',
        'Омская область', 'Курская область', 'Ленинградская область',
        'Псковская область', 'Горный Алтай', 'Санкт-Петербург',
        'Курганская область', 'Мордовская республика', 'Ямало-Ненецкий АО',
        'Карелия', 'Астраханская область', 'Магаданская область',
        'Сахалинская область', 'Удмуртия', 'Тверская область',
        'Марийская республика', 'Владимирская область',
        'Карачаево-Черкесия', 'Хабаровский край', 'Камчатская область',
        'Брянская область', 'Кировская область', 'Чувашия',
        'Ивановская область', 'Иркутская область', 'Рязанская область',
        'Ханты-Мансийский АО', 'Архангельская область',
        'Кабардино-Балкария', 'Смоленская область',
        'Нижегородская область', 'Башкирия', 'Северная Осетия', 'Калмыкия',
        'Хакасия', 'Еврейская АО', 'Москва', 'Эвенкийский АО', 'Дагестан',
        'Агинский Бурятский АО'
    ))

    fl_presence = st.sidebar.toggle("Наличие в собственности квартиры")
    own_car = st.sidebar.toggle("Наличие в собственности автомобиля")
    personal_income = st.sidebar.number_input("Персональный доход клиента", min_value=0)
    fam_income = st.sidebar.selectbox("Семейный доход", (
        'от 10000 до 20000 руб.', 'от 20000 до 50000 руб.',
        'от 5000 до 10000 руб.', 'свыше 50000 руб.', 'до 5000 руб.'))

    loan_total = st.sidebar.number_input("Количество ссуд клиента ", min_value=0)
    loan_closed = st.sidebar.number_input("Количество погашенных ссуд клиента", min_value=0)


    translate = {
        "Мужской": "1",
        "Женский": "0",
        "Да": "1",
        "Нет": "0",
    }

    data = {
        "AGE": age,
        "GENDER": translate[sex],
        "EDUCATION": education,
        "MARITAL_STATUS": martial_status,
        "CHILD_TOTAL": children,
        "DEPENDANTS": dependants,
        "SOCSTATUS_WORK_FL": translate[work_status],
        "SOCSTATUS_PENS_FL": translate[pens_status],
        "REG_ADDRESS_PROVINCE": regadress,
        "FL_PRESENCE_FL": fl_presence,
        "OWN_AUTO": own_car,
        'PERSONAL_INCOME': personal_income,
        "FAMILY_INCOME": fam_income,
        "LOAN_NUM_TOTAL": loan_total,
        "LOAN_NUM_CLOSED": loan_closed,
    }

    df = pd.DataFrame(data, index=[0])

    return df

def process_side_bar_inputs():
    st.sidebar.header('Введите параметры клиента для получения прогноза')
    user_input_df = sidebar_input_features()
    models_folder = pathlib.Path('.').resolve() / 'models'
    model = MarketingCompanyPredictorPreprocessor(models_folder=models_folder)
    prediction, prediction_probas = model.make_prediction(user_input_df)

    if st.sidebar.button('Получить прогноз'):
        with st.spinner('Считаем!'):
                time.sleep(3)
                st.sidebar.subheader("Предсказание")
                st.sidebar.write(prediction)
    tabs(prediction_probas)

if __name__ == "__main__":
    process_main_page()