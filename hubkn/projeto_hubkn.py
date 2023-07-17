import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import streamlit as st


# Page config
st.set_page_config(
    page_title="Funil de Vendas",
    page_icon="img/hubkn_logo.png",
)

# Page title
st.title('Deals - Status Prediction')
st.image('img/hubkn_logo.png')
st.write("\n\n")

st.markdown(
    """
    Este aplicativo tem o intuito de prever o status do negócio no estágio em que ele se encontrada inserindo o número de atividades feitas em cada estágio até o momento atual.
    Os estágios do funil de vendas são mostrados abaixo em ordem:
    - MQL
    - Qualificação
    - Agendamento of children
    - Demonstração
    - Proposta
    - Fechamento
    - Faturamento
    - Ganho
    - Perdido
    """
)
# -- Model Selection -- #

model_number = st.selectbox('Selecione o modelo', list(range(1, 10)))

# -- Parameters -- #

parameters = {
    1: ['MQL'],
    2: ['MQL', 'Qualificação'],
    3: ['MQL', 'Qualificação', 'Agendamento'],
    4: ['MQL', 'Qualificação', 'Agendamento', 'Demonstração'],
    5: ['MQL', 'Qualificação', 'Agendamento', 'Demonstração', 'Proposta'],
    6: ['MQL', 'Qualificação', 'Agendamento', 'Demonstração', 'Proposta', 'Fechamento'],
    7: ['MQL', 'Qualificação', 'Agendamento', 'Demonstração', 'Proposta', 'Fechamento', 'Faturamento'],
    8: ['MQL', 'Qualificação', 'Agendamento', 'Demonstração', 'Proposta', 'Fechamento', 'Faturamento', 'Ganho'],
    9: ['MQL', 'Qualificação', 'Agendamento', 'Demonstração', 'Proposta', 'Fechamento', 'Faturamento', 'Ganho', 'Perdido']
}

selected_parameters = parameters[model_number]

user_input = {}
for parameter in selected_parameters:
    value = st.number_input(label=parameter)
    user_input[parameter] = value

# -- Model -- #

model_path = f'models/modelo_{model_number}.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

def prediction():
    # Create DataFrame from user input
    df_input = pd.DataFrame(user_input, index=[0])

    # Load df_final with medians
    df_final = pd.read_excel('data/df_final.xlsx')
    df_medians = df_final.median()

    # Replace 0 values with respective column medians from df_final
    for col in df_input.columns:
        if (df_input[col] == 0).any():
            df_input.loc[df_input[col] == 0, col] = df_medians[col]

    if isinstance(model, Pipeline):
        classifier = model.named_steps['model']
        prediction = classifier.predict(df_input)
        probabilities = classifier.predict_proba(df_input)
    else:
        prediction = model.predict(df_input)
        probabilities = model.predict_proba(df_input)

    return prediction, probabilities


if st.button('Predict'):
    status_prediction, probabilities = prediction()
    prediction_value = status_prediction[0]
    probability_value = round(probabilities[0][1], 2)*100
    
    if prediction_value == 0:
        prediction_label = "Lost"
    elif prediction_value == 1:
        prediction_label = "Won"
    else:
        prediction_label = "unknown"
    
    st.success(f"Predição: {prediction_label}")
    



