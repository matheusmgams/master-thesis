import os
import datetime
import mplcursors
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Diretório onde estão os arquivos .csv
diretorio = os.getcwd() + '/dataset-samples/'

# Lista para armazenar os DataFrames de cada arquivo
dataframes = []

# Percorre os arquivos no diretório
for arquivo in os.listdir(diretorio):
    if arquivo.endswith('.csv'):  # Verifica se o arquivo é um arquivo .csv
        # Caminho completo do arquivo
        caminho_arquivo = os.path.join(diretorio, arquivo)
        
        # Carrega o arquivo .csv em um DataFrame
        df = pd.read_csv(caminho_arquivo)
        
        # Adiciona o DataFrame à lista
        dataframes.append(df)

# Concatena todos os DataFrames da lista em um único DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Lista das colunas que você deseja descartar
colunas_descartadas = [
    'id_sm', 'samplenumber', 'lotnumber', 'binnumber', 'partnumber', 
    'turnsratio', 'highdcbusvoltage', 'avgdcbusvoltage', 'lowdcbusvoltage', 'highprimarycurrent', 
    'lowprimarycurrent', 'highsecondarycurrent', 
    'lowsecondarycurrent', 'cfactor', 'percentheatorcurrent', 'targetcurrent', 'steppernumber', 
    'totalweldcount', 'stepweldcount', 'spotid', 'contactor', 'avgontime', 
    'hifreqcyclecount', 'weldstatus', 'faultcode', 'alertcode', 'raftmode', 'rd', 're', 'rp', 
    'estthickness', 'toolinteg',  'resistance_data', 'current_data', 'heat_data', 'energy_data', 
    'station', 'stack', 'ipaddress',
    'processinteg', 'nuggetinteg', 'refrd', 'refrp', 'refenergy', 'refheat', 'progthickness', 
    'masteroffset', 'weightone', 'weighttwo', 'weightthree', 'risetime', 'ressumb', 'ressumc', 
    'ressumd', 'wqim', 'wqio', 'expcycle', 'wslide', 'percentsat', 'learnedi', 'preheattime', 
    'vspotid_sm', 'current_data_sm', 'resistance_data_sm', 'energy_data_sm', 'heat_data_sm', 
    'id_gr', 'fk_weld_id', 'record_id', 'timestamp_gr', 'summarydataformat', 'cycledataformat', 
    'cycledataresolution', 'summary_data', 'current_data_gr', 'resistance_data_gr', 'energy_data_gr', 
    'heat_data_gr', 'Unnamed: 0', 'short_tag', 'extended_notes', 'vspotid_gr', 'weld_type'
]

# Verifica se as colunas estão presentes no DataFrame antes de tentar removê-las
colunas_presentes = [coluna for coluna in colunas_descartadas if coluna in df.columns]

# Descarta as colunas presentes no DataFrame
df = df.drop(columns=colunas_presentes)

# Definir os tipos de dados desejados para cada coluna
tipos_de_dados = {
    'stackup': 'string',
    'spot_id': 'string',
    'sequencenumber': 'int64',
    'timestamp_sm': 'datetime64[ns]',
    'avgresistance': 'int64', 
    'avgsecvoltage': 'int64', 
    'totalenergy': 'int64', 
    'totalheat': 'int64',
    'stepnumber': 'int64',
    'avgprimarycurrent':'int64', 
    'avgsecondarycurrent':'int64', 
}

# Define os tipos de dados das colunas no DataFrame
df_final = df.astype(tipos_de_dados)

# Preencher os valores da coluna stackup com base no valor da coluna stepnumber
df_final['stackup'] = df_final.apply(lambda row: 'EMPTY' if pd.isna(row['stackup']) and row['stepnumber'] == 0 else row['stackup'], axis=1)
df_final['stackup'] = df_final['stackup'].fillna('MISSING STACKUP')

# Excluir 'sequencenumber' das colunas numéricas, se presente
colunas_numericas = df_final.select_dtypes(include=['float64', 'int64']).columns
colunas_datetime = df_final.select_dtypes(include=['datetime64[ns]']).columns

if 'sequencenumber' in colunas_numericas:
    colunas_numericas = colunas_numericas.drop('sequencenumber')

if 'stepnumber' in colunas_numericas:
    colunas_numericas = colunas_numericas.drop('stepnumber')
    
if 'spot_id' in colunas_numericas:
    colunas_numericas = colunas_numericas.drop('spot_id')

if 'stackup' in colunas_numericas:
    colunas_numericas = colunas_numericas.drop('stackup')

# Aplicar a normalização min-max apenas nas colunas numéricas
scaler = MinMaxScaler()
dados_normalizados_numericos = scaler.fit_transform(df_final[colunas_numericas].values)

# Converter os dados normalizados de volta para um DataFrame pandas
df_normalizado_numericos = pd.DataFrame(dados_normalizados_numericos, columns=colunas_numericas)

# Concatenar as colunas de data e hora com as colunas normalizadas
df_normalizado = pd.concat([df_final[colunas_datetime], df_final['sequencenumber'], df_final['stepnumber'], df_final['spot_id'], df_final['stackup'], df_normalizado_numericos], axis=1)

# Visualizar
sample_data = df_normalizado.sample(frac=1)

# Ordenar o DataFrame pelo número de sequência
sample_data = sample_data.sort_values(by='sequencenumber')

# Converter a coluna "sequencenumber" para tipo string
sample_data['sequencenumber'] = sample_data['sequencenumber'].astype(str)

# Define isolation variable
variable = "stackup"

# Filtra os dados onde o stackup não é 'EMPTY'
sample_data_filtered = sample_data

# Remove valores NaN antes de mapear as cores
sample_data_filtered = sample_data_filtered.dropna(subset=[variable])

# Mapeia cada tipo de stackup para uma cor
colors = list(mcolors.TABLEAU_COLORS.values())  # Usaremos valores em vez de chaves
stackup_to_color = {stackup: color for stackup, color in zip(sample_data_filtered[variable].unique(), colors)}

# Adiciona uma nova coluna 'color' mapeando o stackup para a cor correspondente
sample_data_filtered['color'] = sample_data_filtered[variable].map(stackup_to_color)

# Substitui os valores NaN na coluna de cores por uma cor padrão (por exemplo, 'gray')
sample_data_filtered['color'] = sample_data_filtered['color'].fillna('gray')

# Calcula o tempo em segundos a partir de um ponto de referência
sample_data_filtered['time_seconds'] = sample_data_filtered['timestamp_sm']

# Cria uma lista de strings para hovertext
hover_text = [f"SpotID: {spot_id}, \n Schedule: {sequencenumber}, \n Energy: {total_energy}, \n Time: {time_seconds}, \n Heat: {total_heat} , \n Stackup: {stackup}" for total_energy, time_seconds, total_heat, stackup, sequencenumber, spot_id in zip(sample_data_filtered['totalenergy'], sample_data_filtered['time_seconds'], sample_data_filtered['totalheat'], sample_data_filtered['stackup'], sample_data_filtered['sequencenumber'], sample_data_filtered['spot_id'])]

# Função para adicionar linhas que se propagam ao longo do eixo do tempo
def add_time_lines(fig, data, enable=True):
    if enable:
        # Ordena os dados pelo datetime
        data_sorted = data.sort_values(by='timestamp_sm')
        # Inicia o rastreamento dos pontos iniciais de cada sequência e spot_id
        seq_spot_lines = {}
        for index, row in data_sorted.iterrows():
            seq_num = row['sequencenumber']
            spot_id = row['spot_id']
            seq_spot_id = (seq_num, spot_id)
            if seq_num not in seq_spot_lines:
                seq_spot_lines[seq_num] = {}
            if spot_id not in seq_spot_lines[seq_num]:
                seq_spot_lines[seq_num][spot_id] = []
            seq_spot_lines[seq_num][spot_id].append((row['totalenergy'], row['time_seconds'], row['totalheat']))
        # Adiciona linhas que se propagam ao longo do eixo do tempo para cada sequência
        for seq_num, spot_lines in seq_spot_lines.items():
            for spot_id, points in spot_lines.items():
                # Ordena os pontos pelo tempo
                points_sorted = sorted(points, key=lambda x: x[1])
                # Coleta as coordenadas dos pontos ordenados
                x_vals, y_vals, z_vals = zip(*points_sorted)
                # Adiciona uma linha que passa por todos os pontos
                fig.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=str(seq_num) + "_" + str(spot_id),  # Define o nome do trace como "seq_num_spot_id"
                    hoverinfo='skip'  # Não mostra o texto de hover para as linhas
                ))

# Cria a figura 3D
fig = go.Figure()

# Plota o gráfico de dispersão 3D
scatter = fig.add_trace(go.Scatter3d(
    x=sample_data_filtered['totalenergy'],
    y=sample_data_filtered['time_seconds'],
    z=sample_data_filtered['totalheat'],
    mode='markers',
    marker=dict(
        size=5,
        color=sample_data_filtered['color'],  # Usa a coluna 'color' para definir as cores
        opacity=0.8,
        colorscale='Viridis'  # Define uma escala de cores
    ),
    hovertext=hover_text,  # Define o texto de hover
    hoverinfo='text'  # Mostra o texto de hover
))

# Adiciona linhas que se propagam ao longo do eixo do tempo
add_time_lines(fig, sample_data_filtered, enable=True)  # Habilita as linhas

# Define os rótulos dos eixos
fig.update_layout(scene=dict(
                    xaxis_title='Total Energy',
                    yaxis_title='Time (seconds)',
                    zaxis_title='Total Heat'))

# Exibe a figura
fig.show()
