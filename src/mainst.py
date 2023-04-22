
from utils_mainst.functionst import *
import pandas as pd

menu = st.sidebar.selectbox('Menu',('Pagina principal','Lectura de datos','Trabajo'))

df = pd.read_csv('src\data\TIC.csv', sep = ';', encoding= 'latin-1')

df = arreglar_datos(df)



if menu == 'Pagina principal':
    pagina_principal()
elif menu == 'Lectura de datos':
    mostrar_datos(df)
elif menu == 'Trabajo':
    trabajo(df)