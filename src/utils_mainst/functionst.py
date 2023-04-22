import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def pagina_principal():
    st.title('Estudio de adicción de las TIC')

    st.subheader('Introducción')
    st.markdown('Texto de introducción')

    #añadir una imagen: img = Image.open(src/data/nombre_imagen.jpg)// st.image(img, use_column_width='auto')

    with st.expander('Sesiones'):
        if st.checkbox('Sesión 1', value = True):
            st.markdown('explicacion e imagen sobre la primera sesion')
        
        if st.checkbox('Sesión 2', value = True):
            st.markdown('explicación e imagen de la sesión segunda')

        if st.checkbox('Sesión 3', value = True):
            st.markdown('explicación e imagen de la sesión tercera')
    
    st.markdown('Finalmente se le pasa una encuesta a los estudiantes que tienen que rellenar, que es la siguiente')
    #insertar o imagen de la encuesta o la encuesta en si misma

def mostrar_datos(df):
    st.markdown('Hasta la fecha un total de 2239 de estudiantes han rellenado la encuesta. Vamos a indagar en los datos para\
                 ver que conclusiones podemos sacar de ellos')
    st.dataframe(df,2000,500)
    
    tab1, tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12 = st.tabs(["Genero",
                                                                                "Edad",'Realización','Interés','Valor de la exposición',
                                                                               'Recomendación','Utilidad de los contenidos',
                                                                               'Satisfacción Sesión 1','Satisfacción Sesión 2',
                                                                               'Satisfacción Sesión 3', 'Satisfacción total','Pregunta'])
    
    

    with tab1:
        st.markdown('Aquí podemos apreciar que se han impartido las sesiones, prácticamente a la misma cantadad de hombres como de mujeres')
        x = df.Genero.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Género'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Genero.describe(),1000,178)

        

    with tab2:
        st.markdown('Se puede apreciar que la mayoría del público estaban cursando educacion secundaria')
        st.bar_chart(df.Edad.value_counts())
        st.dataframe(df.Edad.describe(),1000,318)
        

    with tab3:
        st.markdown('En conjunto, ¿Cómo valora lo desarrollado a lo largo de las tres sesiones realizadas sobre TIC')
        #st.bar_chart(df.Realizacion.value_counts())
        x = df.Realizacion.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Realización'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Realizacion.describe(),1000,318)
        

    with tab4:
        st.markdown('Valore el interés que han tenido los temas tratados')
        #st.bar_chart(df.Interes.value_counts())
        x = df.Interes.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Interés'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Interes.describe(),1000,318)

    with tab5:
        st.markdown('Le han gustado las exposiciones realizadas por la formadora')
        #st.bar_chart(df.Valor_exposicion.value_counts())
        x = df.Valor_exposicion.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Valor de la exposición'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Valor_exposicion.describe(),1000,318)
    with tab6:
        st.markdown('¿Aconsejaría esta actividad?')
        x = df.Recomendacion.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Recomendación'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Recomendacion.describe(),1000,178)
        

    with tab7:
        st.markdown('¿Considera que los contenidos de esta actividad le serán útiles para su vida')
        #st.bar_chart(df.Utilidad_contenidos.value_counts())
        x = df.Utilidad_contenidos.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Utilidad de los contenidos'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Utilidad_contenidos.describe(),1000,318)

    with tab8:
        st.markdown('Grado de satisfacción Sesión 1')
        #st.bar_chart(df.Satisfaccion_s1.value_counts())
        x = df.Satisfaccion_s1.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Satisfacción de la primera sesión'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Satisfaccion_s1.describe(),1000,318)

    with tab9:
        st.markdown('Grado de satisfacción Sesión 2')
        #st.bar_chart(df.Satisfaccion_s2.value_counts())
        x = df.Satisfaccion_s2.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Satisfacción de la segunda sesión'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Satisfaccion_s2.describe(),1000,318)

    with tab10:
        st.markdown('Grado de satisfacción Sesión 3')
        #st.bar_chart(df.Satisfaccion_s3.value_counts())
        x = df.Satisfaccion_s3.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Satisfacción de la tercera sesión'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Satisfaccion_s3.describe(),1000,318)

    with tab11:
        st.markdown('Grado de satisfacción total de las tres Sesiones')
        #st.bar_chart(df.Satisfaccion_total.value_counts())
        x = df.Satisfaccion_total.value_counts()
        fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Satisfacción total'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)
        st.dataframe(df.Satisfaccion_total.describe(),1000,318)
    
    with tab12:
        st.markdown('¿Qué cambiarías o añadirías a las sesiones?')
        x= df.Pregunta
        df1 = pd.DataFrame(x)
        for y in range(len(df1.Pregunta)):
            if type(df1.iloc[y,0])==float:
                df1.iloc[y,0] = 'Sin respuesta'
        df2 = pd.DataFrame(df1.Pregunta.value_counts())
        df2[df2.Pregunta<10] = np.NaN
        df2.dropna(inplace=True)
        lista=[]
        for x in df1.Pregunta.unique():
            if not (x in df2.index):
                lista.append(x)
        df1[df1.Pregunta.isin(lista)] = np.NaN
        df1.dropna(inplace=True)
        plt.figure(figsize=(25,15))
        fig = sb.countplot(x = df1.Pregunta)
        plt.xticks(rotation = -45)
        plt.xlabel('Respuesta')
        f= fig.figure
        x = df1.Pregunta.value_counts()
        fig={'data':[{'values':x[x>34],'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
         'layout':{'title':'Respuestas más populares'}}
        st.plotly_chart(fig,use_container_width=True,theme = None)        
        st.pyplot(f)
        st.subheader('Otras opiniones son:')
        a=0
        for y in lista:
            if len(y)>20:
                a+=1
                y = y[0].upper()+y[1:]

                st.markdown(str(a)+'.- ' + y)


def trabajo(df):

    

    st.markdown('En este apartado vamos a trabajar con diferentes métodos de Maching Learning para realizar \
                una regresión supervisada sobre los datos de la Edad, una clasificación supervisada sobre los datos de la columna\
                del Género, y por último realizaremos una agrupación en dos clases de individuos.')
    tab_info,tab_reg,tab_clas,tab_cluster = st.tabs(['Informacion','Regresión','Clasificador','Agrupación'])
    
    with tab_info:
        st.header('Información general de los datos')
        st.subheader('Matriz de correlación')
        st.markdown('La siguiente matriz de correlación, representada en el siguietne heatmap, muestra que los modelos líneales van a funcionar particularmente mal para clasificar\
                    el género y la edad con estos datos,\
                    puesto que los valores de la matriz son muy próximos al cero')
        #st.dataframe(df.corr(),1500,457)
        plt.figure(figsize=(10,10))
        fig = sb.heatmap(df.corr(),
            vmin=-1,
            vmax=1,
            center=0,
            cmap=sb.diverging_palette(145, 280, s=85, l=25, n=10),
            square=True,
            annot=True,
            linewidths=.5)
        f = fig.figure
        st.pyplot(f)
        
    
    with tab_reg:
        st.markdown('hola')
        regresion(df)
    
    with tab_clas:
        st.subheader('CLASIFICADOR')
        st.markdown('En este apartado vamos a utilizar un modelo para clasificar el género de la población que ha rellenado la encuesta.')
        clasificacion(df)
        
    with tab_cluster:
        kmeans(df)


def arreglar_datos (df):
    df.columns=['a','b','Genero', 'Edad','Realizacion', 'Interes', 'Valor_exposicion', 'Recomendacion',
             'Utilidad_contenidos', 'Satisfaccion_s1', 'Satisfaccion_s2','Satisfaccion_s3','Satisfaccion_total','Pregunta']
    caracteres= []
    palabras = []
    pregunta = []
    for x in df.Pregunta:
        if type(x) == float:
            x=''
        a = x.split()
        pregunta.append(x)
        palabras.append(len(a))
        b =0
        for y in x:
            b+=1
        caracteres.append(b)
    df['Caracteres']=caracteres
    df['Palabras']= palabras
    del(df['Pregunta'])
    recomendacion = []
    for x in df.Recomendacion:
        if x == 'si':
            recomendacion.append(0)
        else:
            recomendacion.append(1)
    Recomendacion = df.Recomendacion
    del(df['Recomendacion'])
    df['Recomendacion_binaria'] = recomendacion
    genero = []

    for x in df.Genero:
        if x == 'Femenino':
            genero.append(1)
        elif x == 'Masculino':
            genero.append(0)
    Genero = df.Genero
    df['Genero_binario'] = genero
    del(df['Genero'])
    del(df['a'])
    del(df['b'])


    df['Genero'] = Genero
    df['Recomendacion']=Recomendacion
    df['Pregunta'] = pregunta
    return df


def kmeans (df):
    df1 = pd.DataFrame(df)
    df1 = df1[df1.Edad <20] 
    del(df1['Caracteres'])
    del(df1['Palabras'])
    X1 = df1.iloc[:,:-3]

    with open('src/model/my_model_kmeans', 'rb') as archivo_entrada:
        Kmeans = pickle.load(archivo_entrada)

    y_pred=Kmeans.fit_predict(X1)
    y_pred1 = pd.Series(y_pred)
    print(y_pred1.value_counts())
    df1['Grupo'] = y_pred

    with st.expander('Grupos'):
        if st.checkbox('Grupo 1', value = True):
            mostrar_datos(df1[df1.Grupo ==0])

        if st.checkbox('Grupo 2', value = True):
            mostrar_datos(df1[df1.Grupo ==1])


def clasificacion(df):
    
    
    df2 = pd.DataFrame(df)
    X= df2.iloc[:,:-4]
    y = df.Genero_binario

    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    x_train['y'] = y_train
    x_train= x_train[x_train.Edad < 20]
    y_train = x_train.y
    del(x_train['y'])
    with open('.\src\model\modelo_clasificacion_rf', 'rb') as archivo_entrada:
        clasificador_rf = pickle.load(archivo_entrada)
    with open('.\src\model\modelo_clasificacion_lg', 'rb') as archivo_entrada:
        clasificador_lg = pickle.load(archivo_entrada)
    
    best_grids = pd.DataFrame( columns = ["Grid", "Score"])
    best_grids.Grid=['gs_reg_log','gs_rand_forest']
    best_grids.Score = [clasificador_lg.score(x_test,y_test),clasificador_rf.score(x_test,y_test)]

    best_grids = best_grids.sort_values(by = "Score", ascending = False)

    st.markdown('Hemos entrenado dos modelos y sus scorings han sido los siguientes')
    st.dataframe(best_grids)


    st.markdown('Gracias a esta información sabemos que utilizaremos el siguiente **clasificador**: ')
    st.markdown( clasificador_rf["classifier"])
    
    df_clas = pd.DataFrame(list(y_test), columns=['y_real'])
    df_clas['y_prediccion'] = clasificador_rf.predict(x_test)
    df_clas['Accuracy'] = df_clas.y_real==df_clas.y_prediccion
    st.dataframe(df_clas)
    x = df_clas.Accuracy.value_counts()
    fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
     'layout':{'title':'Porcentaje de acierto'}}
    st.plotly_chart(fig,use_container_width=True,theme = None)

    
    st.markdown('Así pues, concluimos que una encuesta de satisfacción como esta, no recaba datos relacionados con el género del individuo.\
                Veamos si podemos obtener mejores resultados con la edad de los estudiantes.')
    

def regresion (df):
     
    df3 = pd.DataFrame(df)
    X= df3.iloc[:,1:-3]
    y = df.Edad

    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    x_train['y'] = y_train
    x_train= x_train[x_train.y < 20]
    y_train = x_train.y
    del(x_train['y'])
    with open('.\src\model\modelo_clasificacion1_rf', 'rb') as archivo_entrada:
        clasificador1_rf = pickle.load(archivo_entrada)
    with open('.\src\model\modelo_clasificacion1_lg', 'rb') as archivo_entrada:
        clasificador1_lg = pickle.load(archivo_entrada)
    with open('.\src\model\modelo_regresion_lr', 'rb') as archivo_entrada:
        regresor_lr = pickle.load(archivo_entrada)
    with open('.\src\model\modelo_regresion_tree', 'rb') as archivo_entrada:
        regresor_tree = pickle.load(archivo_entrada)
    
    prediction = regresor_tree.predict(x_test)
    prediction_round_int_tree = list(prediction)
    for x in range(len(prediction)):
        prediction_round_int_tree[x]=round(prediction_round_int_tree[x])
    
    error_tree1 = np.abs(y_test-np.array(prediction_round_int_tree))
    error_tree =round(np.mean(error_tree1), 2)

    prediction = regresor_lr.predict(x_test)
    prediction_round_int = list(prediction)
    for x in range(len(prediction)):
        prediction_round_int[x]=round(prediction_round_int[x])

    error_lr1 = np.abs(y_test-np.array(prediction_round_int))
    error_lr = round(np.mean(error_lr1), 2)
    error_grid_lg1 = np.abs(y_test-clasificador1_lg.predict(x_test))
    error_grid_lg = round(np.mean(error_grid_lg1), 2)
    error_grid_rf1 = np.abs(y_test-clasificador1_rf.predict(x_test))
    error_grid_rf = round(np.mean(error_grid_rf1), 2)
    


    best_grids = pd.DataFrame( columns = ["Grid", "Error_Absoluto"])
    best_grids.Grid=['gs_reg_log','gs_rand_forest','Linear_regression','Tree_regression']
    best_grids.Error_Absoluto = [error_grid_lg,error_grid_rf,error_lr,error_tree]

    best_grids = best_grids.sort_values(by = "Error_Absoluto", ascending = True)

    st.markdown('Hemos entrenado dos modelos y sus scorings han sido los siguientes')
    st.dataframe(best_grids)


    st.markdown('Gracias a esta información sabemos que utilizaremos el siguiente **clasificador**: ')
    st.markdown( regresor_tree)
    
    df_clas = pd.DataFrame(list(y_test), columns=['y_real'])
    df_clas['y_prediccion'] = prediction_round_int_tree
    df_clas['Accuracy'] = df_clas.y_real==df_clas.y_prediccion
    st.dataframe(df_clas)
    x = df_clas.Accuracy.value_counts()
    fig={'data':[{'values':x,'labels':x.index,'domain':{'x':[0,0.7]},'name':'','hoverinfo':'label+value','hole':0.5,'type':'pie'},],
     'layout':{'title':'Porcentaje de acierto'}}
    st.plotly_chart(fig,use_container_width=True,theme = None)

    
    st.markdown('Así pues, concluimos que una encuesta de satisfacción como esta, no recaba datos relacionados con la edad del individuo.\
                Veamos si podemos obtener mejores resultados con la edad de los estudiantes.')
    
    