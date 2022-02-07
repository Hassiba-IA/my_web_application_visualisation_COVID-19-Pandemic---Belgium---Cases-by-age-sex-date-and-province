
##################### LOAD PACKAGES ################################
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling as pp
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
import datetime
import requests
import pandas as pd
from pandas import json_normalize
import streamlit as st
import matplotlib.pyplot as plt
import base64
import altair as alt

########################################################################



















########################################################################


# # Titre de l'application
# st.title("COVID-19 Pandemic - Belgium - Cases by age, sex, date and province")
# liste_choix_reg = ['Toutes les régions', 'Brussels', 'Flanders', 'Wallonia']

# select_graph = ['line_chart', 'bar_chart']


# url = 'https://public.opendatasoft.com/api/records/1.0/search/?dataset=covid-19-pandemic-belgium-cases-agesexprovince&q=&rows=5000&sort=date&facet=date&facet=province&facet=region&facet=agegroup&facet=sex&facet=Cases&facet=Geo+Point'
# r = requests.get(url)
# print(r.status_code)
# print(r.content)

# d = r.json()
# d.keys()

# df = json_normalize(d['records'])
# df.drop(['datasetid', 'fields.geo_point_2d',
#         'geometry.type'], axis=1, inplace=True)
# df.rename(columns={'recordid': "id_enregistrement", 'record_timestamp': "horodatage d'enreg", 'fields.province': "province", 'fields.region': "region",
#           'fields.date': "date", 'fields.agegroup': "groupe_age", 'fields.sex': "sex", 'fields.cases': "nombre_cas", 'geometry.coordinates': "coordonnées_géométrie"}, inplace=True)
# df.dropna(how='any', axis=0, inplace=True)
# df.set_index('date', inplace=True)


# select_graph = st.sidebar.selectbox(
#     "Sélectionnez le type de graphique", options=select_graph)
# # Liste avec les modalités définies au départ
# select_region = st.sidebar.selectbox(
#     "Sélectionnez une région", options=liste_choix_reg)

# if (select_region == 'Brussels'):
#     select_province = st.sidebar.selectbox(
#         "Sélectionnez une Province", options=['Brussels'])

# elif (select_region == 'Flanders'):
#     select_province = st.sidebar.selectbox("Sélectionnez une Province", options=[
#                                            'Toutes les provinces de Flanders', 'Antwerpen', 'Limburg', 'Oost-Vlaanderen', 'Vlaams-Brabant', 'West-Vlaanderen'])

# elif(select_region == 'Wallonia'):
#     select_province = st.sidebar.selectbox("Sélectionnez une Province", options=[
#                                            'Toutes les provinces de Wallonia', 'Brabant Wallon', 'Brussels', 'Liège', 'Hainaut', 'Luxembourg', 'Namur'])

# else:
#     select_province = st.sidebar.selectbox(
#         "Sélectionnez une Province (sélectionnez d'abord la région)", options=['Toutes les provinces'])

# # date=st.sidebar.date_input("Choisissez une date", min_value=datetime.date(2021, 12, 2), max_value=datetime.date(2021, 12, 31))
# genre = st.sidebar.radio("Choisissez le sex", options=['M', 'F'])
# # groupe_age = st.sidebar.slider("Choisissez la tranche d'âge", min_value=0, max_value=100, step=10)
# groupe_age = st.sidebar.selectbox("Choisissez la tranche d'âge", options=[
#                                   '0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+'])


# def ligne_chart_toute_reg():
#     if(select_graph == 'line_chart'):
#         return(alt.Chart(df.groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")))
#     else:
#         return(alt.Chart(df.groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")))


# def ligne_chart_toute_prov(region):
#     if(select_graph == 'line_chart'):
#         return(alt.Chart(df[df.region == region].groupby(['date', 'province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")))

#     else:
#         return(alt.Chart(df[df.region == region].groupby(['date', 'province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")))


# st.write(df)
# st.subheader('le tableau filtré')
# if(select_region == 'Toutes les régions'):
#     st.write(df)
# else:
#     if(select_province == "Toutes les provinces de Wallonia"):
#         st.write(df[(df.region == 'Wallonia')])
#     elif(select_province == 'Toutes les provinces de Flanders'):
#         st.write(df[(df.region == 'Flanders')])
#     else:
#         st.write(df[(df.region == select_region) &
#                  (df.province == select_province)])

# # partie region

# if(select_region == 'Toutes les régions'):

#     st.subheader(f'Le nombre de cas à {select_region}')
# else:
#     # selon jour et la région sélectionnés
#     st.subheader(f'Le nombre de cas à la région {select_region}')

# if (select_region == 'Toutes les régions'):
#      st.altair_chart(ligne_chart_toute_reg(), use_container_width=True)

# else:
#      st.altair_chart(ligne_chart_toute_prov(
#          select_region), use_container_width=True)

# # partie province

# if((select_province == 'Toutes les provinces de Flanders') | (select_province == 'Toutes les provinces de Wallonia') | (select_province == 'Toutes les provinces')):
#     st.subheader(f'Le nombre de cas à {select_province}')
# else:
#     # selon la province ,la date et la région sélectionnés avant
#     st.subheader(f'Le nombre de cas à la province {select_province}')

# if(select_province == 'Toutes les provinces'):
#     st.altair_chart(ligne_chart_toute_reg(), use_container_width=True)

# elif(select_province == 'Brussels'):
#     st.altair_chart(ligne_chart_toute_prov(
#         select_region), use_container_width=True)
# elif(select_province == 'Toutes les provinces de Flanders'):
#     st.altair_chart(ligne_chart_toute_prov(
#         select_region), use_container_width=True)

# elif(select_province == 'Toutes les provinces de Wallonia'):
#     st.altair_chart(ligne_chart_toute_prov(
#         select_region), use_container_width=True)

# else:
#     if(select_graph == 'line_chart'):
#         st.line_chart(df[df.province == select_province].groupby(
#             ['date'])['nombre_cas'].sum())
#     else:
#         st.bar_chart(df[df.province == select_province].groupby(
#             ['date'])['nombre_cas'].sum())

# # la partie gerne
# if(genre == 'M'):
#     # le nombre de cas selon le sexe sélectionné et les critères sélectionnés avant
#     st.subheader(f"Le nombre de cas par rapport aux Masculins")
# else:
#     st.subheader(f"Le nombre de cas par rapport aux Féminins")

# if(select_province == 'Toutes les provinces'):
#     if (select_graph == 'line_chart'):
#         st.altair_chart(alt.Chart(df[(df.sex == genre)].groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index(
#         )).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")), use_container_width=True)
#     else:
#         st.altair_chart(alt.Chart(df[(df.sex == genre)].groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index(
#         )).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")), use_container_width=True)

# elif((select_province == 'Toutes les provinces de Flanders') | (select_province == 'Toutes les provinces de Wallonia') | (select_province == 'Brussels')):
#     if(select_graph == 'line_chart'):
#         st.altair_chart(alt.Chart(df[(df.sex == genre) & (df.region == select_region)].groupby(['date', 'province']).agg(
#             {'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")), use_container_width=True)
#     else:
#         st.altair_chart(alt.Chart(df[(df.sex == genre) & (df.region == select_region)].groupby(['date', 'province']).agg(
#             {'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")), use_container_width=True)

# else:
#     if(select_graph == 'line_chart'):
#         st.line_chart(df[(df.province == select_province) & (
#             df.sex == genre)].groupby(['date'])['nombre_cas'].sum())
#     else:
#         st.bar_chart(df[(df.province == select_province) & (
#             df.sex == genre)].groupby(['date'])['nombre_cas'].sum())


# # partie groupes age
# # le nombre de cas selon la tranche d'âge selectionné et les critères sélectionnés avant
# st.subheader(f"Le nombre de cas par raport aux groupes d'âge ({groupe_age})")

# check_box = st.checkbox("Masculins et Féminins")

# if(check_box):
#     if(select_province == 'Toutes les provinces'):
#         if(select_graph == 'line_chart'):
#             st.altair_chart(alt.Chart(df[df.groupe_age == groupe_age].groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index(
#             )).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")), use_container_width=True)
#         else:
#             st.altair_chart(alt.Chart(df[df.groupe_age == groupe_age].groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index(
#             )).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")), use_container_width=True)

#     elif((select_province == 'Toutes les provinces de Flanders') | (select_province == 'Toutes les provinces de Wallonia') | (select_province == 'Brussels')):
#         if(select_graph == 'line_chart'):
#             st.altair_chart(alt.Chart(df[(df.region == select_region) & (df.groupe_age == groupe_age)].groupby(['date', 'province']).agg(
#                 {'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")), use_container_width=True)
#         else:
#             st.altair_chart(alt.Chart(df[(df.region == select_region) & (df.groupe_age == groupe_age)].groupby(['date', 'province']).agg(
#                 {'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")), use_container_width=True)

#     else:
#         if(select_graph == 'line_chart'):
#             st.line_chart(df[(df.province == select_province) & (
#                 df.groupe_age == groupe_age)].groupby(['date'])['nombre_cas'].sum())

#         else:
#             st.bar_chart(df[(df.province == select_province) & (
#                 df.groupe_age == groupe_age)].groupby(['date'])['nombre_cas'].sum())

# else:
#     if(select_province == 'Toutes les provinces'):
#         if(select_graph == 'line_chart'):
#             st.altair_chart(alt.Chart(df[(df.sex == genre) & (df.groupe_age == groupe_age)].groupby(['date', 'region']).agg(
#                 {'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")), use_container_width=True)

#         else:
#             st.altair_chart(alt.Chart(df[(df.sex == genre) & (df.groupe_age == groupe_age)].groupby(['date', 'region']).agg(
#                 {'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("region")), use_container_width=True)

#     elif((select_province == 'Toutes les provinces de Flanders') | (select_province == 'Toutes les provinces de Wallonia') | (select_province == 'Brussels')):
#         if(select_graph == 'line_chart'):
#             st.altair_chart(alt.Chart(df[(df.sex == genre) & (df.region == select_region) & (df.groupe_age == groupe_age)].groupby(['date', 'province']).agg(
#                 {'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")), use_container_width=True)

#         else:
#             st.altair_chart(alt.Chart(df[(df.sex == genre) & (df.region == select_region) & (df.groupe_age == groupe_age)].groupby(['date', 'province']).agg(
#                 {'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'), y=('nombre_cas'), color=alt.Color("province")), use_container_width=True)

#     else:
#         if(select_graph == 'line_chart'):
#             st.line_chart(df[(df.province == select_province) & (df.sex == genre) & (
#                 df.groupe_age == groupe_age)].groupby(['date'])['nombre_cas'].sum())

#         else:
#             st.bar_chart(df[(df.province == select_province) & (df.sex == genre) & (
#                 df.groupe_age == groupe_age)].groupby(['date'])['nombre_cas'].sum())


################### 2ème Partie ###################

# ------------------------------- CLEANING DATA ----------------------
import missingno as msno

# --------------------- Test Statistique -------------------
import statsmodels.api

# ------------------ Librairie de Manipulation de données ------------------------------

from pandas import json_normalize
import pandas as pd
import requests
from datetime import date, time, datetime as dt
import numpy as np

# ------------------ Librairie de visualisation -----------------------------------

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns 
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.family'] = 'sans-serif'
import folium

# ------------------ Librairie ML ----------------------------------------------------

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split





#☺st.title(" II. 2ème Partie")
url1='https://public.opendatasoft.com//api/records/1.0/search/?dataset=covid-19-pandemic-belgium-deaths-agesexdate&q=&rows=800&facet=date&facet=region&facet=agegroup&facet=sex&facet=deaths'
r1 = requests.get(url1)
print(r1.status_code)
print(r1.content)

d = r1.json()
d.keys()

df1 = json_normalize(d['records'])
#st.write(df1)

######## Supprimer les colonnnes #############

df1.drop(["datasetid", 'recordid','record_timestamp', 'fields.geo_point_2d','fields.nis', 'geometry.type', 'geometry.coordinates'], axis = 1, inplace = True)

########

df1.rename(columns = {'fields.region':"region",'fields.date':"date",'fields.agegroup':"groupe_age",'fields.sex':"sex",'fields.deaths':"nombre_cas"},inplace=True)
df1.dropna(how='any',axis=0,inplace=True)
#df.set_index('date',inplace=True)
st.write(df1.head())
############# séparation de la colonne date: Anne, mois, jour #############

from datetime import date, time, datetime as dt
df1['date'] = pd.to_datetime(df1['date'], utc = True)
df1['Annee'] = df1['date'].dt.year

df1['Mois'] = df1['date'].dt.month
df1['Jour'] = df1['date'].dt.day
df1.drop('date',axis=1)
st.write(df1.head())

############## Présentation des variables qui nous intéressent #############


if st.checkbox("Vous retrouverez ici :"):
    st.markdown(
        """
        ###  groupe_age

        ###  sex

        ###  region

        ###  nombre_cas

        ###  Annee

        ###  Mois

        ###  Jour

        ###
        
        """
        )
# # Visualize missingness

# Décors
liste_choix = ['NON', 'VRAI']
select_clean = st.sidebar.selectbox(
    "Sélectionnez graphique des données manquantes", options=liste_choix)

if (select_clean == 'NON'):
    select_lean_right = st.write(df1.isna())

elif(select_clean == 'VRAI'):
    select_lean_right = st.write(df1.isna().sum())


if st.button("PRINT MISSING GRAPH"):
    fig, ax = plt.subplots()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    msno.bar(df1)
    plt.show()
    st.pyplot()
    
    
############## Profile Visualisation  ################



# def app(title=None):

#     st.title(title)
#     profile = pp.ProfileReport(df1)
#     st_profile_report(profile)
    

# app(title='Profile Visualization')

# ############## COMPARAISON PAR CATEGORIE ##############
#Comparatif du nombre de cas en fonction  par tranche d'age, régions, par sexe et par annéee
st.table(df1.groupby(['sex']).agg({'nombre_cas' : 'sum'}))
st.table(df1.groupby(['region']).agg({'nombre_cas' : 'sum'}))
#display(df.groupby(['province']).agg({'nombre_cas' : 'sum'}))
st.table(df1.groupby('groupe_age').agg({'nombre_cas' : 'sum'}))
st.table(df1.groupby(['Annee']).agg({'nombre_cas' : 'sum'}))
#display(df.groupby(['Mois']).agg({'nombre_cas' : 'sum'}))

plt.style.use('ggplot')
st.set_option('deprecation.showPyplotGlobalUse', False)


Masc=df1[df1.sex=='M']
Fem=df1[df1.sex=='F']
df1_sex=df1.groupby(['sex']).agg({'nombre_cas' : 'sum'})
df1_reg=df1.groupby(['region']).agg({'nombre_cas' : 'sum'})


df1_sex.plot(kind='barh')
#sns.barplot(x="sex",y="nombre_cas", data= df_sex)
df1_reg.plot(kind='barh')

st.pyplot()
df1_age=df1.groupby('groupe_age').agg({'nombre_cas' : 'sum'})
df1_age.plot(kind='barh')

st.pyplot()
df1_year=df1.groupby(['Annee']).agg({'nombre_cas' : 'sum'})
df1_year.plot(kind='barh', ylabel='Annee')

st.pyplot()
############### GRAPHE DE COMPARAISON #############
#---------création d'un dataframe contenant les infos des 5 meilleurs ---------
df1_top5 = df1.copy()

#Tri du dataframe sur les 5 compteurs les plus fréquentés
df1_top5 = df1_top5.groupby(['sex'], as_index = False).agg({'nombre_cas':'mean'}).sort_values('nombre_cas', ascending = False).head()

#Récupération de la liste des 5 meilleurs compteurs
top5 = df1_top5['sex']

#Récupération des données complètes pour le top5 compteur
df1_top5 = df1.loc[df1['sex'].isin(top5)]

#Transformation du type datetime.time en integer pour utiliser dans notre graphique
df1_top5['region'] = df1_top5['region']#.apply(lambda x: x.strftime('%Y'))
#fig, ax = plt.subplots(figsize =(12,5)
sns.boxplot(x='sex', y='nombre_cas', hue = 'groupe_age', data=df1_top5) #ci=None,
st.title("Classement des groupes d'âges les plus atteints en fonction du sexe");

st.pyplot()



#---------création d'un dataframe contenant les infos des 5 meilleurs compteurs---------
df1_top3 = df1.copy()

#Tri du dataframe sur les 5 compteurs les plus fréquentés
df1_top3 = df1_top5.groupby(['groupe_age'], as_index = False).agg({'nombre_cas':'mean'}).sort_values('nombre_cas', ascending = False).head()

#Récupération de la liste des 5 meilleurs compteurs
top5 = df1_top5['groupe_age']

#Récupération des données complètes pour le top5 compteur
df1_top3 = df1.loc[df1['groupe_age'].isin(top5)]

#Transformation du type datetime.time en integer pour utiliser dans notre graphique
#df_top3['Annee'] = df_top3['Annee']#.apply(lambda x : x[-1:]).astype(int)
sns.boxplot(x='Annee', y='nombre_cas',hue='region' , data=df1_top5)#ci=None,
st.title("Classement du nombre de cas en année en fonction des régions");
st.pyplot()

from datetime import date, time, datetime as dt
#---------création d'un dataframe contenant les infos des 5 meilleurs compteurs---------
df1_top3 = df1.copy()

#Tri du dataframe sur les 5 compteurs les plus fréquentés
df1_top3 = df1_top5.groupby(['region'], as_index = False).agg({'nombre_cas':'mean'}).sort_values('nombre_cas', ascending = False).head()

#Récupération de la liste des 5 meilleurs compteurs
top5 = df1_top5['region']

#Récupération des données complètes pour le top5 compteur
df1_top3 = df1.loc[df1['region'].isin(top5)]

#Transformation du type datetime.time en integer pour utiliser dans notre graphique
#df_top3[''] = df_top3['Annee']#.apply(lambda x : x[-1:]).astype(int)

sns.boxplot(x='groupe_age', y='nombre_cas',hue = 'sex', data=df1_top5) #ci=None,
st.title("Classement des categories d'age les plus atteint en fonction du sexe"); #hue = 'sex',kind='line',
st.pyplot()

# -------------------- TEST STATISQUE -------------------
st.title(" TEST STATISTIQUE : ANOVA ")

#-----------CORRELATION ANOVA ENTRE DATES ET TRAFIC--------------
st.title("---- CORRELATION ANOVA ENTRE NOMBRE DE CAS  ET ANNEE ----")
result = statsmodels.formula.api.ols('nombre_cas ~ Annee', data=df1).fit()
table = statsmodels.api.stats.anova_lm(result)
st.table(table)
#-----------CORRELATION ANOVA ENTRE JOURS ET TRAFIC--------------
st.title("---- CORRELATION ANOVA ENTRE NOMBRE DE CAS  ET GROUPE D'AGE ----")
result = statsmodels.formula.api.ols('nombre_cas ~ groupe_age ', data=df1).fit()
table = statsmodels.api.stats.anova_lm(result)
st.table(table)
#-----------CORRELATION ANOVA ENTRE JOURS ET TRAFIC--------------
st.title("---- CORRELATION ANOVA ENTRE NOMBRE DE CAS  ET SEX ----")
result = statsmodels.formula.api.ols('nombre_cas ~ sex', data=df1).fit()
table = statsmodels.api.stats.anova_lm(result)
st.table(table)
#-----------CORRELATION ANOVA ENTRE HEURES ET TRAFIC--------------
st.title("---- CORRELATION ANOVA ENTRE NOMBRES DE CAS  ET REGION ----")
result = statsmodels.formula.api.ols('nombre_cas ~ region', data=df1).fit()
table = statsmodels.api.stats.anova_lm(result)
st.table(table)

# ----------- ENCODAGE -------------------------
st.subheader("Encodage: GROUPE D'AGE, SEX, REGION, ANNEE")
df1=pd.get_dummies(df1, columns=["region"], drop_first=True,prefix='region')
df1=pd.get_dummies(df1, columns=["sex"], drop_first=True,prefix='sex')
df1=pd.get_dummies(df1, columns=["groupe_age"], drop_first=True,prefix='groupe_age')
df1=pd.get_dummies(df1, columns=["Annee"], drop_first=True,prefix='Annee')
    
X = df1[['region_Flanders', 'region_Wallonia', 'sex_M', 'sex_NA', 'groupe_age_25-44', 'groupe_age_45-64', 'groupe_age_65-74', 'groupe_age_75-84', 'groupe_age_85+', 'groupe_age_NA', 'Annee_2021', 'Annee_2022']]
y = df1["nombre_cas"]

X.shape, y.shape
y=y.values.reshape(-1,1)
# -------------- Visualisation ------------------------------------
if st.checkbox("Visualisation de la relation entre la cible et les variables caractérisques"):
    verification_variable=st.selectbox(
        'Selectionne une variable caractéristique',
        X.columns
        #df1.drop(columns="nombre_cas").columns
        )
    # Plot
    fig, ax =plt.subplots(figsize=(5,3))
    ax.scatter(x=df1[verification_variable], y = df1["nombre_cas"])
    plt.xlabel(verification_variable)
    plt.ylabel("nombre_cas")
    st.pyplot(fig)

#---------------- Split the dataset ---------------------
left_column, right_column=st.columns(2)
test_size=left_column.number_input(
    "Validation data size(rate:0.0-1.0):",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
    )
#random_seed=right_column.number_input(
#    "Random seed(Nonnegative integer):",
#    value=0,
#    step=1,
#    min_value=0
#   )
#random_state=random_seed
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
st.write(X_train.shape)
st.write(X_test.shape)
st.write(y_train.shape)
st.write(y_test.shape)

# --------------- Instancions, ajustons, et faisons des predictions
model=LinearRegression()
model.fit(X_train,y_train)
y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)


# ------------- Precision du model -----------------
r2=model.score(X_test, y_test)
st.write(r2*100)

# ------------- Plot th result ------------------
left_column, right_column=st.columns(2)
show_train= left_column.radio(
    'plot the reult of the tarining dataset:',
    ('Oui','Non')
    )
show_test=right_column.radio(
    'plot the reult of the test dataset:',
    ('Oui','Non')
    )

#Get the maimum value of all objective variable data,
#including predicted values
y_max_train=max(max(y_train), max(y_pred_train))
y_max_test=max(max(y_test), max(y_pred_test))
y_max=int(max([y_max_train, y_max_test]))

#Allows the axis range to be changed dynamically
left_column, right_column=st.columns(2)
x_min=left_column.number_input('x_min:', value=0, step=1)
x_max=right_column.number_input('x_max:', value=y_max, step=1)
left_column, right_column=st.columns(2)
y_min=left_column.number_input('y_min:', value=0, step=1)
y_max=right_column.number_input('y_max:', value=y_max, step=1)

#Show the result
fig=plt.figure(figsize=(3,3))
if show_train=='Oui':
    plt.scatter(y_train, y_pred_train, color="m",label="trainig data")
if show_train=='Oui':
    plt.scatter(y_test, y_pred_test, color="b",label="validation data")
plt.xlabel("nombre_cas", fontsize=8)
plt.ylabel("Prediction nombre_cas", fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
st.pyplot(fig)

########################### Plot X_test
st.header("Plot the X_train")
X_train.value_counts().plot(kind='bar', figsize=(20,10))
st.pyplot()
st.header("Plot the X_test")
X_test.value_counts().plot(kind='bar', figsize=(20,10))
st.pyplot()


# comparaison of score
# get importance
st.header("get the importance")
slope = model.coef_[0]
intercept = model.intercept_[0]
print("slope=", slope) 
print("intercept=", intercept)  # 0.99410415 (close to 1)
fittedline = slope*X + intercept # generate prediction line (y=ax+b)
fittedline
plt.plot(X, fittedline)
st.pyplot()
#plt.scatter(X, fittedline, label='X')

from matplotlib import pyplot
import numpy as np
# summarize feature importance
st.header("plot the feature importance")
fig=plt.figure(figsize=(3,3))
for i,v in enumerate(slope):
    print((i,v))
# plot feature importance
sns.barplot([x for x in range(len(slope))], slope)
st.pyplot(fig)







