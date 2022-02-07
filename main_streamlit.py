
import datetime
import requests
import pandas as pd
from pandas import json_normalize
import streamlit as st
import matplotlib.pyplot as plt
import base64
import altair as alt
from PIL import Image

#image aeeière plan
st.markdown(
    """
    <style>
    .reportview-container { 
        background: url("https://i.pinimg.com/originals/94/bd/cb/94bdcbbeb8b7569395a062593499ac6b.png")
    }
   
    </style>
    """,
    unsafe_allow_html=True
)

st.title("COVID-19 Pandemic - Belgium -") #### Titre de l'application
image = Image.open('C:\\Users\\ymmel\\OneDrive\\Bureau\\exam_dasboard\\image.png')
st.image(image) 

liste_choix_reg=['Toutes les régions', 'Brussels','Flanders', 'Wallonia']

select_graph=['line_chart','bar_chart']

#url pour récupérer nos infrmation sous forme d'in fichier json
url ='https://public.opendatasoft.com/api/records/1.0/search/?dataset=covid-19-pandemic-belgium-cases-agesexprovince&q=&rows=5000&sort=date&facet=date&facet=province&facet=region&facet=agegroup&facet=sex&facet=Cases&facet=Geo+Point'
r=requests.get(url)
print(r.status_code)
print(r.content)

d=r.json()
d.keys()

df = json_normalize(d['records'])#lire le ficher jscon sous forme d'un dataframe
#supprimer les colonnes qui sont pas importantes
df.drop(['datasetid', 'fields.geo_point_2d','geometry.type'], axis = 1,inplace=True) 

#renommer les colonnes 
df.rename(columns = {'recordid': "id_enregistrement",'record_timestamp':"horodatage d'enreg",'fields.province':"province",'fields.region':"region",'fields.date':"date",'fields.agegroup':"groupe_age",'fields.sex':"sex",'fields.cases':"nombre_cas",'geometry.coordinates':"coordonnées_géométrie"},inplace=True)
#supprimer les lignes qui contient des valeurs manquantes  
df.dropna(how='any',axis=0,inplace=True)

#mettre l'indx est la date
df.set_index('date',inplace=True)


select_graph = st.sidebar.selectbox("Sélectionnez le type de graphique", options=select_graph) 
select_region = st.sidebar.selectbox("Sélectionnez une région", options=liste_choix_reg) #### Liste avec les modalités définies au départ

#afficher la liste des provinces par rapport au region sélectonnée
if (select_region == 'Brussels'):
    select_province =st.sidebar.selectbox("Sélectionnez une Provice",options=['Brussels'])

elif (select_region=='Flanders'): 
    select_province=st.sidebar.selectbox("Sélectionnez une Provice",options=['Toutes les provinces de Flanders','Antwerpen','Limburg','Oost-Vlaanderen','Vlaams-Brabant','West-Vlaanderen'])

elif(select_region=='Wallonia'):
    select_province=st.sidebar.selectbox("Sélectionnez une Provice",options=['Toutes les provinces de Wallonia','Brabant Wallon', 'Brussels','Liège','Hainaut','Luxembourg','Namur'])

else:
    select_province=st.sidebar.selectbox("Sélectionnez une Provice (sélectionnez d'abord la région)",options=['Toutes les provinces'])

#date=st.sidebar.date_input("Choisissez une date", min_value=datetime.date(2021, 12, 2), max_value=datetime.date(2021, 12, 31))
genre = st.sidebar.radio("Choisissez le sex",options=['M', 'F'])
#groupe_age = st.sidebar.slider("Choisissez la tranche d'âge", min_value=0, max_value=100, step=10)
groupe_age = st.sidebar.selectbox("Choisissez la tranche d'âge",options=['0-9','10-19','20-29','30-39','40-49', '50-59','60-69','70-79','80-89','90+'])


def ligne_chart_toute_reg():
    if(select_graph=='line_chart'):
        return(alt.Chart(df.groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")))
    else:
        return(alt.Chart(df.groupby(['date', 'region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")))

def ligne_chart_toute_prov(region):
    if(select_graph=='line_chart'):
        return(alt.Chart(df[df.region==region].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")))
    
    else:
        return(alt.Chart(df[df.region==region].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")))
def grapg(type_graphe):
    return(alt.type_graphe(df[df.region==region].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")))


st.write(df)
st.subheader('le tableau filtré')
if(select_region=='Toutes les régions'):
    st.write(df)
#afficher le tableau filtré
else:
    if(select_province=="Toutes les provinces de Wallonia"):
        st.write(df[(df.region=='Wallonia')])
    elif(select_province=='Toutes les provinces de Flanders'):
        st.write(df[(df.region=='Flanders')])
    else:
        st.write(df[(df.region==select_region)&(df.province==select_province)])

#################################partie region

if(select_region=='Toutes les régions'):
    
    st.subheader(f'Le nombre de cas à {select_region}')
else:
    st.subheader(f'Le nombre de cas à la région {select_region}')#selon jour et la région sélectionnés

if (select_region=='Toutes les régions'):
     st.altair_chart(ligne_chart_toute_reg(), use_container_width=True)
    
else: 
     st.altair_chart(ligne_chart_toute_prov(select_region), use_container_width=True)

##############################partie province
    
if((select_province=='Toutes les provinces de Flanders')|(select_province=='Toutes les provinces de Wallonia')|(select_province=='Toutes les provinces')):
    st.subheader(f'Le nombre de cas à {select_province}')
else:
    st.subheader(f'Le nombre de cas à la province {select_province}')#selon la province ,la date et la région sélectionnés avant

if(select_province=='Toutes les provinces'):
    st.altair_chart(ligne_chart_toute_reg(), use_container_width=True)

elif(select_province=='Brussels'):
    st.altair_chart(ligne_chart_toute_prov(select_region), use_container_width=True)
elif(select_province=='Toutes les provinces de Flanders'):
    st.altair_chart(ligne_chart_toute_prov(select_region), use_container_width=True)

elif(select_province=='Toutes les provinces de Wallonia'):
    st.altair_chart(ligne_chart_toute_prov(select_region), use_container_width=True)

else:
    if(select_graph=='line_chart'):
        st.line_chart(df[df.province==select_province].groupby(['date'])['nombre_cas'].sum())
    else:
        st.bar_chart(df[df.province==select_province].groupby(['date'])['nombre_cas'].sum())

##################################la partie gerne
if(genre=='M'):
    st.subheader(f"Le nombre de cas par rapport aux Masculins")#le nombre de cas selon le sexe sélectionné et les critères sélectionnés avant
else:
    st.subheader(f"Le nombre de cas par rapport aux Féminins") 

if(select_province=='Toutes les provinces'):
    if (select_graph=='line_chart'):
        st.altair_chart(alt.Chart(df[(df.sex==genre)].groupby(['date','region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")),use_container_width=True)
    else:
        st.altair_chart(alt.Chart(df[(df.sex==genre)].groupby(['date','region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")),use_container_width=True)

elif((select_province=='Toutes les provinces de Flanders')|(select_province=='Toutes les provinces de Wallonia')|(select_province=='Brussels')): 
    if(select_graph=='line_chart'):
        st.altair_chart(alt.Chart(df[(df.sex==genre)&(df.region==select_region)].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")),use_container_width=True)
    else:
        st.altair_chart(alt.Chart(df[(df.sex==genre)&(df.region==select_region)].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")),use_container_width=True)

else:
    if(select_graph=='line_chart'):
        st.line_chart(df[(df.province==select_province)&(df.sex==genre)].groupby(['date'])['nombre_cas'].sum())
    else:
        st.bar_chart(df[(df.province==select_province)&(df.sex==genre)].groupby(['date'])['nombre_cas'].sum())


################################partie groupes age 
st.subheader(f"Le nombre de cas par raport aux groupes d'âge ({groupe_age})")#le nombre de cas selon la tranche d'âge selectionné et les critères sélectionnés avant

check_box = st.checkbox("Masculins et Féminins")

if(check_box):
    if(select_province=='Toutes les provinces'):
        if(select_graph=='line_chart'):
            st.altair_chart(alt.Chart(df[df.groupe_age==groupe_age].groupby(['date','region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")),use_container_width=True)
        else:
            st.altair_chart(alt.Chart(df[df.groupe_age==groupe_age].groupby(['date','region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")),use_container_width=True)
        '''liste=[]
        for i in (df[df.groupe_age==groupe_age]['coordonnées_géométrie']):
            liste.append(i)
        #st.write(liste)
        df_map = pd.DataFrame(liste,columns=['longitude','latitude'])
        st.map(data=df_map,use_container_width=True)'''
    elif((select_province=='Toutes les provinces de Flanders')|(select_province=='Toutes les provinces de Wallonia')|(select_province=='Brussels')): 
        if(select_graph=='line_chart'):
            st.altair_chart(alt.Chart(df[(df.region==select_region)&(df.groupe_age==groupe_age)].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")),use_container_width=True)
        else:
            st.altair_chart(alt.Chart(df[(df.region==select_region)&(df.groupe_age==groupe_age)].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")),use_container_width=True)
    

    else:
        if(select_graph=='line_chart'):
            st.line_chart(df[(df.province==select_province)&(df.groupe_age==groupe_age)].groupby(['date'])['nombre_cas'].sum())

        else:
            st.bar_chart(df[(df.province==select_province)&(df.groupe_age==groupe_age)].groupby(['date'])['nombre_cas'].sum())

else:
    if(select_province=='Toutes les provinces'):
        if(select_graph=='line_chart'):
            st.altair_chart(alt.Chart(df[(df.sex==genre)&(df.groupe_age==groupe_age)].groupby(['date','region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")),use_container_width=True)

        else:
            st.altair_chart(alt.Chart(df[(df.sex==genre)&(df.groupe_age==groupe_age)].groupby(['date','region']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("region")),use_container_width=True)

    elif((select_province=='Toutes les provinces de Flanders')|(select_province=='Toutes les provinces de Wallonia')|(select_province=='Brussels')): 
        if(select_graph=='line_chart'):
            st.altair_chart(alt.Chart(df[(df.sex==genre)&(df.region==select_region)&(df.groupe_age==groupe_age)].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_line().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")),use_container_width=True)
    
        else:
            st.altair_chart(alt.Chart(df[(df.sex==genre)&(df.region==select_region)&(df.groupe_age==groupe_age)].groupby(['date','province']).agg({'nombre_cas': 'sum'}).reset_index()).mark_bar().encode(x=('date'),y=('nombre_cas'),color=alt.Color("province")),use_container_width=True)
    
    else:
        if(select_graph=='line_chart'):
            st.line_chart(df[(df.province==select_province)&(df.sex==genre)&(df.groupe_age==groupe_age)].groupby(['date'])['nombre_cas'].sum())

        else:
            st.bar_chart(df[(df.province==select_province)&(df.sex==genre)&(df.groupe_age==groupe_age)].groupby(['date'])['nombre_cas'].sum())






 

