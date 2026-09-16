import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/train.csv')


# Analisis Exploratorio inicial
print("Información General")
print(df.info())
print("Número de pasajeros y columnas")
print(df.shape)
print("Valores faltantes")
print(df.isnull().sum())
print("Duplicados")
print(df.duplicated().sum())
print("Estadísticas descriptivas")
print(df.describe())



# Requerimiento del proyecto nuevas columnas y visualizaciones 
# Creación de FamilySize ya mero ya mero
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Creación de categorías de edad este si funciona 
bins = [0, 12, 18, 60, 100]
labels = ['Niño', 'Joven', 'Adulto', 'Adulto mayor']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)



# Grafica de supervivencia por sexo
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Supervivencia por Sexo')
plt.savefig('outputs/resultados/graf_sexo.png') # Guardamos la imagen con este codigo en la carpeta de resultados 
plt.show()

# Grafica sobre supervivencia por clase
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Supervivencia por Clase')
plt.savefig('outputs/resultados/graf_clase.png')
plt.show()

# Mapa de calor de correlaciones
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.savefig('outputs/resultados/heatmap.png')
plt.show()