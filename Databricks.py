#Leitura de Dados e Visualização com PySpark

# Carregar os dados de um arquivo CSV
df = spark.read.csv('/mnt/data/sample_data.csv', header=True, inferSchema=True)

# Mostrar as primeiras linhas do DataFrame
df.show()

# Exibir informações sobre o DataFrame (como tipos de dados)
df.printSchema()

#****************************************************************************************#******************************************************************************************************#
#Transformação de Dados (Filtragem e Seleção de Colunas)

# Filtrar os dados para incluir apenas linhas onde a coluna 'age' é maior que 30
df_filtered = df.filter(df.age > 30)

# Selecionar algumas colunas específicas
df_selected = df_filtered.select("name", "age", "city")

# Mostrar o resultado
df_selected.show()

#Escrita de Dados em Parquet
# Escrever o DataFrame em formato Parquet
df_selected.write.parquet('/mnt/data/output_data.parquet')

# Verificar se o arquivo Parquet foi salvo corretamente
df_parquet = spark.read.parquet('/mnt/data/output_data.parquet')
df_parquet.show()

#****************************************************************************************#******************************************************************************************************#
#Machine Learning com PySpark MLlib

# Carregar dados de exemplo
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# Criar um DataFrame de exemplo (substitua por seu próprio conjunto de dados)
data = spark.createDataFrame([
    (1, 1.0, 1.1),
    (2, 2.0, 1.9),
    (3, 3.0, 3.0),
    (4, 4.0, 4.1)
], ["id", "feature", "label"])

# Usar VectorAssembler para combinar as colunas de recursos
assembler = VectorAssembler(inputCols=["feature"], outputCol="features")
data_assembled = assembler.transform(data)

# Dividir os dados em treino e teste
train_data, test_data = data_assembled.randomSplit([0.8, 0.2], seed=1234)

# Inicializar o modelo de regressão linear
lr = LinearRegression(featuresCol="features", labelCol="label")

# Treinar o modelo
lr_model = lr.fit(train_data)

# Fazer previsões no conjunto de teste
predictions = lr_model.transform(test_data)

# Mostrar os resultados
predictions.select("id", "features", "label", "prediction").show()

#****************************************************************************************#******************************************************************************************************#
#Leitura de Dados do Azure Blob Storage

# Configurar o armazenamento do Azure Blob
spark.conf.set("spark.hadoop.fs.azure.account.key.<your_account_name>.blob.core.windows.net", "<your_account_key>")

# Ler dados de um arquivo CSV do Blob Storage
df_blob = spark.read.csv("wasbs://<your_container>@<your_account_name>.blob.core.windows.net/<your_file>.csv", header=True, inferSchema=True)

# Mostrar as primeiras linhas
df_blob.show()

#****************************************************************************************#******************************************************************************************************#
#Usando Spark SQL para Consultas

# Registrar o DataFrame como uma tabela temporária
df.createOrReplaceTempView("people")

# Executar uma consulta SQL
result = spark.sql("SELECT name, age FROM people WHERE age > 30")

# Mostrar os resultados
result.show()

#****************************************************************************************#******************************************************************************************************#
#Modelo de Classificação com MLflow no Databricks

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar um conjunto de dados de exemplo
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar o experimento MLflow
with mlflow.start_run():
    # Treinar o modelo
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    
    # Logar o modelo e a acurácia no MLflow
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy}")

#****************************************************************************************#******************************************************************************************************#
#Databricks Autoloader para Processamento de Arquivos em Tempo Real

# Caminho para o diretório de arquivos
file_path = "/mnt/data/raw_data/"

# Usar o Autoloader para ler arquivos em tempo real
df_streaming = (spark.readStream
                .format("cloudFiles")
                .option("cloudFiles.format", "csv")
                .load(file_path))

# Realizar transformações e exibir o conteúdo
df_transformed = df_streaming.select("column1", "column2")

# Exibir a saída do stream
query = df_transformed.writeStream.outputMode("append").format("console").start()
query.awaitTermination()

#****************************************************************************************#******************************************************************************************************#
Delta Lake para Leitura e Escrita de Dados

# Criar um DataFrame de exemplo
df = spark.createDataFrame([
    ("Alice", 30),
    ("Bob", 25),
    ("Charlie", 35)
], ["name", "age"])

# Escrever o DataFrame como uma tabela Delta
df.write.format("delta").save("/mnt/data/delta_table")

# Ler os dados da tabela Delta
df_delta = spark.read.format("delta").load("/mnt/data/delta_table")

# Mostrar os dados lidos
df_delta.show()

#****************************************************************************************#******************************************************************************************************#
# Usando MLlib para K-Means Clustering

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# Criar um DataFrame de exemplo
df = spark.createDataFrame([
    (1, 2.0, 3.0),
    (2, 3.0, 4.0),
    (3, 4.0, 5.0),
    (4, 5.0, 6.0)
], ["id", "x", "y"])

# Usar VectorAssembler para combinar as colunas 'x' e 'y' em um vetor de recursos
assembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
df_assembled = assembler.transform(df)

# Inicializar o modelo KMeans
kmeans = KMeans().setK(2).setSeed(1)

# Treinar o modelo KMeans
model = kmeans.fit(df_assembled)

# Fazer previsões
predictions = model.transform(df_assembled)

# Mostrar as previsões
predictions.show()
