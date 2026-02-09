import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reservatorio = "example/reservatorio/4ree_decomp_agosto/out/simulation/operation_hydros.parquet"
sem_reservatorio = "example/sem_reservatorio/4ree_decomp_agosto/out/simulation/operation_hydros.parquet"

df_reservatorio = pd.read_parquet(reservatorio)
df_sem_reservatorio = pd.read_parquet(sem_reservatorio)

df_reservatorio = df_reservatorio[df_reservatorio["variable_name"] == "HYDRO_GENERATION"]
df_reservatorio = df_reservatorio[df_reservatorio["entity_id"] == 4]
df_reservatorio = df_reservatorio[df_reservatorio["stage"] == 1]
df_reservatorio["modelo"] = "Reservatorio"

df_sem_reservatorio = df_sem_reservatorio[df_sem_reservatorio["variable_name"] == "HYDRO_GENERATION"]
df_sem_reservatorio = df_sem_reservatorio[df_sem_reservatorio["entity_id"] == 4]
df_sem_reservatorio = df_sem_reservatorio[df_sem_reservatorio["stage"] == 1]
df_sem_reservatorio["modelo"] = "Sem Reservatório"

df = pd.concat([df_reservatorio, df_sem_reservatorio])
print(df)

sns.barplot(df, x = "modelo", y = "value")

plt.show()