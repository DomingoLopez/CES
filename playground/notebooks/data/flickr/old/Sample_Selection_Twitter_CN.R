## SELECTION######
library(tidyverse)
library(sp)
library(sf)
library(plotly)
library(gridExtra)

# Unir base de datos con cluster 
# 
# # Load the data of cluster 
# df<- read.delim("Flickr_cls_300.csv", sep = ",", header = TRUE) %>% 
#   rename(filename = image_id, cluster = prediction) 
# 
# # Load defintion of cluster
# df2 <- read.delim("Flickr_cls_300_Centroids_human_regroup.csv", sep = ";", header = TRUE) %>% 
#   rename(cluster = cluster_id) %>% 
#   select(cluster, stoten_label, level3_label, valid)
# 
# # Si quiero excluir los que dijimos que eran candidatos a ser eliminados
# # df2 <- df2 %>% filter(valid == "SI")
# 
# # Cargo la pasarela para unir la informacion de los otros niveles 
# df3 <- read.delim("CESLabelsTree_asv.csv", sep = ";", header = TRUE) %>% 
#   rename(level3_label = X3rd.level, level2_label = X2nd.level, level1_label = X1st.level) %>% 
#   select(level3_label, level2_label, level1_label)
# 
# df3$level3_label
# 
# ## Como Towns and villages está como villages en esta version le cambié el nombre en el excel a Towns and villages
# df3 <- df3 %>% 
#   mutate(level3_label = ifelse(level3_label == "Villages", "Towns and villages", level3_label))
# 
# # Primero uno las columnas de la pasarela con la informacion de los niveles
# 
# df2 <- merge(df2,df3, by = "level3_label")
# 
# # Seleccionar solo valores unicos de la columna cluster
# df4 <- df2 %>% distinct()
# 
# 
# # Unir por la columna cluster
# df <- left_join(df,df4, by = "cluster")
# 
# # ver todas las categorias posibles en Stoten y Level3
# df2 %>% 
#   select(stoten_label) %>% 
#   distinct() %>% 
#   arrange(stoten_label)
# 
# # Level 3
# df2 %>% 
#   select(level3_label) %>% 
#   distinct() %>% 
#   arrange(level3_label)

# -----------------------------------------------------------


drive <- "H:/.shortcut-targets-by-id/1M44eQfgiKNtbHWnOemX3695BodORG2B6/LW_SEM/WP4/EarthCul/Data/"


#### FLICKR ####

data_images_1 <- read.csv(paste0(drive,"Twitter/images/aiguestortes.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "aiguestortes")
data_images_2 <- read.csv(paste0(drive,"Twitter/images/ordesa.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "ordesa")
data_images_3 <- read.csv(paste0(drive,"Twitter/images/teide.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "teide")
data_images_4 <- read.csv(paste0(drive,"Twitter/images/peneda.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "peneda")
data_images_5 <- read.csv(paste0(drive,"Twitter/images/picos.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "picos")
data_images_6 <- read.csv(paste0(drive,"Twitter/images/guadarrama.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "guadarrama")
data_images_7 <- read.csv(paste0(drive,"Twitter/images/sierra_nieves.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "sierra_nieves")
data_images_8 <- read.csv(paste0(drive,"Twitter/images/snev.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "snev")

data <- rbind(data_images_1, data_images_2, data_images_3, data_images_4, data_images_5, data_images_6, data_images_7, data_images_8)

rm(data_images_1, data_images_2, data_images_3, data_images_4, data_images_5, data_images_6, data_images_7, data_images_8)

names(data)

# Unir con informacion de cluster

data <- left_join(data,df, by = "filename")
metricas <- data

"
Los cluster calculadas por Rohaifa fueron asignados a las etiquetas del nivel 3. 
Not relevant: es cuando el cluster es claro, pero no se corresponde con ningun CES por ejemplo selfies
Miscelaneo: basicamente es ruido, no se puede asignar claramente a un cluster 


Pasos 
1) Seleccionar una imagen por la combinacion de todas las variables que nos interesan (parque(8), nivel3(21+1)- basado en el cluster, nivel de protección (7), municipio (344))
2) Crear una submuestra excluyendo municipio para quedarnos con lo minimo indispensable
3) Crear un data.frame de la diferencia entre 1 y 2
4) Seleccionar un 20%, 50% de la diferencia (3)
5) Unir 1 y 4

"


# 1) Seleccionar una imagen por la combinacion de todas las variables que nos interesan (parque, nivel3, nivel de protección, municipio)

selected_images_initial <- metricas %>%
  # filter(level3_label != "Not relevant") %>%
  group_by(area_of_interest, level3_label, location_id, municipality) %>%
  slice_sample(n = 1) %>%  
  ungroup() %>% 
  mutate(seleccion = "completa")

nrow(selected_images_initial)

# 2) Crear una submuestra excluyendo municipio para quedarnos con lo minimo indispensable

selected_images_a <- selected_images_initial %>%
  group_by(area_of_interest, level3_label, location_id) %>%
  slice_sample(n = 1) %>%  
  ungroup() %>% 
  mutate(seleccion = "escenciales")

nrow(selected_images_a)

# 3) Crear un data.frame de la diferencia entre 1 y 2


selected_images_b <- selected_images_initial %>%
  anti_join(selected_images_a, by = c("photo_id"))

nrow(selected_images_b)

# 4) Seleccionar un 20%, 50% de la diferencia (3)

selected_images_c <- selected_images_b %>%
  group_by(area_of_interest) %>%
  sample_frac(0.50) %>% 
  mutate(seleccion = "50%")

selected_images_d <- selected_images_c %>%
  # group_by(area_of_interest) %>%
  sample_frac(0.20) %>%
  mutate(seleccion = "20%")

# 5) Unir 1 y 4
selected_images <- rbind(selected_images_a, selected_images_c, selected_images_d)

# Me quedo con valores unicos pero siempre seleccionando los que en seleccion digan 20%
# Esto para no tener filas repetidas que sean 20% y 50% a la vez 

# Suponiendo que tu tabla se llama "mi_tabla" y la columna se llama "columna_seleccion"
selected_images_x <- selected_images %>%
  arrange(seleccion) %>%  # Ordenar por columna_seleccion para poner "20%" primero
  distinct(across(-seleccion), .keep_all = TRUE) %>%  # Mantener filas únicas en el resto de columnas
  filter(seleccion == "20%" | !duplicated(across(-seleccion)))

# # Para tener en un unico archivo todos los datos con la columna seleccion
intermedio <- selected_images_initial  %>%
  anti_join(selected_images_x, by = c("photo_id"))

selected_images2 <- rbind(selected_images_x, intermedio)
nrow(selected_images2)


## guardar los tres archivos importantes 

# 1) Tabla con las fotos escenciales 

escenciales <- selected_images2 %>%
  filter(seleccion %in% c("escenciales"))

write.csv(escenciales, "Seleccion final/Twitter/escenciales/Sample_AOIProtClu_955_Twitter.csv")

# 2) Tabla con las fotos escenciales más un 20%

Mun_20 <- selected_images2 %>%
  filter(seleccion %in% c("20%"))

write.csv(Mun_20, "Seleccion final/Twitter/20 por ciento/Sample_20%_575_Twitter.csv")

# 3) Tabla con las fotos escenciales más un 50%

Mun_50 <- selected_images2 %>%
  filter(seleccion %in% c("50%"))

write.csv(Mun_50, "Seleccion final/Twitter/50 por ciento/Sample_50%_2307_Twitter.csv")

# 4) Tabla con todas las fotos 
write.csv(selected_images_initial, "Seleccion final/Twitter/completa/Sample_AOIProtCluMun_6760_Twitter.csv")


### Histogramas de box ids 

# Crear columna de año
data_images_1 <- data_images_1 %>% 
  mutate(year = substr(tweet_created_at, 1, 4))


# Reemplazar cadenas vacías o ceros por NA
data_images_1 <- data_images_1 %>%
  mutate(tweet_latitude = ifelse(tweet_latitude == "" | tweet_latitude == 0, NA, tweet_latitude))

# Reemplazar cadenas "NA" por valores NA reales
data_images_1 <- data_images_1 %>%
  mutate(tweet_latitude = na_if(tweet_latitude, "NA"))


# Volver a realizar el conteo
resultado <- data_images_1 %>%
  group_by(year) %>%
  summarise(n = sum(is.na(tweet_latitude), na.rm = TRUE))

print(resultado)


## cantidad de nulos en la columna tweet_latitude por año
data_images_1 %>% 
  group_by(year) %>% 
  summarise(n = sum(is.na(tweet_latitude), na.rm = TRUE)) %>% 
  ggplot(aes(x = year, y = n)) +
  geom_bar(stat = "identity") +
  labs(title = "Cantidad de nulos en la columna tweet_latitude por año",
       x = "Año",
       y = "Cantidad de nulos")


