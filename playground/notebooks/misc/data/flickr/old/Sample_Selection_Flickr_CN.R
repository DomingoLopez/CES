## SELECTION######
library(tidyverse)
library(sp)
library(sf)
library(plotly)
library(gridExtra)

# Unir base de datos con cluster 

# Load the data of cluster 
df<- read.delim("Flickr_cls_300.csv", sep = ",", header = TRUE) %>% 
  rename(filename = image_id, cluster = prediction) 

# Load defintion of cluster
df2 <- read.delim("Flickr_cls_300_Centroids_human_regroup.csv", sep = ";", header = TRUE) %>% 
  rename(cluster = cluster_id) %>% 
  select(cluster, stoten_label, level3_label, valid)

# Si quiero excluir los que dijimos que eran candidatos a ser eliminados
# df2 <- df2 %>% filter(valid == "SI")

# Cargo la pasarela para unir la informacion de los otros niveles 
df3 <- read.delim("CESLabelsTree_asv.csv", sep = ";", header = TRUE) %>% 
  rename(level3_label = X3rd.level, level2_label = X2nd.level, level1_label = X1st.level) %>% 
  select(level3_label, level2_label, level1_label)

df3$level3_label

## Como Towns and villages está como villages en esta version le cambié el nombre en el excel a Towns and villages
df3 <- df3 %>% 
  mutate(level3_label = ifelse(level3_label == "Villages", "Towns and villages", level3_label))


# Primero uno las columnas de la pasarela con la informacion de los niveles

df2 <- merge(df2,df3, by = "level3_label")

# Seleccionar solo valores unicos de la columna cluster
df4 <- df2 %>% distinct()


# Unir por la columna cluster
df <- left_join(df,df4, by = "cluster")

# Función personalizada para capitalizar solo la primera palabra
capitalize_first_word <- function(text) {
  gsub("^(\\w)(\\w*)", "\\U\\1\\L\\2", text, perl = TRUE)
}

# Aplicar la transformación
df <- df %>%
  mutate(level3_label = tolower(level3_label)) %>%
  mutate(level3_label = sapply(level3_label, capitalize_first_word))


# -----------------------------------------------------------
# Corregir la etiqueta accomodation
df <- df %>% 
  mutate(level3_label = ifelse(level3_label == "Accomodation", "Accommodation", level3_label))

# ver todas las categorias posibles en Stoten y Level3
df %>% 
  select(stoten_label) %>% 
  distinct() %>% 
  arrange(stoten_label)

# Level 3
df %>% 
  select(level3_label) %>% 
  distinct() %>% 
  arrange(level3_label)

# -----------------------------------------------------------


drive <- "H:/.shortcut-targets-by-id/1M44eQfgiKNtbHWnOemX3695BodORG2B6/LW_SEM/WP4/EarthCul/Data/"


#### FLICKR ####

data_images_1 <- read.csv(paste0(drive,"Flickr/aiguestortes.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "aiguestortes")
data_images_2 <- read.csv(paste0(drive,"Flickr/ordesa.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "ordesa")
data_images_3 <- read.csv(paste0(drive,"Flickr/teide.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "teide")
data_images_4 <- read.csv(paste0(drive,"Flickr/peneda.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "peneda")
data_images_5 <- read.csv(paste0(drive,"Flickr/picos.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "picos")
data_images_6 <- read.csv(paste0(drive,"Flickr/guadarrama.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "guadarrama")
data_images_7 <- read.csv(paste0(drive,"Flickr/sierra_nieves.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "sierra_nieves")
data_images_8 <- read.csv(paste0(drive,"Flickr/snev.csv"), na.strings = c("[]", "None"), sep = ",") %>% 
  mutate(area_of_interest = "snev")

data <- rbind(data_images_1, data_images_2, data_images_3, data_images_4, data_images_5, data_images_6, data_images_7, data_images_8)

rm(data_images_1, data_images_2, data_images_3, data_images_4, data_images_5, data_images_6, data_images_7, data_images_8)


# Unir con informacion de cluster

data <- left_join(data,df, by = "filename")
metricas <- data

# Export metricas
write.csv(metricas, "cluster_results_completo.csv")



"
Los cluster calculadas por Rohaifa fueron asignados a las etiquetas del nivel 3. 
Not relevant: es cuando el cluster es claro, pero no se corresponde con ningun CES por ejemplo selfies
Miscelaneo: basicamente es ruido, no se puede asignar claramente a un cluster 


Pasos 
1) Seleccionar una imagen por la combinacion de todas las variables que nos interesan (parque(8), nivel3(22 CES+2 (Miscelaneo y not relevant))- basado en el cluster, nivel de protección (7), municipio (344))
2) Crear una submuestra excluyendo municipio para quedarnos con lo minimo indispensable
3) Crear un data.frame de la diferencia entre 1 y 2
4) Seleccionar un 20%, 50% de la diferencia (3)
5) Unir 1 y 4

"


# Establecer una semilla para garantizar reproducibilidad
set.seed(123)

# 1) Seleccionar una imagen por la combinacion de todas las variables que nos interesan (parque, nivel3, nivel de protección, municipio)

selected_images_initial <- metricas %>%
  # filter(level3_label != "Not relevant") %>%
  group_by(area_of_interest, level3_label, location_id, municipality) %>%
  slice_sample(n = 1) %>%  
  ungroup() %>% 
  mutate(seleccion = "completa")

nrow(selected_images_initial)
print(
selected_images_initial %>% 
  count(level3_label), n = 50
)
# 2) Crear una submuestra excluyendo municipio para quedarnos con lo minimo indispensable

selected_images_a <- selected_images_initial %>%
  group_by(area_of_interest, level3_label, location_id) %>%
  slice_sample(n = 1) %>%  
  ungroup() %>% 
  mutate(seleccion = "escenciales")

nrow(selected_images_a)

# 3) Crear un data.frame de la diferencia entre 1 y 2


selected_images_b <- selected_images_initial %>%
  anti_join(selected_images_a, by = c("image_url"))

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
  anti_join(selected_images_x, by = c("image_url"))

selected_images_todo <- rbind(selected_images_x, intermedio)
nrow(selected_images2)


# -----------------------------------------------------------
#              Base de datos limpias para Manolo 
# -----------------------------------------------------------
selected_images2 <- selected_images_todo %>% 
  select(c("filename", "conversation_id", "area_of_interest", "image_path", "image_url", "text", "description" ,  "latitude", "longitude", "cluster", "level3_label", "seleccion"))

# renombrar algunas columnas 

selected_images2 <- selected_images2 %>% 
  rename( original_id = conversation_id, filename = filename, area_of_interest_id = area_of_interest, image_path = image_path, image_url = image_url, main_text = text, secondary_text = description, latitude = latitude, longitude = longitude, cluster = cluster, level3_label = level3_label)


# Agregar algunas columnas 
selected_images2 <- selected_images2 %>% 
  arrange(level3_label, cluster) %>%
  mutate(
    source = "Flickr", 
    labeller_id = paste0("labeller", ntile(row_number(), 6)),
    cluster_class_id = case_when(
      level3_label == "Accommodation" ~ 1,
      level3_label == "Air activities" ~ 2,
      level3_label == "Algae" ~ 3,
      level3_label == "Animals" ~ 4,
      level3_label == "Breakwater" ~ 5,
      level3_label == "Bridge" ~ 6,
      level3_label == "Commerce facilities" ~ 7,
      level3_label == "Cities" ~ 8,
      level3_label == "Dam" ~ 9,
      level3_label == "Dock" ~ 10,
      level3_label == "Fungus" ~ 11,
      level3_label == "Gardens" ~ 12,
      level3_label == "Heritage and culture" ~ 13,
      level3_label == "Knowledge" ~ 14,
      level3_label == "Landforms" ~ 15,
      level3_label == "Lichens" ~ 16,
      level3_label == "Lighthouse" ~ 17,
      level3_label == "Not relevant" ~ 18,
      level3_label == "Other abiotic features" ~ 19,
      level3_label == "Plants" ~ 20,
      level3_label == "Roads" ~ 21,
      level3_label == "Shelter" ~ 22,
      level3_label == "Spiritual, symbolic and related connotations" ~ 23,
      level3_label == "Terrestrial activities" ~ 24,
      level3_label == "Towns and villages" ~ 25,
      level3_label == "Tracks and trails" ~ 26,
      level3_label == "Vegetation and habitats" ~ 27,
      level3_label == "Vehicle" ~ 28,
      level3_label == "Water activities" ~ 29,
      level3_label == "Wind farm" ~ 30,
      level3_label == "Winter activities" ~ 31,
      level3_label == "Miscellaneous" ~ 32,
      TRUE ~ NA_real_  # En caso de que haya valores que no coincidan
    )
  )


# Reemplazar en labeller_id los valores de labeller por los nombres de los labellers a.roscandeira, ricuni, jcperezgiron,kampax7,siham,isabel.uc95

selected_images2 <- selected_images2 %>% 
  mutate(labeller_id = case_when(
    labeller_id == "labeller1" ~ "a.roscandeira",
    labeller_id == "labeller2" ~ "ricuni",
    labeller_id == "labeller3" ~ "jcperezgiron",
    labeller_id == "labeller4" ~ "kampax7",
    labeller_id == "labeller5" ~ "siham",
    labeller_id == "labeller6" ~ "isabel.uc95"
  ))

# Quitar la columna level3_label

selected_images3 <- selected_images2 %>% 
  select(-level3_label)

# Ordenar las columnas de la tabla en este orden: original_id	filename,	source,	area_of_interest_id,	image_path,	image_url,	main_text,	secondary_text,	latitude,	longitude,	labeller_id,	cluster_class_id,	cluster	seleccion

selected_images3 <- selected_images3 %>% 
  select(original_id, filename, source, area_of_interest_id, image_path, image_url, main_text, secondary_text, latitude, longitude, labeller_id, cluster_class_id, cluster, seleccion)




## guardar los tres archivos importantes 

# 1) Tabla con las fotos escenciales 

escenciales <- selected_images3 %>%
  filter(seleccion %in% c("escenciales"))

escenciales <- escenciales %>% 
  select(-seleccion)

write.csv(escenciales, "Seleccion final/Flickr/escenciales/Sample_AOIProtClu_1042_Flickr.csv")

# 2) Tabla con las fotos escenciales más un 20%

Mun_20 <- selected_images3 %>%
  filter(seleccion %in% c("20%"))

Mun_20 <- Mun_20 %>% 
  select(-seleccion)

write.csv(Mun_20, "Seleccion final/Flickr/20 por ciento/Sample_20%_602_Flickr.csv")

# 3) Tabla con las fotos escenciales más un 50%

Mun_50 <- selected_images3 %>%
  filter(seleccion %in% c("50%"))

Mun_50 <- Mun_50 %>% 
  select(-seleccion)

write.csv(Mun_50, "Seleccion final/Flickr/50 por ciento/Sample_50%_2413_Flickr.csv")

# 4) Tabla con todas las fotos 

# conteo por factor de la columna seleccion 
selected_images3 <- selected_images3 %>% 
  select(-seleccion) 

write.csv(selected_images3, "Seleccion final/Flickr/completa/Sample_AOIProtCluMun_7072_Flickr.csv")


# -----------------------------------------------------------
# Convertir el csv as un objeto spatial (sf)
selected_images_sf <- st_as_sf(selected_images, coords = c("longitude", "latitude"), crs = "EPSG:4326")

# test
data_sf_test <- st_transform(selected_images_sf, crs = 32630)

#
park_name <- "snev"

# Plotear los puntos en un mapa
p<-
  data_sf_test %>% filter(area_of_interest == "snev") %>%
  ggplot() +
  geom_sf(aes(color = level3_label,
              text = paste("Label:", level3_label, "<br>Photo ID:", photo_id), na.translate = FALSE)) +
  geom_sf(data = get(paste0("aoi_",park_name)), fill = NA, color = "blue", linewidth=1.1) +
  geom_sf(data = get(paste0("pn_",park_name)), fill = NA, color = "darkgreen", linewidth=1.2) +
  geom_sf(data = get(paste0("mun_",park_name)), fill = NA, color = "yellow", linewidth=0.5) +
  theme_minimal() +
  labs(title = "Flickr photos selected of snev",
       x = "Longitude",
       y = "Latitude")

ggplotly(p, tooltip = "text")

# -----------------------------------------------------------
# TODOS LOS PARQUES 
# -----------------------------------------------------------

# Vector con las categorías de Level3
categories <- c(unique(selected_images_sf$stoten_label))

# Lista de parques a procesar
parks <- c("snev", "aiguestortes", "picos", "teide", "guadarrama", "sierra_nieves", "peneda", "ordesa")

# Loop para cada parque
for (park_name in parks) {
  
  # Seleccionar el CRS correspondiente al parque actual
  crs_park <- switch(park_name,
                     "snev" = 32630,
                     "aiguestortes" = 25831,
                     "picos" = 32629,
                     "teide" = 32628,
                     "guadarrama" = 32630,
                     "sierra_nieves" = 32630,
                     "peneda" = 32629,
                     "ordesa" = 32631
  )
  
  # Reproyectar el data frame al CRS seleccionado
  data_sf_park <- st_transform(selected_images_sf, crs = crs_park)
  
  # Crear un archivo PDF para el parque actual
  pdf(paste0("PDF_Seleccion/Stoten/", park_name, "_maps_by_stoten_category50.pdf"), width = 11, height = 8.5)
  

    
    # Filtrar los datos para el parque y la categoría actuales
    data_sf_filtered <- data_sf_park %>% 
      filter(area_of_interest == park_name)
    
    # Crear el gráfico para la categoría
    p <- ggplot(data_sf_filtered) +
      geom_sf(aes(color = stoten_label, text = paste("Label:", stoten_label, "<br>Photo ID:", photo_id))) +
      geom_sf(data = get(paste0("aoi_", park_name)), fill = NA, color = "blue", linewidth = 1.1) +
      geom_sf(data = get(paste0("pn_", park_name)), fill = NA, color = "darkgreen", linewidth = 1.2) +
      geom_sf(data = get(paste0("mun_", park_name)), fill = NA, color = "yellow", linewidth = 0.5) +
      theme_minimal() +
      labs(title = paste("Flickr photos of", park_name, "-"),
           x = "Longitude",
           y = "Latitude")
    
    # Dibujar el gráfico en el archivo PDF
    print(p)
  
  # Cerrar el archivo PDF después de graficar todas las categorías
  dev.off()
}




