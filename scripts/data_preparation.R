
library (dplyr)
library (tidyr)

pdat = read.csv ("../data/ALFAM2-data-3.0/data-output/04/ALFAM2_plot.csv")

dim (pdat)

pdat = pdat %>% 

  filter (meas.tech2 == "micro met") %>%
  filter (e.rel.final <= 1.2) %>%

  # Remplacer les NAs de rain.rate.24 par des 0
  mutate(rain.rate.24 = replace_na(rain.rate.24, 0)) %>%

  # Enlever les plots avec des valeurs manquantes
  filter(

    if_all(

      c(e.rel.72,
        soil.type2,
        air.temp.24, wind.2m.24, rain.rate.24,
        man.source, man.trt1, man.tan, man.ph, man.dm, 
        app.method, app.rate, tan.app, incorp),

        ~ !is.na(.x))
  ) %>%

  mutate (
    soil.type2.origin = soil.type2,
    man.source.origin = man.source,
    man.trt1.origin = man.trt1,
    app.method.origin = app.method,
    incorp.origin = incorp
  ) %>%

  # remplace les '.' par des '_' dans le noms des colonnes (nécessaire pour python)
  rename_with (~ gsub (".", "_", .x, fixed = TRUE)) %>%

  # transformation numériques des variables catégorielles pour le modèle
  mutate (
    across(
      c(soil_type2, man_source, man_trt1, app_method, incorp),
      ~ as.integer(factor(.x))
    )
  ) 

dim (pdat)

write.csv (pdat, file = "data.csv")

