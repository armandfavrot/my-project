
library(kableExtra)   
library(dplyr) 
library(IRdisplay)
library(ggplot2)
library (paletteer)


# Couleurs #

Couleurs = list(Dark2 = paletteer_d("RColorBrewer::Dark2"), 
                Pastel1 = paletteer_d("RColorBrewer::Pastel1"), 
                Pastel2 = paletteer_d("RColorBrewer::Pastel2"), 
                category20b_d3 = paletteer_d("ggsci::category20b_d3"))


Dark2 = Couleurs$Dark2


# Theme ggplot #

axis_text_size = 18 
axis_title_size = 22
title_size = 28

theme_replace(
    
    axis.text = element_text(size = axis_text_size), 
    axis.title.x = element_text(size = axis_title_size, angle = 0, margin = ggplot2::margin (t = 10)), 
    axis.title.y = element_text(size = axis_title_size, angle = 90, margin = ggplot2::margin (r = 10)), 
    
    plot.title = element_text(size = title_size, hjust = 0.5, face = "bold", margin = ggplot2::margin (b = 20, t = 20)), 
    
    legend.text = element_text(size = axis_title_size), 
    legend.title = element_text(size = axis_title_size, face = "bold"), 
    
    strip.text = element_text(size = axis_title_size, face = "bold", color = "white"), 
    strip.text.x = element_text(margin = ggplot2::margin(t = 5, b = 5, r = 0, l = 0)),
    strip.background = element_rect(fill = "#000080"), 
    
    panel.background = element_rect(fill = "#f3faff"), panel.grid.major = element_line(colour = "white"), 
    panel.spacing = unit(2, "lines"),
    legend.background = element_rect (fill = "transparent", color = NA),
    plot.background = element_rect (fill = "transparent", color = NA)
        
)

options(ggplot2.discrete.fill = Dark2)
options(ggplot2.discrete.colour = Dark2)

update_geom_defaults("point", list(size = 3)) 
update_geom_defaults("line", list(linewidth = 1.3)) 


# Fonctions pour jupyter #

size <- function(x,y){
  return(options(repr.plot.width=x, repr.plot.height=y))
}

embed <- function(df){
    return(df %>% kable() %>% kable_styling() %>% as.character() %>% display_html())
}

# Fonction pour visualiser les valeurs manquantes

visualize_NA <- function (df){
    
  data.frame(
      id = rep(seq_len(nrow(df)), times = ncol(df)),
      key = rep(names(df), each = nrow(df)),
      isna = as.vector(is.na(df))
  ) %>%
      ggplot() +
          geom_raster(aes(x = id, y = key, fill = isna)) +
          scale_fill_manual(name = "", values = c('steelblue', 'tomato3')) +
          theme (legend.position = "none") +
            xlab ("Row number") +
            ylab ("")
}
