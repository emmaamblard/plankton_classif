#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Generate performance increase figure and a nice classification report
# Date: 30/08/2022
# Author: Thelma Panaiotis
#--------------------------------------------------------------------------#

library(tidyverse)
library(googlesheets4)
library(glue)

times_100 <- function(x){x * 100}

# Dataset to work with
dataset <- "zooscan"


## Read data ----
#--------------------------------------------------------------------------#
# Spreadsheet for classification reports
ss <- "https://docs.google.com/spreadsheets/d/1Y1_sOIpWDA7S9ABdzZxysHKTG5Wjk_VQsXrb3NAZZuI/edit#gid=0"

# Sheet with taxonomy
taxo <- read_sheet("https://docs.google.com/spreadsheets/d/1C57EPnnOljtFKWrkdvQj2-UNc0bg0PMHb6km0YhaGSw/edit#gid=0", sheet = dataset) %>% 
  select(taxon, grouped = level2, plankton)

# Read classification reports for given dataset
files <- list.files("perf", full.names = TRUE, pattern = "report*") %>% str_subset(dataset)


## Performance increase ----
#--------------------------------------------------------------------------#
## Detailed metrics
df <- read_csv(files %>% str_subset("detailed")) %>% 
  select(taxon, everything()) %>% 
  # reformat to longer
  pivot_longer(`precision-native_rf`:`f1-effnetv2s`) %>% 
  # get metric and model from name
  separate(name, into = c("metric", "model"), sep = "-") %>% 
  mutate(level = "detailed")

# Separate RF values which are the reference
df_rf <- df %>% 
  filter(model == "native_rf") %>% 
  rename(ref = value) %>% 
  select(-model)

# Compute the difference between metrics and reference
df <- df %>% 
  filter(model != "native_rf") %>% 
  left_join(df_rf) %>% 
  mutate(value = value - ref) %>% 
  select(-ref)

## Grouped metrics
df_g <- read_csv(files %>% str_subset("grouped")) %>% 
  select(taxon, everything()) %>% 
  # reformat to longer
  pivot_longer(`precision-native_rf`:`f1-effnetv2s`) %>% 
  # get metric and model from name
  separate(name, into = c("metric", "model"), sep = "-") %>% 
  mutate(level = "grouped")

# Separate RF values which are the reference
df_g_rf <- df_g %>% 
  filter(model == "native_rf") %>% 
  rename(ref = value) %>% 
  select(-model)

# Compute the difference between metrics and reference
df_g <- df_g %>% 
  filter(model != "native_rf") %>% 
  left_join(df_g_rf) %>% 
  mutate(value = value - ref) %>% 
  select(-ref)
  
# Generate nice colours
my_cols <- c(
  "Mob + MLP600" = "#3a62bfff",
  "Eff S + MLP600" = "#77d1daff",
  "Mob + PCA + RF" = "#b0aaf8ff"
)


# Prepare nice names for models
models <- tibble(
  model = c("mobilenet600", "effnetv2s", "mobilenet_pca50_rf"),
  name = c("Mob + MLP600", "Eff S + MLP600", "Mob + PCA + RF")
) %>% 
  mutate(name = fct_inorder(name))


# Join detailed and grouped metrics together
df_all <- df %>% 
  bind_rows(df_g) %>% 
  mutate(value = value * 100) %>% 
  # metric as factor for plotting
  mutate(
    metric = str_to_sentence(metric),
    level = str_to_sentence(level)
    ) %>% 
  mutate(metric = factor(metric, levels = c("Precision", "Recall", "F1"))) %>% 
  left_join(models) %>% 
  select(-model) %>% 
  rename(model = name)

override.linetype <- c(1, 3, 2) 
# Plot it
df_all %>% 
  ggplot(aes(x = value, color = model, linetype = model)) +
  geom_vline(xintercept = 0, colour = "gray80", linewidth = 1) +
  geom_density(adjust=0.75) +
  scale_color_manual(
    guide = none,
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Eff S + MLP600` = expression(Eff~S + MLP[600])
    )) +
  scale_y_continuous(breaks = c(0, 0.01, 0.02)) +
  theme_classic() +
  theme(strip.background = element_blank(), legend.text.align = 0, legend.position = "bottom") +
  facet_grid(rows = vars(level), cols = vars(metric)) +
  labs(x = "Point increase in metric, from RF on native features", y = "Estimated probability density", color = "Model") +
  guides(colour = guide_legend(override.aes = list(linetype = override.linetype))) +
  scale_linetype(guide = FALSE)
ggsave(file = "figures/figure_5.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")



## Write classification reports to Gsheet ----
#--------------------------------------------------------------------------#
cr <- read_csv(files %>% str_subset("detailed")) %>% 
  select(taxon, grouped, contains("f1")) %>% 
  rename(nat_rf = `f1-native_rf`, mob600 = `f1-mobilenet_pca50_rf`, effs_600 = `f1-mobilenet600`, mob_pca_rf = `f1-effnetv2s`) %>% 
  mutate(across(nat_rf:mob_pca_rf, ~times_100(.)))

# Generate report for plankton and for non plankton
cr_plank <- cr %>% left_join(taxo) %>% filter(plankton) %>% arrange(grouped) %>% select(-plankton)
cr_noplank <- cr %>% left_join(taxo) %>% filter(!plankton) %>% arrange(grouped) %>% select(-plankton)


range_write(ss, data = cr_plank, range = glue("{dataset}_report!A4"), col_names = FALSE, reformat = FALSE)
range_write(ss, data = cr_noplank, range = glue("{dataset}_report!A{4+nrow(cr_plank)+2}"), col_names = FALSE, reformat = FALSE)

