#--------------------------------------------------------------------------#
# Project: plankton_classif_benchmark
# Script purpose: Generate plots for paper
# Date: 02/08/2023
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

library(tidyverse)


# Path to save figures
dir.create("figures", showWarnings = FALSE)


## Read and format results ----
#--------------------------------------------------------------------------#

df <- read_csv("classification_performance/prediction_metrics.csv", col_types = cols())

# is "mobilenet_acp50_rf_deep" the same as "mobilenet_pca50_rf"
# is "mobilenet_noacp_rf_deep" the same as "mobilenet_rf"
# No, the difference is RF hyperparameters. Use mobilenet_pca50_rf and mobilenet_rf

# Prepare nice names for models
models <- tibble(
  model = c("mobilenet600nw", "mobilenet600", "mobilenet_rf", "effnetv2s", "effnetv2xl", "mobilenet50", "mobilenet_pca50_rf", "native_rf_noweights", "native_rf", "random"),
  name = c("Mob + MLP600 (NW)", "Mob + MLP600", "Mob + RF", "Eff S + MLP600", "Eff XL + MLP600", "Mob + MLP50", "Mob + PCA + RF", "Nat + RF (NW)", "Nat + RF", "Random")
) %>% 
  mutate(name = fct_inorder(name))

# Rename models
df <- df %>% 
  left_join(models, by = join_by(model)) %>% 
  select(-model) %>% 
  rename(model = name) %>% 
  drop_na(model)

# Generate nice names for datasets
datasets <- tibble(
  dataset = c("flowcam", "ifcb", "isiis", "uvp6", "zoocam", "zooscan"),
  name = c("FlowCam", "IFCB", "ISIIS", "UVP6", "ZooCam", "ZooScan")
)

# Rename datasets
df <- df %>% 
  left_join(datasets, by = join_by(dataset)) %>% 
  select(-dataset) %>% 
  rename(dataset = name)

# Reformat results in a long dataframe
# For this we need to separate detailed and grouped results
df_d <- df %>% 
  select(model, dataset, accuracy:plankton_recall) %>% 
  pivot_longer(accuracy:plankton_recall, names_to = "metric", values_to = "score")

df_g <- df %>% 
  select(model, dataset, accuracy_g:plankton_recall_g) %>% 
  pivot_longer(accuracy_g:plankton_recall_g, names_to = "metric_g", values_to = "score_g")

df <- df_d %>% 
  mutate(metric_g = str_c(metric, "_g")) %>% 
  left_join(df_g)


# Create labels for faceting
labels <- c(
  accuracy = "Accuracy", 
  balanced_accuracy = "Balanced accuracy",
  plankton_precision = "Plankton averaged precision",
  plankton_recall = "Plankton averaged recall"
)

# Generate nice colours
my_cols <- c(
  "Random" = "#bebebeff",
  "Mob + MLP600" = "#3a62bfff",
  "Mob + MLP600 (NW)" = "#839fe0ff",
  "Nat + RF" = "#ffbc63ff",
  "Nat + RF (NW)" = "#ffd194ff",
  "Eff S + MLP600" = "#77d1daff",
  "Eff XL + MLP600" = "#29a8b4ff",
  "Mob + MLP50" = "#63c6ffff",
  "Mob + RF" = "#7f7bc5ff",
  "Mob + PCA + RF" = "#b0aaf8ff"
)

dodge <- position_dodge(width=0.9) # define dodge for linerange


## Random VS. MobileNet (W & NW) VS. RF (W & NW) ----
#--------------------------------------------------------------------------#
df %>% 
  filter(model %in% c("Random", "Mob + MLP600", "Mob + MLP600 (NW)", "Nat + RF", "Nat + RF (NW)")) %>% 
  ggplot(aes(x = dataset, y = score, fill = model)) +
  geom_col(position = "dodge") +
  geom_linerange(aes(ymax = score_g, ymin = score, color = model), position = dodge, show.legend = F, linewidth = 1) +
  scale_fill_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Mob + MLP600 (NW)` = expression(Mob + MLP[600] ~(NW))
    )) +
  scale_colour_manual(values = my_cols) +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  facet_wrap(~metric, labeller=labeller(metric = labels)) +
  theme_minimal() +
  labs(x = "Dataset", y = "Score", fill = "Model") +
  theme(
    legend.position = "bottom", 
    #axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(colour="white", fill="white"),
    text = element_text(size = 10)
  )
ggsave(file = "figures/figure_2.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")



## Bigger CNN do not improve classification performance and a smaller CNN performs just as well ----
#--------------------------------------------------------------------------#
df %>% 
  filter(model %in% c("Mob + MLP600", "Eff S + MLP600", "Eff XL + MLP600", "Mob + MLP50")) %>% 
  ggplot(aes(x = dataset, y = score, fill = model)) +
  geom_col(position = "dodge") +
  geom_linerange(aes(ymax = score_g, ymin = score, color = model), position = dodge, show.legend = F, linewidth = 1) +
  scale_fill_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Eff S + MLP600` = expression(Eff~S + MLP[600]),
      `Eff XL + MLP600` = expression(Eff~XL + MLP[600]),
      `Mob + MLP50` = expression(Mob + MLP[50])
    )) +
  scale_colour_manual(values = my_cols) +
  facet_wrap(~metric, labeller=labeller(metric = labels)) +
  labs(x = "Dataset", y = "Metric value", fill = "Model") +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  theme_minimal() +
  theme(
    strip.background = element_rect(colour="white", fill="white"),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    text = element_text(size = 10)
  ) 
ggsave(file = "figures/figure_3.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")


## Is it the features or the classifier?  ----
#--------------------------------------------------------------------------#
df %>% 
  filter(model %in% c("Mob + MLP600", "Mob + RF", "Mob + PCA + RF", "Nat + RF")) %>% 
  ggplot(aes(x = dataset, y = score, fill = model)) +
  geom_col(position = "dodge") +
  geom_linerange(aes(ymax = score_g, ymin = score, color = model), position = dodge, show.legend = F, linewidth = 1) +
  scale_fill_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600])
    )) +
  scale_colour_manual(values = my_cols) +
  facet_wrap(~metric, labeller=labeller(metric = labels)) +
  labs(x = "Dataset", y = "Metric value", fill = "Model") +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  theme_minimal() +
  theme(
    strip.background = element_rect(colour="white", fill="white"),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    text = element_text(size = 10)
  ) 
ggsave(file = "figures/figure_4.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")
