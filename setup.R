library(aws.s3)
library(aws.signature)
library(purrr)
library(readr)
library(DT)
library(knitr)
library(scales)
library(stringr)
library(tibble)
library(Amelia)
library(lubridate)
library(feather)
library(xtable)
library(ggplot2)
library(dplyr)
library(glue)
library(tidyr)
library(agricolae)
library(reticulate)
library(kableExtra)
library(ggpubr)
library(broom)

# *** MUST SET THE FOLLOWING IN NOTEBOOK: ***
# SHARELATEX_PROJ <- ''
# BUCKET <- ''
# AWS_PROFILE <- ''

options(scipen=10000)
use_credentials(profile = AWS_PROFILE)
Sys.setenv('AWS_PROFILE' = AWS_PROFILE)

latex_table <- function(df, name, cap = '', digits = 3, kable = T, position = '!t') {
  out <- xtable(df, caption = cap, label = str_c('tab:', name), digits = digits)
  print(out, include.rownames = F, file = file.path(SHARELATEX_PROJ, str_c('tab_', name, '.tex')), table.placement = position)
  if (kable) {
    print(kable(df, digits = digits))
  }
}

latex_figure <- function(p, name, cap = '', display = T, latex_width = '3.5in', width = NA, height = NA, position = '!t') {
  ggsave(str_c(name, '.pdf'), p, path = SHARELATEX_PROJ, width = width, height = height)
  table_str <- str_c("\\begin{figure}[", position, "]
  \\centering
  \\includegraphics[width=", latex_width, "]{", name, ".pdf}
  \\caption{", cap, "}
  \\label{fig:", name, "}
  \\end{figure}
  ")
  file_con <- file(file.path(SHARELATEX_PROJ, str_c('fig_', name, '.tex')))
  write(table_str, file_con)
  close(file_con)
  if (display) {
    print(p)
  }
}

fit_curve <- function(df, predict_size, adj=0) {
  N <- nrow(df)
  X <- df %>% pull(num_pos)
  Y <- df %>% pull(test_roc_auc)
  W <- lapply(X, function(i) { i / N}) %>% unlist
  
  i <- predict_size
  x <- X[0:i]
  y <- Y[0:i]
  w <- W[0:i]
  gradientF<-deriv3(~(1-a)-(b*(x^c)), c("a","b","c"), function(a,b,c,x) NULL)
  startParams <- list(a=0, b=1, c=-0.5)
  
  m <- nls(y~gradientF(a,b,c,x), start = startParams, weights=w,
           control = list(maxiter=1000, warnOnly = TRUE),
           algorithm = "port", upper = list(a=10, b = 10, c = -0.1), lower = list(a = 0, b = 0, c=-10),
           data = data.frame(y=y, x=x))
  
  testX<-X[((i):N)]
  testY<-Y[((i):N)]
  testW<-W[((i):N)]
  
  predictY <- predict(m, list(x=testX))
  predictY <- predictY + adj
  res <- predictY - testY
  mae <- mean(abs(res))
  rmse <- sqrt(sum(res^2) / length(res))
  
  out <- df %>% rename(actual = test_roc_auc)
  out$prediction <- c(rep(NA, i-1), predictY) 
  out$predict_size <- X[i]
  out$mae <- mae
  out$rmse <- rmse
  out$residual <- c(rep(NA, i-1), res)
  out
}

fit_curves <- function(df, range, adj=0) {
  map(range, function(i) { fit_curve(df, i, adj)}) %>% bind_rows
}
