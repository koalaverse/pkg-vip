# Setup ------------------------------------------------------------------------

# Simulate training data
set.seed(101)  # for reproducibility
trn <- as.data.frame(mlbench::mlbench.friedman1(500))  # ?mlbench.friedman1
names(trn) <- gsub("^x\\.", replacement = "X", x = names(trn))

# Inspect data
tibble::as_tibble(trn)

# Load required packages
library(dplyr)          # for easy data wrangling
library(ggplot2)        # for nicer graphics
library(rpart)          # for fitting CART-like decision trees
library(randomForest)   # for fitting random forests
library(xgboost)        # for fitting GBMs
library(vip)            # for variable importance plots


# Model-specific VI ------------------------------------------------------------

# Fit a single regression tree
tree <- rpart(y ~ ., data = trn)

# Fit a random forest
set.seed(101)
rfo <- randomForest(y ~ ., data = trn, importance = TRUE)

# Fit a GBM
set.seed(102)
bst <- xgboost(
  data = data.matrix(subset(trn, select = -y)),
  label = trn$y, 
  objective = "reg:linear",
  nrounds = 100, 
  max_depth = 5, 
  eta = 0.3,
  verbose = 0  # suppress printing
)

# VI plot for single regression tree
vi_tree <- tree$variable.importance %>%
  data.frame("Importance" = .) %>% 
  tibble::rownames_to_column("Feature")

# VI plot for RF
vi_rfo <- rfo$importance %>%
  data.frame("Importance" = .) %>% 
  tibble::rownames_to_column("Feature")

# VI plot for GMB
vi_bst <- bst %>%
  xgb.importance(model = .) %>%
  as.data.frame() %>%
  select(Feature, Importance = Gain)

# Plot results
library(ggplot2)

p1 <- vip(tree) + ggtitle("Single tree")
p2 <- vip(rfo) + ggtitle("Random forest")
p3 <- vip(bst) + ggtitle("Gradient boosting")

# 
pdf("figures/vi-plots.pdf", width = 9, height = 3)
grid.arrange(p1, p2, p3, nrow = 1)
dev.off()

library(vip)

# Extract (tibble of) VI scores
vi(tree)  # CART-like decision tree
vi(rfo)   # RF
vi(bst)   # GBM

library(ggplot2)  # for theme_light() function
pdf("figures/dot-plot.pdf", width = 7, height = 4.326)
vip(bst, num_features = 5, bar = FALSE, color, horizontal = FALSE, 
    color = "red", shape = 17, size = 4) +
  theme_light()
dev.off()

# Fit a LM
linmod <- lm(y ~ .^2, data = trn)
backward <- step(linmod, direction = "backward", trace = 0)

# Extract VI scores
vi(backward)
#> # A tibble: 21 x 3
#>    Variable Importance Sign 
#>    <chr>         <dbl> <chr>
#>  1 X4           14.2  POS  
#>  2 X2            7.31 POS  
#>  3 X1            5.63 POS  
#>  4 X5            5.21 POS  
#>  5 X3:X5        2.46 POS  
#>  6 X1:X10       2.41 NEG  
#>  7 X2:X6        2.41 NEG  
#>  8 X1:X5        2.37 NEG  
#>  9 X10           2.21 POS  
#> 10 X3:X4        2.01 NEG  
#> # … with 11 more rows

# Plot VI scores
pdf("figures/vip-step.pdf", width = 7, height = 4.326)
vip(backward, num_features = length(coef(backward)))
dev.off()

# Load required packages
library(earth)

# Fit a MARS model
mars <- earth(y ~ ., data = trn, degree = 2, pmethod = "exhaustive")

# Extract VI scores
vi(mars)
#> # A tibble: 10 x 2
#>    Variable Importance
#>    <chr>         <dbl>
#>  1 X4           100  
#>  2 X1            83.2
#>  3 X2            83.2
#>  4 X5            59.3
#>  5 X3            43.5
#>  6 X6             0  
#>  7 X7             0  
#>  8 X8             0  
#>  9 X9             0  
#> 10 X10            0

# Plot VI scores
pdf("figures/vip-earth.pdf", width = 7, height = 4.326)
vip(mars)
dev.off()

# Load required packages
library(nnet)

# Fit a neural network
set.seed(0803)
nn <- nnet(y ~ ., data = trn, size = 7, decay = 0.1, linout = TRUE, maxit = 500)

# VIPs
p1 <- vip(nn, type = "garson")
p2 <- vip(nn, type = "olden")

# Figure X
pdf("figures/vip-model-nn.pdf", width = 7, height = 3.5)
grid.arrange(p1, p2, nrow = 1)
dev.off()


# PDP method -------------------------------------------------------------------

# Load required packages
library(pdp)

# Fit a PPR model (nterms was chosen using the caret package with 5 repeats of 
# 5-fold cross-validation)
pp <- ppr(y ~ ., data = trn, nterms = 11)  

# PDPs for all 10 features
features <- paste0("X", 1:10)
pdps <- lapply(features, FUN = function(feature) {
  pd <- partial(pp, pred.var = feature)
  autoplot(pd) + 
    ylim(range(trn$y)) + 
    theme_light()
})

pdf("figures/pdp-ppr.pdf", width = 15, height = 5)
grid.arrange(grobs = pdps, ncol = 5)
dev.off()

# Plot VI scores
p1 <- vip(pp, method = "pdp") + ggtitle("PPR")
p2 <- vip(nn, method = "pdp") + ggtitle("NN")

# Figure X
pdf("figures/vip-ppr-nn.pdf", width = 7, height = 3.5)
grid.arrange(p1, p2, nrow = 1)
dev.off()


# ICE curve method -------------------------------------------------------------

# ICE curves for all 10 features
ice_curves <- lapply(features, FUN = function(feature) {
  ice <- partial(pp, pred.var = feature, ice = TRUE)
  autoplot(ice, alpha = 0.1) + 
    ylim(range(trn$y)) +
    theme_light()
})

pdf("figures/ice-ppr.pdf", width = 15, height = 5)
grid.arrange(grobs = ice_curves, ncol = 5)
dev.off()

# Plot VI scores
p1 <- vip(pp, method = "ice") + ggtitle("PPR")
p2 <- vip(nn, method = "ice") + ggtitle("NN")

# Figure X
pdf("figures/vip-ice-ppr-nn.pdf", width = 7, height = 3.5)
grid.arrange(p1, p2, ncol = 2)
dev.off()


vis <- vi(pp, method = "pdp")
vis
# Figure X
pdf("figures/pdp-from-attr.pdf", width = 10, height = 5)
par(mfrow = c(2, 5))
for (name in paste0("X", 1:10)) {
  plot(attr(vis, which = "pdp")[[name]], type = "l", ylim = c(9, 19), las = 1)
}
dev.off()


# Permutation method -----------------------------------------------------------

# Plot VI scores
set.seed(2021)  # for reproducibility
p1 <- vip(pp, method = "permute", target = "y", metric = "rsquared",
          pred_wrapper = predict) + ggtitle("PPR")
p2 <- vip(nn, method = "permute", target = "y", metric = "rsquared",
          pred_wrapper = predict) + ggtitle("NN")

# Figure X
pdf("figures/vip-permute-ppr-nn.pdf", width = 7, height = 3.5)
grid.arrange(p1, p2, ncol = 2)
dev.off()

# Use 10 Monte Carlo reps
set.seed(403)  # for reproducibility
vi(pp, method = "permute", target = "y", metric = "rsquared",
   pred_wrapper = predict, nsim = 10)

# Custom loss function: mean absolute error
mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

# Figure X
set.seed(2321)  # for reproducibility
pdf("figures/vip-permute-nn-mae.pdf", width = 7, height = 4.326)
vip(nn, method = "permute", target = "y", metric = mae, 
    smaller_is_better = TRUE, pred_wrapper = nnet:::predict.nnet) + 
  ggtitle("Custom loss function: MAE")
dev.off()

# Figure X
set.seed(2327)  # for reproducibility
pdf("figures/vip-permute-nn-sample.pdf", width = 7, height = 4.326)
vip(nn, method = "permute", 
    train = trn[sample(nrow(trn), size = 400), ],  # sample 400 observations
    target = "y", metric = "rmse") +
  ggtitle("Using a random subset of training data")
dev.off()


# Use sparklines to characterize feature effects -------------------------------

# First, compute a tibble of variable importance scores using any method
var_imp <- vi(rfo, method = "permute", metric = "rmse", target = "y")

# Next, convert to an html-based data table with sparklines
add_sparklines(var_imp, fit = rfo)


# Ames housing example ---------------------------------------------------------

library(SuperLearner)

boston <- pdp::boston
X <- subset(boston, select = -cmedv)

# Load the data and assign column names
url <- paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/",
              "00291/airfoil_self_noise.dat")
airfoil <- read.table(url, header = FALSE)
names(airfoil) <- c(
  "frequency", 
  "angle_of_attack", 
  "chord_length", 
  "free_stream_velocity", 
  "suction_side_displacement_thickness", 
  "scaled_sound_pressure_level"
)
X <- subset(airfoil, select = -scaled_sound_pressure_level)
y <- airfoil$scaled_sound_pressure_level
sl_lib <- c("SL.xgboost", "SL.ranger", "SL.glmnet")

# Stack an XGBoost, RF, Lasso, and an SVM
set.seed(840)
sl <- SuperLearner(Y = y, X = X, SL.library = sl_lib)
sl

pfun <- function(object, newdata) {
  predict(object, newdata = newdata)$pred
}
parfun <- function(object, newdata) {
  mean(predict(object, newdata = newdata)$pred)
}

set.seed(278)
var_imp <- vi(sl, method = "permute", train = X, target = y, metric = "rmse",
              pred_wrapper = pfun, nsim = 10)
add_sparklines(var_imp, fit = sl, pred.fun = parfun, train = X)

library(doParallel) # load the parallel backend
cl <- makeCluster(8) # use 8 workers
registerDoParallel(cl) # register the parallel backend
set.seed(278)
res <- vi(sl, method = "permute", train = X, target = boston$cmedv, 
          metric = "rmse", pred_wrapper = pfun, nsim = 10, parallel = TRUE, 
          paropts = list(.packages = "SuperLearner"))
stopCluster(cl) # good practice

vip(res)


# Predict the sale price for a home --------------------------------------------

# Load the Ames housing data
ames <- AmesHousing::make_ames()
X <- subset(ames, select = -Sale_Price)
y <- ames$Sale_Price

# Load required packages
library(SuperLearner)

# List of base learners
learners <- c("SL.xgboost", "SL.ranger", "SL.earth", "SL.glmnet", "SL.ksvm")

# Stack models
set.seed(840)
ctrl <- SuperLearner.CV.control(V = 5L, shuffle = TRUE)
sl <- SuperLearner(Y = y, X = X, SL.library = learners, verbose = TRUE,
                   cvControl = ctrl)
sl

# Prediction wrapper functions
imp_fun <- function(object, newdata) {
  predict(object, newdata = newdata)$pred
}
par_fun <- function(object, newdata) {
  mean(predict(object, newdata = newdata)$pred)
}

# Setup parallel backend
library(doParallel) # load the parallel backend
cl <- makeCluster(5) # use 5 workers
registerDoParallel(cl) # register the parallel backend

# Permutation-based feature importance
set.seed(278)
var_imp <- vi(sl, method = "permute", train = X, target = y, metric = "rmse",
              pred_wrapper = imp_fun, nsim = 5, parallel = TRUE)

# Add sparline representation of feature effects
add_sparklines(var_imp[1L:10L, ], fit = sl, pred.fun = par_fun, train = X, 
               digits = 2, verbose = TRUE, trim.outliers = TRUE, 
               grid.resolution = 20, parallel = TRUE)

# Shut down cluster
stopCluster(cl)
