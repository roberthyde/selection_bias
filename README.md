
<!-- badges: start -->
<!-- badges: end -->

# Inferential modelling with wide data

## Introduction

Welcome to our workshop on inferential modelling with wide data. We hope
you enjoy the session.

This workshop will cover the problems associated with inferential
modelling of high dimensional, wide data, suggest approaches to overcome
them and provide hands-on training in the implementation of
regularisation techniques, covariate selection stability and
triangulation.

The following packages are required for these exercises.

``` r
library(Hmisc)
library(broom)
library(glmnet)
library(ncvreg)
library(bigstep)
library(rsample)
library(tidyverse)
library(stabiliser)
```

## Simulating data

In order to appreciate the issues we will be discussing today, we have
provided functions to simulate datasets for exploration.

The following function generates a dataset with “ncols” as the number of
variables and “nrows” as the number of rows.

``` r
generate_uncor_variables <- function(ncols, nrows) {
  data.frame(replicate(ncols, rnorm(nrows, 0, 1)))
}
```

A dataset with 197 rows and 130 variables can then be generated using
this function as follows:

``` r
variables <- generate_uncor_variables(ncols = 130, nrows = 197)
```

This results in the following dataset being generated:

``` r
variables %>%
  as_tibble()
```

    ## # A tibble: 197 x 130
    ##         X1     X2      X3     X4     X5      X6     X7      X8      X9     X10
    ##      <dbl>  <dbl>   <dbl>  <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>   <dbl>
    ##  1 -0.890  -1.16   0.720  -1.62  -0.778 -0.470  -0.146 -0.668  -1.68    0.955 
    ##  2 -0.382   1.24  -1.31   -1.53  -0.813 -0.0171 -0.500 -0.679   0.524  -0.244 
    ##  3 -0.343  -0.223  0.323  -0.239 -0.515  0.110   0.973 -0.601   0.293   0.101 
    ##  4  0.258  -1.34  -1.18   -0.681  0.186 -0.392   0.829 -0.778  -0.519  -0.449 
    ##  5  1.06   -0.663 -0.488   0.600  1.74   1.16    0.406 -0.396   1.66    0.229 
    ##  6  1.16   -2.71   0.0598  0.287  0.821 -0.412  -0.925  0.830   0.439   0.836 
    ##  7 -0.0183  0.760 -0.582   0.677  1.50   2.12    0.602 -0.615   0.654   0.276 
    ##  8  1.30    1.21  -0.669   1.46  -0.123  0.346  -1.33   1.30   -1.02   -0.294 
    ##  9  0.280  -1.20   0.802  -0.857 -1.61   0.900  -1.26   0.0435  0.120   1.31  
    ## 10 -1.20   -1.33   0.516   1.11   0.164 -2.48   -1.41  -1.08    0.0445 -0.0363
    ## # ... with 187 more rows, and 120 more variables: X11 <dbl>, X12 <dbl>,
    ## #   X13 <dbl>, X14 <dbl>, X15 <dbl>, X16 <dbl>, X17 <dbl>, X18 <dbl>,
    ## #   X19 <dbl>, X20 <dbl>, X21 <dbl>, X22 <dbl>, X23 <dbl>, X24 <dbl>,
    ## #   X25 <dbl>, X26 <dbl>, X27 <dbl>, X28 <dbl>, X29 <dbl>, X30 <dbl>,
    ## #   X31 <dbl>, X32 <dbl>, X33 <dbl>, X34 <dbl>, X35 <dbl>, X36 <dbl>,
    ## #   X37 <dbl>, X38 <dbl>, X39 <dbl>, X40 <dbl>, X41 <dbl>, X42 <dbl>,
    ## #   X43 <dbl>, X44 <dbl>, X45 <dbl>, X46 <dbl>, X47 <dbl>, X48 <dbl>, ...

We can also generate an outcome variable, in this case randomly
generated in the same manner, but renaming as “outcome”

``` r
generate_uncor_outcome <- function(nrows) {
  data.frame(replicate(1, rnorm(nrows, 0, 1))) %>%
    rename("outcome" = 1)
}

outcome <- generate_uncor_outcome(nrows = 197)

outcome %>%
  as_tibble()
```

    ## # A tibble: 197 x 1
    ##    outcome
    ##      <dbl>
    ##  1  -1.31 
    ##  2  -1.85 
    ##  3  -1.55 
    ##  4  -1.94 
    ##  5  -0.571
    ##  6  -0.903
    ##  7   1.41 
    ##  8  -0.344
    ##  9  -0.483
    ## 10   0.937
    ## # ... with 187 more rows

We can now bind together the uncorrelated, randomly generated variables,
with the randomly generated outcome.

``` r
df_no_signal <- outcome %>%
  bind_cols(variables)
```

This results in a dataset of 197 rows, with a single outcome variable,
which has no relationship to the 130 columns as shown below.

``` r
df_no_signal %>%
  as_tibble()
```

    ## # A tibble: 197 x 131
    ##    outcome      X1     X2      X3     X4     X5      X6     X7      X8      X9
    ##      <dbl>   <dbl>  <dbl>   <dbl>  <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>
    ##  1  -1.31  -0.890  -1.16   0.720  -1.62  -0.778 -0.470  -0.146 -0.668  -1.68  
    ##  2  -1.85  -0.382   1.24  -1.31   -1.53  -0.813 -0.0171 -0.500 -0.679   0.524 
    ##  3  -1.55  -0.343  -0.223  0.323  -0.239 -0.515  0.110   0.973 -0.601   0.293 
    ##  4  -1.94   0.258  -1.34  -1.18   -0.681  0.186 -0.392   0.829 -0.778  -0.519 
    ##  5  -0.571  1.06   -0.663 -0.488   0.600  1.74   1.16    0.406 -0.396   1.66  
    ##  6  -0.903  1.16   -2.71   0.0598  0.287  0.821 -0.412  -0.925  0.830   0.439 
    ##  7   1.41  -0.0183  0.760 -0.582   0.677  1.50   2.12    0.602 -0.615   0.654 
    ##  8  -0.344  1.30    1.21  -0.669   1.46  -0.123  0.346  -1.33   1.30   -1.02  
    ##  9  -0.483  0.280  -1.20   0.802  -0.857 -1.61   0.900  -1.26   0.0435  0.120 
    ## 10   0.937 -1.20   -1.33   0.516   1.11   0.164 -2.48   -1.41  -1.08    0.0445
    ## # ... with 187 more rows, and 121 more variables: X10 <dbl>, X11 <dbl>,
    ## #   X12 <dbl>, X13 <dbl>, X14 <dbl>, X15 <dbl>, X16 <dbl>, X17 <dbl>,
    ## #   X18 <dbl>, X19 <dbl>, X20 <dbl>, X21 <dbl>, X22 <dbl>, X23 <dbl>,
    ## #   X24 <dbl>, X25 <dbl>, X26 <dbl>, X27 <dbl>, X28 <dbl>, X29 <dbl>,
    ## #   X30 <dbl>, X31 <dbl>, X32 <dbl>, X33 <dbl>, X34 <dbl>, X35 <dbl>,
    ## #   X36 <dbl>, X37 <dbl>, X38 <dbl>, X39 <dbl>, X40 <dbl>, X41 <dbl>,
    ## #   X42 <dbl>, X43 <dbl>, X44 <dbl>, X45 <dbl>, X46 <dbl>, X47 <dbl>, ...

## Conventional approaches

The following function conducts univariable analysis to determine the
association between a given variable and the outcome. A pearson/spearman
rank correlation matrix is another option for this.

``` r
univariable_analysis <- function(data, variable) {
  data %>%
    lm(outcome ~ variable, .) %>%
    tidy() %>%
    filter(term != "(Intercept)")
}
```

This function can then be applied using map_df() to each column of the
dataset individually and return a dataframe.

``` r
univariable_outcomes <- map_df(df_no_signal, ~ univariable_analysis(data = df_no_signal, variable = .), .id = "variable")
```

A conventional approach would then filter at a given threshold (for
example P\<0.2).

``` r
univariable_outcomes_filtered <- univariable_outcomes %>%
  filter(p.value < 0.2)
```

This results in a table below of all of the variables that have a
p-value of \<0.2 to be carried forward into a multivariable model.

``` r
univariable_outcomes_filtered %>%
  as_tibble()
```

    ## # A tibble: 30 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable   1.00    4.95e-17   2.02e16  0     
    ##  2 X1       variable  -0.106   7.95e- 2  -1.33e 0  0.184 
    ##  3 X6       variable  -0.116   7.34e- 2  -1.58e 0  0.115 
    ##  4 X10      variable   0.115   7.50e- 2   1.54e 0  0.126 
    ##  5 X13      variable   0.129   7.90e- 2   1.64e 0  0.103 
    ##  6 X21      variable  -0.131   7.43e- 2  -1.77e 0  0.0789
    ##  7 X23      variable   0.0952  7.30e- 2   1.30e 0  0.194 
    ##  8 X27      variable  -0.124   6.42e- 2  -1.93e 0  0.0556
    ##  9 X37      variable   0.126   7.75e- 2   1.62e 0  0.106 
    ## 10 X41      variable   0.103   7.99e- 2   1.29e 0  0.197 
    ## # ... with 20 more rows

A list of variables to be included is as follows:

``` r
variables_for_stepwise <- univariable_outcomes_filtered %>%
  pull(variable)
```

These variables would subsequently be offered into a stepwise selection
process such as the following

``` r
stepwise_model <- function(data, variables) {
  data_selected <- data %>%
    select(variables)

  lm(outcome ~ ., data = data_selected) %>%
    step(., direction = "backward", trace = FALSE) %>%
    tidy() %>%
    filter(p.value < 0.05) %>%
    rename(variable = term)
}

prefiltration_results <- stepwise_model(data = df_no_signal, variables = variables_for_stepwise)

prefiltration_results %>%
  as_tibble()
```

    ## # A tibble: 8 x 5
    ##   variable estimate std.error statistic p.value
    ##   <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 X1         -0.173    0.0756     -2.29  0.0234
    ## 2 X44         0.138    0.0662      2.08  0.0390
    ## 3 X61         0.150    0.0679      2.21  0.0282
    ## 4 X70        -0.174    0.0735     -2.37  0.0188
    ## 5 X73         0.170    0.0723      2.36  0.0194
    ## 6 X104        0.174    0.0691      2.51  0.0128
    ## 7 X125       -0.153    0.0655     -2.33  0.0207
    ## 8 X129        0.138    0.0695      1.99  0.0477

## Data with a true signal

We will test a variety of models on this dataset. For future comparison
let’s set up a list where we can store model results

``` r
model_results <- list()
```

``` r
generate_data_with_signal <- function(nrow, ncol, n_causal_vars, amplitude) {
  # Generate the variables from a multivariate normal distribution
  mu <- rep(0, ncol)
  rho <- 0.25
  sigma <- toeplitz(rho^(0:(ncol - 1))) #  Symmetric Toeplitz Matrix
  X <- matrix(rnorm(nrow * ncol), nrow) %*% chol(sigma) # multiply matrices Choleski Decomposition. Description. Compute the Choleski factorization of a real symmetric positive-definite square matrix)

  # Generate the response from a linear model
  nonzero <- sample(ncol, n_causal_vars) # select the id of 'true' variables
  beta <- amplitude * (1:ncol %in% nonzero) / sqrt(nrow) # vector of effect sizes to pick out true varaiables
  beta_value <- amplitude / sqrt(nrow)
  outcome.sample <- function(X) X %*% beta + rnorm(nrow) # calculate outcome from true vars and error
  outcome <- outcome.sample(X)

  ## Rename true variables
  X_data <- as.data.frame(X)
  for (i in c(nonzero)) {
    X_data1 <- X_data %>%
      rename_with(.cols = i, ~ paste("causal_", i, sep = ""))
    X_data <- X_data1
  }

  dataset_sim <- as.data.frame(cbind(outcome, X_data1))
}
```

We can now simulate a dataset df_signal with 300 rows and 300 variables,
8 of which have a relationship with the outcome.

We can also alter the signal strenght of causal variables by changing
the “amplitute” paramater.

``` r
df_signal <- generate_data_with_signal(nrow = 300, ncol = 300, n_causal_vars = 8, amplitude = 7)
```

## “Cheat” model

As we have simulated this dataset, we can “cheat” and create the perfect
model to check everything’s worked correctly.

``` r
df_signal %>%
  select(outcome, contains("causal_")) %>%
  lm(outcome~., data=.) %>%
  tidy()
```

    ## # A tibble: 9 x 5
    ##   term        estimate std.error statistic  p.value
    ##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ## 1 (Intercept)  -0.0111    0.0584    -0.189 8.50e- 1
    ## 2 causal_52     0.350     0.0579     6.06  4.31e- 9
    ## 3 causal_73     0.364     0.0553     6.59  2.05e-10
    ## 4 causal_120    0.470     0.0607     7.75  1.58e-13
    ## 5 causal_186    0.478     0.0619     7.73  1.77e-13
    ## 6 causal_227    0.378     0.0589     6.42  5.39e-10
    ## 7 causal_250    0.363     0.0594     6.10  3.33e- 9
    ## 8 causal_258    0.412     0.0566     7.27  3.28e-12
    ## 9 causal_266    0.470     0.0573     8.19  8.11e-15

## Conventional stepwise approach

We can now repeat out prefiltration and stepwise selection approach as
before

``` r
univariable_outcomes <- map_df(df_signal, ~ univariable_analysis(data = df_signal, variable = .), .id = "variable")
univariable_outcomes_filtered <- univariable_outcomes %>%
  filter(p.value < 0.2)
variables_for_stepwise <- univariable_outcomes_filtered %>%
  pull(variable)
model_results$prefiltration <- stepwise_model(data = df_signal, variables = variables_for_stepwise)
model_results$prefiltration %>%
  as_tibble()
```

    ## # A tibble: 18 x 5
    ##    variable   estimate std.error statistic  p.value
    ##    <chr>         <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 V27          -0.118    0.0549     -2.14 3.31e- 2
    ##  2 V31          -0.128    0.0514     -2.50 1.31e- 2
    ##  3 causal_52     0.379    0.0530      7.15 8.07e-12
    ##  4 V63          -0.163    0.0560     -2.90 3.98e- 3
    ##  5 causal_73     0.329    0.0510      6.45 5.22e-10
    ##  6 causal_120    0.439    0.0572      7.67 3.01e-13
    ##  7 V164          0.106    0.0522      2.03 4.33e- 2
    ##  8 causal_186    0.449    0.0602      7.47 1.13e-12
    ##  9 V223          0.255    0.0562      4.54 8.39e- 6
    ## 10 causal_227    0.366    0.0547      6.69 1.30e-10
    ## 11 V245          0.191    0.0542      3.52 5.10e- 4
    ## 12 causal_250    0.308    0.0546      5.64 4.32e- 8
    ## 13 causal_258    0.324    0.0558      5.80 1.84e- 8
    ## 14 V264         -0.122    0.0551     -2.22 2.71e- 2
    ## 15 causal_266    0.396    0.0532      7.43 1.38e-12
    ## 16 V280         -0.110    0.0548     -2.00 4.64e- 2
    ## 17 V293          0.129    0.0526      2.45 1.47e- 2
    ## 18 V297          0.131    0.0555      2.36 1.90e- 2

## Regularisation

There are several regularisation methods available. Here, we will use
the lasso. The following function enables the use of the lasso algorithm
from the *glmnet* package.

``` r
model_lasso <- function(data) {
  y_temp <- data %>%
    select("outcome") %>%
    as.matrix()

  x_temp <- data %>%
    select(-"outcome") %>%
    as.matrix()

  fit_lasso <- cv.glmnet(x = x_temp, y = y_temp, alpha = 1)

  coefs <- coef(fit_lasso, s = "lambda.min")

  data.frame(name = coefs@Dimnames[[1]][coefs@i + 1], coefficient = coefs@x) %>%
    rename(
      variable = name,
      estimate = coefficient
    ) %>%
    filter(variable != "(Intercept)") %>%
    select(variable, estimate)
}

model_results$lasso <- model_lasso(df_signal)
```

MCP can also be used in a similar manner using the *ncvreg* package.

``` r
model_mcp <- function(data) {
  y_temp <- data %>%
    select("outcome") %>%
    as.matrix()

  x_temp <- data %>%
    select(-"outcome")

  fit_mcp <- cv.ncvreg(X = x_temp, y = y_temp)

  fit_mcp %>%
    coef() %>%
    as_tibble(rownames = "variable") %>%
    rename(
      estimate = value
    ) %>%
    filter(
      variable != "(Intercept)",
      estimate != 0,
      !grepl("(Intercept)", variable),
      !grepl("Xm[, -1]", variable)
    ) %>%
    mutate(variable = str_remove_all(variable, "`"))
}

model_results$mcp <- model_mcp(df_signal)
```

MBIC can also be used from the *bigstep* package

``` r
model_mbic <- function(data) {
  y_temp <- data %>%
    select("outcome") %>%
    as.matrix()

  x_temp <- data %>%
    select(-"outcome")

  bigstep_prepped <- bigstep::prepare_data(y_temp, x_temp, verbose = FALSE)

  bigstep_prepped %>%
    reduce_matrix(minpv = 0.01) %>%
    fast_forward(crit = mbic) %>%
    multi_backward(crit = mbic) %>%
    summary() %>%
    stats::coef() %>%
    as.data.frame() %>%
    rownames_to_column(., var = "variable") %>%
    mutate(variable = str_remove_all(variable, "`")) %>%
    filter(
      !grepl("(Intercept)", variable),
      !grepl("Xm[, -1]", variable)
    ) %>%
    rename(estimate = Estimate) %>%
    select(variable, estimate)
}

model_results$mbic <- model_mbic(df_signal)
```

A comparison of the number of True/False positives is shown below:

``` r
calculate_tp_fp <- function(results) {
  results %>%
    mutate(causal = case_when(
      grepl("causal", variable) ~ "tp",
      !grepl("causal", variable) ~ "fp"
    )) %>%
    group_by(model) %>%
    summarise(
      tp = sum(causal == "tp", na.rm = TRUE),
      fp = sum(causal == "fp", na.rm = TRUE)
    ) %>%
    mutate("total_selected" = tp + fp)
}

conventional_results <- model_results %>%
  bind_rows(., .id = "model") %>%
  calculate_tp_fp()

conventional_results
```

    ## # A tibble: 4 x 4
    ##   model            tp    fp total_selected
    ##   <chr>         <int> <int>          <int>
    ## 1 lasso             8    16             24
    ## 2 mbic              8     1              9
    ## 3 mcp               8     1              9
    ## 4 prefiltration     8    10             18

## Stability selection

Function for bootstrapping

``` r
boot_sample <- function(data, boot_reps) {
  rsample::bootstraps(data, boot_reps)
}

bootstrapped_datasets <- boot_sample(data = df_signal, boot_reps = 10)
```

Bootstrapped data is presented here as a table of 10 different nested
tables.

``` r
bootstrapped_datasets
```

    ## # Bootstrap sampling 
    ## # A tibble: 10 x 2
    ##    splits            id         
    ##    <list>            <chr>      
    ##  1 <split [300/116]> Bootstrap01
    ##  2 <split [300/118]> Bootstrap02
    ##  3 <split [300/109]> Bootstrap03
    ##  4 <split [300/109]> Bootstrap04
    ##  5 <split [300/98]>  Bootstrap05
    ##  6 <split [300/101]> Bootstrap06
    ##  7 <split [300/113]> Bootstrap07
    ##  8 <split [300/112]> Bootstrap08
    ##  9 <split [300/120]> Bootstrap09
    ## 10 <split [300/108]> Bootstrap10

If we extract a single bootstrapped dataset and sort by the outcome, we
can see that several rows have been resampled. Consequently as the
dataset length is the same as the original, several rows will be omitted
completely.

``` r
bootstrapped_datasets$splits[[1]] %>%
  as_tibble() %>%
  arrange(outcome)
```

    ## # A tibble: 300 x 301
    ##    outcome     V1     V2      V3     V4     V5     V6      V7     V8     V9
    ##      <dbl>  <dbl>  <dbl>   <dbl>  <dbl>  <dbl>  <dbl>   <dbl>  <dbl>  <dbl>
    ##  1   -3.75 -0.955 -1.61   0.601   1.08  -0.821  1.00  -1.20   -0.621 -1.09 
    ##  2   -3.68  0.178  1.19   0.0911  0.212 -0.632  0.166  0.670  -0.860  1.66 
    ##  3   -3.68  0.178  1.19   0.0911  0.212 -0.632  0.166  0.670  -0.860  1.66 
    ##  4   -3.68  0.178  1.19   0.0911  0.212 -0.632  0.166  0.670  -0.860  1.66 
    ##  5   -2.80  0.667  0.505 -0.242  -1.66  -2.65   1.35   0.981   0.109  1.18 
    ##  6   -2.72  0.794 -1.04   1.00   -0.940 -0.501 -0.686 -0.282  -1.04  -1.08 
    ##  7   -2.59 -1.13  -2.18  -1.60   -1.64  -0.295 -1.46  -0.308  -1.62  -0.565
    ##  8   -2.57 -1.72  -1.82  -0.0605  0.842  0.164 -1.38  -0.0773  0.819  0.620
    ##  9   -2.57 -1.72  -1.82  -0.0605  0.842  0.164 -1.38  -0.0773  0.819  0.620
    ## 10   -2.57 -1.72  -1.82  -0.0605  0.842  0.164 -1.38  -0.0773  0.819  0.620
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
    ## #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...

## Model for bootstraps

We can apply our previous lasso function over each one of these
bootstrapped resamples.

``` r
model_lasso_bootstrapped <- bootstrapped_datasets %>%
  map_df(.x = .$splits, .f = ~ as.data.frame(.) %>% model_lasso(.), .id = "bootstrap")
```

## Permutation

To identify a null threshold, first we must permute the outcome.

Our original dataset looks like this:

``` r
df_signal %>%
  as_tibble()
```

    ## # A tibble: 300 x 301
    ##    outcome     V1     V2      V3     V4      V5     V6     V7      V8      V9
    ##      <dbl>  <dbl>  <dbl>   <dbl>  <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>
    ##  1 -3.08   -0.748 -0.882 -0.112   0.508 -0.208   0.353  0.412 -2.83   -0.845 
    ##  2 -1.63   -1.10  -0.409  0.558   0.897 -0.439   1.26   1.13   2.00   -0.0597
    ##  3 -2.02    0.528 -0.139  0.270  -0.833  0.821   0.639 -0.647  0.532   0.252 
    ##  4  0.358  -1.17  -0.677  0.381  -0.728 -0.768  -0.310 -0.178 -0.525   0.378 
    ##  5  0.807   0.497  1.67   1.79    1.66   1.99   -0.150 -1.33  -0.190   1.01  
    ##  6 -0.0379  1.21   1.21  -0.945  -1.03  -0.940  -0.903 -0.862  0.499  -0.597 
    ##  7  0.888   1.24   0.471  1.09    0.280  0.900  -0.304  0.134  0.0495  0.802 
    ##  8 -0.982   0.663  0.659  0.0946 -0.318  0.0104  1.03   1.69  -0.641  -0.582 
    ##  9  0.986  -1.91  -0.295 -0.0342  0.879  0.724   0.563 -0.261  1.53   -1.66  
    ## 10  1.92   -0.349 -0.747  0.559   0.219 -0.108   0.393  1.33   1.32   -0.677 
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
    ## #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...

By permuting the outcome variable we sever all ties between the outcome
and the explanatory variables. We might want to conduct this 5 times.

``` r
permuted_datasets <- rsample::permutations(data = df_signal, permute = outcome, times = 5)
```

A single dataset would look like this. Note the structure of the
explanatory variables remains the same, but the outcome is randomly
shuffled.

``` r
permuted_datasets$splits[[1]] %>%
  as_tibble()
```

    ## # A tibble: 300 x 301
    ##    outcome     V1     V2      V3     V4      V5     V6     V7      V8      V9
    ##      <dbl>  <dbl>  <dbl>   <dbl>  <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>
    ##  1 -2.80   -0.748 -0.882 -0.112   0.508 -0.208   0.353  0.412 -2.83   -0.845 
    ##  2 -0.0908 -1.10  -0.409  0.558   0.897 -0.439   1.26   1.13   2.00   -0.0597
    ##  3 -1.40    0.528 -0.139  0.270  -0.833  0.821   0.639 -0.647  0.532   0.252 
    ##  4  0.0496 -1.17  -0.677  0.381  -0.728 -0.768  -0.310 -0.178 -0.525   0.378 
    ##  5 -0.931   0.497  1.67   1.79    1.66   1.99   -0.150 -1.33  -0.190   1.01  
    ##  6  0.615   1.21   1.21  -0.945  -1.03  -0.940  -0.903 -0.862  0.499  -0.597 
    ##  7 -1.33    1.24   0.471  1.09    0.280  0.900  -0.304  0.134  0.0495  0.802 
    ##  8  2.27    0.663  0.659  0.0946 -0.318  0.0104  1.03   1.69  -0.641  -0.582 
    ##  9  0.307  -1.91  -0.295 -0.0342  0.879  0.724   0.563 -0.261  1.53   -1.66  
    ## 10  1.81   -0.349 -0.747  0.559   0.219 -0.108   0.393  1.33   1.32   -0.677 
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
    ## #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...

We can then apply our bootstrap function to each one of these 5 permuted
datasets. We might perform 10 bootstrap samples for each of the 5
permuted datasets. The model would then be applied to each dataset
within the following table.

``` r
permuted_bootstrapped_datasets <- permuted_datasets %>%
  map_df(.x = .$splits, .f = ~ as.data.frame(.) %>% boot_sample(., boot_reps = 10), .id = "permutation")

permuted_bootstrapped_datasets
```

    ## # A tibble: 50 x 3
    ##    permutation splits            id         
    ##    <chr>       <list>            <chr>      
    ##  1 1           <split [300/114]> Bootstrap01
    ##  2 1           <split [300/112]> Bootstrap02
    ##  3 1           <split [300/111]> Bootstrap03
    ##  4 1           <split [300/115]> Bootstrap04
    ##  5 1           <split [300/108]> Bootstrap05
    ##  6 1           <split [300/111]> Bootstrap06
    ##  7 1           <split [300/116]> Bootstrap07
    ##  8 1           <split [300/113]> Bootstrap08
    ##  9 1           <split [300/124]> Bootstrap09
    ## 10 1           <split [300/113]> Bootstrap10
    ## # ... with 40 more rows

This code is relatively lengthy, and is therefore deliberately omitted
from the workshop, however is present within the *stabiliser* package
and freely available at www.github.com/roberthyde/stabiliser.

Using the ecdf() function, the stability of each variable within each
permutation can be calculated.

By choosing the value where quantile(., probs=1), the highest stability
that might have occurred by chance can be calculated.

The mean threshold across all permutations can then be calculated. This
represents the “null threshold”; i.e., the mean highest stability a
variable might acheive across all permuted datasets (where we know there
should be no links between variables and outcome).

Variables in the true model (i.e., non-permuted data) that have a
stability value in excess of this null threshold are highly likely to be
truly correlated with the outcome.

## The *stabiliser* approach

``` r
stab_output <- stabilise(outcome = "outcome", data = df_signal, models = c("mbic"), type = "linear")

stab_output$mbic$stability
```

    ## # A tibble: 301 x 7
    ##    variable   mean_coefficient ci_lower ci_upper bootstrap_p stability stable
    ##    <chr>                 <dbl>    <dbl>    <dbl>       <dbl>     <dbl> <chr> 
    ##  1 causal_120            0.463    0.323    0.594           0        99 *     
    ##  2 causal_73             0.366    0.263    0.480           0        99 *     
    ##  3 causal_258            0.410    0.293    0.510           0        98 *     
    ##  4 causal_266            0.452    0.331    0.562           0        97 *     
    ##  5 causal_227            0.371    0.274    0.478           0        96 *     
    ##  6 causal_186            0.473    0.343    0.588           0        94 *     
    ##  7 causal_52             0.351    0.247    0.500           0        91 *     
    ##  8 causal_250            0.354    0.272    0.442           0        88 *     
    ##  9 V223                  0.297    0.222    0.383           0        45 *     
    ## 10 V31                  -0.224   -0.258   -0.199           0         6 <NA>  
    ## # ... with 291 more rows

## All models

The *stabiliser* package allows multiple models to be run
simultaneously. Just select the models you wish to run in the “models”
argument.

MCP is omitted here for speed. To include it, just add it to the list of
models using: models = c(“mbic”, “lasso”, “mcp”)

``` r
#stab_output <- stabilise(outcome = "outcome", data = df_signal, models = c("mbic", "lasso"), type = "linear")
stab_output <- stabilise(outcome = "outcome", data = df_signal, models = c("mbic"), type = "linear")
```

## Results

Calculate the number of true and false positives selected through
stability approaches, and rename columns to include “\_stability”.

``` r
stability_results <- stab_output %>%
  map_df(., ~ .x$stability, .id = "model") %>%
  filter(stable == "*") %>%
  calculate_tp_fp(.) %>%
  rename_all(., ~ paste0(., "_stability"))

stability_results
```

    ## # A tibble: 1 x 4
    ##   model_stability tp_stability fp_stability total_selected_stability
    ##   <chr>                  <int>        <int>                    <int>
    ## 1 mbic                       8            1                        9

Compare this with the non-stability approach

``` r
conventional_results %>%
  left_join(stability_results, by = c("model" = "model_stability"))
```

    ## # A tibble: 4 x 7
    ##   model      tp    fp total_selected tp_stability fp_stability total_selected_s~
    ##   <chr>   <int> <int>          <int>        <int>        <int>             <int>
    ## 1 lasso       8    16             24           NA           NA                NA
    ## 2 mbic        8     1              9            8            1                 9
    ## 3 mcp         8     1              9           NA           NA                NA
    ## 4 prefil~     8    10             18           NA           NA                NA

# Triangulation

The *stabiliser* package allows the stability selection results from
multiple models to be used synergistically, and by leveraging the
strenghts of various models, a more robust method of variable selection
is often achieved.

``` r
triangulated_stability <- triangulate(stab_output)

triangulated_stability
```

    ## $combi
    ## $combi$stability
    ## # A tibble: 301 x 4
    ##    variable   stability bootstrap_p stable
    ##    <chr>          <dbl>       <dbl> <chr> 
    ##  1 causal_266        98           0 *     
    ##  2 causal_186        97           0 *     
    ##  3 causal_73         97           0 *     
    ##  4 causal_120        96           0 *     
    ##  5 causal_258        95           0 *     
    ##  6 causal_227        89           0 *     
    ##  7 causal_250        88           0 *     
    ##  8 causal_52         86           0 *     
    ##  9 V223              42           0 *     
    ## 10 V280               4           0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 25

``` r
stab_plot(triangulated_stability)
```

    ## $combi

![](README_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

## No signal datasets

We can now return to our original dataset that we simulated to have no
signal.

Our conventional approach performed relatively poorly, selecting the
following variables as being significantly associated with the outcome
variable.

``` r
prefiltration_results
```

    ## # A tibble: 8 x 5
    ##   variable estimate std.error statistic p.value
    ##   <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 X1         -0.173    0.0756     -2.29  0.0234
    ## 2 X44         0.138    0.0662      2.08  0.0390
    ## 3 X61         0.150    0.0679      2.21  0.0282
    ## 4 X70        -0.174    0.0735     -2.37  0.0188
    ## 5 X73         0.170    0.0723      2.36  0.0194
    ## 6 X104        0.174    0.0691      2.51  0.0128
    ## 7 X125       -0.153    0.0655     -2.33  0.0207
    ## 8 X129        0.138    0.0695      1.99  0.0477

Using *stabiliser*, the following variables are selected from the
dataset.

``` r
# stab_output_no_signal <- stabilise(outcome = "outcome", data = df_no_signal, models = c("mbic", "lasso"), type = "linear")
stab_output_no_signal <- stabilise(outcome = "outcome", data = df_no_signal, models = c("mbic"), type = "linear")
triangulated_output_no_signal <- triangulate(stab_output_no_signal)

triangulated_output_no_signal$combi$stability %>%
  filter(stable == "*")
```

    ## # A tibble: 0 x 4
    ## # ... with 4 variables: variable <chr>, stability <dbl>, bootstrap_p <dbl>,
    ## #   stable <chr>

## Conclusions

Thank you for attending this workshop. We hope you enjoyed the session,
and have a good understanding of where some conventional modelling
approaches might not be appropriate in wider datasets.

If you have any further questions after the workshop, please feel free
to contact Martin Green (<martin.green@nottingham.ac.uk>) or Robert Hyde
(<robert.hyde4@nottingham.ac.uk>).
