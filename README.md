
<!-- badges: start -->

## <!-- badges: end -->

title: “Inferential modelling with wide data” output: html_document:
df_print: paged —

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

FUNCTION EXAMPLE

``` r
example_function <- function(x, y){
  x + y
}

example_function(x = 2, y = 4)
```

    ## [1] 6

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
    ##        X1      X2     X3      X4      X5     X6     X7      X8      X9      X10
    ##     <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>    <dbl>
    ##  1 -0.361 -0.702   0.613 -1.87    0.358  -0.981  1.26   0.404  -2.53    1.29   
    ##  2  1.30  -0.180  -0.114 -1.12   -1.76    0.532  2.00  -1.02    1.11   -1.02   
    ##  3  0.726 -0.344   1.92  -0.799  -1.89    0.494  1.65   1.50    0.495   0.00787
    ##  4  1.77  -0.507   0.126  0.711  -1.44   -1.55  -1.29   1.37    0.0817 -0.362  
    ##  5 -0.544  0.642  -0.715 -1.52    0.600   0.333 -1.36  -0.563   0.353   1.12   
    ##  6 -0.883 -0.0232  1.00  -0.310   0.484   0.647 -0.468 -0.436   0.611  -1.08   
    ##  7  0.555  0.162   1.07   1.86   -0.0711 -1.15  -1.77   0.0526  1.04   -0.184  
    ##  8  1.82  -0.788   0.385 -0.254   0.438  -0.430 -1.15   0.239  -0.891  -1.39   
    ##  9 -0.353 -0.690  -0.201 -2.50   -1.68   -0.649 -2.49   0.508  -1.04    0.313  
    ## 10  0.363  1.88   -0.686 -0.0204 -0.270  -1.24  -1.62   0.502  -0.0875  1.02   
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
    ##  1   0.357
    ##  2   1.36 
    ##  3   0.741
    ##  4   0.224
    ##  5   0.803
    ##  6   1.70 
    ##  7  -0.714
    ##  8  -0.615
    ##  9  -0.236
    ## 10   0.688
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
    ##    outcome     X1      X2     X3      X4      X5     X6     X7      X8      X9
    ##      <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>
    ##  1   0.357 -0.361 -0.702   0.613 -1.87    0.358  -0.981  1.26   0.404  -2.53  
    ##  2   1.36   1.30  -0.180  -0.114 -1.12   -1.76    0.532  2.00  -1.02    1.11  
    ##  3   0.741  0.726 -0.344   1.92  -0.799  -1.89    0.494  1.65   1.50    0.495 
    ##  4   0.224  1.77  -0.507   0.126  0.711  -1.44   -1.55  -1.29   1.37    0.0817
    ##  5   0.803 -0.544  0.642  -0.715 -1.52    0.600   0.333 -1.36  -0.563   0.353 
    ##  6   1.70  -0.883 -0.0232  1.00  -0.310   0.484   0.647 -0.468 -0.436   0.611 
    ##  7  -0.714  0.555  0.162   1.07   1.86   -0.0711 -1.15  -1.77   0.0526  1.04  
    ##  8  -0.615  1.82  -0.788   0.385 -0.254   0.438  -0.430 -1.15   0.239  -0.891 
    ##  9  -0.236 -0.353 -0.690  -0.201 -2.50   -1.68   -0.649 -2.49   0.508  -1.04  
    ## 10   0.688  0.363  1.88   -0.686 -0.0204 -0.270  -1.24  -1.62   0.502  -0.0875
    ## # ... with 187 more rows, and 121 more variables: X10 <dbl>, X11 <dbl>,
    ## #   X12 <dbl>, X13 <dbl>, X14 <dbl>, X15 <dbl>, X16 <dbl>, X17 <dbl>,
    ## #   X18 <dbl>, X19 <dbl>, X20 <dbl>, X21 <dbl>, X22 <dbl>, X23 <dbl>,
    ## #   X24 <dbl>, X25 <dbl>, X26 <dbl>, X27 <dbl>, X28 <dbl>, X29 <dbl>,
    ## #   X30 <dbl>, X31 <dbl>, X32 <dbl>, X33 <dbl>, X34 <dbl>, X35 <dbl>,
    ## #   X36 <dbl>, X37 <dbl>, X38 <dbl>, X39 <dbl>, X40 <dbl>, X41 <dbl>,
    ## #   X42 <dbl>, X43 <dbl>, X44 <dbl>, X45 <dbl>, X46 <dbl>, X47 <dbl>, ...

## Conventional approaches

### Univariable prefiltration

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

### Stepwise regression

This results in a table below of all of the variables that have a
p-value of \<0.2 to be carried forward into a multivariable model.

``` r
univariable_outcomes_filtered %>%
  as_tibble()
```

    ## # A tibble: 24 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable    1.00   5.55e-17   1.80e16  0     
    ##  2 X8       variable    0.128  6.93e- 2   1.84e 0  0.0672
    ##  3 X18      variable   -0.165  6.82e- 2  -2.42e 0  0.0163
    ##  4 X23      variable   -0.122  7.66e- 2  -1.59e 0  0.114 
    ##  5 X36      variable   -0.125  7.06e- 2  -1.77e 0  0.0786
    ##  6 X44      variable    0.117  7.47e- 2   1.57e 0  0.119 
    ##  7 X51      variable   -0.184  7.21e- 2  -2.56e 0  0.0113
    ##  8 X54      variable    0.120  7.39e- 2   1.62e 0  0.106 
    ##  9 X55      variable   -0.104  7.00e- 2  -1.49e 0  0.137 
    ## 10 X56      variable    0.110  6.82e- 2   1.62e 0  0.107 
    ## # ... with 14 more rows

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

stepwise_selected_vars <- stepwise_model(data = df_no_signal, variables = variables_for_stepwise)
```

We can then extract the variables from this stepwise approach and refit
in a final model.

``` r
variables_final_model <- stepwise_selected_vars %>%
  filter(variable != "(Intercept)")

prefiltration_results  <- df_no_signal %>% 
  select(outcome, variables_final_model$variable) %>%
  lm(outcome ~ ., data = .)

prefiltration_results %>%
  tidy()
```

    ## # A tibble: 6 x 5
    ##   term        estimate std.error statistic p.value
    ##   <chr>          <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 (Intercept)  -0.0603    0.0714    -0.844 0.400  
    ## 2 X18          -0.171     0.0660    -2.59  0.0105 
    ## 3 X51          -0.161     0.0697    -2.31  0.0220 
    ## 4 X70          -0.100     0.0719    -1.39  0.166  
    ## 5 X110          0.240     0.0735     3.26  0.00131
    ## 6 X112          0.135     0.0739     1.82  0.0696

We can also calculate the R2 for this model.

``` r
prefiltration_results %>%
  glance()
```

    ## # A tibble: 1 x 12
    ##   r.squared adj.r.squared sigma statistic   p.value    df logLik   AIC   BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>     <dbl> <dbl>  <dbl> <dbl> <dbl>
    ## 1     0.131         0.108 0.975      5.76 0.0000560     5  -271.  557.  580.
    ## # ... with 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

## Data with a true signal

We will test a variety of models on this dataset. For future comparison
let’s set up a list where we can store model results

``` r
model_results <- list()
```

We will also want to explore some simulated datasets with a true signal;
i.e., some of the variables in our dataset are truly associated with the
outcome.

The following function generates a dataset with *nrow* rows and *ncol*
variables, of which *n_causal_vars* variables are truly associated with
the outcome with a signal strength of *amplitude*.

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

We can also alter the signal strength of causal variables by changing
the *amplitute* paramater.

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
    ## 1 (Intercept)    0.134    0.0546      2.45 1.49e- 2
    ## 2 causal_32      0.291    0.0557      5.22 3.45e- 7
    ## 3 causal_66      0.331    0.0548      6.04 4.80e- 9
    ## 4 causal_105     0.323    0.0546      5.92 8.97e- 9
    ## 5 causal_159     0.360    0.0502      7.18 5.98e-12
    ## 6 causal_217     0.340    0.0549      6.20 1.93e- 9
    ## 7 causal_237     0.395    0.0531      7.44 1.17e-12
    ## 8 causal_287     0.383    0.0554      6.91 3.12e-11
    ## 9 causal_292     0.421    0.0553      7.62 3.51e-13

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

    ## # A tibble: 23 x 5
    ##    variable    estimate std.error statistic       p.value
    ##    <chr>          <dbl>     <dbl>     <dbl>         <dbl>
    ##  1 (Intercept)    0.130    0.0482      2.68 0.00772      
    ##  2 V14           -0.126    0.0534     -2.35 0.0193       
    ##  3 causal_32      0.213    0.0514      4.15 0.0000453    
    ##  4 V53           -0.111    0.0447     -2.48 0.0137       
    ##  5 causal_66      0.250    0.0508      4.93 0.00000145   
    ##  6 V77           -0.118    0.0461     -2.55 0.0114       
    ##  7 V79            0.161    0.0548      2.94 0.00362      
    ##  8 V93            0.103    0.0509      2.03 0.0431       
    ##  9 V100           0.116    0.0512      2.26 0.0248       
    ## 10 causal_105     0.293    0.0487      6.01 0.00000000609
    ## # ... with 13 more rows

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

A comparison of the number of True/False positives is shown below, by
using a *calculate_tp_fp()* function.

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
    ## 1 lasso             8    43             51
    ## 2 mbic              8     0              8
    ## 3 mcp               8     4             12
    ## 4 prefiltration     8    15             23

## Stability selection

Stability selection relies heavily on bootstrapping.

Let’s set the number of bootstraps to 10.

``` r
bootstrap_n <- 10
```

An example of the bootstrapping approach is shown below (in reality
100-200 bootstrap resamples might be conducted).

``` r
boot_sample <- function(data, boot_reps) {
  rsample::bootstraps(data, boot_reps)
}

bootstrapped_datasets <- boot_sample(data = df_signal, boot_reps = bootstrap_n)
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
    ##  1 <split [300/107]> Bootstrap01
    ##  2 <split [300/112]> Bootstrap02
    ##  3 <split [300/112]> Bootstrap03
    ##  4 <split [300/106]> Bootstrap04
    ##  5 <split [300/105]> Bootstrap05
    ##  6 <split [300/105]> Bootstrap06
    ##  7 <split [300/107]> Bootstrap07
    ##  8 <split [300/105]> Bootstrap08
    ##  9 <split [300/110]> Bootstrap09
    ## 10 <split [300/115]> Bootstrap10

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
    ##    outcome     V1     V2     V3       V4     V5        V6     V7     V8     V9
    ##      <dbl>  <dbl>  <dbl>  <dbl>    <dbl>  <dbl>     <dbl>  <dbl>  <dbl>  <dbl>
    ##  1   -3.32 -0.106  0.165  0.745  0.156    0.542  0.135     0.348  0.676  1.36 
    ##  2   -2.73 -0.269  0.564  1.08   0.00949 -0.202 -0.643    -1.83   0.588  2.17 
    ##  3   -2.73 -0.269  0.564  1.08   0.00949 -0.202 -0.643    -1.83   0.588  2.17 
    ##  4   -2.59  0.714 -0.713  0.829 -0.0943   0.632  1.42      0.871 -0.102 -0.880
    ##  5   -2.59  0.714 -0.713  0.829 -0.0943   0.632  1.42      0.871 -0.102 -0.880
    ##  6   -2.53  0.238 -0.614 -0.506 -1.63    -1.02  -0.000565 -0.379  1.86  -1.01 
    ##  7   -2.53  0.238 -0.614 -0.506 -1.63    -1.02  -0.000565 -0.379  1.86  -1.01 
    ##  8   -2.53  0.238 -0.614 -0.506 -1.63    -1.02  -0.000565 -0.379  1.86  -1.01 
    ##  9   -2.53  0.238 -0.614 -0.506 -1.63    -1.02  -0.000565 -0.379  1.86  -1.01 
    ## 10   -2.46 -0.462 -0.500 -0.520 -1.06     1.58   0.774     0.767 -1.43  -0.334
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, causal_32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
    ## #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...

## Model for bootstraps

We can apply our previous lasso function over each one of these
bootstrapped resamples.

``` r
model_lasso_bootstrapped <- bootstrapped_datasets %>%
  map_df(.x = .$splits, .f = ~ as.data.frame(.) %>% model_lasso(.), .id = "bootstrap")
```

The output from this shows the variables selected by lasso for each
bootstrap repeat

``` r
model_lasso_bootstrapped %>%
  as_tibble()
```

    ## # A tibble: 1,129 x 3
    ##    bootstrap variable   estimate
    ##    <chr>     <chr>         <dbl>
    ##  1 1         V2        0.122    
    ##  2 1         V6        0.0208   
    ##  3 1         V9       -0.0578   
    ##  4 1         V11      -0.0933   
    ##  5 1         V14      -0.0758   
    ##  6 1         V15       0.0746   
    ##  7 1         V16      -0.0788   
    ##  8 1         V17      -0.0000480
    ##  9 1         V18      -0.0399   
    ## 10 1         V21      -0.00269  
    ## # ... with 1,119 more rows

We can calculate stability for each variable by the number of times it
was selected across bootstraps.

``` r
model_lasso_bootstrapped %>%
  group_by(variable) %>%
  summarise(stability = (n()/bootstrap_n) * 100) %>%
  arrange(desc(stability))
```

    ## # A tibble: 289 x 2
    ##    variable   stability
    ##    <chr>          <dbl>
    ##  1 causal_105       100
    ##  2 causal_159       100
    ##  3 causal_217       100
    ##  4 causal_237       100
    ##  5 causal_287       100
    ##  6 causal_292       100
    ##  7 causal_32        100
    ##  8 causal_66        100
    ##  9 V203             100
    ## 10 V77              100
    ## # ... with 279 more rows

## Permutation

To identify a null threshold, first we must permute the outcome.

By permuting the outcome variable we sever all ties between the outcome
and the explanatory variables. We might want to conduct this 5 times.

We can then apply our bootstrap function to each one of these 5 permuted
datasets. We might perform 3 bootstrap samples for each of the 5
permuted datasets for this example (this is typically 20 bootstraps for
each of 5 permutations in reality). The model would then be applied to
each dataset.

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

The goal of the *stabiliser* package is to provide a flexible method of
applying stability selection (Meinshausen and Buhlmann, 2010) with
various model types, and the framework for triangulating the results for
multiple models (Lima et al., 2021).

-   `stabilise()` performs stability selection on a range of models to
    identify causal models.
-   `triangulate()` identifies which variables are most likely to be
    causal across all models.
-   `stab_plot()` allows visualisation of either `stabilise()` or
    `triangulate()` outputs.

To attempt to identify which variables are truly “causal” in this
dataset using a selection stability approach, use the `stabilise()`
function as follows:

``` r
stab_output <- stabilise(outcome = "outcome", data = df_signal, models = c("mbic"), type = "linear")
```

Access the stability (percentage of bootstrap resamples where a given
variable was selected by a given model) results for elastic net as
follows:

``` r
stab_output$mbic$stability
```

    ## # A tibble: 301 x 7
    ##    variable   mean_coefficient ci_lower ci_upper bootstrap_p stability stable
    ##    <chr>                 <dbl>    <dbl>    <dbl>       <dbl>     <dbl> <chr> 
    ##  1 causal_287            0.397    0.268    0.504           0       100 *     
    ##  2 causal_237            0.396    0.287    0.513           0        99 *     
    ##  3 causal_159            0.352    0.255    0.438           0        98 *     
    ##  4 causal_292            0.407    0.291    0.522           0        98 *     
    ##  5 causal_105            0.323    0.231    0.416           0        89 *     
    ##  6 causal_66             0.331    0.239    0.431           0        89 *     
    ##  7 causal_217            0.339    0.249    0.431           0        83 *     
    ##  8 causal_32             0.289    0.221    0.379           0        71 *     
    ##  9 V115                 -0.225   -0.275   -0.188           0        20 <NA>  
    ## 10 V234                 -0.219   -0.247   -0.195           0         8 <NA>  
    ## # ... with 291 more rows

This ranks the variables by stability, and displays the mean
coefficients, 95% confidence interval and bootstrap p-value. It also
displays whether the variable is deemed “stable”.

By default, this implements an elastic net algorithm over a number of
bootstrap resamples of the dataset (200 resamples for small datasets).
The stability of each variable is then calculated as the proportion of
bootstrap repeats where that variable is selected in the model.

`stabilise()` also permutes the outcome several times (10 by default for
small datasets) and performs the same process on each permuted dataset
(20 bootstrap resamples for each by default).

This allows a permutation threshold to be calculated. Variables with a
non-permuted stability % above this threshold are deemed “stable” as
they were selected in a higher proportion of bootstrap resamples than in
the permuted datasets, where we know there is no association between
variables and the outcome.

The permutation threshold is available as follows:

``` r
stab_output$mbic$perm_thresh
```

    ## [1] 31

The *stabiliser* package allows multiple models to be run
simultaneously. Just select the models you wish to run in the “models”
argument.

MCP is omitted here for speed. To include it, just add it to the list of
models using: models = c(“mbic”, “lasso”, “mcp”)

``` r
stab_output <- stabilise(outcome = "outcome", data = df_signal, models = c("mbic", "lasso"), type = "linear")
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

    ## # A tibble: 2 x 4
    ##   model_stability tp_stability fp_stability total_selected_stability
    ##   <chr>                  <int>        <int>                    <int>
    ## 1 lasso                      8            0                        8
    ## 2 mbic                       8            0                        8

Compare this with the non-stability approach

``` r
conventional_results %>%
  left_join(stability_results, by = c("model" = "model_stability"))
```

    ## # A tibble: 4 x 7
    ##   model      tp    fp total_selected tp_stability fp_stability total_selected_s~
    ##   <chr>   <int> <int>          <int>        <int>        <int>             <int>
    ## 1 lasso       8    43             51            8            0                 8
    ## 2 mbic        8     0              8            8            0                 8
    ## 3 mcp         8     4             12           NA           NA                NA
    ## 4 prefil~     8    15             23           NA           NA                NA

# Triangulation

Our confidence that a given variable is truly associated with a given
outcome might be increased if it is identified in multiple model types.

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
    ##  1 causal_287     100             0 *     
    ##  2 causal_292     100             0 *     
    ##  3 causal_159      99.5           0 *     
    ##  4 causal_237      99             0 *     
    ##  5 causal_105      94             0 *     
    ##  6 causal_217      91             0 *     
    ##  7 causal_66       90             0 *     
    ##  8 causal_32       77.5           0 *     
    ##  9 V115            54.5           0 <NA>  
    ## 10 V79             48.5           0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 61.5

``` r
stab_plot(triangulated_stability)
```

    ## $combi

![](README_files/figure-gfm/unnamed-chunk-41-1.png)<!-- -->

## No signal datasets

We can now return to our original dataset that we simulated to have no
signal.

Our conventional approach performed relatively poorly, selecting the
following variables as being significantly associated with the outcome
variable.

``` r
prefiltration_results %>%
  tidy()
```

    ## # A tibble: 6 x 5
    ##   term        estimate std.error statistic p.value
    ##   <chr>          <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 (Intercept)  -0.0603    0.0714    -0.844 0.400  
    ## 2 X18          -0.171     0.0660    -2.59  0.0105 
    ## 3 X51          -0.161     0.0697    -2.31  0.0220 
    ## 4 X70          -0.100     0.0719    -1.39  0.166  
    ## 5 X110          0.240     0.0735     3.26  0.00131
    ## 6 X112          0.135     0.0739     1.82  0.0696

The `stabilise()` and `triangulate()` functions from the *stabiliser*
package can be used to perform stability selection with multiple models,
and utilise a robust threhsold to identify which variables are
associated with the outcome.

``` r
stab_output_no_signal <- stabilise(outcome = "outcome", data = df_no_signal, models = c("mbic", "lasso"), type = "linear")
triangulated_output_no_signal <- triangulate(stab_output_no_signal)
```

The following table includes all variables selected from the dataset
with no signal.

``` r
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
