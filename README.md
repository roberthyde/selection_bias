
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
    ##         X1       X2     X3     X4     X5     X6     X7     X8     X9     X10
    ##      <dbl>    <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>   <dbl>
    ##  1  0.980   1.51    -0.500 -1.62   2.84   0.785 -0.189 -0.988 -2.79  -0.737 
    ##  2  1.05    0.829   -0.571 -0.152  0.589 -0.252  0.494  0.966  0.370  0.0823
    ##  3  0.953   0.134   -0.806 -0.720  0.747  1.08  -1.12   1.24  -0.169 -0.300 
    ##  4 -1.07   -0.548   -1.53   0.553  0.177 -2.09  -1.42  -0.989 -0.276 -0.489 
    ##  5 -0.485   0.116    0.657  0.122 -0.430  0.410 -0.914 -0.951  1.17   0.151 
    ##  6  0.909   0.253    0.927 -0.627 -1.95  -0.951  1.39  -0.172  1.90  -0.198 
    ##  7  1.08    0.00664 -1.86  -2.01   0.701 -0.634 -0.897  0.728  0.593  0.169 
    ##  8  1.16    0.360   -0.683  0.424  0.171 -0.711  0.524 -1.05  -1.52  -1.56  
    ##  9 -3.08   -0.131    0.260 -0.663 -0.265 -0.100 -0.190 -0.717 -0.599  0.458 
    ## 10 -0.0842 -1.32     0.120  0.101 -0.805 -0.170 -0.175 -3.34   1.74   0.632 
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
    ##  1   0.822
    ##  2  -0.406
    ##  3  -0.993
    ##  4  -0.670
    ##  5  -0.182
    ##  6  -0.345
    ##  7   0.138
    ##  8  -1.74 
    ##  9  -1.46 
    ## 10  -0.153
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
    ##    outcome      X1       X2     X3     X4     X5     X6     X7     X8     X9
    ##      <dbl>   <dbl>    <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ##  1   0.822  0.980   1.51    -0.500 -1.62   2.84   0.785 -0.189 -0.988 -2.79 
    ##  2  -0.406  1.05    0.829   -0.571 -0.152  0.589 -0.252  0.494  0.966  0.370
    ##  3  -0.993  0.953   0.134   -0.806 -0.720  0.747  1.08  -1.12   1.24  -0.169
    ##  4  -0.670 -1.07   -0.548   -1.53   0.553  0.177 -2.09  -1.42  -0.989 -0.276
    ##  5  -0.182 -0.485   0.116    0.657  0.122 -0.430  0.410 -0.914 -0.951  1.17 
    ##  6  -0.345  0.909   0.253    0.927 -0.627 -1.95  -0.951  1.39  -0.172  1.90 
    ##  7   0.138  1.08    0.00664 -1.86  -2.01   0.701 -0.634 -0.897  0.728  0.593
    ##  8  -1.74   1.16    0.360   -0.683  0.424  0.171 -0.711  0.524 -1.05  -1.52 
    ##  9  -1.46  -3.08   -0.131    0.260 -0.663 -0.265 -0.100 -0.190 -0.717 -0.599
    ## 10  -0.153 -0.0842 -1.32     0.120  0.101 -0.805 -0.170 -0.175 -3.34   1.74 
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

    ## # A tibble: 23 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable   1       2.92e-17   3.42e16  0     
    ##  2 X5       variable   0.0968  6.71e- 2   1.44e 0  0.150 
    ##  3 X24      variable   0.112   7.08e- 2   1.59e 0  0.114 
    ##  4 X29      variable   0.130   7.38e- 2   1.76e 0  0.0805
    ##  5 X35      variable   0.110   6.53e- 2   1.69e 0  0.0929
    ##  6 X36      variable   0.0964  6.40e- 2   1.51e 0  0.133 
    ##  7 X41      variable  -0.109   6.76e- 2  -1.61e 0  0.109 
    ##  8 X42      variable  -0.169   6.91e- 2  -2.45e 0  0.0151
    ##  9 X54      variable  -0.113   7.16e- 2  -1.58e 0  0.117 
    ## 10 X58      variable   0.125   6.68e- 2   1.88e 0  0.0621
    ## # ... with 13 more rows

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

    ## # A tibble: 7 x 5
    ##   variable estimate std.error statistic p.value
    ##   <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 X29         0.146    0.0695      2.10 0.0374 
    ## 2 X35         0.120    0.0604      1.99 0.0477 
    ## 3 X67         0.190    0.0661      2.87 0.00454
    ## 4 X83         0.126    0.0598      2.11 0.0367 
    ## 5 X92        -0.147    0.0681     -2.16 0.0321 
    ## 6 X101        0.176    0.0643      2.74 0.00673
    ## 7 X121       -0.145    0.0612     -2.36 0.0193

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
    ## 1 (Intercept)  -0.0732    0.0616     -1.19 2.36e- 1
    ## 2 causal_37     0.420     0.0653      6.43 5.13e-10
    ## 3 causal_65     0.342     0.0606      5.64 3.92e- 8
    ## 4 causal_98     0.478     0.0596      8.01 2.77e-14
    ## 5 causal_133    0.328     0.0624      5.25 2.94e- 7
    ## 6 causal_173    0.453     0.0612      7.40 1.46e-12
    ## 7 causal_200    0.334     0.0657      5.08 6.71e- 7
    ## 8 causal_214    0.399     0.0651      6.13 2.85e- 9
    ## 9 causal_243    0.433     0.0619      6.99 1.93e-11

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

    ## # A tibble: 16 x 5
    ##    variable   estimate std.error statistic  p.value
    ##    <chr>         <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 causal_37     0.389    0.0629      6.18 2.32e- 9
    ##  2 V64           0.114    0.0537      2.12 3.50e- 2
    ##  3 causal_65     0.243    0.0603      4.03 7.23e- 5
    ##  4 V66           0.121    0.0612      1.98 4.86e- 2
    ##  5 causal_98     0.457    0.0573      7.98 4.18e-14
    ##  6 V132          0.138    0.0607      2.27 2.38e- 2
    ##  7 causal_133    0.284    0.0610      4.65 5.12e- 6
    ##  8 V144          0.122    0.0571      2.13 3.37e- 2
    ##  9 causal_173    0.379    0.0592      6.39 6.99e-10
    ## 10 causal_200    0.328    0.0629      5.21 3.82e- 7
    ## 11 V202          0.130    0.0597      2.17 3.06e- 2
    ## 12 causal_214    0.379    0.0617      6.15 2.78e- 9
    ## 13 causal_243    0.373    0.0612      6.10 3.71e- 9
    ## 14 V272          0.139    0.0641      2.17 3.08e- 2
    ## 15 V297          0.116    0.0583      2.00 4.69e- 2
    ## 16 V299         -0.125    0.0591     -2.11 3.58e- 2

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
    ## 1 lasso             8    14             22
    ## 2 mbic              8     0              8
    ## 3 mcp               8     2             10
    ## 4 prefiltration     8     8             16

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
    ##  1 <split [300/110]> Bootstrap01
    ##  2 <split [300/110]> Bootstrap02
    ##  3 <split [300/115]> Bootstrap03
    ##  4 <split [300/115]> Bootstrap04
    ##  5 <split [300/106]> Bootstrap05
    ##  6 <split [300/107]> Bootstrap06
    ##  7 <split [300/109]> Bootstrap07
    ##  8 <split [300/121]> Bootstrap08
    ##  9 <split [300/116]> Bootstrap09
    ## 10 <split [300/113]> Bootstrap10

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
    ##    outcome      V1      V2     V3     V4      V5      V6     V7     V8     V9
    ##      <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>  <dbl>
    ##  1   -4.22  1.14    1.08    1.69  -0.816 -0.0547 -0.0661 -0.107 -0.521 -0.627
    ##  2   -4.22  1.14    1.08    1.69  -0.816 -0.0547 -0.0661 -0.107 -0.521 -0.627
    ##  3   -3.77  0.877   1.66    0.166 -0.337  0.426   0.981  -0.487 -0.389 -0.821
    ##  4   -3.77  0.877   1.66    0.166 -0.337  0.426   0.981  -0.487 -0.389 -0.821
    ##  5   -3.20  0.907   1.48   -0.704 -2.16  -1.14    1.23    0.616  1.62  -1.11 
    ##  6   -3.20  0.907   1.48   -0.704 -2.16  -1.14    1.23    0.616  1.62  -1.11 
    ##  7   -3.08  0.953   0.850   1.08  -1.13  -0.487  -0.218   1.22  -1.31  -1.49 
    ##  8   -2.98 -0.0508 -0.180   0.256 -0.486  0.183   0.575  -0.320 -0.577 -2.31 
    ##  9   -2.98 -1.60   -1.16    1.01  -0.121 -1.45   -1.11    1.74   0.337 -0.987
    ## 10   -2.96 -0.694  -0.0820 -0.749  0.902 -0.659   1.05    0.756  1.15  -1.81 
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, causal_37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
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
model_lasso_bootstrapped
```

    ##      bootstrap   variable      estimate
    ## 1            1         V1 -2.383687e-03
    ## 2            1         V2 -3.670307e-03
    ## 3            1         V3  1.176783e-01
    ## 4            1         V4  1.301705e-01
    ## 5            1         V7  3.356030e-02
    ## 6            1        V12  3.635317e-03
    ## 7            1        V13 -4.540059e-02
    ## 8            1        V15  2.366895e-01
    ## 9            1        V16 -2.933762e-02
    ## 10           1        V20  7.366131e-02
    ## 11           1        V25  1.562734e-02
    ## 12           1        V27  5.304032e-02
    ## 13           1        V28 -1.083662e-01
    ## 14           1        V29 -4.952049e-02
    ## 15           1        V32 -2.445304e-02
    ## 16           1        V34 -7.948455e-02
    ## 17           1  causal_37  4.160727e-01
    ## 18           1        V38 -6.625022e-02
    ## 19           1        V40 -3.207347e-02
    ## 20           1        V41 -1.226839e-01
    ## 21           1        V43  4.508787e-02
    ## 22           1        V44  2.713802e-02
    ## 23           1        V45  9.230616e-03
    ## 24           1        V47  7.801516e-03
    ## 25           1        V50 -4.420169e-02
    ## 26           1        V54 -8.800739e-02
    ## 27           1        V56 -2.726264e-02
    ## 28           1        V57  8.535021e-02
    ## 29           1        V58  1.225780e-01
    ## 30           1        V60 -3.580107e-02
    ## 31           1        V61 -1.322181e-01
    ## 32           1        V64  9.910597e-02
    ## 33           1  causal_65  1.507816e-01
    ## 34           1        V66  1.389551e-01
    ## 35           1        V72 -1.290511e-01
    ## 36           1        V73  9.228831e-02
    ## 37           1        V79 -7.885916e-02
    ## 38           1        V81 -3.901520e-02
    ## 39           1        V86  1.921823e-01
    ## 40           1        V89 -4.438192e-02
    ## 41           1        V95 -1.801005e-02
    ## 42           1        V96  1.143699e-01
    ## 43           1        V97  4.097718e-02
    ## 44           1  causal_98  4.376542e-01
    ## 45           1       V101  2.411081e-02
    ## 46           1       V102 -1.173418e-01
    ## 47           1       V105 -1.740732e-02
    ## 48           1       V109 -6.449952e-02
    ## 49           1       V111  5.845206e-02
    ## 50           1       V112  1.127991e-01
    ## 51           1       V113 -1.104705e-01
    ## 52           1       V114  3.130867e-02
    ## 53           1       V119 -1.064559e-01
    ## 54           1       V121 -1.440621e-02
    ## 55           1       V123 -1.927316e-02
    ## 56           1       V132  1.639302e-01
    ## 57           1 causal_133  2.506440e-01
    ## 58           1       V138  3.857285e-02
    ## 59           1       V139 -1.889011e-01
    ## 60           1       V140  1.237009e-01
    ## 61           1       V143 -2.410101e-02
    ## 62           1       V144  1.450034e-01
    ## 63           1       V147  4.616737e-02
    ## 64           1       V148 -6.260092e-02
    ## 65           1       V149  1.750455e-02
    ## 66           1       V150 -1.932706e-02
    ## 67           1       V153  3.198533e-02
    ## 68           1       V154  1.433117e-01
    ## 69           1       V155 -3.093929e-02
    ## 70           1       V165 -6.513997e-02
    ## 71           1       V169 -6.921092e-02
    ## 72           1       V170 -4.705441e-02
    ## 73           1 causal_173  3.815835e-01
    ## 74           1       V174 -8.709639e-02
    ## 75           1       V176 -1.106697e-01
    ## 76           1       V178 -2.875721e-02
    ## 77           1       V180  8.440043e-02
    ## 78           1       V181  9.180058e-02
    ## 79           1       V182 -1.331962e-01
    ## 80           1       V184  4.421312e-03
    ## 81           1       V185  9.757211e-02
    ## 82           1       V186 -8.659504e-03
    ## 83           1       V188  4.397761e-02
    ## 84           1       V189 -1.036268e-01
    ## 85           1       V191  6.530593e-03
    ## 86           1       V195  4.676621e-02
    ## 87           1       V197  2.524189e-02
    ## 88           1       V198  1.391818e-02
    ## 89           1       V199 -2.344304e-02
    ## 90           1 causal_200  3.825225e-01
    ## 91           1       V201 -1.902246e-01
    ## 92           1       V202  1.853867e-04
    ## 93           1       V203 -5.941092e-02
    ## 94           1       V210 -1.521059e-02
    ## 95           1 causal_214  3.254636e-01
    ## 96           1       V215 -2.944831e-02
    ## 97           1       V216 -3.753920e-02
    ## 98           1       V220 -4.109677e-03
    ## 99           1       V221  1.513405e-02
    ## 100          1       V224 -1.088394e-01
    ## 101          1       V225 -2.272488e-02
    ## 102          1       V226  1.108228e-01
    ## 103          1       V227 -1.269881e-02
    ## 104          1       V228  2.882491e-02
    ## 105          1       V233  2.257120e-01
    ## 106          1       V234 -1.637631e-02
    ## 107          1       V235 -1.858531e-02
    ## 108          1       V236 -5.408731e-02
    ## 109          1       V237  8.358143e-02
    ## 110          1       V239  1.156604e-01
    ## 111          1       V241  6.597313e-02
    ## 112          1 causal_243  3.144059e-01
    ## 113          1       V246  7.701341e-02
    ## 114          1       V248 -3.895146e-02
    ## 115          1       V249 -8.542240e-02
    ## 116          1       V252 -2.937029e-02
    ## 117          1       V254  5.310232e-03
    ## 118          1       V258  2.895197e-02
    ## 119          1       V261 -2.378872e-02
    ## 120          1       V262 -6.721193e-02
    ## 121          1       V263 -2.369272e-02
    ## 122          1       V269 -1.070040e-01
    ## 123          1       V270  1.051526e-01
    ## 124          1       V273 -8.021325e-03
    ## 125          1       V274  1.682097e-02
    ## 126          1       V276  4.709919e-02
    ## 127          1       V280  1.008765e-01
    ## 128          1       V282 -1.619460e-01
    ## 129          1       V284 -8.897961e-03
    ## 130          1       V285 -5.707660e-02
    ## 131          1       V286 -5.712186e-02
    ## 132          1       V287 -2.002040e-02
    ## 133          1       V289 -3.074557e-02
    ## 134          1       V291 -9.278164e-02
    ## 135          1       V292 -9.970404e-02
    ## 136          1       V293  1.508861e-01
    ## 137          1       V296 -4.961764e-02
    ## 138          1       V297  3.403860e-02
    ## 139          1       V300 -5.889902e-02
    ## 140          2         V2 -9.655366e-02
    ## 141          2         V5  7.551584e-02
    ## 142          2        V12  8.730627e-02
    ## 143          2        V14 -3.185060e-03
    ## 144          2        V15  9.270355e-02
    ## 145          2        V20  1.390001e-01
    ## 146          2        V25  1.626289e-02
    ## 147          2        V26 -2.361398e-01
    ## 148          2        V27  2.834845e-01
    ## 149          2        V28 -2.381417e-02
    ## 150          2        V29 -1.659529e-01
    ## 151          2        V31 -5.576129e-02
    ## 152          2        V34  2.867195e-02
    ## 153          2        V36 -1.251497e-02
    ## 154          2  causal_37  3.792599e-01
    ## 155          2        V39  5.019568e-02
    ## 156          2        V45  2.616178e-02
    ## 157          2        V46 -5.322673e-03
    ## 158          2        V47  1.152410e-02
    ## 159          2        V48  1.456102e-02
    ## 160          2        V53  9.694535e-02
    ## 161          2        V55 -9.219866e-02
    ## 162          2        V60 -6.661065e-02
    ## 163          2        V61 -8.943131e-02
    ## 164          2  causal_65  2.393224e-01
    ## 165          2        V66  7.928959e-02
    ## 166          2        V67 -1.063344e-01
    ## 167          2        V68  1.264023e-01
    ## 168          2        V71  1.830527e-02
    ## 169          2        V75 -1.642579e-01
    ## 170          2        V76 -1.724407e-02
    ## 171          2        V77  7.590071e-02
    ## 172          2        V79 -3.116425e-02
    ## 173          2        V80  1.434452e-02
    ## 174          2        V82  2.864383e-02
    ## 175          2        V86  4.509476e-02
    ## 176          2        V88  7.437871e-02
    ## 177          2        V91 -1.906509e-01
    ## 178          2        V92 -1.486985e-02
    ## 179          2        V93 -2.396742e-02
    ## 180          2  causal_98  4.042343e-01
    ## 181          2        V99  5.622281e-02
    ## 182          2       V102 -1.407722e-01
    ## 183          2       V104  5.295726e-02
    ## 184          2       V105 -2.571403e-03
    ## 185          2       V106 -3.570870e-03
    ## 186          2       V107  1.359252e-01
    ## 187          2       V109 -3.830059e-02
    ## 188          2       V110 -1.425442e-02
    ## 189          2       V111  3.768041e-02
    ## 190          2       V113 -4.102868e-02
    ## 191          2       V118 -7.613368e-02
    ## 192          2       V120  7.198019e-02
    ## 193          2       V121  4.792649e-02
    ## 194          2       V122 -1.695195e-01
    ## 195          2       V123 -5.329209e-02
    ## 196          2       V125 -1.645794e-01
    ## 197          2       V128 -4.625530e-02
    ## 198          2 causal_133  1.781046e-01
    ## 199          2       V134 -3.766113e-02
    ## 200          2       V135 -4.696788e-02
    ## 201          2       V138 -9.578249e-02
    ## 202          2       V139 -8.584629e-02
    ## 203          2       V140  9.116861e-03
    ## 204          2       V144  2.730776e-01
    ## 205          2       V147  5.575755e-03
    ## 206          2       V149  8.171019e-02
    ## 207          2       V153  1.832439e-02
    ## 208          2       V154  4.801348e-02
    ## 209          2       V157 -1.651942e-03
    ## 210          2       V158 -4.678272e-02
    ## 211          2       V161  2.381726e-02
    ## 212          2       V162  3.413904e-02
    ## 213          2       V163  1.942705e-02
    ## 214          2       V169  8.819940e-02
    ## 215          2       V170 -1.949192e-01
    ## 216          2       V171 -6.697625e-02
    ## 217          2       V172  4.609982e-02
    ## 218          2 causal_173  2.956540e-01
    ## 219          2       V175 -2.255840e-02
    ## 220          2       V176 -4.607815e-02
    ## 221          2       V179  4.125816e-03
    ## 222          2       V182 -4.331549e-02
    ## 223          2       V183  2.035203e-02
    ## 224          2       V185  3.542218e-02
    ## 225          2       V188  2.479814e-03
    ## 226          2       V191 -6.269876e-02
    ## 227          2       V195  9.137563e-02
    ## 228          2       V196  4.233397e-02
    ## 229          2       V197  1.761912e-02
    ## 230          2       V198 -4.830716e-03
    ## 231          2       V199 -2.162110e-02
    ## 232          2 causal_200  2.122990e-01
    ## 233          2       V202  4.183530e-02
    ## 234          2       V206 -2.851886e-02
    ## 235          2       V211  1.148588e-01
    ## 236          2 causal_214  2.370197e-01
    ## 237          2       V216 -1.241234e-01
    ## 238          2       V217 -1.491850e-02
    ## 239          2       V219  7.839817e-02
    ## 240          2       V224 -6.197327e-02
    ## 241          2       V226  2.041686e-02
    ## 242          2       V230  5.177569e-02
    ## 243          2       V231 -1.586930e-01
    ## 244          2       V233  2.513104e-03
    ## 245          2       V234 -5.385859e-03
    ## 246          2       V235 -1.244674e-01
    ## 247          2       V239  5.947117e-02
    ## 248          2       V240 -1.565860e-01
    ## 249          2       V241  1.790512e-01
    ## 250          2 causal_243  3.299751e-01
    ## 251          2       V245 -8.905559e-02
    ## 252          2       V246  4.576680e-02
    ## 253          2       V248 -2.535466e-02
    ## 254          2       V252 -9.829170e-02
    ## 255          2       V253 -1.107832e-02
    ## 256          2       V254 -7.942646e-03
    ## 257          2       V257 -1.218078e-01
    ## 258          2       V259 -1.271588e-01
    ## 259          2       V261 -1.002593e-04
    ## 260          2       V262 -1.566265e-01
    ## 261          2       V263  1.839225e-02
    ## 262          2       V264  2.819354e-02
    ## 263          2       V265  3.338008e-02
    ## 264          2       V266 -5.477896e-02
    ## 265          2       V267  6.050023e-02
    ## 266          2       V269 -2.563139e-02
    ## 267          2       V272  1.368500e-02
    ## 268          2       V273 -1.485856e-02
    ## 269          2       V280 -2.011645e-02
    ## 270          2       V282 -1.809618e-01
    ## 271          2       V284  7.096065e-02
    ## 272          2       V285  5.294244e-03
    ## 273          2       V288 -3.108376e-02
    ## 274          2       V289 -1.007729e-01
    ## 275          2       V291  1.194563e-02
    ## 276          2       V294  1.877374e-01
    ## 277          2       V297  1.149741e-01
    ## 278          3         V4  5.229686e-02
    ## 279          3         V6  1.296917e-01
    ## 280          3         V7  5.017908e-03
    ## 281          3         V8 -4.133449e-02
    ## 282          3         V9 -1.860387e-02
    ## 283          3        V15  1.192030e-01
    ## 284          3        V18 -5.463362e-03
    ## 285          3        V29 -9.860682e-02
    ## 286          3        V34 -7.356726e-03
    ## 287          3        V36  1.615295e-02
    ## 288          3  causal_37  2.240732e-01
    ## 289          3        V40 -7.206269e-02
    ## 290          3        V41 -1.080294e-02
    ## 291          3        V45  1.922941e-02
    ## 292          3        V48  2.062612e-02
    ## 293          3        V53  2.047798e-02
    ## 294          3        V60 -1.917785e-02
    ## 295          3        V61 -1.093283e-02
    ## 296          3        V64  5.655495e-02
    ## 297          3  causal_65  2.283431e-01
    ## 298          3        V66  1.663785e-02
    ## 299          3        V68  4.818085e-02
    ## 300          3        V75 -8.786976e-03
    ## 301          3        V80  6.527759e-02
    ## 302          3        V86  7.259814e-02
    ## 303          3        V89 -2.485922e-02
    ## 304          3        V94 -5.807175e-02
    ## 305          3        V96  1.539414e-02
    ## 306          3  causal_98  4.337718e-01
    ## 307          3       V100 -9.693715e-03
    ## 308          3       V108 -5.696737e-03
    ## 309          3       V109 -9.722609e-02
    ## 310          3       V113 -7.915096e-02
    ## 311          3       V123 -1.618699e-02
    ## 312          3       V124 -5.044960e-03
    ## 313          3       V130  3.141781e-02
    ## 314          3 causal_133  3.378316e-01
    ## 315          3       V139 -1.205862e-02
    ## 316          3       V147  3.447615e-02
    ## 317          3       V151  1.034306e-02
    ## 318          3       V159  4.503106e-02
    ## 319          3       V161 -8.146318e-05
    ## 320          3       V163  1.481348e-02
    ## 321          3       V165 -1.392412e-01
    ## 322          3       V170 -5.692797e-02
    ## 323          3 causal_173  3.092926e-01
    ## 324          3       V177  3.913158e-02
    ## 325          3       V180  2.312946e-02
    ## 326          3       V181  3.032747e-03
    ## 327          3       V183  4.564344e-02
    ## 328          3       V184  7.679441e-02
    ## 329          3       V186 -4.065596e-03
    ## 330          3       V189 -1.179272e-02
    ## 331          3       V193 -1.767749e-02
    ## 332          3       V199 -7.781011e-02
    ## 333          3 causal_200  3.584713e-01
    ## 334          3       V202  8.715818e-02
    ## 335          3       V205  6.296467e-03
    ## 336          3       V211  1.904901e-03
    ## 337          3 causal_214  2.951509e-01
    ## 338          3       V216 -8.900097e-02
    ## 339          3       V222 -8.219577e-02
    ## 340          3       V230  2.836197e-02
    ## 341          3       V231 -6.917009e-02
    ## 342          3       V233  1.414335e-01
    ## 343          3       V237  3.689728e-02
    ## 344          3       V238  2.743782e-03
    ## 345          3       V239  1.053915e-01
    ## 346          3       V240 -6.076460e-02
    ## 347          3 causal_243  1.660328e-01
    ## 348          3       V244  1.542144e-01
    ## 349          3       V247 -7.929278e-02
    ## 350          3       V249 -9.904208e-02
    ## 351          3       V252 -3.017045e-02
    ## 352          3       V256 -2.046108e-02
    ## 353          3       V257 -1.854906e-02
    ## 354          3       V261 -7.044619e-02
    ## 355          3       V272  1.284988e-01
    ## 356          3       V274  3.465823e-02
    ## 357          3       V275  3.870515e-03
    ## 358          3       V278 -1.705805e-02
    ## 359          3       V284 -6.703892e-03
    ## 360          3       V287 -2.086958e-02
    ## 361          3       V289 -1.633733e-02
    ## 362          3       V290 -7.524175e-02
    ## 363          3       V293  2.258588e-03
    ## 364          3       V295  2.191095e-02
    ## 365          3       V297  1.058753e-01
    ## 366          3       V299 -3.731639e-02
    ## 367          3       V300 -8.190424e-02
    ## 368          4         V2 -8.417087e-02
    ## 369          4         V3  3.764815e-02
    ## 370          4         V4  1.097604e-02
    ## 371          4         V5  5.021931e-02
    ## 372          4         V8  3.833841e-02
    ## 373          4         V9  7.633734e-02
    ## 374          4        V10  3.911592e-02
    ## 375          4        V12  1.076977e-02
    ## 376          4        V16 -4.828728e-02
    ## 377          4        V20  1.894487e-02
    ## 378          4        V26 -4.632089e-02
    ## 379          4        V27  2.287837e-02
    ## 380          4        V29 -7.083793e-02
    ## 381          4        V34  5.704678e-02
    ## 382          4  causal_37  2.442015e-01
    ## 383          4        V50  5.024420e-02
    ## 384          4        V51 -3.586940e-02
    ## 385          4        V57  9.538090e-02
    ## 386          4        V58  2.530612e-03
    ## 387          4        V60 -8.261974e-03
    ## 388          4        V61 -1.717629e-01
    ## 389          4        V62  1.377858e-01
    ## 390          4        V64  8.319209e-03
    ## 391          4  causal_65  1.585570e-01
    ## 392          4        V66  1.735943e-01
    ## 393          4        V67 -1.836642e-02
    ## 394          4        V68  5.988186e-02
    ## 395          4        V71  1.066570e-02
    ## 396          4        V75 -7.288855e-03
    ## 397          4        V78  3.733145e-02
    ## 398          4        V82  5.403945e-02
    ## 399          4        V86  4.486323e-02
    ## 400          4        V87  5.423566e-02
    ## 401          4        V92 -1.104153e-02
    ## 402          4        V93 -1.185345e-02
    ## 403          4        V95 -3.729740e-02
    ## 404          4        V96  1.334843e-01
    ## 405          4        V97  1.469577e-01
    ## 406          4  causal_98  3.979089e-01
    ## 407          4       V103 -7.015373e-03
    ## 408          4       V113 -6.217759e-02
    ## 409          4       V120  5.618852e-02
    ## 410          4       V121 -2.192264e-02
    ## 411          4       V123 -4.998547e-03
    ## 412          4       V127  1.171738e-02
    ## 413          4       V131  4.056664e-03
    ## 414          4       V132  2.455121e-03
    ## 415          4 causal_133  2.542275e-01
    ## 416          4       V136  1.242514e-01
    ## 417          4       V139 -2.674864e-01
    ## 418          4       V141 -3.991565e-02
    ## 419          4       V143  7.038571e-03
    ## 420          4       V144  7.547605e-02
    ## 421          4       V146  5.838528e-02
    ## 422          4       V148 -6.389109e-02
    ## 423          4       V149  4.014777e-03
    ## 424          4       V154  5.806386e-02
    ## 425          4       V158 -1.521057e-02
    ## 426          4       V163  2.650158e-02
    ## 427          4       V170 -1.291280e-01
    ## 428          4       V172  4.580594e-03
    ## 429          4 causal_173  4.112921e-01
    ## 430          4       V176 -9.809955e-02
    ## 431          4       V182 -1.098812e-01
    ## 432          4       V185  7.253142e-02
    ## 433          4       V190  5.753492e-02
    ## 434          4       V194 -9.397555e-02
    ## 435          4       V195  2.268776e-02
    ## 436          4 causal_200  4.523916e-01
    ## 437          4       V201 -4.147957e-02
    ## 438          4       V202  1.082187e-01
    ## 439          4       V203 -5.593179e-02
    ## 440          4       V205  2.535100e-01
    ## 441          4       V206 -1.192675e-01
    ## 442          4       V211  1.508138e-01
    ## 443          4       V212  1.682955e-01
    ## 444          4 causal_214  2.252539e-01
    ## 445          4       V216 -7.419739e-02
    ## 446          4       V224 -4.967713e-03
    ## 447          4       V225 -2.977164e-02
    ## 448          4       V234  5.544659e-02
    ## 449          4       V238  1.072518e-01
    ## 450          4       V239  2.431659e-02
    ## 451          4       V240 -4.848598e-02
    ## 452          4 causal_243  3.229507e-01
    ## 453          4       V244  1.116926e-01
    ## 454          4       V245 -9.336002e-03
    ## 455          4       V247  2.019909e-02
    ## 456          4       V250  2.843847e-02
    ## 457          4       V251 -1.164466e-01
    ## 458          4       V259 -8.943766e-02
    ## 459          4       V265  1.188822e-01
    ## 460          4       V275 -9.113715e-02
    ## 461          4       V282 -9.183642e-03
    ## 462          4       V285  1.106926e-01
    ## 463          4       V288 -1.801005e-02
    ## 464          4       V292 -1.289910e-02
    ## 465          4       V295  2.208127e-02
    ## 466          5         V4  2.347364e-01
    ## 467          5         V7  2.305753e-02
    ## 468          5        V12  8.049193e-02
    ## 469          5        V13 -5.209363e-02
    ## 470          5        V15  5.699580e-02
    ## 471          5        V19  4.537390e-02
    ## 472          5        V29 -1.357154e-01
    ## 473          5        V31  3.711137e-02
    ## 474          5        V32 -9.788741e-02
    ## 475          5        V34 -2.728244e-02
    ## 476          5  causal_37  2.489834e-01
    ## 477          5        V39 -4.141862e-02
    ## 478          5        V43  5.432594e-02
    ## 479          5        V48  8.511541e-02
    ## 480          5        V51 -1.893927e-02
    ## 481          5        V54 -5.035213e-02
    ## 482          5        V55 -9.840815e-03
    ## 483          5        V61 -1.149030e-01
    ## 484          5        V62 -5.542174e-03
    ## 485          5        V64  8.425573e-02
    ## 486          5  causal_65  1.266598e-01
    ## 487          5        V66  2.458294e-02
    ## 488          5        V67  1.365184e-03
    ## 489          5        V69 -3.304757e-02
    ## 490          5        V75 -6.357725e-02
    ## 491          5        V78  2.488722e-03
    ## 492          5        V79 -7.124286e-03
    ## 493          5        V80  6.630494e-02
    ## 494          5        V82  1.280112e-01
    ## 495          5        V83 -2.688832e-03
    ## 496          5        V87  2.517133e-02
    ## 497          5        V91 -4.016559e-03
    ## 498          5  causal_98  4.610304e-01
    ## 499          5       V103  2.547338e-02
    ## 500          5       V104  4.893730e-03
    ## 501          5       V106  2.292198e-03
    ## 502          5       V109 -1.380997e-02
    ## 503          5       V111  4.954283e-02
    ## 504          5       V114 -7.078852e-02
    ## 505          5       V115  5.367358e-03
    ## 506          5       V118 -5.682361e-02
    ## 507          5       V120  7.834071e-02
    ## 508          5       V123 -1.658784e-01
    ## 509          5       V124 -9.112306e-04
    ## 510          5       V128  1.193691e-02
    ## 511          5       V131  1.088067e-02
    ## 512          5       V132  1.096313e-01
    ## 513          5 causal_133  3.540235e-01
    ## 514          5       V139 -9.852153e-02
    ## 515          5       V140  6.253703e-02
    ## 516          5       V142 -6.722108e-02
    ## 517          5       V145 -1.160426e-02
    ## 518          5       V147  1.352188e-02
    ## 519          5       V149  9.936348e-03
    ## 520          5       V150 -4.464142e-02
    ## 521          5       V153  8.804682e-02
    ## 522          5       V160  7.992629e-02
    ## 523          5       V161  5.784656e-02
    ## 524          5       V164  8.313066e-02
    ## 525          5       V165 -1.962052e-02
    ## 526          5       V168 -1.144213e-01
    ## 527          5       V170 -6.321368e-02
    ## 528          5       V172  6.871210e-02
    ## 529          5 causal_173  3.329475e-01
    ## 530          5       V175 -3.885770e-02
    ## 531          5       V177  5.646303e-02
    ## 532          5       V180  4.186955e-02
    ## 533          5       V181  1.827119e-02
    ## 534          5       V184  1.038762e-01
    ## 535          5       V185  7.698572e-02
    ## 536          5       V186 -8.860305e-02
    ## 537          5       V189 -1.620398e-03
    ## 538          5       V190  5.412753e-03
    ## 539          5       V192  3.618620e-03
    ## 540          5       V193 -3.664546e-03
    ## 541          5       V195  1.319086e-01
    ## 542          5 causal_200  2.821660e-01
    ## 543          5       V202  6.646097e-02
    ## 544          5       V205  6.388197e-02
    ## 545          5       V207 -5.799316e-02
    ## 546          5       V208 -7.052641e-02
    ## 547          5       V211  4.567579e-02
    ## 548          5       V212  2.726945e-02
    ## 549          5 causal_214  3.046277e-01
    ## 550          5       V216 -4.083567e-02
    ## 551          5       V220 -6.643219e-02
    ## 552          5       V224  6.215551e-02
    ## 553          5       V226  3.264372e-03
    ## 554          5       V228  1.173604e-01
    ## 555          5       V229  1.244929e-02
    ## 556          5       V231 -6.741109e-02
    ## 557          5       V232 -7.972779e-03
    ## 558          5       V237  1.299362e-01
    ## 559          5       V239  6.134589e-02
    ## 560          5       V240 -1.113445e-03
    ## 561          5       V242  8.411556e-03
    ## 562          5 causal_243  2.700720e-01
    ## 563          5       V244  9.978253e-02
    ## 564          5       V249 -1.702799e-01
    ## 565          5       V252 -9.546290e-02
    ## 566          5       V253  1.559135e-02
    ## 567          5       V256 -1.068267e-02
    ## 568          5       V258  2.059393e-02
    ## 569          5       V261 -5.457499e-03
    ## 570          5       V264 -1.611749e-02
    ## 571          5       V266 -1.668686e-02
    ## 572          5       V269 -4.949258e-02
    ## 573          5       V271  5.654415e-02
    ## 574          5       V272  4.947316e-02
    ## 575          5       V275  7.308561e-02
    ## 576          5       V279 -4.172986e-02
    ## 577          5       V282 -9.648710e-03
    ## 578          5       V290 -3.357916e-02
    ## 579          5       V291 -8.668240e-02
    ## 580          5       V292 -1.002559e-01
    ## 581          5       V293  1.409295e-02
    ## 582          5       V297  1.063937e-01
    ## 583          6         V1  5.195420e-02
    ## 584          6         V3  8.129998e-03
    ## 585          6         V4  9.233413e-02
    ## 586          6         V7  1.187024e-02
    ## 587          6         V9  2.212042e-02
    ## 588          6        V11  5.130646e-02
    ## 589          6        V16 -1.065538e-02
    ## 590          6        V29 -6.176829e-02
    ## 591          6  causal_37  3.243924e-01
    ## 592          6        V43  2.003562e-02
    ## 593          6        V46  3.427145e-02
    ## 594          6        V47  9.970778e-02
    ## 595          6        V49 -3.209526e-02
    ## 596          6        V51 -1.636544e-01
    ## 597          6        V52  6.129903e-02
    ## 598          6        V55 -7.564191e-02
    ## 599          6        V57  7.445110e-03
    ## 600          6        V61 -7.284060e-02
    ## 601          6        V64  7.013263e-02
    ## 602          6  causal_65  2.223619e-01
    ## 603          6        V66  7.533060e-02
    ## 604          6        V67  2.224982e-02
    ## 605          6        V70 -4.660240e-02
    ## 606          6        V73  6.144434e-02
    ## 607          6        V79 -9.815443e-03
    ## 608          6        V81 -4.238367e-02
    ## 609          6        V82  1.004464e-02
    ## 610          6        V84  7.602912e-02
    ## 611          6        V86  1.909817e-02
    ## 612          6        V90  5.330454e-03
    ## 613          6        V92 -2.651146e-02
    ## 614          6        V93 -3.950195e-04
    ## 615          6  causal_98  3.856605e-01
    ## 616          6       V101  8.359195e-03
    ## 617          6       V102 -7.977974e-02
    ## 618          6       V103 -2.528974e-02
    ## 619          6       V113 -6.223253e-02
    ## 620          6       V114  8.966275e-03
    ## 621          6       V115 -2.536929e-02
    ## 622          6       V120  8.645774e-02
    ## 623          6       V122 -6.721553e-03
    ## 624          6       V123 -2.110375e-02
    ## 625          6 causal_133  2.561705e-01
    ## 626          6       V138 -1.221805e-03
    ## 627          6       V139 -3.232075e-02
    ## 628          6       V141 -7.076172e-03
    ## 629          6       V144  2.179991e-02
    ## 630          6       V146 -2.580530e-02
    ## 631          6       V149  7.321823e-02
    ## 632          6       V154  1.270843e-02
    ## 633          6       V165 -3.032114e-02
    ## 634          6       V168 -6.167803e-02
    ## 635          6       V170 -3.430953e-02
    ## 636          6 causal_173  2.670589e-01
    ## 637          6       V180  8.886534e-02
    ## 638          6       V184  5.842547e-02
    ## 639          6       V186  1.122301e-03
    ## 640          6       V187  1.491534e-02
    ## 641          6       V188  6.022406e-02
    ## 642          6       V193  2.683505e-02
    ## 643          6       V195  7.338684e-02
    ## 644          6 causal_200  2.847025e-01
    ## 645          6       V201 -1.554629e-01
    ## 646          6       V202  1.268913e-01
    ## 647          6       V203 -1.267309e-02
    ## 648          6       V204 -6.814826e-02
    ## 649          6       V205  6.742787e-02
    ## 650          6 causal_214  3.707369e-01
    ## 651          6       V219  2.613976e-02
    ## 652          6       V229  7.216534e-02
    ## 653          6       V233  1.398838e-03
    ## 654          6       V237  1.091477e-01
    ## 655          6       V239  2.705296e-02
    ## 656          6       V240 -1.990624e-02
    ## 657          6 causal_243  3.140890e-01
    ## 658          6       V257 -1.521838e-02
    ## 659          6       V259 -9.068194e-02
    ## 660          6       V270  6.381327e-02
    ## 661          6       V272  1.676210e-01
    ## 662          6       V282 -2.407037e-02
    ## 663          6       V283 -4.443140e-02
    ## 664          6       V285 -4.468289e-02
    ## 665          6       V289 -9.840252e-02
    ## 666          6       V291 -7.667822e-03
    ## 667          6       V293  9.880479e-02
    ## 668          6       V299 -1.114065e-01
    ## 669          7         V2 -1.496032e-01
    ## 670          7         V3  8.118234e-02
    ## 671          7         V4  1.189680e-02
    ## 672          7         V6  1.376443e-02
    ## 673          7        V17  1.105654e-03
    ## 674          7        V19  3.557587e-02
    ## 675          7        V20  2.842984e-02
    ## 676          7        V23  8.295576e-03
    ## 677          7        V27  3.869345e-03
    ## 678          7        V28 -5.198641e-02
    ## 679          7        V31  4.931929e-02
    ## 680          7        V34 -5.034003e-02
    ## 681          7        V35 -5.270163e-02
    ## 682          7        V36 -2.750091e-02
    ## 683          7  causal_37  4.048637e-01
    ## 684          7        V39 -6.214508e-02
    ## 685          7        V49  5.747391e-02
    ## 686          7        V51 -4.012105e-02
    ## 687          7        V52  1.452545e-02
    ## 688          7        V53  1.479967e-01
    ## 689          7        V54  7.673338e-03
    ## 690          7        V64  2.121951e-02
    ## 691          7  causal_65  2.161507e-01
    ## 692          7        V73  9.464479e-04
    ## 693          7        V86  5.045713e-02
    ## 694          7        V91 -9.916066e-03
    ## 695          7        V96  2.048025e-01
    ## 696          7        V97  2.914691e-02
    ## 697          7  causal_98  3.558655e-01
    ## 698          7       V102 -2.548388e-01
    ## 699          7       V105 -3.159273e-02
    ## 700          7       V110  3.324299e-02
    ## 701          7       V112  4.993300e-02
    ## 702          7       V113 -1.864071e-02
    ## 703          7       V123 -2.852440e-02
    ## 704          7       V124 -3.678017e-02
    ## 705          7       V131 -4.508412e-02
    ## 706          7       V132  1.088889e-01
    ## 707          7 causal_133  4.087001e-01
    ## 708          7       V144  1.176614e-01
    ## 709          7       V154  3.952405e-03
    ## 710          7       V166  3.311027e-03
    ## 711          7       V169  2.743961e-03
    ## 712          7       V170 -1.641148e-03
    ## 713          7 causal_173  2.963787e-01
    ## 714          7       V181  3.678726e-02
    ## 715          7       V187  5.190866e-03
    ## 716          7 causal_200  2.276017e-01
    ## 717          7       V201 -4.347716e-02
    ## 718          7       V202  9.069877e-02
    ## 719          7       V203 -1.458193e-02
    ## 720          7       V205  1.909071e-02
    ## 721          7       V208 -5.043569e-02
    ## 722          7       V211  7.942564e-02
    ## 723          7 causal_214  3.400059e-01
    ## 724          7       V216 -8.609438e-02
    ## 725          7       V220 -2.247085e-02
    ## 726          7       V228  5.553567e-02
    ## 727          7       V229  1.097216e-01
    ## 728          7       V231 -8.569428e-04
    ## 729          7       V233  1.637748e-01
    ## 730          7 causal_243  4.065902e-01
    ## 731          7       V244  6.241474e-02
    ## 732          7       V251 -8.755614e-03
    ## 733          7       V253  9.589795e-02
    ## 734          7       V257 -6.741059e-03
    ## 735          7       V260 -4.540507e-02
    ## 736          7       V269 -9.425942e-03
    ## 737          7       V274  2.903028e-02
    ## 738          7       V280  3.249064e-02
    ## 739          7       V287 -8.755878e-04
    ## 740          7       V290 -3.924744e-02
    ## 741          7       V296  3.144674e-03
    ## 742          7       V297  1.121359e-02
    ## 743          8         V1  9.460976e-02
    ## 744          8         V2 -7.683038e-02
    ## 745          8         V3  4.855721e-02
    ## 746          8         V5  8.837953e-02
    ## 747          8         V9  4.257820e-02
    ## 748          8        V11  1.857361e-01
    ## 749          8        V12  5.855378e-02
    ## 750          8        V14  2.544151e-02
    ## 751          8        V18 -2.477284e-02
    ## 752          8        V20  5.046616e-02
    ## 753          8        V26 -1.117603e-01
    ## 754          8        V29 -1.725017e-02
    ## 755          8        V30 -3.438376e-03
    ## 756          8        V31  6.541930e-02
    ## 757          8        V33  2.525591e-02
    ## 758          8  causal_37  3.837971e-01
    ## 759          8        V38 -8.417974e-04
    ## 760          8        V43  3.759407e-02
    ## 761          8        V45  3.726338e-02
    ## 762          8        V50 -1.508408e-02
    ## 763          8        V56  6.130100e-02
    ## 764          8        V58  1.287972e-01
    ## 765          8        V61 -1.876361e-01
    ## 766          8        V64  3.488574e-02
    ## 767          8  causal_65  2.010896e-01
    ## 768          8        V66  1.501872e-01
    ## 769          8        V67  5.940248e-03
    ## 770          8        V68  4.921152e-03
    ## 771          8        V70  1.544608e-02
    ## 772          8        V71  1.046210e-02
    ## 773          8        V75 -4.946409e-02
    ## 774          8        V76 -2.974361e-02
    ## 775          8        V77 -2.125869e-02
    ## 776          8        V80  3.038089e-02
    ## 777          8        V81 -3.640634e-02
    ## 778          8        V82  4.630474e-02
    ## 779          8        V85 -1.014974e-01
    ## 780          8        V86  4.796897e-02
    ## 781          8        V87  1.102918e-02
    ## 782          8        V92 -3.798245e-02
    ## 783          8        V93 -2.001569e-01
    ## 784          8  causal_98  2.922843e-01
    ## 785          8       V100 -1.390578e-02
    ## 786          8       V102 -1.143421e-01
    ## 787          8       V103 -1.715287e-02
    ## 788          8       V104  1.918677e-02
    ## 789          8       V105 -1.012537e-01
    ## 790          8       V107  1.606729e-02
    ## 791          8       V113 -8.329741e-02
    ## 792          8       V116  6.014373e-02
    ## 793          8       V117 -2.226973e-02
    ## 794          8       V120  1.959954e-02
    ## 795          8       V124 -6.390381e-02
    ## 796          8       V125 -4.003461e-02
    ## 797          8       V128 -8.360109e-03
    ## 798          8       V130  1.906770e-02
    ## 799          8       V132  1.743527e-01
    ## 800          8 causal_133  3.305167e-01
    ## 801          8       V144  4.566098e-02
    ## 802          8       V146 -3.658621e-02
    ## 803          8       V148 -2.690733e-02
    ## 804          8       V154  2.217358e-02
    ## 805          8       V155  4.298048e-02
    ## 806          8       V157 -2.654490e-02
    ## 807          8       V162 -3.729332e-03
    ## 808          8       V165 -3.516261e-02
    ## 809          8       V169  7.426386e-03
    ## 810          8 causal_173  1.533497e-01
    ## 811          8       V177  3.596173e-02
    ## 812          8       V182 -7.116260e-02
    ## 813          8       V188 -5.926693e-02
    ## 814          8       V193 -1.588568e-02
    ## 815          8 causal_200  2.337147e-01
    ## 816          8       V201 -2.039258e-01
    ## 817          8       V202  8.648461e-02
    ## 818          8       V203 -2.584339e-02
    ## 819          8       V205  5.573797e-02
    ## 820          8       V208 -7.064165e-02
    ## 821          8       V211  1.649091e-02
    ## 822          8       V212  2.467944e-02
    ## 823          8 causal_214  5.278634e-01
    ## 824          8       V215 -1.441158e-01
    ## 825          8       V217 -1.238123e-02
    ## 826          8       V221  5.301505e-02
    ## 827          8       V222 -1.067925e-01
    ## 828          8       V223 -3.792572e-02
    ## 829          8       V229  1.798488e-01
    ## 830          8       V230  9.364223e-02
    ## 831          8       V232  3.730232e-02
    ## 832          8       V235  5.520600e-03
    ## 833          8       V237  1.785584e-02
    ## 834          8       V239  6.307135e-02
    ## 835          8       V240 -6.505871e-02
    ## 836          8 causal_243  3.328508e-01
    ## 837          8       V252 -4.637383e-02
    ## 838          8       V254 -1.596550e-02
    ## 839          8       V257 -4.086699e-02
    ## 840          8       V259 -5.800934e-02
    ## 841          8       V260 -7.702211e-03
    ## 842          8       V262 -1.203113e-01
    ## 843          8       V266 -4.255777e-02
    ## 844          8       V268 -3.178464e-02
    ## 845          8       V269 -8.375674e-02
    ## 846          8       V270  1.169145e-01
    ## 847          8       V272  5.465741e-02
    ## 848          8       V276  2.335100e-01
    ## 849          8       V279 -1.355365e-01
    ## 850          8       V282 -8.865635e-02
    ## 851          8       V284  1.121797e-01
    ## 852          8       V286 -8.470739e-02
    ## 853          8       V287 -1.056851e-01
    ## 854          8       V289 -9.061072e-02
    ## 855          8       V292 -1.407899e-01
    ## 856          8       V293  2.721635e-02
    ## 857          8       V295  1.052042e-02
    ## 858          8       V296 -5.476498e-04
    ## 859          8       V297  1.567768e-01
    ## 860          8       V299 -1.484647e-01
    ## 861          9         V2 -5.271393e-02
    ## 862          9         V7  6.875007e-02
    ## 863          9        V10 -1.123273e-02
    ## 864          9        V13 -1.160658e-01
    ## 865          9        V15  2.685093e-03
    ## 866          9        V16 -7.800534e-02
    ## 867          9        V17  3.715833e-02
    ## 868          9        V18  3.649972e-02
    ## 869          9        V20  1.826215e-01
    ## 870          9        V22 -2.297145e-01
    ## 871          9        V23 -5.207848e-02
    ## 872          9        V24 -2.226985e-02
    ## 873          9        V26 -3.847710e-02
    ## 874          9        V29 -1.619305e-01
    ## 875          9        V30  1.374794e-01
    ## 876          9        V31 -5.287530e-02
    ## 877          9        V35  9.477097e-02
    ## 878          9  causal_37  3.783651e-01
    ## 879          9        V39  4.733121e-02
    ## 880          9        V41 -3.071962e-02
    ## 881          9        V47  6.850310e-02
    ## 882          9        V48  6.337603e-02
    ## 883          9        V49  7.734475e-02
    ## 884          9        V52 -5.603009e-02
    ## 885          9        V55 -1.943248e-01
    ## 886          9        V57  1.448941e-01
    ## 887          9        V60 -2.907504e-02
    ## 888          9        V61 -5.484939e-02
    ## 889          9        V63 -5.380180e-02
    ## 890          9  causal_65  8.688745e-02
    ## 891          9        V66  8.240787e-02
    ## 892          9        V67 -6.778669e-03
    ## 893          9        V70  5.322740e-02
    ## 894          9        V75 -1.002293e-01
    ## 895          9        V79 -1.928926e-01
    ## 896          9        V80  9.931804e-02
    ## 897          9        V83  5.496923e-02
    ## 898          9        V85  5.641984e-02
    ## 899          9        V87  2.501236e-01
    ## 900          9        V92 -4.977131e-02
    ## 901          9        V93 -1.209595e-01
    ## 902          9        V96  1.012751e-01
    ## 903          9  causal_98  4.201593e-01
    ## 904          9        V99  3.789293e-02
    ## 905          9       V104  1.094084e-02
    ## 906          9       V105 -7.250692e-02
    ## 907          9       V110  6.454806e-02
    ## 908          9       V119 -2.106112e-01
    ## 909          9       V120  1.488926e-01
    ## 910          9       V121 -7.284462e-02
    ## 911          9       V122  6.601883e-02
    ## 912          9       V123 -8.091160e-02
    ## 913          9       V124 -9.487176e-02
    ## 914          9       V125 -4.599333e-02
    ## 915          9       V126  3.562792e-02
    ## 916          9       V129 -1.061501e-03
    ## 917          9 causal_133  4.903845e-01
    ## 918          9       V135  7.019330e-02
    ## 919          9       V137  4.743760e-02
    ## 920          9       V139 -1.496474e-01
    ## 921          9       V140  1.237100e-01
    ## 922          9       V142 -2.226889e-01
    ## 923          9       V144  1.220856e-01
    ## 924          9       V145 -1.259412e-01
    ## 925          9       V148 -1.344613e-01
    ## 926          9       V149  3.266079e-03
    ## 927          9       V154  9.823064e-02
    ## 928          9       V156  4.487294e-02
    ## 929          9       V158 -4.205289e-04
    ## 930          9       V159  1.731229e-01
    ## 931          9       V163  4.946972e-02
    ## 932          9       V165 -7.546000e-02
    ## 933          9       V166  4.494907e-02
    ## 934          9       V169  2.043638e-03
    ## 935          9       V171  1.073848e-01
    ## 936          9       V172  4.367649e-02
    ## 937          9 causal_173  4.577328e-01
    ## 938          9       V175 -2.655601e-02
    ## 939          9       V177  1.023202e-01
    ## 940          9       V179 -5.918930e-02
    ## 941          9       V180  6.030219e-02
    ## 942          9       V181  8.362567e-02
    ## 943          9       V185  5.604899e-02
    ## 944          9       V186 -1.475180e-01
    ## 945          9       V188  1.200768e-01
    ## 946          9       V190  5.098053e-03
    ## 947          9       V191 -1.170922e-02
    ## 948          9       V193 -4.359836e-02
    ## 949          9 causal_200  1.371381e-01
    ## 950          9       V201 -5.604624e-02
    ## 951          9       V202  1.933517e-01
    ## 952          9       V203 -7.821233e-02
    ## 953          9       V205  7.748538e-02
    ## 954          9       V206 -2.098135e-02
    ## 955          9       V210  3.962906e-03
    ## 956          9       V213  9.846240e-02
    ## 957          9 causal_214  3.255706e-01
    ## 958          9       V216 -1.494017e-02
    ## 959          9       V218 -5.256582e-02
    ## 960          9       V222 -9.028251e-02
    ## 961          9       V223  1.073001e-01
    ## 962          9       V224 -7.640417e-02
    ## 963          9       V225 -7.764146e-02
    ## 964          9       V227 -7.894933e-02
    ## 965          9       V228  5.845975e-02
    ## 966          9       V229  1.754148e-03
    ## 967          9       V230  4.656658e-02
    ## 968          9       V232  3.707007e-02
    ## 969          9       V233  1.239674e-01
    ## 970          9       V234  1.304624e-01
    ## 971          9       V235 -2.076649e-02
    ## 972          9       V242 -4.902901e-03
    ## 973          9 causal_243  1.708401e-01
    ## 974          9       V244  1.463689e-01
    ## 975          9       V245 -1.337953e-01
    ## 976          9       V246  1.939436e-02
    ## 977          9       V248 -3.136203e-02
    ## 978          9       V249 -1.894156e-04
    ## 979          9       V253 -1.547562e-03
    ## 980          9       V255  1.695227e-02
    ## 981          9       V258  8.210233e-02
    ## 982          9       V259 -1.137915e-02
    ## 983          9       V261 -1.605691e-01
    ## 984          9       V264 -2.671836e-02
    ## 985          9       V265  1.188801e-01
    ## 986          9       V266 -7.298077e-02
    ## 987          9       V268  3.754349e-02
    ## 988          9       V271  5.607883e-02
    ## 989          9       V273 -2.703847e-02
    ## 990          9       V277 -6.428035e-02
    ## 991          9       V280  1.012521e-01
    ## 992          9       V283  6.640888e-02
    ## 993          9       V288 -9.310408e-02
    ## 994          9       V290 -1.029374e-01
    ## 995          9       V291 -1.728235e-02
    ## 996          9       V292 -1.292624e-01
    ## 997          9       V293  1.760854e-01
    ## 998          9       V296  5.841500e-02
    ## 999          9       V297  6.129667e-02
    ## 1000         9       V299 -9.802520e-02
    ## 1001        10         V2 -7.314946e-02
    ## 1002        10         V3  8.398238e-02
    ## 1003        10         V6  2.607280e-02
    ## 1004        10         V7  4.255355e-02
    ## 1005        10        V11  4.432465e-02
    ## 1006        10        V12  2.008888e-03
    ## 1007        10        V15  1.449555e-01
    ## 1008        10        V18 -1.234912e-02
    ## 1009        10        V20  1.322929e-02
    ## 1010        10        V25 -9.281308e-03
    ## 1011        10        V26 -2.176761e-01
    ## 1012        10        V30  1.081433e-02
    ## 1013        10        V32  4.832739e-02
    ## 1014        10        V34 -3.660620e-03
    ## 1015        10  causal_37  1.543074e-01
    ## 1016        10        V38 -4.469197e-02
    ## 1017        10        V40 -2.470311e-02
    ## 1018        10        V52  8.082693e-03
    ## 1019        10        V57  5.287339e-02
    ## 1020        10        V58  9.402980e-02
    ## 1021        10        V60 -3.165794e-03
    ## 1022        10        V61 -9.812648e-02
    ## 1023        10        V64  2.672123e-02
    ## 1024        10  causal_65  1.469041e-01
    ## 1025        10        V68  1.867266e-02
    ## 1026        10        V73  2.103494e-02
    ## 1027        10        V74 -2.260279e-03
    ## 1028        10        V82  1.277973e-01
    ## 1029        10        V86  1.120229e-02
    ## 1030        10        V93 -6.335213e-02
    ## 1031        10        V95 -3.615813e-02
    ## 1032        10  causal_98  2.999621e-01
    ## 1033        10       V104 -4.305320e-03
    ## 1034        10       V106 -5.977799e-03
    ## 1035        10       V107  1.677450e-02
    ## 1036        10       V108 -1.909382e-02
    ## 1037        10       V110 -1.337100e-02
    ## 1038        10       V113 -2.662421e-02
    ## 1039        10       V120  1.210658e-01
    ## 1040        10       V123 -7.699795e-02
    ## 1041        10       V124 -5.004264e-02
    ## 1042        10       V126  7.005780e-02
    ## 1043        10       V127  9.270249e-03
    ## 1044        10       V131 -1.181925e-02
    ## 1045        10       V132  4.188982e-02
    ## 1046        10 causal_133  1.774771e-01
    ## 1047        10       V137 -4.744819e-02
    ## 1048        10       V139 -1.331108e-01
    ## 1049        10       V143  9.778323e-02
    ## 1050        10       V144  7.770625e-03
    ## 1051        10       V147  7.958115e-02
    ## 1052        10       V150 -2.190605e-02
    ## 1053        10       V151 -7.750800e-02
    ## 1054        10       V153  1.628805e-02
    ## 1055        10       V154  5.387967e-02
    ## 1056        10       V155  2.886469e-02
    ## 1057        10       V157 -1.918515e-02
    ## 1058        10 causal_173  3.820541e-01
    ## 1059        10       V176 -2.833606e-02
    ## 1060        10       V184  1.043485e-01
    ## 1061        10       V185  2.903940e-02
    ## 1062        10       V188 -1.062254e-01
    ## 1063        10       V189 -4.765225e-02
    ## 1064        10       V190  8.014187e-02
    ## 1065        10       V192  2.166155e-02
    ## 1066        10       V195  7.207882e-02
    ## 1067        10       V197  8.592515e-03
    ## 1068        10       V199 -2.149537e-02
    ## 1069        10 causal_200  2.496148e-01
    ## 1070        10       V202  3.683498e-02
    ## 1071        10       V205  1.581006e-01
    ## 1072        10       V207 -1.219850e-01
    ## 1073        10 causal_214  3.625812e-01
    ## 1074        10       V216 -4.059390e-02
    ## 1075        10       V226  4.141696e-02
    ## 1076        10       V229  4.205196e-02
    ## 1077        10       V231 -1.172305e-02
    ## 1078        10       V233  4.464039e-02
    ## 1079        10       V240 -4.496315e-03
    ## 1080        10 causal_243  4.519433e-01
    ## 1081        10       V244  1.775433e-01
    ## 1082        10       V249 -4.185524e-02
    ## 1083        10       V255 -3.672674e-02
    ## 1084        10       V259 -6.193582e-02
    ## 1085        10       V261 -3.841488e-02
    ## 1086        10       V266 -1.870738e-01
    ## 1087        10       V268 -2.196978e-02
    ## 1088        10       V269 -3.104156e-02
    ## 1089        10       V270  5.287813e-02
    ## 1090        10       V271  5.501964e-02
    ## 1091        10       V283  5.493116e-02
    ## 1092        10       V287 -2.963158e-02
    ## 1093        10       V288 -1.384160e-02
    ## 1094        10       V291  6.384434e-03
    ## 1095        10       V296  7.211559e-02
    ## 1096        10       V299 -1.876986e-02

We can calculate stability for each variable by the number of times it
was selected across bootstraps.

``` r
model_lasso_bootstrapped %>%
  group_by(variable) %>%
  summarise(stability = (n()/bootstrap_n) * 100) %>%
  arrange(desc(stability))
```

    ## # A tibble: 292 x 2
    ##    variable   stability
    ##    <chr>          <dbl>
    ##  1 causal_133       100
    ##  2 causal_173       100
    ##  3 causal_200       100
    ##  4 causal_214       100
    ##  5 causal_243       100
    ##  6 causal_37        100
    ##  7 causal_65        100
    ##  8 causal_98        100
    ##  9 V202             100
    ## 10 V123              90
    ## # ... with 282 more rows

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
    ##  1 causal_243            0.429    0.290    0.546           0        97 *     
    ##  2 causal_173            0.437    0.301    0.587           0        96 *     
    ##  3 causal_214            0.423    0.289    0.584           0        96 *     
    ##  4 causal_98             0.454    0.331    0.601           0        94 *     
    ##  5 causal_133            0.343    0.254    0.488           0        78 *     
    ##  6 causal_37             0.410    0.310    0.521           0        70 *     
    ##  7 causal_200            0.358    0.269    0.477           0        68 *     
    ##  8 causal_65             0.351    0.238    0.464           0        64 *     
    ##  9 V297                  0.293    0.239    0.363           0         6 <NA>  
    ## 10 V64                   0.252    0.203    0.300           0         6 <NA>  
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

    ## [1] 23

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
    ## 1 lasso       8    14             22            8            0                 8
    ## 2 mbic        8     0              8            8            0                 8
    ## 3 mcp         8     2             10           NA           NA                NA
    ## 4 prefil~     8     8             16           NA           NA                NA

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
    ##  1 causal_243      99.5           0 *     
    ##  2 causal_214      98.5           0 *     
    ##  3 causal_173      98             0 *     
    ##  4 causal_98       97.5           0 *     
    ##  5 causal_133      87.5           0 *     
    ##  6 causal_200      87             0 *     
    ##  7 causal_37       86.5           0 *     
    ##  8 causal_65       80.5           0 *     
    ##  9 V96             46             0 <NA>  
    ## 10 V61             45             0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 56.5

``` r
stab_plot(triangulated_stability)
```

    ## $combi

![](README_files/figure-gfm/unnamed-chunk-38-1.png)<!-- -->

## No signal datasets

We can now return to our original dataset that we simulated to have no
signal.

Our conventional approach performed relatively poorly, selecting the
following variables as being significantly associated with the outcome
variable.

``` r
prefiltration_results
```

    ## # A tibble: 7 x 5
    ##   variable estimate std.error statistic p.value
    ##   <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 X29         0.146    0.0695      2.10 0.0374 
    ## 2 X35         0.120    0.0604      1.99 0.0477 
    ## 3 X67         0.190    0.0661      2.87 0.00454
    ## 4 X83         0.126    0.0598      2.11 0.0367 
    ## 5 X92        -0.147    0.0681     -2.16 0.0321 
    ## 6 X101        0.176    0.0643      2.74 0.00673
    ## 7 X121       -0.145    0.0612     -2.36 0.0193

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

    ## # A tibble: 1 x 4
    ##   variable stability bootstrap_p stable
    ##   <chr>        <dbl>       <dbl> <chr> 
    ## 1 X67           56.2           0 *

## Conclusions

Thank you for attending this workshop. We hope you enjoyed the session,
and have a good understanding of where some conventional modelling
approaches might not be appropriate in wider datasets.

If you have any further questions after the workshop, please feel free
to contact Martin Green (<martin.green@nottingham.ac.uk>) or Robert Hyde
(<robert.hyde4@nottingham.ac.uk>).
