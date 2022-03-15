
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
    ##        X1      X2     X3      X4      X5     X6      X7     X8      X9    X10
    ##     <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>   <dbl>  <dbl>   <dbl>  <dbl>
    ##  1 -1.82  -1.14    0.849  0.497   1.15    1.37   1.39    1.48   0.737  -0.705
    ##  2  0.372  0.112   0.319  1.51    1.03    0.717 -0.691  -1.61  -1.10    0.620
    ##  3 -1.08  -0.338  -0.199 -0.669  -0.0606  0.887 -0.0592 -0.390  0.574   0.935
    ##  4 -0.634  0.736  -1.21   0.303   1.24    1.86   0.379  -0.665 -0.469  -3.03 
    ##  5 -2.50  -0.687   0.919 -0.638   0.0265 -2.27  -0.964   0.261  2.19   -2.14 
    ##  6 -0.490  0.399  -1.25   0.247  -0.222   0.923  0.982  -0.739 -1.06    0.217
    ##  7  0.468 -0.0978  0.548  0.307  -0.172   0.793  1.27   -0.729  0.594  -0.293
    ##  8  1.07  -0.559   0.796 -0.254  -0.155  -0.626 -0.219  -2.18   0.460   2.29 
    ##  9 -0.332  0.558   0.464  0.426  -0.573   1.89   1.71   -0.680  0.822   0.639
    ## 10  0.256 -0.955   0.952 -0.0737  1.34    0.180 -1.00    0.822  0.0926  0.622
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
    ##  1  1.47  
    ##  2  1.70  
    ##  3 -0.955 
    ##  4 -0.775 
    ##  5  0.492 
    ##  6 -0.0189
    ##  7 -0.419 
    ##  8 -2.22  
    ##  9 -1.51  
    ## 10 -0.176 
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
    ##    outcome     X1      X2     X3      X4      X5     X6      X7     X8      X9
    ##      <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>   <dbl>  <dbl>   <dbl>
    ##  1  1.47   -1.82  -1.14    0.849  0.497   1.15    1.37   1.39    1.48   0.737 
    ##  2  1.70    0.372  0.112   0.319  1.51    1.03    0.717 -0.691  -1.61  -1.10  
    ##  3 -0.955  -1.08  -0.338  -0.199 -0.669  -0.0606  0.887 -0.0592 -0.390  0.574 
    ##  4 -0.775  -0.634  0.736  -1.21   0.303   1.24    1.86   0.379  -0.665 -0.469 
    ##  5  0.492  -2.50  -0.687   0.919 -0.638   0.0265 -2.27  -0.964   0.261  2.19  
    ##  6 -0.0189 -0.490  0.399  -1.25   0.247  -0.222   0.923  0.982  -0.739 -1.06  
    ##  7 -0.419   0.468 -0.0978  0.548  0.307  -0.172   0.793  1.27   -0.729  0.594 
    ##  8 -2.22    1.07  -0.559   0.796 -0.254  -0.155  -0.626 -0.219  -2.18   0.460 
    ##  9 -1.51   -0.332  0.558   0.464  0.426  -0.573   1.89   1.71   -0.680  0.822 
    ## 10 -0.176   0.256 -0.955   0.952 -0.0737  1.34    0.180 -1.00    0.822  0.0926
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

    ## # A tibble: 28 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable   1       2.45e-17   4.09e16  0     
    ##  2 X3       variable  -0.153   7.23e- 2  -2.12e 0  0.0353
    ##  3 X11      variable  -0.113   7.14e- 2  -1.58e 0  0.115 
    ##  4 X17      variable  -0.152   7.45e- 2  -2.04e 0  0.0424
    ##  5 X21      variable  -0.147   6.57e- 2  -2.24e 0  0.0262
    ##  6 X24      variable   0.127   8.65e- 2   1.47e 0  0.143 
    ##  7 X27      variable   0.112   6.95e- 2   1.61e 0  0.110 
    ##  8 X30      variable   0.152   7.55e- 2   2.01e 0  0.0459
    ##  9 X32      variable   0.0981  7.25e- 2   1.35e 0  0.178 
    ## 10 X41      variable  -0.119   6.85e- 2  -1.74e 0  0.0834
    ## # ... with 18 more rows

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

    ## # A tibble: 9 x 5
    ##   variable estimate std.error statistic p.value
    ##   <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 X21        -0.146    0.0599     -2.44 0.0157 
    ## 2 X30         0.169    0.0691      2.45 0.0153 
    ## 3 X46         0.190    0.0631      3.01 0.00302
    ## 4 X81         0.173    0.0628      2.76 0.00643
    ## 5 X96         0.140    0.0684      2.05 0.0415 
    ## 6 X98         0.182    0.0679      2.69 0.00787
    ## 7 X100        0.179    0.0664      2.69 0.00781
    ## 8 X103        0.163    0.0616      2.64 0.00905
    ## 9 X116        0.140    0.0665      2.11 0.0363

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
    ## 1 (Intercept)   0.0134    0.0577     0.233 8.16e- 1
    ## 2 causal_24     0.358     0.0551     6.50  3.55e-10
    ## 3 causal_41     0.343     0.0557     6.16  2.40e- 9
    ## 4 causal_92     0.423     0.0589     7.19  5.59e-12
    ## 5 causal_108    0.390     0.0567     6.87  3.91e-11
    ## 6 causal_117    0.348     0.0563     6.17  2.22e- 9
    ## 7 causal_179    0.445     0.0587     7.58  4.69e-13
    ## 8 causal_258    0.489     0.0552     8.86  7.97e-17
    ## 9 causal_261    0.404     0.0588     6.86  4.13e-11

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
    ##  1 V3           -0.166    0.0535     -3.10 2.12e- 3
    ##  2 causal_24     0.322    0.0513      6.27 1.40e- 9
    ##  3 causal_41     0.313    0.0516      6.07 4.34e- 9
    ##  4 causal_92     0.375    0.0557      6.74 9.24e-11
    ##  5 causal_108    0.390    0.0522      7.46 1.14e-12
    ##  6 causal_117    0.335    0.0517      6.48 4.16e-10
    ##  7 V127         -0.180    0.0529     -3.40 7.78e- 4
    ##  8 V148          0.114    0.0493      2.30 2.21e- 2
    ##  9 V164         -0.127    0.0523     -2.43 1.57e- 2
    ## 10 V175         -0.114    0.0530     -2.14 3.30e- 2
    ## 11 causal_179    0.400    0.0566      7.05 1.42e-11
    ## 12 causal_258    0.458    0.0517      8.87 9.93e-17
    ## 13 V260          0.124    0.0556      2.24 2.60e- 2
    ## 14 causal_261    0.319    0.0581      5.48 9.49e- 8
    ## 15 V276          0.121    0.0490      2.46 1.45e- 2
    ## 16 V299          0.162    0.0551      2.95 3.49e- 3

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
    ## 1 lasso             8    25             33
    ## 2 mbic              8     0              8
    ## 3 mcp               8     9             17
    ## 4 prefiltration     8     8             16

## Stability selection

Stability selection relies heavily on bootstrapping. An example of the
bootstrapping approach is shown below.

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
    ##  1 <split [300/106]> Bootstrap01
    ##  2 <split [300/112]> Bootstrap02
    ##  3 <split [300/115]> Bootstrap03
    ##  4 <split [300/115]> Bootstrap04
    ##  5 <split [300/111]> Bootstrap05
    ##  6 <split [300/107]> Bootstrap06
    ##  7 <split [300/115]> Bootstrap07
    ##  8 <split [300/112]> Bootstrap08
    ##  9 <split [300/112]> Bootstrap09
    ## 10 <split [300/109]> Bootstrap10

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
    ##    outcome     V1     V2     V3     V4     V5     V6     V7      V8     V9
    ##      <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>   <dbl>  <dbl>
    ##  1   -3.78  0.916  0.573  1.76   0.246 -1.14  -2.03  -1.88  -0.0222  1.41 
    ##  2   -3.76  0.164  1.64  -0.104 -0.628  1.23   2.53  -0.105  0.576  -0.561
    ##  3   -3.76  0.164  1.64  -0.104 -0.628  1.23   2.53  -0.105  0.576  -0.561
    ##  4   -3.28  0.888 -0.128  1.46   0.218  1.67   1.84   1.10   1.32    0.911
    ##  5   -3.23  0.234 -0.375  0.969 -0.196  1.08   2.30   1.46   0.364   0.903
    ##  6   -2.87  0.548 -0.299  1.05   0.101 -1.09  -1.91   0.301  1.27    0.972
    ##  7   -2.87  0.548 -0.299  1.05   0.101 -1.09  -1.91   0.301  1.27    0.972
    ##  8   -2.84  0.862  0.910 -0.649  2.14  -0.799  0.508  0.370 -0.831   0.399
    ##  9   -2.82 -0.231 -0.657  1.23  -1.71   0.381  1.19   0.646  0.0458  0.781
    ## 10   -2.82 -0.231 -0.657  1.23  -1.71   0.381  1.19   0.646  0.0458  0.781
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   causal_24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, causal_41 <dbl>,
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
    ##    outcome      V1      V2     V3      V4     V5      V6      V7      V8     V9
    ##      <dbl>   <dbl>   <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ##  1  -2.21  -0.229   2.13    1.23  -0.143  -0.158  0.667  -0.526  -0.462  -1.84 
    ##  2   0.169 -1.46   -0.102  -0.384  0.0715  0.580  0.933   1.30    0.741   1.46 
    ##  3   0.192 -1.24   -0.703  -1.43   0.996   0.277 -1.44   -1.79    0.0583  0.207
    ##  4  -1.06  -0.303  -0.707   1.07   0.333  -1.08  -0.136   1.06   -0.553   0.326
    ##  5  -2.70   0.361  -0.301   0.555  1.12   -1.09  -2.07   -0.784  -1.13   -0.656
    ##  6   1.14  -0.0232 -0.324  -0.575  1.99   -0.720 -0.268  -0.517   1.41   -2.10 
    ##  7  -0.622 -0.238  -1.14    0.676 -0.200  -1.32   0.275   1.05    0.264  -0.180
    ##  8  -1.41  -0.0434  0.247  -0.701 -0.602  -0.376  0.440   0.0708 -1.75   -0.108
    ##  9   0.516 -0.974   3.12    1.16   1.54    0.770  0.0462  0.0744 -0.327  -0.989
    ## 10  -0.122  0.525   0.0999 -1.15   2.29   -0.847 -0.976  -0.974  -1.50   -0.838
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   causal_24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, causal_41 <dbl>,
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
    ##    outcome      V1      V2     V3      V4     V5      V6      V7      V8     V9
    ##      <dbl>   <dbl>   <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ##  1   1.34  -0.229   2.13    1.23  -0.143  -0.158  0.667  -0.526  -0.462  -1.84 
    ##  2   0.859 -1.46   -0.102  -0.384  0.0715  0.580  0.933   1.30    0.741   1.46 
    ##  3  -1.28  -1.24   -0.703  -1.43   0.996   0.277 -1.44   -1.79    0.0583  0.207
    ##  4  -1.48  -0.303  -0.707   1.07   0.333  -1.08  -0.136   1.06   -0.553   0.326
    ##  5   2.44   0.361  -0.301   0.555  1.12   -1.09  -2.07   -0.784  -1.13   -0.656
    ##  6  -0.223 -0.0232 -0.324  -0.575  1.99   -0.720 -0.268  -0.517   1.41   -2.10 
    ##  7  -2.82  -0.238  -1.14    0.676 -0.200  -1.32   0.275   1.05    0.264  -0.180
    ##  8  -3.76  -0.0434  0.247  -0.701 -0.602  -0.376  0.440   0.0708 -1.75   -0.108
    ##  9  -0.999 -0.974   3.12    1.16   1.54    0.770  0.0462  0.0744 -0.327  -0.989
    ## 10   0.795  0.525   0.0999 -1.15   2.29   -0.847 -0.976  -0.974  -1.50   -0.838
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   causal_24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, causal_41 <dbl>,
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
    ##  1 1           <split [300/115]> Bootstrap01
    ##  2 1           <split [300/117]> Bootstrap02
    ##  3 1           <split [300/107]> Bootstrap03
    ##  4 1           <split [300/109]> Bootstrap04
    ##  5 1           <split [300/104]> Bootstrap05
    ##  6 1           <split [300/109]> Bootstrap06
    ##  7 1           <split [300/100]> Bootstrap07
    ##  8 1           <split [300/111]> Bootstrap08
    ##  9 1           <split [300/111]> Bootstrap09
    ## 10 1           <split [300/114]> Bootstrap10
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
variable was selected by a given model) results for as follows:

``` r
stab_output$mbic$stability
```

    ## # A tibble: 301 x 7
    ##    variable   mean_coefficient ci_lower ci_upper bootstrap_p stability stable
    ##    <chr>                 <dbl>    <dbl>    <dbl>       <dbl>     <dbl> <chr> 
    ##  1 causal_179            0.446    0.322    0.539           0       100 *     
    ##  2 causal_258            0.490    0.388    0.588           0       100 *     
    ##  3 causal_108            0.398    0.290    0.529           0        98 *     
    ##  4 causal_261            0.393    0.279    0.504           0        96 *     
    ##  5 causal_24             0.359    0.271    0.463           0        93 *     
    ##  6 causal_92             0.407    0.297    0.503           0        92 *     
    ##  7 causal_41             0.337    0.249    0.449           0        85 *     
    ##  8 causal_117            0.349    0.256    0.456           0        84 *     
    ##  9 V3                   -0.254   -0.330   -0.206           0        22 *     
    ## 10 V67                  -0.248   -0.298   -0.204           0        14 <NA>  
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

    ## [1] 22

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
    ## 1 mbic                       8            0                        8

Compare this with the non-stability approach

``` r
conventional_results %>%
  left_join(stability_results, by = c("model" = "model_stability"))
```

    ## # A tibble: 4 x 7
    ##   model      tp    fp total_selected tp_stability fp_stability total_selected_s~
    ##   <chr>   <int> <int>          <int>        <int>        <int>             <int>
    ## 1 lasso       8    25             33           NA           NA                NA
    ## 2 mbic        8     0              8            8            0                 8
    ## 3 mcp         8     9             17           NA           NA                NA
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
    ##  1 causal_179       100           0 *     
    ##  2 causal_258       100           0 *     
    ##  3 causal_108        99           0 *     
    ##  4 causal_24         98           0 *     
    ##  5 causal_261        97           0 *     
    ##  6 causal_117        93           0 *     
    ##  7 causal_41         92           0 *     
    ##  8 causal_92         92           0 *     
    ##  9 V3                18           0 <NA>  
    ## 10 V67                8           0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 23

``` r
stab_plot(triangulated_stability)
```

![](README_files/figure-gfm/stab_plot-1.png)<!-- -->

## No signal datasets

We can now return to our original dataset that we simulated to have no
signal.

Our conventional approach performed relatively poorly, selecting the
following variables as being significantly associated with the outcome
variable.

``` r
prefiltration_results
```

    ## # A tibble: 9 x 5
    ##   variable estimate std.error statistic p.value
    ##   <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ## 1 X21        -0.146    0.0599     -2.44 0.0157 
    ## 2 X30         0.169    0.0691      2.45 0.0153 
    ## 3 X46         0.190    0.0631      3.01 0.00302
    ## 4 X81         0.173    0.0628      2.76 0.00643
    ## 5 X96         0.140    0.0684      2.05 0.0415 
    ## 6 X98         0.182    0.0679      2.69 0.00787
    ## 7 X100        0.179    0.0664      2.69 0.00781
    ## 8 X103        0.163    0.0616      2.64 0.00905
    ## 9 X116        0.140    0.0665      2.11 0.0363

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
