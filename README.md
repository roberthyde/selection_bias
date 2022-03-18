
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
library(rms)
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
    ##         X1      X2     X3     X4     X5      X6      X7      X8     X9    X10
    ##      <dbl>   <dbl>  <dbl>  <dbl>  <dbl>   <dbl>   <dbl>   <dbl>  <dbl>  <dbl>
    ##  1 -0.212  -0.295   0.265  1.11   0.621 -2.23   -0.182  -1.99   -0.365 -0.395
    ##  2 -0.770  -0.0571  0.713  0.612 -0.187  0.142   0.414  -0.408  -0.252 -0.122
    ##  3  1.41    0.831  -0.269  0.403  0.843  0.0808  1.13    0.130  -0.387 -0.180
    ##  4  0.0349  0.537   0.668  0.130 -1.79  -0.734   1.17   -0.624  -0.787 -1.34 
    ##  5 -1.16   -1.24   -2.53   1.22   0.398 -0.513   1.04    0.387  -1.37  -2.18 
    ##  6  0.920   0.390   0.237 -1.32   0.871 -1.72   -0.696  -0.670  -1.69  -1.07 
    ##  7  1.88    1.14    0.834  0.993 -0.917 -0.602   0.422  -1.06    0.705  0.374
    ##  8  3.58    1.95    0.556  1.20  -1.36  -0.159  -1.18    1.51    0.268 -1.34 
    ##  9  0.235   0.324  -0.638  1.15   1.88   1.47   -1.47   -0.419  -0.265 -1.11 
    ## 10 -0.665   0.101   0.203 -0.418 -1.89  -0.0500 -0.0686  0.0397  1.29   1.17 
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
    ##  1 -1.44  
    ##  2 -0.479 
    ##  3 -0.0689
    ##  4 -0.219 
    ##  5  1.11  
    ##  6 -0.478 
    ##  7  1.04  
    ##  8  0.297 
    ##  9  0.276 
    ## 10  0.428 
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
    ##    outcome      X1      X2     X3     X4     X5      X6      X7      X8     X9
    ##      <dbl>   <dbl>   <dbl>  <dbl>  <dbl>  <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ##  1 -1.44   -0.212  -0.295   0.265  1.11   0.621 -2.23   -0.182  -1.99   -0.365
    ##  2 -0.479  -0.770  -0.0571  0.713  0.612 -0.187  0.142   0.414  -0.408  -0.252
    ##  3 -0.0689  1.41    0.831  -0.269  0.403  0.843  0.0808  1.13    0.130  -0.387
    ##  4 -0.219   0.0349  0.537   0.668  0.130 -1.79  -0.734   1.17   -0.624  -0.787
    ##  5  1.11   -1.16   -1.24   -2.53   1.22   0.398 -0.513   1.04    0.387  -1.37 
    ##  6 -0.478   0.920   0.390   0.237 -1.32   0.871 -1.72   -0.696  -0.670  -1.69 
    ##  7  1.04    1.88    1.14    0.834  0.993 -0.917 -0.602   0.422  -1.06    0.705
    ##  8  0.297   3.58    1.95    0.556  1.20  -1.36  -0.159  -1.18    1.51    0.268
    ##  9  0.276   0.235   0.324  -0.638  1.15   1.88   1.47   -1.47   -0.419  -0.265
    ## 10  0.428  -0.665   0.101   0.203 -0.418 -1.89  -0.0500 -0.0686  0.0397  1.29 
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

    ## # A tibble: 37 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable    1      8.49e-18   1.18e17  0     
    ##  2 X2       variable    0.124  6.83e- 2   1.81e 0  0.0718
    ##  3 X3       variable    0.183  7.06e- 2   2.59e 0  0.0103
    ##  4 X4       variable    0.152  6.90e- 2   2.20e 0  0.0290
    ##  5 X7       variable   -0.127  6.32e- 2  -2.01e 0  0.0458
    ##  6 X12      variable   -0.117  7.10e- 2  -1.65e 0  0.101 
    ##  7 X17      variable   -0.147  6.96e- 2  -2.11e 0  0.0357
    ##  8 X20      variable   -0.104  7.23e- 2  -1.44e 0  0.151 
    ##  9 X23      variable   -0.104  6.31e- 2  -1.65e 0  0.101 
    ## 10 X28      variable    0.123  6.30e- 2   1.95e 0  0.0526
    ## # ... with 27 more rows

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
    dplyr::select(variables)
  model1 <- ols(outcome ~ ., data=data_selected)
  model2 <- fastbw(fit = model1, rule = "p", sls = 0.05)
  final_terms <- model2$names.kept
  final_data <- data %>% 
    select(outcome, final_terms)
  final_model <- summary(lm(outcome~., data = final_data ))
}

prefiltration_model <- stepwise_model(data = df_no_signal, variables = variables_for_stepwise)

prefiltration_results <- prefiltration_model$coefficients %>%
  as.data.frame() %>%
  rownames_to_column(var = "variable")
```

We can also calculate the R2 for this model.

``` r
prefiltration_model$r.squared
```

    ## [1] 0.3236893

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
    ## 1 (Intercept) -0.00737    0.0568    -0.130 8.97e- 1
    ## 2 causal_27    0.435      0.0571     7.61  3.76e-13
    ## 3 causal_36    0.447      0.0572     7.82  9.67e-14
    ## 4 causal_98    0.354      0.0576     6.14  2.63e- 9
    ## 5 causal_122   0.487      0.0551     8.83  9.72e-17
    ## 6 causal_199   0.414      0.0569     7.27  3.31e-12
    ## 7 causal_224   0.395      0.0571     6.91  3.05e-11
    ## 8 causal_255   0.269      0.0565     4.77  2.93e- 6
    ## 9 causal_277   0.280      0.0563     4.98  1.11e- 6

## Conventional stepwise approach

We can now repeat out prefiltration and stepwise selection approach as
before

``` r
univariable_outcomes <- map_df(df_signal, ~ univariable_analysis(data = df_signal, variable = .), .id = "variable")
univariable_outcomes_filtered <- univariable_outcomes %>%
  filter(p.value < 0.2)
variables_for_stepwise <- univariable_outcomes_filtered %>%
  pull(variable)

model_results$prefiltration <- stepwise_model(data = df_signal, variables = variables_for_stepwise) %>%
  tidy() %>%
  rename(variable = term)

model_results$prefiltration %>%
  as_tibble()
```

    ## # A tibble: 15 x 5
    ##    variable    estimate std.error statistic  p.value
    ##    <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)   0.0331    0.0545     0.608 5.44e- 1
    ##  2 causal_27     0.436     0.0546     7.99  3.43e-14
    ##  3 V29           0.193     0.0516     3.73  2.27e- 4
    ##  4 causal_36     0.428     0.0543     7.88  7.12e-14
    ##  5 causal_98     0.290     0.0554     5.23  3.27e- 7
    ##  6 causal_122    0.438     0.0526     8.34  3.20e-15
    ##  7 V124          0.183     0.0595     3.07  2.37e- 3
    ##  8 V125         -0.174     0.0558    -3.13  1.96e- 3
    ##  9 V137         -0.153     0.0551    -2.78  5.84e- 3
    ## 10 causal_199    0.402     0.0539     7.46  1.05e-12
    ## 11 causal_224    0.400     0.0540     7.40  1.53e-12
    ## 12 causal_255    0.244     0.0538     4.54  8.42e- 6
    ## 13 V261         -0.122     0.0539    -2.26  2.43e- 2
    ## 14 causal_277    0.263     0.0535     4.92  1.48e- 6
    ## 15 V288         -0.131     0.0533    -2.45  1.49e- 2

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
    ## 1 lasso             8    19             27
    ## 2 mbic              8     0              8
    ## 3 mcp               8     2             10
    ## 4 prefiltration     8     7             15

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
    ##  1 <split [300/113]> Bootstrap01
    ##  2 <split [300/108]> Bootstrap02
    ##  3 <split [300/113]> Bootstrap03
    ##  4 <split [300/111]> Bootstrap04
    ##  5 <split [300/110]> Bootstrap05
    ##  6 <split [300/106]> Bootstrap06
    ##  7 <split [300/122]> Bootstrap07
    ##  8 <split [300/118]> Bootstrap08
    ##  9 <split [300/117]> Bootstrap09
    ## 10 <split [300/118]> Bootstrap10

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
    ##    outcome      V1      V2     V3     V4       V5     V6      V7     V8      V9
    ##      <dbl>   <dbl>   <dbl>  <dbl>  <dbl>    <dbl>  <dbl>   <dbl>  <dbl>   <dbl>
    ##  1   -3.11  0.0787  0.876  -0.425 -0.544 -1.43     0.440 -0.707   0.383  0.388 
    ##  2   -2.75 -1.03   -0.105   0.656  0.850  0.682   -0.800 -1.21    1.51  -1.71  
    ##  3   -2.75 -1.03   -0.105   0.656  0.850  0.682   -0.800 -1.21    1.51  -1.71  
    ##  4   -2.75 -1.03   -0.105   0.656  0.850  0.682   -0.800 -1.21    1.51  -1.71  
    ##  5   -2.51 -0.402  -0.290   0.304  0.260 -0.172    0.247  0.0296  1.96   0.0313
    ##  6   -2.42 -1.12    0.0976 -1.09  -0.374 -1.03     1.51  -0.174   0.286  0.371 
    ##  7   -2.22 -0.480  -0.875  -1.05  -0.404 -0.265    1.09  -0.370  -0.308 -0.668 
    ##  8   -2.22 -0.480  -0.875  -1.05  -0.404 -0.265    1.09  -0.370  -0.308 -0.668 
    ##  9   -2.22 -0.429   1.85    1.81   0.355 -0.00507  0.862  1.57    1.54   1.65  
    ## 10   -2.16  0.367   2.30    0.346 -0.299  0.903   -0.339  0.175   1.95   1.28  
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, causal_27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   causal_36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
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

    ## # A tibble: 1,068 x 3
    ##    bootstrap variable   estimate
    ##    <chr>     <chr>         <dbl>
    ##  1 1         V9         0.0100  
    ##  2 1         V12       -0.000604
    ##  3 1         V18        0.0382  
    ##  4 1         V26        0.00485 
    ##  5 1         causal_27  0.217   
    ##  6 1         V29        0.0500  
    ##  7 1         V30        0.0931  
    ##  8 1         causal_36  0.297   
    ##  9 1         V39        0.0454  
    ## 10 1         V45       -0.0350  
    ## # ... with 1,058 more rows

We can calculate stability for each variable by the number of times it
was selected across bootstraps.

``` r
model_lasso_bootstrapped %>%
  group_by(variable) %>%
  summarise(stability = (n()/bootstrap_n) * 100) %>%
  arrange(desc(stability))
```

    ## # A tibble: 286 x 2
    ##    variable   stability
    ##    <chr>          <dbl>
    ##  1 causal_122       100
    ##  2 causal_199       100
    ##  3 causal_224       100
    ##  4 causal_27        100
    ##  5 causal_277       100
    ##  6 causal_36        100
    ##  7 causal_98        100
    ##  8 V168             100
    ##  9 V29              100
    ## 10 V85              100
    ## # ... with 276 more rows

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
    ##  1 causal_122            0.484    0.377    0.577           0       100 *     
    ##  2 causal_36             0.456    0.347    0.601           0       100 *     
    ##  3 causal_199            0.421    0.316    0.537           0        99 *     
    ##  4 causal_224            0.400    0.287    0.515           0        99 *     
    ##  5 causal_27             0.416    0.295    0.541           0        97 *     
    ##  6 causal_98             0.343    0.251    0.458           0        90 *     
    ##  7 causal_277            0.295    0.218    0.383           0        78 *     
    ##  8 causal_255            0.283    0.216    0.374           0        61 *     
    ##  9 V29                   0.250    0.206    0.298           0        22 *     
    ## 10 V125                 -0.240   -0.281   -0.193           0         8 <NA>  
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

    ## [1] 17

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
    ## 2 mbic                       8            1                        9

Compare this with the non-stability approach

``` r
conventional_results %>%
  left_join(stability_results, by = c("model" = "model_stability"))
```

    ## # A tibble: 4 x 7
    ##   model      tp    fp total_selected tp_stability fp_stability total_selected_s~
    ##   <chr>   <int> <int>          <int>        <int>        <int>             <int>
    ## 1 lasso       8    19             27            8            0                 8
    ## 2 mbic        8     0              8            8            1                 9
    ## 3 mcp         8     2             10           NA           NA                NA
    ## 4 prefil~     8     7             15           NA           NA                NA

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
    ##  1 causal_122     100             0 *     
    ##  2 causal_27      100             0 *     
    ##  3 causal_36      100             0 *     
    ##  4 causal_224      99.5           0 *     
    ##  5 causal_199      98.5           0 *     
    ##  6 causal_98       94.5           0 *     
    ##  7 causal_277      87.5           0 *     
    ##  8 causal_255      80             0 *     
    ##  9 V29             63             0 *     
    ## 10 V125            50.5           0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 58.5

``` r
stab_plot(triangulated_stability)
```

    ## $combi

![](README_files/figure-gfm/unnamed-chunk-40-1.png)<!-- -->

## No signal datasets

We can now return to our original dataset that we simulated to have no
signal.

Our conventional approach performed relatively poorly, selecting the
following variables as being significantly associated with the outcome
variable.

``` r
prefiltration_results
```

    ##       variable    Estimate Std. Error   t value    Pr(>|t|)
    ## 1  (Intercept) -0.09490423 0.06057225 -1.566794 0.118899917
    ## 2           X3  0.20658587 0.06350083  3.253278 0.001359882
    ## 3          X17 -0.19014772 0.06163029 -3.085297 0.002350938
    ## 4          X42  0.13143862 0.06174171  2.128846 0.034611780
    ## 5          X55 -0.15921049 0.05802032 -2.744047 0.006676409
    ## 6          X56 -0.12171417 0.05899493 -2.063129 0.040520166
    ## 7          X61 -0.11811351 0.05943904 -1.987137 0.048407320
    ## 8          X79 -0.18418595 0.06599951 -2.790717 0.005819939
    ## 9          X81  0.15562877 0.05540635  2.808862 0.005514845
    ## 10         X87 -0.16754576 0.06131192 -2.732678 0.006901647
    ## 11        X102  0.12392063 0.05517565  2.245930 0.025911047
    ## 12        X112 -0.17226982 0.05814023 -2.963005 0.003453723
    ## 13        X123 -0.15925218 0.06226692 -2.557573 0.011355638
    ## 14        X125 -0.20715392 0.06222587 -3.329064 0.001054713
    ## 15        X129 -0.11979740 0.06087710 -1.967857 0.050603661

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
