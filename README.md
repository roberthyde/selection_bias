
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
    ##        X1     X2      X3      X4     X5     X6      X7      X8      X9     X10
    ##     <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ##  1 -1.13   0.917  0.0579 -1.15   -2.58  -2.55  -0.484  -0.546  -0.336  -1.35  
    ##  2  1.04   2.01   0.241   0.829   0.170  0.214  0.0393  0.994  -1.76   -1.22  
    ##  3  0.646  1.02   0.891  -0.509   0.101  0.596 -0.713   0.430  -0.573   1.65  
    ##  4 -1.44   0.114 -0.512  -0.912  -1.06  -0.688 -1.36    0.533  -1.27   -0.0297
    ##  5  1.94   0.273 -0.720   1.80    0.484  0.201  0.886   0.771  -1.26    1.53  
    ##  6 -0.571 -1.36  -0.526   0.131  -1.58  -1.12  -0.623   0.0310 -0.155   0.567 
    ##  7 -0.590  0.770 -0.0878  0.0851 -0.638  0.365 -0.512   0.514  -0.754  -0.952 
    ##  8 -0.876 -0.571 -0.951   1.49    1.07   0.303 -0.595   0.900  -0.333   0.900 
    ##  9 -0.533 -0.710  2.39    0.833   0.710  1.38   1.81   -1.14   -0.709   0.383 
    ## 10  1.01  -1.17   0.0419 -0.710  -0.742 -1.73   1.31   -1.05   -0.0118 -0.358 
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
    ##  1  0.0951
    ##  2 -0.0973
    ##  3  1.41  
    ##  4 -1.34  
    ##  5  0.426 
    ##  6  0.390 
    ##  7 -2.09  
    ##  8  0.352 
    ##  9  0.338 
    ## 10  0.325 
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
    ##    outcome     X1     X2      X3      X4     X5     X6      X7      X8      X9
    ##      <dbl>  <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>   <dbl>
    ##  1  0.0951 -1.13   0.917  0.0579 -1.15   -2.58  -2.55  -0.484  -0.546  -0.336 
    ##  2 -0.0973  1.04   2.01   0.241   0.829   0.170  0.214  0.0393  0.994  -1.76  
    ##  3  1.41    0.646  1.02   0.891  -0.509   0.101  0.596 -0.713   0.430  -0.573 
    ##  4 -1.34   -1.44   0.114 -0.512  -0.912  -1.06  -0.688 -1.36    0.533  -1.27  
    ##  5  0.426   1.94   0.273 -0.720   1.80    0.484  0.201  0.886   0.771  -1.26  
    ##  6  0.390  -0.571 -1.36  -0.526   0.131  -1.58  -1.12  -0.623   0.0310 -0.155 
    ##  7 -2.09   -0.590  0.770 -0.0878  0.0851 -0.638  0.365 -0.512   0.514  -0.754 
    ##  8  0.352  -0.876 -0.571 -0.951   1.49    1.07   0.303 -0.595   0.900  -0.333 
    ##  9  0.338  -0.533 -0.710  2.39    0.833   0.710  1.38   1.81   -1.14   -0.709 
    ## 10  0.325   1.01  -1.17   0.0419 -0.710  -0.742 -1.73   1.31   -1.05   -0.0118
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

    ## # A tibble: 41 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable   1       3.92e-17   2.55e16  0     
    ##  2 X7       variable   0.0989  7.30e- 2   1.35e 0  0.177 
    ##  3 X8       variable  -0.112   7.61e- 2  -1.48e 0  0.141 
    ##  4 X10      variable  -0.115   6.95e- 2  -1.66e 0  0.0987
    ##  5 X11      variable   0.141   7.19e- 2   1.96e 0  0.0518
    ##  6 X17      variable   0.181   7.42e- 2   2.44e 0  0.0155
    ##  7 X18      variable   0.109   7.26e- 2   1.50e 0  0.134 
    ##  8 X23      variable   0.155   7.58e- 2   2.05e 0  0.0422
    ##  9 X28      variable   0.0972  6.84e- 2   1.42e 0  0.157 
    ## 10 X29      variable   0.110   7.26e- 2   1.52e 0  0.130 
    ## # ... with 31 more rows

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

    ## # A tibble: 16 x 5
    ##    variable estimate std.error statistic  p.value
    ##    <chr>       <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 X7          0.131    0.0627      2.08 0.0389  
    ##  2 X10        -0.134    0.0590     -2.27 0.0246  
    ##  3 X11         0.131    0.0648      2.02 0.0446  
    ##  4 X29         0.163    0.0633      2.57 0.0111  
    ##  5 X31         0.160    0.0654      2.45 0.0152  
    ##  6 X33        -0.157    0.0620     -2.53 0.0122  
    ##  7 X40        -0.147    0.0673     -2.18 0.0306  
    ##  8 X45         0.172    0.0661      2.61 0.00984 
    ##  9 X50        -0.127    0.0610     -2.09 0.0382  
    ## 10 X61        -0.143    0.0637     -2.24 0.0266  
    ## 11 X72        -0.149    0.0682     -2.18 0.0306  
    ## 12 X81         0.127    0.0636      2.01 0.0465  
    ## 13 X93        -0.229    0.0652     -3.51 0.000574
    ## 14 X100        0.153    0.0694      2.20 0.0293  
    ## 15 X103       -0.129    0.0638     -2.03 0.0438  
    ## 16 X108        0.155    0.0642      2.42 0.0164

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
    ## 1 (Intercept)  -0.0194    0.0562    -0.346 7.29e- 1
    ## 2 causal_33     0.359     0.0541     6.64  1.52e-10
    ## 3 causal_122    0.422     0.0567     7.45  1.10e-12
    ## 4 causal_153    0.479     0.0565     8.47  1.26e-15
    ## 5 causal_160    0.394     0.0555     7.11  9.20e-12
    ## 6 causal_244    0.444     0.0577     7.70  2.20e-13
    ## 7 causal_284    0.400     0.0581     6.90  3.32e-11
    ## 8 causal_291    0.342     0.0584     5.85  1.36e- 8
    ## 9 causal_299    0.438     0.0554     7.91  5.27e-14

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
    ##  1 V17          -0.131    0.0531     -2.46 1.44e- 2
    ##  2 causal_33     0.361    0.0504      7.16 7.59e-12
    ##  3 V49           0.182    0.0551      3.31 1.07e- 3
    ##  4 V103          0.163    0.0549      2.97 3.29e- 3
    ##  5 V111          0.115    0.0573      2.01 4.51e- 2
    ##  6 V117          0.174    0.0604      2.88 4.30e- 3
    ##  7 V120          0.134    0.0545      2.46 1.46e- 2
    ##  8 causal_122    0.417    0.0538      7.76 1.75e-13
    ##  9 V127          0.164    0.0534      3.07 2.37e- 3
    ## 10 V128         -0.138    0.0561     -2.46 1.47e- 2
    ## 11 V143         -0.166    0.0541     -3.07 2.34e- 3
    ## 12 causal_153    0.394    0.0559      7.06 1.38e-11
    ## 13 causal_160    0.359    0.0516      6.94 2.83e-11
    ## 14 V227         -0.148    0.0571     -2.59 1.00e- 2
    ## 15 causal_244    0.326    0.0571      5.70 3.07e- 8
    ## 16 causal_284    0.351    0.0558      6.29 1.30e- 9
    ## 17 causal_291    0.340    0.0540      6.29 1.27e- 9
    ## 18 causal_299    0.385    0.0520      7.40 1.70e-12

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
    ## 3 mcp               8     5             13
    ## 4 prefiltration     8    10             18

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
    ##  2 <split [300/115]> Bootstrap02
    ##  3 <split [300/111]> Bootstrap03
    ##  4 <split [300/115]> Bootstrap04
    ##  5 <split [300/109]> Bootstrap05
    ##  6 <split [300/113]> Bootstrap06
    ##  7 <split [300/116]> Bootstrap07
    ##  8 <split [300/107]> Bootstrap08
    ##  9 <split [300/108]> Bootstrap09
    ## 10 <split [300/111]> Bootstrap10

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
    ##    outcome      V1     V2     V3     V4      V5     V6     V7       V8       V9
    ##      <dbl>   <dbl>  <dbl>  <dbl>  <dbl>   <dbl>  <dbl>  <dbl>    <dbl>    <dbl>
    ##  1   -4.02  0.419  -0.365 -1.56  -1.36  -0.122  -0.450  0.342 -0.708    0.0142 
    ##  2   -3.98  0.515   1.01   0.525  1.10   0.332  -0.943  0.892  0.0955   1.20   
    ##  3   -3.98  0.515   1.01   0.525  1.10   0.332  -0.943  0.892  0.0955   1.20   
    ##  4   -3.62 -0.0480 -0.556 -1.58   0.155 -0.750  -1.12  -0.789 -0.00159 -0.626  
    ##  5   -3.58 -0.520  -1.02   0.341 -0.463 -1.02    0.643 -1.86  -0.925   -0.00182
    ##  6   -3.35 -0.0419 -0.453  0.151 -1.62  -0.135   0.133  2.11   0.399   -0.625  
    ##  7   -3.24  0.544   1.60  -0.771 -0.216  0.592   1.06   0.475  1.45    -0.0746 
    ##  8   -3.04 -0.252  -0.804  1.73  -1.06  -0.0954 -0.541 -0.744 -1.66     0.0312 
    ##  9   -2.97  0.147   1.22   1.55   0.291 -0.493  -0.339 -1.43  -0.283    0.703  
    ## 10   -2.91 -0.301   0.169  0.676 -1.38   0.592  -2.01  -0.822 -1.01    -0.747  
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, causal_33 <dbl>, V34 <dbl>, V35 <dbl>,
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

    ## # A tibble: 1,029 x 3
    ##    bootstrap variable estimate
    ##    <chr>     <chr>       <dbl>
    ##  1 1         V3       -0.0718 
    ##  2 1         V4       -0.0832 
    ##  3 1         V5       -0.00916
    ##  4 1         V7        0.116  
    ##  5 1         V8       -0.0366 
    ##  6 1         V13      -0.0279 
    ##  7 1         V17      -0.149  
    ##  8 1         V20       0.0527 
    ##  9 1         V27      -0.0296 
    ## 10 1         V28       0.0213 
    ## # ... with 1,019 more rows

We can calculate stability for each variable by the number of times it
was selected across bootstraps.

``` r
model_lasso_bootstrapped %>%
  group_by(variable) %>%
  summarise(stability = (n()/bootstrap_n) * 100) %>%
  arrange(desc(stability))
```

    ## # A tibble: 281 x 2
    ##    variable   stability
    ##    <chr>          <dbl>
    ##  1 causal_122       100
    ##  2 causal_153       100
    ##  3 causal_160       100
    ##  4 causal_244       100
    ##  5 causal_284       100
    ##  6 causal_291       100
    ##  7 causal_299       100
    ##  8 causal_33        100
    ##  9 V117             100
    ## 10 V143             100
    ## # ... with 271 more rows

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
    ##  1 causal_153            0.475    0.381    0.582           0       100 *     
    ##  2 causal_299            0.431    0.315    0.527           0       100 *     
    ##  3 causal_244            0.421    0.297    0.561           0        98 *     
    ##  4 causal_160            0.390    0.288    0.487           0        97 *     
    ##  5 causal_284            0.371    0.262    0.485           0        89 *     
    ##  6 causal_291            0.362    0.276    0.477           0        87 *     
    ##  7 causal_122            0.408    0.314    0.525           0        86 *     
    ##  8 causal_33             0.372    0.246    0.480           0        53 *     
    ##  9 V143                 -0.251   -0.324   -0.211           0        11 <NA>  
    ## 10 V17                  -0.268   -0.322   -0.230           0         9 <NA>  
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

    ## [1] 21

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
    ## 1 lasso                      8            1                        9
    ## 2 mbic                       8            0                        8

Compare this with the non-stability approach

``` r
conventional_results %>%
  left_join(stability_results, by = c("model" = "model_stability"))
```

    ## # A tibble: 4 x 7
    ##   model      tp    fp total_selected tp_stability fp_stability total_selected_s~
    ##   <chr>   <int> <int>          <int>        <int>        <int>             <int>
    ## 1 lasso       8    25             33            8            1                 9
    ## 2 mbic        8     0              8            8            0                 8
    ## 3 mcp         8     5             13           NA           NA                NA
    ## 4 prefil~     8    10             18           NA           NA                NA

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
    ##  1 causal_153     100             0 *     
    ##  2 causal_160     100             0 *     
    ##  3 causal_244     100             0 *     
    ##  4 causal_299     100             0 *     
    ##  5 causal_122      93             0 *     
    ##  6 causal_291      92             0 *     
    ##  7 causal_284      89.5           0 *     
    ##  8 causal_33       80             0 *     
    ##  9 V117            51             0 <NA>  
    ## 10 V17             50             0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 60.5

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

    ## # A tibble: 16 x 5
    ##    variable estimate std.error statistic  p.value
    ##    <chr>       <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 X7          0.131    0.0627      2.08 0.0389  
    ##  2 X10        -0.134    0.0590     -2.27 0.0246  
    ##  3 X11         0.131    0.0648      2.02 0.0446  
    ##  4 X29         0.163    0.0633      2.57 0.0111  
    ##  5 X31         0.160    0.0654      2.45 0.0152  
    ##  6 X33        -0.157    0.0620     -2.53 0.0122  
    ##  7 X40        -0.147    0.0673     -2.18 0.0306  
    ##  8 X45         0.172    0.0661      2.61 0.00984 
    ##  9 X50        -0.127    0.0610     -2.09 0.0382  
    ## 10 X61        -0.143    0.0637     -2.24 0.0266  
    ## 11 X72        -0.149    0.0682     -2.18 0.0306  
    ## 12 X81         0.127    0.0636      2.01 0.0465  
    ## 13 X93        -0.229    0.0652     -3.51 0.000574
    ## 14 X100        0.153    0.0694      2.20 0.0293  
    ## 15 X103       -0.129    0.0638     -2.03 0.0438  
    ## 16 X108        0.155    0.0642      2.42 0.0164

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
