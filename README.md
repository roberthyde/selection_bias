
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
    ##        X1      X2     X3      X4       X5     X6      X7      X8      X9    X10
    ##     <dbl>   <dbl>  <dbl>   <dbl>    <dbl>  <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ##  1 -0.595  1.81   -0.213  0.552   2.15     0.312 -0.884   2.60    0.0972  0.311
    ##  2 -0.740 -2.00   -0.591  0.335  -0.290    1.84   1.20    0.0227 -0.198  -0.928
    ##  3 -0.642  0.887  -0.254  0.293   1.02    -1.19  -0.347  -1.44   -1.42   -1.36 
    ##  4 -1.63  -0.229  -0.183  0.699   0.00524  0.644 -0.481   1.75   -1.95    1.86 
    ##  5 -1.49  -0.265  -0.113  1.45   -0.951   -0.312  0.382  -0.756   0.626  -0.862
    ##  6  0.646  0.700  -0.994  2.90   -1.22    -0.607 -0.0800 -1.23   -0.480  -0.955
    ##  7 -1.18   0.0833 -0.468  1.62   -1.75     0.948 -0.966  -0.645   0.849   1.43 
    ##  8 -0.217 -0.141  -0.212  0.0442 -0.201   -1.59   0.379  -0.399   0.930   0.774
    ##  9  0.281  1.44    1.18  -0.636  -1.10     0.211 -0.501  -0.280   2.18   -0.404
    ## 10 -0.598  1.63   -1.24   0.0358 -1.20     0.886  1.02    0.406   0.610  -0.508
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
    ##  1  1.62  
    ##  2 -1.68  
    ##  3 -1.24  
    ##  4  0.852 
    ##  5  2.08  
    ##  6 -0.0541
    ##  7 -0.0127
    ##  8  0.252 
    ##  9 -0.914 
    ## 10 -0.777 
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
    ##    outcome     X1      X2     X3      X4       X5     X6      X7      X8      X9
    ##      <dbl>  <dbl>   <dbl>  <dbl>   <dbl>    <dbl>  <dbl>   <dbl>   <dbl>   <dbl>
    ##  1  1.62   -0.595  1.81   -0.213  0.552   2.15     0.312 -0.884   2.60    0.0972
    ##  2 -1.68   -0.740 -2.00   -0.591  0.335  -0.290    1.84   1.20    0.0227 -0.198 
    ##  3 -1.24   -0.642  0.887  -0.254  0.293   1.02    -1.19  -0.347  -1.44   -1.42  
    ##  4  0.852  -1.63  -0.229  -0.183  0.699   0.00524  0.644 -0.481   1.75   -1.95  
    ##  5  2.08   -1.49  -0.265  -0.113  1.45   -0.951   -0.312  0.382  -0.756   0.626 
    ##  6 -0.0541  0.646  0.700  -0.994  2.90   -1.22    -0.607 -0.0800 -1.23   -0.480 
    ##  7 -0.0127 -1.18   0.0833 -0.468  1.62   -1.75     0.948 -0.966  -0.645   0.849 
    ##  8  0.252  -0.217 -0.141  -0.212  0.0442 -0.201   -1.59   0.379  -0.399   0.930 
    ##  9 -0.914   0.281  1.44    1.18  -0.636  -1.10     0.211 -0.501  -0.280   2.18  
    ## 10 -0.777  -0.598  1.63   -1.24   0.0358 -1.20     0.886  1.02    0.406   0.610 
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

    ## # A tibble: 21 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable   1       9.15e-18   1.09e17 0      
    ##  2 X6       variable  -0.127   5.95e- 2  -2.14e 0 0.0338 
    ##  3 X10      variable   0.125   6.28e- 2   2.00e 0 0.0472 
    ##  4 X17      variable  -0.0860  6.31e- 2  -1.36e 0 0.174  
    ##  5 X18      variable   0.121   6.94e- 2   1.74e 0 0.0840 
    ##  6 X23      variable   0.210   6.45e- 2   3.25e 0 0.00136
    ##  7 X28      variable   0.115   6.27e- 2   1.83e 0 0.0687 
    ##  8 X35      variable   0.121   7.45e- 2   1.62e 0 0.107  
    ##  9 X44      variable  -0.0952  6.65e- 2  -1.43e 0 0.154  
    ## 10 X46      variable  -0.132   6.12e- 2  -2.16e 0 0.0323 
    ## # ... with 11 more rows

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

prefiltration_results
```

    ##      variable    Estimate Std. Error    t value     Pr(>|t|)
    ## 1 (Intercept) -0.04487799 0.06220364 -0.7214688 0.4715126645
    ## 2         X10  0.18394352 0.06025412  3.0527958 0.0025939184
    ## 3         X23  0.20707712 0.06169409  3.3565147 0.0009541922
    ## 4         X28  0.15728754 0.05944961  2.6457288 0.0088380413
    ## 5         X35  0.18384579 0.07140231  2.5747878 0.0107950696
    ## 6         X51  0.14480115 0.06052760  2.3923161 0.0177224878
    ## 7         X95  0.13931089 0.05880681  2.3689584 0.0188466989
    ## 8        X105  0.10955658 0.06676068  1.6410344 0.1024533423

We can also calculate the R2 for this model.

``` r
prefiltration_model$r.squared
```

    ## [1] 0.1699614

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
    ## 1 (Intercept)  -0.0169    0.0600    -0.282 7.78e- 1
    ## 2 causal_42     0.474     0.0626     7.58  4.52e-13
    ## 3 causal_71     0.409     0.0572     7.14  7.30e-12
    ## 4 causal_83     0.349     0.0564     6.19  1.99e- 9
    ## 5 causal_107    0.486     0.0611     7.95  4.14e-14
    ## 6 causal_114    0.371     0.0589     6.30  1.09e- 9
    ## 7 causal_143    0.442     0.0616     7.18  5.71e-12
    ## 8 causal_257    0.511     0.0618     8.27  4.80e-15
    ## 9 causal_266    0.403     0.0573     7.03  1.43e-11

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

    ## # A tibble: 13 x 5
    ##    variable    estimate std.error statistic  p.value
    ##    <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)  -0.0501    0.0578    -0.867 3.87e- 1
    ##  2 causal_42     0.425     0.0611     6.96  2.37e-11
    ##  3 causal_71     0.395     0.0548     7.20  5.32e-12
    ##  4 causal_83     0.329     0.0541     6.08  3.82e- 9
    ##  5 causal_107    0.470     0.0588     8.00  3.14e-14
    ##  6 causal_114    0.380     0.0564     6.73  9.01e-11
    ##  7 causal_143    0.425     0.0592     7.19  5.81e-12
    ##  8 V150          0.148     0.0561     2.63  8.97e- 3
    ##  9 V159          0.175     0.0612     2.86  4.60e- 3
    ## 10 V187         -0.144     0.0527    -2.74  6.58e- 3
    ## 11 causal_257    0.511     0.0599     8.53  8.54e-16
    ## 12 causal_266    0.398     0.0548     7.26  3.62e-12
    ## 13 V292         -0.172     0.0546    -3.14  1.86e- 3

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
    ## 1 lasso             8    27             35
    ## 2 mbic              8     0              8
    ## 3 mcp               8     5             13
    ## 4 prefiltration     8     5             13

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
    ##  1 <split [300/100]> Bootstrap01
    ##  2 <split [300/114]> Bootstrap02
    ##  3 <split [300/113]> Bootstrap03
    ##  4 <split [300/112]> Bootstrap04
    ##  5 <split [300/112]> Bootstrap05
    ##  6 <split [300/114]> Bootstrap06
    ##  7 <split [300/111]> Bootstrap07
    ##  8 <split [300/94]>  Bootstrap08
    ##  9 <split [300/112]> Bootstrap09
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
    ##    outcome     V1      V2     V3      V4       V5     V6       V7     V8     V9
    ##      <dbl>  <dbl>   <dbl>  <dbl>   <dbl>    <dbl>  <dbl>    <dbl>  <dbl>  <dbl>
    ##  1   -3.93  0.271  0.968  -0.261 -0.567   0.00468  0.261 -0.218   -1.58  -0.605
    ##  2   -3.93  0.271  0.968  -0.261 -0.567   0.00468  0.261 -0.218   -1.58  -0.605
    ##  3   -3.89  0.495 -0.0649 -0.630  0.0335  0.658    0.201  0.913   -0.384 -0.897
    ##  4   -3.52 -0.458 -1.13   -1.71   0.789   0.694    0.469  1.58     0.204 -0.551
    ##  5   -3.52 -0.458 -1.13   -1.71   0.789   0.694    0.469  1.58     0.204 -0.551
    ##  6   -3.52 -0.458 -1.13   -1.71   0.789   0.694    0.469  1.58     0.204 -0.551
    ##  7   -3.44  0.860  0.489   0.898  0.409  -0.827    0.303  0.567   -1.09   2.42 
    ##  8   -3.03  0.173 -1.54   -0.529 -0.719  -0.713   -0.271  0.440    0.484 -0.866
    ##  9   -3.03  0.173 -1.54   -0.529 -0.719  -0.713   -0.271  0.440    0.484 -0.866
    ## 10   -2.87 -0.275 -0.339   0.741 -0.361  -1.18     1.47  -0.00704 -0.857 -0.331
    ## # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
    ## #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
    ## #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
    ## #   V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>, V29 <dbl>,
    ## #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
    ## #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
    ## #   causal_42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...

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

    ## # A tibble: 987 x 3
    ##    bootstrap variable estimate
    ##    <chr>     <chr>       <dbl>
    ##  1 1         V2        0.00827
    ##  2 1         V3        0.0132 
    ##  3 1         V4        0.0470 
    ##  4 1         V5        0.00344
    ##  5 1         V9        0.00410
    ##  6 1         V13       0.00971
    ##  7 1         V24      -0.0100 
    ##  8 1         V26      -0.0748 
    ##  9 1         V37      -0.0553 
    ## 10 1         V38      -0.00979
    ## # ... with 977 more rows

We can calculate stability for each variable by the number of times it
was selected across bootstraps.

``` r
model_lasso_bootstrapped %>%
  group_by(variable) %>%
  summarise(stability = (n()/bootstrap_n) * 100) %>%
  arrange(desc(stability))
```

    ## # A tibble: 277 x 2
    ##    variable   stability
    ##    <chr>          <dbl>
    ##  1 causal_107       100
    ##  2 causal_114       100
    ##  3 causal_143       100
    ##  4 causal_257       100
    ##  5 causal_266       100
    ##  6 causal_42        100
    ##  7 causal_71        100
    ##  8 causal_83        100
    ##  9 V187             100
    ## 10 V292             100
    ## # ... with 267 more rows

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
    ##  1 causal_107            0.483    0.374    0.613           0       100 *     
    ##  2 causal_42             0.454    0.360    0.588           0       100 *     
    ##  3 causal_257            0.500    0.349    0.643           0        99 *     
    ##  4 causal_266            0.398    0.288    0.538           0        97 *     
    ##  5 causal_143            0.437    0.329    0.564           0        91 *     
    ##  6 causal_71             0.390    0.290    0.505           0        91 *     
    ##  7 causal_114            0.368    0.271    0.456           0        82 *     
    ##  8 causal_83             0.352    0.243    0.487           0        81 *     
    ##  9 V187                 -0.243   -0.306   -0.204           0        16 *     
    ## 10 V292                 -0.251   -0.315   -0.211           0        15 <NA>  
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

    ## [1] 16

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
    ## 1 lasso       8    27             35            8            0                 8
    ## 2 mbic        8     0              8            8            0                 8
    ## 3 mcp         8     5             13           NA           NA                NA
    ## 4 prefil~     8     5             13           NA           NA                NA

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
    ##  1 causal_257     100             0 *     
    ##  2 causal_107      99.5           0 *     
    ##  3 causal_42       98.5           0 *     
    ##  4 causal_143      98             0 *     
    ##  5 causal_266      97.5           0 *     
    ##  6 causal_71       94             0 *     
    ##  7 causal_83       92.5           0 *     
    ##  8 causal_114      86             0 *     
    ##  9 V282            53             0 <NA>  
    ## 10 V261            52             0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 62.5

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

    ##      variable    Estimate Std. Error    t value     Pr(>|t|)
    ## 1 (Intercept) -0.04487799 0.06220364 -0.7214688 0.4715126645
    ## 2         X10  0.18394352 0.06025412  3.0527958 0.0025939184
    ## 3         X23  0.20707712 0.06169409  3.3565147 0.0009541922
    ## 4         X28  0.15728754 0.05944961  2.6457288 0.0088380413
    ## 5         X35  0.18384579 0.07140231  2.5747878 0.0107950696
    ## 6         X51  0.14480115 0.06052760  2.3923161 0.0177224878
    ## 7         X95  0.13931089 0.05880681  2.3689584 0.0188466989
    ## 8        X105  0.10955658 0.06676068  1.6410344 0.1024533423

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
