
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
    ##         X1     X2     X3      X4     X5      X6      X7     X8     X9    X10
    ##      <dbl>  <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>  <dbl>
    ##  1  1.26    0.727  0.416  1.34   -0.297  0.221  -0.793   0.717 -0.515  0.221
    ##  2  0.696  -1.49  -0.229  1.19    1.94   0.677  -0.0438 -1.44   1.82   0.552
    ##  3  0.500   1.01   1.26  -0.374  -0.196 -0.122  -0.434   0.259 -0.340 -1.34 
    ##  4 -0.241   0.569 -2.24   0.860   1.19   0.642   1.04    0.735  0.735 -0.404
    ##  5 -1.16   -0.491 -1.35   2.56   -1.81   0.197   0.0831  2.22  -0.522  1.97 
    ##  6  0.0291  2.02  -0.898  1.30    1.16  -0.0930 -0.467  -1.62  -0.302  0.336
    ##  7  0.268  -0.447  0.657 -0.0136  0.440  0.192   0.0668  0.428  2.09  -0.761
    ##  8  1.55   -2.16  -0.485 -0.726   0.310  0.471   1.87   -0.455  0.550 -2.70 
    ##  9 -0.837  -1.32   0.303 -0.717   1.02  -0.720  -0.650  -1.19  -0.364  1.52 
    ## 10  1.82    0.748  0.125 -2.42    0.360 -1.01   -1.22   -0.326  0.350  0.200
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
    ##  1   0.735
    ##  2   0.219
    ##  3  -0.712
    ##  4   0.145
    ##  5   1.48 
    ##  6   0.541
    ##  7   0.572
    ##  8  -0.656
    ##  9  -0.956
    ## 10  -1.24 
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
    ##    outcome      X1     X2     X3      X4     X5      X6      X7     X8     X9
    ##      <dbl>   <dbl>  <dbl>  <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>
    ##  1   0.735  1.26    0.727  0.416  1.34   -0.297  0.221  -0.793   0.717 -0.515
    ##  2   0.219  0.696  -1.49  -0.229  1.19    1.94   0.677  -0.0438 -1.44   1.82 
    ##  3  -0.712  0.500   1.01   1.26  -0.374  -0.196 -0.122  -0.434   0.259 -0.340
    ##  4   0.145 -0.241   0.569 -2.24   0.860   1.19   0.642   1.04    0.735  0.735
    ##  5   1.48  -1.16   -0.491 -1.35   2.56   -1.81   0.197   0.0831  2.22  -0.522
    ##  6   0.541  0.0291  2.02  -0.898  1.30    1.16  -0.0930 -0.467  -1.62  -0.302
    ##  7   0.572  0.268  -0.447  0.657 -0.0136  0.440  0.192   0.0668  0.428  2.09 
    ##  8  -0.656  1.55   -2.16  -0.485 -0.726   0.310  0.471   1.87   -0.455  0.550
    ##  9  -0.956 -0.837  -1.32   0.303 -0.717   1.02  -0.720  -0.650  -1.19  -0.364
    ## 10  -1.24   1.82    0.748  0.125 -2.42    0.360 -1.01   -1.22   -0.326  0.350
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

    ## # A tibble: 26 x 6
    ##    variable term     estimate std.error statistic p.value
    ##    <chr>    <chr>       <dbl>     <dbl>     <dbl>   <dbl>
    ##  1 outcome  variable   1       2.65e-17   3.77e16 0      
    ##  2 X4       variable   0.150   6.70e- 2   2.24e 0 0.0259 
    ##  3 X6       variable   0.159   7.95e- 2   1.99e 0 0.0475 
    ##  4 X7       variable  -0.115   7.32e- 2  -1.57e 0 0.117  
    ##  5 X12      variable   0.0916  7.02e- 2   1.31e 0 0.193  
    ##  6 X15      variable  -0.103   7.11e- 2  -1.45e 0 0.149  
    ##  7 X25      variable  -0.198   6.90e- 2  -2.87e 0 0.00459
    ##  8 X27      variable   0.116   7.00e- 2   1.65e 0 0.0998 
    ##  9 X28      variable  -0.138   7.46e- 2  -1.85e 0 0.0652 
    ## 10 X34      variable   0.113   6.89e- 2   1.64e 0 0.102  
    ## # ... with 16 more rows

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

tidy(prefiltration_results)
```

    ## # A tibble: 12 x 5
    ##    term        estimate std.error statistic  p.value
    ##    <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)  -0.0749    0.0622     -1.20 0.230   
    ##  2 X4            0.118     0.0607      1.95 0.0530  
    ##  3 X6            0.194     0.0703      2.76 0.00640 
    ##  4 X7           -0.145     0.0644     -2.25 0.0253  
    ##  5 X12           0.145     0.0628      2.30 0.0224  
    ##  6 X25          -0.178     0.0615     -2.90 0.00415 
    ##  7 X46          -0.184     0.0608     -3.03 0.00277 
    ##  8 X59           0.230     0.0615      3.74 0.000246
    ##  9 X68           0.157     0.0657      2.40 0.0176  
    ## 10 X111          0.164     0.0609      2.70 0.00758 
    ## 11 X120          0.188     0.0592      3.18 0.00174 
    ## 12 X123         -0.131     0.0631     -2.08 0.0390

We can also calculate the R2 for this model.

``` r
glance(prefiltration_results)
```

    ## # A tibble: 1 x 12
    ##   r.squared adj.r.squared sigma statistic  p.value    df logLik   AIC   BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>    <dbl> <dbl>  <dbl> <dbl> <dbl>
    ## 1     0.298         0.257 0.857      7.15 4.33e-10    11  -243.  512.  555.
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
    ## 1 (Intercept)    0.139    0.0549      2.52 1.21e- 2
    ## 2 causal_9       0.417    0.0568      7.35 2.08e-12
    ## 3 causal_18      0.421    0.0546      7.71 2.00e-13
    ## 4 causal_45      0.589    0.0616      9.56 5.20e-19
    ## 5 causal_76      0.388    0.0558      6.94 2.51e-11
    ## 6 causal_84      0.466    0.0531      8.77 1.54e-16
    ## 7 causal_116     0.495    0.0511      9.69 1.96e-19
    ## 8 causal_121     0.440    0.0512      8.59 5.27e-16
    ## 9 causal_221     0.380    0.0538      7.06 1.26e-11

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
    ##    variable    estimate std.error statistic  p.value
    ##    <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)   0.119     0.0524      2.27 2.38e- 2
    ##  2 causal_9      0.350     0.0547      6.40 6.63e-10
    ##  3 causal_18     0.440     0.0549      8.01 3.32e-14
    ##  4 causal_45     0.539     0.0573      9.41 2.13e-18
    ##  5 causal_76     0.403     0.0525      7.68 2.82e-13
    ##  6 V80          -0.131     0.0514     -2.55 1.12e- 2
    ##  7 causal_84     0.438     0.0499      8.79 1.76e-16
    ##  8 V98          -0.142     0.0511     -2.78 5.86e- 3
    ##  9 V109         -0.120     0.0516     -2.33 2.06e- 2
    ## 10 causal_116    0.443     0.0485      9.14 1.46e-17
    ## 11 causal_121    0.463     0.0482      9.60 5.33e-19
    ## 12 V129         -0.105     0.0481     -2.19 2.94e- 2
    ## 13 V159          0.0971    0.0486      2.00 4.67e- 2
    ## 14 V177         -0.112     0.0506     -2.21 2.82e- 2
    ## 15 V206          0.120     0.0535      2.25 2.55e- 2
    ## 16 causal_221    0.314     0.0518      6.05 4.76e- 9
    ## 17 V231          0.165     0.0534      3.08 2.26e- 3
    ## 18 V272         -0.114     0.0564     -2.01 4.50e- 2

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
    ## 1 lasso             8    24             32
    ## 2 mbic              8     0              8
    ## 3 mcp               8     1              9
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
    ##  1 <split [300/113]> Bootstrap01
    ##  2 <split [300/105]> Bootstrap02
    ##  3 <split [300/107]> Bootstrap03
    ##  4 <split [300/121]> Bootstrap04
    ##  5 <split [300/107]> Bootstrap05
    ##  6 <split [300/112]> Bootstrap06
    ##  7 <split [300/105]> Bootstrap07
    ##  8 <split [300/113]> Bootstrap08
    ##  9 <split [300/110]> Bootstrap09
    ## 10 <split [300/107]> Bootstrap10

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
    ##    outcome      V1     V2      V3      V4      V5      V6       V7      V8
    ##      <dbl>   <dbl>  <dbl>   <dbl>   <dbl>   <dbl>   <dbl>    <dbl>   <dbl>
    ##  1   -5.59 -0.0449 -0.430 -1.03   -0.234   0.537   0.0803 -0.162    0.121 
    ##  2   -5.30  1.81    1.02  -0.933   1.79    0.815  -1.50   -0.0758  -0.0575
    ##  3   -4.84  1.52    0.983  0.0247  0.928   1.97    0.135   1.50     1.63  
    ##  4   -3.68  0.213   0.337 -0.892   0.0355  0.272  -1.56    0.0991   0.722 
    ##  5   -3.49 -0.0162 -0.529 -0.274   0.287   1.19   -0.559  -0.823   -1.26  
    ##  6   -3.43 -1.58   -1.79   0.525  -0.163  -0.0325  0.392   0.219   -0.371 
    ##  7   -2.91 -0.145   1.44  -1.21    0.823  -0.431  -0.309  -0.00417 -0.393 
    ##  8   -2.91 -0.145   1.44  -1.21    0.823  -0.431  -0.309  -0.00417 -0.393 
    ##  9   -2.82 -1.77   -0.885  1.37   -0.651   0.686  -0.673  -0.717    0.0153
    ## 10   -2.82 -1.77   -0.885  1.37   -0.651   0.686  -0.673  -0.717    0.0153
    ## # ... with 290 more rows, and 292 more variables: causal_9 <dbl>, V10 <dbl>,
    ## #   V11 <dbl>, V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>,
    ## #   V17 <dbl>, causal_18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>,
    ## #   V23 <dbl>, V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>, V28 <dbl>,
    ## #   V29 <dbl>, V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>,
    ## #   V35 <dbl>, V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>,
    ## #   V41 <dbl>, V42 <dbl>, V43 <dbl>, V44 <dbl>, causal_45 <dbl>, V46 <dbl>, ...

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

    ## # A tibble: 999 x 3
    ##    bootstrap variable   estimate
    ##    <chr>     <chr>         <dbl>
    ##  1 1         V6         0.000279
    ##  2 1         causal_9   0.260   
    ##  3 1         causal_18  0.285   
    ##  4 1         V32       -0.0394  
    ##  5 1         V35        0.0554  
    ##  6 1         V42       -0.00410 
    ##  7 1         causal_45  0.541   
    ##  8 1         V46        0.0209  
    ##  9 1         V56        0.0370  
    ## 10 1         V57        0.0579  
    ## # ... with 989 more rows

We can calculate stability for each variable by the number of times it
was selected across bootstraps.

``` r
model_lasso_bootstrapped %>%
  group_by(variable) %>%
  summarise(stability = (n()/bootstrap_n) * 100) %>%
  arrange(desc(stability))
```

    ## # A tibble: 272 x 2
    ##    variable   stability
    ##    <chr>          <dbl>
    ##  1 causal_116       100
    ##  2 causal_121       100
    ##  3 causal_18        100
    ##  4 causal_221       100
    ##  5 causal_45        100
    ##  6 causal_76        100
    ##  7 causal_84        100
    ##  8 causal_9         100
    ##  9 V206              90
    ## 10 V231              90
    ## # ... with 262 more rows

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
    ##  1 causal_116            0.493    0.382    0.602           0       100 *     
    ##  2 causal_121            0.434    0.330    0.551           0       100 *     
    ##  3 causal_18             0.416    0.332    0.520           0       100 *     
    ##  4 causal_45             0.589    0.464    0.716           0       100 *     
    ##  5 causal_84             0.471    0.370    0.572           0       100 *     
    ##  6 causal_9              0.405    0.316    0.526           0        96 *     
    ##  7 causal_76             0.376    0.281    0.481           0        88 *     
    ##  8 causal_221            0.373    0.250    0.484           0        81 *     
    ##  9 V231                  0.253    0.220    0.296           0         7 <NA>  
    ## 10 V69                   0.244    0.195    0.336           0         7 <NA>  
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

    ## [1] 29

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
    ## 1 lasso       8    24             32            8            0                 8
    ## 2 mbic        8     0              8            8            0                 8
    ## 3 mcp         8     1              9           NA           NA                NA
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
    ##  1 causal_116     100             0 *     
    ##  2 causal_121     100             0 *     
    ##  3 causal_45      100             0 *     
    ##  4 causal_84      100             0 *     
    ##  5 causal_18       99.5           0 *     
    ##  6 causal_9        97             0 *     
    ##  7 causal_76       95.5           0 *     
    ##  8 causal_221      91             0 *     
    ##  9 V129            46             0 <NA>  
    ## 10 V80             46             0 <NA>  
    ## # ... with 291 more rows
    ## 
    ## $combi$perm_thresh
    ## [1] 60

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
prefiltration_results
```

    ## 
    ## Call:
    ## lm(formula = outcome ~ ., data = .)
    ## 
    ## Coefficients:
    ## (Intercept)           X4           X6           X7          X12          X25  
    ##     -0.0749       0.1183       0.1938      -0.1452       0.1446      -0.1784  
    ##         X46          X59          X68         X111         X120         X123  
    ##     -0.1843       0.2299       0.1573       0.1643       0.1881      -0.1312

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

    ## # A tibble: 2 x 4
    ##   variable stability bootstrap_p stable
    ##   <chr>        <dbl>       <dbl> <chr> 
    ## 1 X59           68.8           0 *     
    ## 2 X25           60             0 *

## Conclusions

Thank you for attending this workshop. We hope you enjoyed the session,
and have a good understanding of where some conventional modelling
approaches might not be appropriate in wider datasets.

If you have any further questions after the workshop, please feel free
to contact Martin Green (<martin.green@nottingham.ac.uk>) or Robert Hyde
(<robert.hyde4@nottingham.ac.uk>).
