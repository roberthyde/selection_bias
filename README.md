
<!-- badges: start -->
<!-- badges: end -->

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
  data_frame(replicate(ncols, rnorm(nrows, 0, 1)))
}
```

A dataset with 197 rows and 130 variables can then be generated using
this function as follows:

``` r
variables <- generate_uncor_variables(ncols = 130, nrows = 197)
```

This results in the following dataset being generated:

``` r
variables
#> # A tibble: 197 x 1
#>    `replicate(ncols,~   [,2]    [,3]    [,4]   [,5]    [,6]   [,7]   [,8]   [,9]
#>                 <dbl>  <dbl>   <dbl>   <dbl>  <dbl>   <dbl>  <dbl>  <dbl>  <dbl>
#>  1             -1.82   0.492  1.22   -0.239  -0.673 -1.18    1.01  -2.01   0.278
#>  2             -0.927 -2.71  -1.28   -0.844   0.112  1.01   -0.172  3.28  -1.08 
#>  3              0.382  1.09  -0.923  -0.759   0.182 -0.737  -0.307  0.980  0.468
#>  4              0.234  0.989  1.53   -0.430   1.86   0.371  -0.640 -0.376 -0.189
#>  5              0.135  0.706  1.36    0.645   0.103 -0.292   0.972  0.407 -0.390
#>  6              0.667 -0.618 -0.0112  1.08   -2.89  -2.11   -0.803  0.265  0.887
#>  7              0.209 -1.46  -0.650  -0.0792  0.482 -1.74   -0.522 -1.16  -0.883
#>  8              0.317 -0.590  0.525   0.394  -0.800  0.180  -0.653 -0.443  0.509
#>  9             -0.282 -1.74  -0.705  -0.826  -0.790 -0.0855 -0.332  0.113 -0.343
#> 10              0.468  1.33  -1.39   -0.382  -1.19   0.737   0.993  1.36   1.87 
#> # ... with 187 more rows
```

We can also generate an outcome variable, in this case randomly
generated in the same manner, but renaming as “outcome”

``` r
generate_uncor_outcome <- function(nrows) {
  data_frame(replicate(1, rnorm(nrows, 0, 1))) %>%
    rename("outcome" = 1)
}

outcome <- generate_uncor_outcome(nrows = 197)

outcome
#> # A tibble: 197 x 1
#>    outcome[,1]
#>          <dbl>
#>  1     -1.93  
#>  2      0.318 
#>  3     -0.588 
#>  4     -0.0161
#>  5     -0.593 
#>  6      0.797 
#>  7      2.23  
#>  8      0.857 
#>  9     -0.198 
#> 10     -0.371 
#> # ... with 187 more rows
```

We can now bind together the uncorrelated, randomly generated variables,
with the randomly generated outcome.

``` r
df_no_signal <- outcome %>%
  bind_cols(variables)
```

This results in a dataset of 197 rows, with a single outcome variable,
which has no relationship to the 130 columns as shown below.

``` r
df_no_signal
#> # A tibble: 197 x 2
#>    outcome[,1] `replicate(ncols, r~   [,2]    [,3]    [,4]   [,5]    [,6]   [,7]
#>          <dbl>                <dbl>  <dbl>   <dbl>   <dbl>  <dbl>   <dbl>  <dbl>
#>  1     -1.93                 -1.82   0.492  1.22   -0.239  -0.673 -1.18    1.01 
#>  2      0.318                -0.927 -2.71  -1.28   -0.844   0.112  1.01   -0.172
#>  3     -0.588                 0.382  1.09  -0.923  -0.759   0.182 -0.737  -0.307
#>  4     -0.0161                0.234  0.989  1.53   -0.430   1.86   0.371  -0.640
#>  5     -0.593                 0.135  0.706  1.36    0.645   0.103 -0.292   0.972
#>  6      0.797                 0.667 -0.618 -0.0112  1.08   -2.89  -2.11   -0.803
#>  7      2.23                  0.209 -1.46  -0.650  -0.0792  0.482 -1.74   -0.522
#>  8      0.857                 0.317 -0.590  0.525   0.394  -0.800  0.180  -0.653
#>  9     -0.198                -0.282 -1.74  -0.705  -0.826  -0.790 -0.0855 -0.332
#> 10     -0.371                 0.468  1.33  -1.39   -0.382  -1.19   0.737   0.993
#> # ... with 187 more rows
```

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
univariable_outcomes_filtered
#> # A tibble: 18 x 6
#>    variable                       term      estimate std.error statistic p.value
#>    <chr>                          <chr>        <dbl>     <dbl>     <dbl>   <dbl>
#>  1 outcome                        variable     1      3.60e-18   2.78e17  0     
#>  2 replicate(ncols, rnorm(nrows,~ variable7   -0.323  1.38e- 1  -2.33e 0  0.0228
#>  3 replicate(ncols, rnorm(nrows,~ variable~    0.176  1.31e- 1   1.35e 0  0.182 
#>  4 replicate(ncols, rnorm(nrows,~ variable~   -0.176  1.33e- 1  -1.32e 0  0.190 
#>  5 replicate(ncols, rnorm(nrows,~ variable~   -0.205  1.26e- 1  -1.62e 0  0.110 
#>  6 replicate(ncols, rnorm(nrows,~ variable~   -0.163  1.07e- 1  -1.53e 0  0.132 
#>  7 replicate(ncols, rnorm(nrows,~ variable~    0.238  1.36e- 1   1.75e 0  0.0844
#>  8 replicate(ncols, rnorm(nrows,~ variable~    0.291  1.45e- 1   2.01e 0  0.0491
#>  9 replicate(ncols, rnorm(nrows,~ variable~   -0.314  1.26e- 1  -2.48e 0  0.0157
#> 10 replicate(ncols, rnorm(nrows,~ variable~   -0.216  1.50e- 1  -1.44e 0  0.154 
#> 11 replicate(ncols, rnorm(nrows,~ variable~   -0.166  1.18e- 1  -1.40e 0  0.165 
#> 12 replicate(ncols, rnorm(nrows,~ variable~    0.246  1.45e- 1   1.69e 0  0.0953
#> 13 replicate(ncols, rnorm(nrows,~ variable~    0.304  1.34e- 1   2.27e 0  0.0264
#> 14 replicate(ncols, rnorm(nrows,~ variable~    0.220  1.38e- 1   1.59e 0  0.116 
#> 15 replicate(ncols, rnorm(nrows,~ variable~   -0.195  1.32e- 1  -1.47e 0  0.146 
#> 16 replicate(ncols, rnorm(nrows,~ variable~    0.246  1.47e- 1   1.67e 0  0.0989
#> 17 replicate(ncols, rnorm(nrows,~ variable~    0.186  1.34e- 1   1.39e 0  0.169 
#> 18 replicate(ncols, rnorm(nrows,~ variable~    0.216  1.39e- 1   1.56e 0  0.124
```

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

prefiltration_results
#> # A tibble: 0 x 5
#> # ... with 5 variables: variable <chr>, estimate <dbl>, std.error <dbl>,
#> #   statistic <dbl>, p.value <dbl>
```

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
  X_data <- as_data_frame(X)
  for (i in c(nonzero)) {
    X_data1 <- X_data %>%
      rename_with(.cols = i, ~ paste("causal_", i, sep = ""))
    X_data <- X_data1
  }

  dataset_sim <- as_data_frame(cbind(outcome, X_data1))
}
```

Plot signal strengths

True “cheat” model

``` r
df_signal <- generate_data_with_signal(nrow = 300, ncol = 300, n_causal_vars = 8, amplitude = 7)
univariable_outcomes <- map_df(df_signal, ~ univariable_analysis(data = df_signal, variable = .), .id = "variable")
univariable_outcomes_filtered <- univariable_outcomes %>%
  filter(p.value < 0.2)
variables_for_stepwise <- univariable_outcomes_filtered %>%
  pull(variable)
stepwise_model(data = df_signal, variables = variables_for_stepwise)
#> # A tibble: 19 x 5
#>    variable   estimate std.error statistic  p.value
#>    <chr>         <dbl>     <dbl>     <dbl>    <dbl>
#>  1 causal_27     0.349    0.0561      6.22 1.91e- 9
#>  2 V28           0.120    0.0552      2.17 3.07e- 2
#>  3 V79          -0.140    0.0529     -2.65 8.55e- 3
#>  4 V81          -0.133    0.0584     -2.27 2.40e- 2
#>  5 causal_104    0.381    0.0537      7.11 1.09e-11
#>  6 causal_110    0.274    0.0586      4.68 4.49e- 6
#>  7 V138         -0.170    0.0582     -2.92 3.83e- 3
#>  8 V146         -0.139    0.0521     -2.66 8.28e- 3
#>  9 causal_156    0.389    0.0516      7.54 7.46e-13
#> 10 V165          0.128    0.0521      2.45 1.51e- 2
#> 11 causal_169    0.377    0.0555      6.79 7.10e-11
#> 12 V178          0.153    0.0560      2.72 6.90e- 3
#> 13 causal_179    0.352    0.0546      6.45 5.20e-10
#> 14 V207         -0.141    0.0554     -2.54 1.15e- 2
#> 15 causal_228    0.388    0.0548      7.08 1.30e-11
#> 16 V235         -0.107    0.0538     -1.99 4.80e- 2
#> 17 V251         -0.112    0.0485     -2.31 2.16e- 2
#> 18 V266         -0.182    0.0568     -3.19 1.57e- 3
#> 19 causal_286    0.455    0.0550      8.28 5.91e-15
```

## Regularisation

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

  data_frame(name = coefs@Dimnames[[1]][coefs@i + 1], coefficient = coefs@x) %>%
    rename(
      variable = name,
      estimate = coefficient
    ) %>%
    filter(variable != "(Intercept)") %>%
    select(variable, estimate)
}

model_results$lasso <- model_lasso(df_signal)
```

MCP can also be used

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

MBIC can also be used

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
    as_data_frame() %>%
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
```

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
#> # Bootstrap sampling 
#> # A tibble: 10 x 2
#>    splits            id         
#>    <list>            <chr>      
#>  1 <split [300/108]> Bootstrap01
#>  2 <split [300/100]> Bootstrap02
#>  3 <split [300/98]>  Bootstrap03
#>  4 <split [300/110]> Bootstrap04
#>  5 <split [300/122]> Bootstrap05
#>  6 <split [300/103]> Bootstrap06
#>  7 <split [300/113]> Bootstrap07
#>  8 <split [300/113]> Bootstrap08
#>  9 <split [300/106]> Bootstrap09
#> 10 <split [300/108]> Bootstrap10
```

If we extract a single bootstrapped dataset and sort by the outcome, we
can see that several rows have been resampled. Consequently as the
dataset length is the same as the original, several rows will be omitted
completely.

``` r
bootstrapped_datasets$splits[[1]] %>%
  as_tibble() %>%
  arrange(outcome)
#> # A tibble: 300 x 301
#>    outcome     V1     V2      V3       V4      V5     V6      V7     V8     V9
#>      <dbl>  <dbl>  <dbl>   <dbl>    <dbl>   <dbl>  <dbl>   <dbl>  <dbl>  <dbl>
#>  1   -4.46  0.296  1.31   1.90    0.00317 -0.369  -2.49  -0.446  -0.339 -1.19 
#>  2   -4.13 -0.738 -1.37  -0.0636  0.569   -0.0802  0.580 -0.0842  0.234 -0.720
#>  3   -4.13 -0.738 -1.37  -0.0636  0.569   -0.0802  0.580 -0.0842  0.234 -0.720
#>  4   -3.41 -1.85   0.666  2.68    2.55     0.725  -0.420 -0.105   0.487 -0.854
#>  5   -3.41 -1.85   0.666  2.68    2.55     0.725  -0.420 -0.105   0.487 -0.854
#>  6   -3.21  1.36   1.42  -0.370  -1.03     0.770   0.777  0.333  -2.06  -0.701
#>  7   -3.21  1.36   1.42  -0.370  -1.03     0.770   0.777  0.333  -2.06  -0.701
#>  8   -3.12 -0.342 -0.606 -0.150  -1.82    -1.99   -0.120 -1.45    0.108  1.84 
#>  9   -2.98 -1.08   0.327  0.931  -0.472    0.535  -0.372 -0.177   1.05   0.454
#> 10   -2.98 -1.08   0.327  0.931  -0.472    0.535  -0.372 -0.177   1.05   0.454
#> # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
#> #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
#> #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
#> #   V24 <dbl>, V25 <dbl>, V26 <dbl>, causal_27 <dbl>, V28 <dbl>, V29 <dbl>,
#> #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
#> #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
#> #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...
```

## Model for bootstraps (lasso)

We can apply our previous lasso function over each one of these
bootstrapped resamples.

``` r
model_lasso_bootstrapped <- bootstrapped_datasets %>%
  map_df(.x = .$splits, .f = ~ as_data_frame(.) %>% model_lasso(.), .id = "bootstrap")
```

## Permutation

To identify a null threshold, first we must permute the outcome.

Our original dataset looks like this:

``` r
df_signal
#> # A tibble: 300 x 301
#>     outcome      V1     V2      V3      V4     V5     V6      V7      V8      V9
#>       <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>   <dbl>
#>  1  1.80     0.144  -0.100 -0.0382  0.879   0.278 -1.27  -0.539  -0.998   1.53  
#>  2  0.471   -1.50    1.05   0.484  -0.0587  0.237 -0.372 -0.144  -0.558   0.529 
#>  3 -1.49     1.60   -0.404 -0.392  -0.474  -0.505 -0.366  1.36   -0.868   1.29  
#>  4 -1.76     0.554   1.19   0.323   0.0162 -1.05  -0.424  0.679  -0.709  -0.0551
#>  5  1.67    -1.23   -0.672 -0.434   0.492   0.336 -0.509 -1.42   -0.0872  0.551 
#>  6  1.49    -0.314   0.376  1.17   -0.351  -0.763 -0.651  2.25    0.546   0.250 
#>  7  0.0688  -0.318  -1.26  -1.42   -0.177   0.109 -0.155  1.27   -0.763   1.61  
#>  8  1.14    -0.0992 -0.905  1.23    0.397   0.498  0.502  0.360  -0.165   1.63  
#>  9  4.52    -0.358   1.19  -1.11   -1.99   -1.09   0.811  0.752   0.156  -0.0143
#> 10 -0.00138 -0.717   0.801 -0.139  -0.877  -2.71   0.705 -0.0844  0.0420  1.43  
#> # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
#> #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
#> #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
#> #   V24 <dbl>, V25 <dbl>, V26 <dbl>, causal_27 <dbl>, V28 <dbl>, V29 <dbl>,
#> #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
#> #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
#> #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...
```

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
#> # A tibble: 300 x 301
#>    outcome      V1     V2      V3      V4     V5     V6      V7      V8      V9
#>      <dbl>   <dbl>  <dbl>   <dbl>   <dbl>  <dbl>  <dbl>   <dbl>   <dbl>   <dbl>
#>  1 -1.20    0.144  -0.100 -0.0382  0.879   0.278 -1.27  -0.539  -0.998   1.53  
#>  2 -0.0189 -1.50    1.05   0.484  -0.0587  0.237 -0.372 -0.144  -0.558   0.529 
#>  3  0.713   1.60   -0.404 -0.392  -0.474  -0.505 -0.366  1.36   -0.868   1.29  
#>  4  0.560   0.554   1.19   0.323   0.0162 -1.05  -0.424  0.679  -0.709  -0.0551
#>  5  2.12   -1.23   -0.672 -0.434   0.492   0.336 -0.509 -1.42   -0.0872  0.551 
#>  6  0.481  -0.314   0.376  1.17   -0.351  -0.763 -0.651  2.25    0.546   0.250 
#>  7 -0.960  -0.318  -1.26  -1.42   -0.177   0.109 -0.155  1.27   -0.763   1.61  
#>  8 -4.46   -0.0992 -0.905  1.23    0.397   0.498  0.502  0.360  -0.165   1.63  
#>  9  0.249  -0.358   1.19  -1.11   -1.99   -1.09   0.811  0.752   0.156  -0.0143
#> 10  0.108  -0.717   0.801 -0.139  -0.877  -2.71   0.705 -0.0844  0.0420  1.43  
#> # ... with 290 more rows, and 291 more variables: V10 <dbl>, V11 <dbl>,
#> #   V12 <dbl>, V13 <dbl>, V14 <dbl>, V15 <dbl>, V16 <dbl>, V17 <dbl>,
#> #   V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>, V22 <dbl>, V23 <dbl>,
#> #   V24 <dbl>, V25 <dbl>, V26 <dbl>, causal_27 <dbl>, V28 <dbl>, V29 <dbl>,
#> #   V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>, V34 <dbl>, V35 <dbl>,
#> #   V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>, V40 <dbl>, V41 <dbl>,
#> #   V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>, V46 <dbl>, V47 <dbl>, ...
```

We can then apply our bootstrap function to each one of these 5 permuted
datasets. We might perform 10 bootstrap samples for each of the 5
permuted datasets. The model would then be applied to each dataset
within the following table.

``` r
permuted_bootstrapped_datasets <- permuted_datasets %>%
  map_df(.x = .$splits, .f = ~ as_data_frame(.) %>% boot_sample(., boot_reps = 10), .id = "permutation")

permuted_bootstrapped_datasets
#> # A tibble: 50 x 3
#>    permutation splits            id         
#>    <chr>       <list>            <chr>      
#>  1 1           <split [300/123]> Bootstrap01
#>  2 1           <split [300/116]> Bootstrap02
#>  3 1           <split [300/107]> Bootstrap03
#>  4 1           <split [300/112]> Bootstrap04
#>  5 1           <split [300/109]> Bootstrap05
#>  6 1           <split [300/115]> Bootstrap06
#>  7 1           <split [300/106]> Bootstrap07
#>  8 1           <split [300/107]> Bootstrap08
#>  9 1           <split [300/121]> Bootstrap09
#> 10 1           <split [300/104]> Bootstrap10
#> # ... with 40 more rows
```

This code is relatively lengthy, and is therefore deliberately omitted
from the workshop, however is present within the stabiliser package and
freely available.

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

## stabiliser approach

``` r
stab_output <- stabilise(outcome = "outcome", data = df_signal, models = c("mbic"), type = "linear")

stab_output$mbic$stability
#> # A tibble: 301 x 7
#>    variable   mean_coefficient ci_lower ci_upper bootstrap_p stability stable
#>    <chr>                 <dbl>    <dbl>    <dbl>       <dbl>     <dbl> <chr> 
#>  1 causal_179            0.433    0.288    0.582           0       100 *     
#>  2 causal_228            0.420    0.309    0.528           0        98 *     
#>  3 causal_27             0.458    0.346    0.595           0        98 *     
#>  4 causal_156            0.407    0.320    0.517           0        97 *     
#>  5 causal_104            0.394    0.293    0.509           0        95 *     
#>  6 causal_169            0.454    0.322    0.569           0        95 *     
#>  7 causal_110            0.403    0.297    0.528           0        91 *     
#>  8 causal_286            0.430    0.338    0.545           0        89 *     
#>  9 V207                 -0.265   -0.307   -0.236           0        13 <NA>  
#> 10 V235                 -0.243   -0.277   -0.205           0         6 <NA>  
#> # ... with 291 more rows
```

## All models

The stabiliser package allows multiple models to be run simultaneously.
Just select the models you wish to run in the “models” argument.

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
#> # A tibble: 2 x 4
#>   model_stability tp_stability fp_stability total_selected_stability
#>   <chr>                  <int>        <int>                    <int>
#> 1 lasso                      8            0                        8
#> 2 mbic                       8            0                        8
```

Compare this with the non-stability approach

``` r
conventional_results %>%
  left_join(stability_results, by = c("model" = "model_stability"))
#> # A tibble: 3 x 7
#>   model    tp    fp total_selected tp_stability fp_stability total_selected_sta~
#>   <chr> <int> <int>          <int>        <int>        <int>               <int>
#> 1 lasso     8    34             42            8            0                   8
#> 2 mbic      0     9              9            8            0                   8
#> 3 mcp       8     3             11           NA           NA                  NA
```

# Triangulation

The stabiliser package allows the stability selection results from
multiple models to be used synergistically, and by leveraging the
strenghts of various models, a more robust method of variable selection
is often acheived.

``` r
triangulated_stability <- triangulate(stab_output)

triangulated_stability
#> $combi
#> $combi$stability
#> # A tibble: 301 x 4
#>    variable   stability bootstrap_p stable
#>    <chr>          <dbl>       <dbl> <chr> 
#>  1 causal_156     100             0 *     
#>  2 causal_179     100             0 *     
#>  3 causal_228      98.5           0 *     
#>  4 causal_104      98             0 *     
#>  5 causal_27       98             0 *     
#>  6 causal_169      96.5           0 *     
#>  7 causal_110      95             0 *     
#>  8 causal_286      91.5           0 *     
#>  9 V207            50             0 <NA>  
#> 10 V165            46.5           0 <NA>  
#> # ... with 291 more rows
#> 
#> $combi$perm_thresh
#> [1] 65.5
```

``` r
stab_plot(triangulated_stability)
#> $combi
```

<img src="README_files/figure-gfm/unnamed-chunk-35-1.png" width="100%" />

## No signal datasets

We can now return to our original dataset that we simulated to have no
signal.

Our conventional approach performed relatively poorly, selecting the
following variables as being significantly associated with the outcome
variable.

``` r
prefiltration_results
#> # A tibble: 0 x 5
#> # ... with 5 variables: variable <chr>, estimate <dbl>, std.error <dbl>,
#> #   statistic <dbl>, p.value <dbl>
```

Using stabiliser, the following variables are selected from the dataset.

``` r
stab_output_no_signal <- stabilise(outcome = "outcome", data = df_no_signal, models = c("mbic", "lasso"), type = "linear")

triangulated_output_no_signal <- triangulate(stab_output_no_signal)

triangulated_output_no_signal$combi$stability %>%
  filter(stable == "*")
#> # A tibble: 2 x 4
#>   variable                             stability bootstrap_p stable
#>   <chr>                                    <dbl>       <dbl> <chr> 
#> 1 outcome                                      0         NaN *     
#> 2 replicate(ncols, rnorm(nrows, 0, 1))         0         NaN *
```

## Conclusions

Thank you for attending this workshop. We hope you enjoyed the session,
and have a good understanding of where some conventional modelling
approaches might not be appropriate in wider datasets.

If you have any further questions after the workshop, please feel free
to contact Martin Green (<martin.green@nottingham.ac.uk>) or Robert Hyde
(<robert.hyde4@nottingham.ac.uk>).
