# Results

Run on GPU: Tesla T4 · tabpfn 10.2.0 (TabPFN-2 weights) · xgboost 3.4.1 · lightgbm 4.7.0 · TabPFN context cap 10,000

German credit source: OpenML 46562 (as in Prior Labs' example)  
Bank marketing source: UCI bank-additional-full.csv (original, date-ordered)

## A1. German credit: single 80/20 split (as in Prior Labs' example)

|    | Setting                     | Model                       |   ROC-AUC |   PR-AUC |   Log loss |   Brier |   Mean pred |   Base rate |   Fit s |   Predict s |
|---:|:----------------------------|:----------------------------|----------:|---------:|-----------:|--------:|------------:|------------:|--------:|------------:|
|  0 | German credit, single split | Prior (base rate)           |    0.5    |   0.3    |     0.6109 |  0.21   |      0.3    |         0.3 |  0      |      0.0002 |
|  1 | German credit, single split | Logistic regression         |    0.737  |   0.5663 |     0.5474 |  0.1822 |      0.3038 |         0.3 |  0.0455 |      0.0033 |
|  2 | German credit, single split | XGBoost (sensible defaults) |    0.7475 |   0.6445 |     0.5253 |  0.1723 |      0.2932 |         0.3 |  0.1351 |      0.0064 |
|  3 | German credit, single split | XGBoost (CV-tuned rounds)   |    0.7502 |   0.6467 |     0.5172 |  0.1701 |      0.2991 |         0.3 |  0.6359 |      0.0241 |
|  4 | German credit, single split | LightGBM (defaults)         |    0.7167 |   0.6089 |     0.6819 |  0.1977 |      0.2612 |         0.3 |  0.3407 |      0.0069 |
|  5 | German credit, single split | TabPFN v2 (defaults)        |    0.7555 |   0.6461 |     0.5223 |  0.1716 |      0.2909 |         0.3 |  0.6087 |      1.1244 |

## A2. Bootstrap 95% CIs on the 200-row test set

|    | Model                       |   ROC-AUC |   95% CI low |   95% CI high |
|---:|:----------------------------|----------:|-------------:|--------------:|
|  0 | Logistic regression         |     0.737 |        0.66  |         0.811 |
|  1 | XGBoost (sensible defaults) |     0.748 |        0.671 |         0.818 |
|  2 | XGBoost (CV-tuned rounds)   |     0.75  |        0.671 |         0.821 |
|  3 | LightGBM (defaults)         |     0.717 |        0.634 |         0.797 |
|  4 | TabPFN v2 (defaults)        |     0.755 |        0.677 |         0.822 |

## A3. Repeated stratified 5-fold CV (paired)

| Model                       |   AUC_mean |   AUC_sd |   LogLoss_mean |   Seconds_mean |   TabPFN minus model (mean ΔAUC) | TabPFN wins (folds)   |
|:----------------------------|-----------:|---------:|---------------:|---------------:|---------------------------------:|:----------------------|
| TabPFN v2 (defaults)        |     0.8005 |   0.0306 |         0.4861 |         1.0766 |                           0      | 0/15                  |
| XGBoost (sensible defaults) |     0.7963 |   0.03   |         0.4924 |         0.1157 |                           0.0042 | 9/15                  |
| XGBoost (CV-tuned rounds)   |     0.7929 |   0.0346 |         0.4939 |         0.4787 |                           0.0076 | 9/15                  |
| Logistic regression         |     0.7883 |   0.0376 |         0.5012 |         0.016  |                           0.0123 | 12/15                 |
| LightGBM (defaults)         |     0.7771 |   0.0278 |         0.5896 |         0.0531 |                           0.0234 | 12/15                 |

## B1. Leakage: with vs without `duration` (random stratified split)

|    | Setting                      | Model                       |   ROC-AUC |   PR-AUC |   Log loss |   Brier |   Mean pred |   Base rate |   Fit s |   Predict s |
|---:|:-----------------------------|:----------------------------|----------:|---------:|-----------:|--------:|------------:|------------:|--------:|------------:|
|  0 | WITH duration (leaky)        | Logistic regression         |    0.9424 |   0.6356 |     0.1969 |  0.0599 |      0.113  |      0.1126 |  1.7208 |      0.0893 |
|  1 | WITH duration (leaky)        | XGBoost (sensible defaults) |    0.9561 |   0.7086 |     0.1633 |  0.0517 |      0.1119 |      0.1126 |  0.7771 |      0.046  |
|  2 | WITH duration (leaky)        | TabPFN v2 (defaults)        |    0.9521 |   0.6925 |     0.1691 |  0.0537 |      0.1074 |      0.1126 |  1.084  |     33.9919 |
|  0 | WITHOUT duration (realistic) | Logistic regression         |    0.8008 |   0.4648 |     0.2717 |  0.0769 |      0.1116 |      0.1126 |  1.1717 |      0.1169 |
|  1 | WITHOUT duration (realistic) | XGBoost (sensible defaults) |    0.816  |   0.4932 |     0.2634 |  0.0744 |      0.1118 |      0.1126 |  0.9288 |      0.0861 |
|  2 | WITHOUT duration (realistic) | TabPFN v2 (defaults)        |    0.806  |   0.4711 |     0.2715 |  0.0767 |      0.0934 |      0.1126 |  1.467  |     32.949  |

## B2. Random vs temporal split (no `duration`)

|    | Setting                             | Model                       |   ROC-AUC |   PR-AUC |   Log loss |   Brier |   Mean pred |   Base rate |   Fit s |   Predict s |
|---:|:------------------------------------|:----------------------------|----------:|---------:|-----------:|--------:|------------:|------------:|--------:|------------:|
|  0 | Random stratified                   | Prior (base rate)           |    0.5    |   0.1126 |     0.352  |  0.1    |      0.1127 |      0.1126 |  0      |      0.0002 |
|  1 | Random stratified                   | Logistic regression         |    0.8008 |   0.4648 |     0.2717 |  0.0769 |      0.1116 |      0.1126 |  0.6659 |      0.0379 |
|  2 | Random stratified                   | XGBoost (sensible defaults) |    0.816  |   0.4932 |     0.2634 |  0.0744 |      0.1118 |      0.1126 |  0.6688 |      0.0593 |
|  3 | Random stratified                   | XGBoost (CV-tuned rounds)   |    0.8164 |   0.4892 |     0.2725 |  0.0772 |      0.1119 |      0.1126 |  3.1785 |      0.0344 |
|  4 | Random stratified                   | LightGBM (defaults)         |    0.8097 |   0.4956 |     0.2641 |  0.0742 |      0.1119 |      0.1126 |  0.4618 |      0.0508 |
|  5 | Random stratified                   | XGBoost @ TabPFN's N        |    0.8072 |   0.4563 |     0.2712 |  0.0771 |      0.113  |      0.1126 |  0.2401 |      0.0554 |
|  6 | Random stratified                   | TabPFN v2 (defaults)        |    0.806  |   0.4711 |     0.2715 |  0.0767 |      0.0934 |      0.1126 |  1.1574 |     34.4314 |
|  0 | Temporal (train 2008–09, test 2010) | Prior (base rate)           |    0.5    |   0.5214 |     1.2945 |  0.4346 |      0.0912 |      0.5214 |  0      |      0.0002 |
|  1 | Temporal (train 2008–09, test 2010) | Logistic regression         |    0.5943 |   0.6196 |     0.9026 |  0.3007 |      0.7126 |      0.5214 |  1.2725 |      0.0403 |
|  2 | Temporal (train 2008–09, test 2010) | XGBoost (sensible defaults) |    0.6738 |   0.6755 |     0.7608 |  0.2755 |      0.3045 |      0.5214 |  0.9027 |      0.0179 |
|  3 | Temporal (train 2008–09, test 2010) | XGBoost (CV-tuned rounds)   |    0.6733 |   0.6774 |     0.7742 |  0.2815 |      0.2909 |      0.5214 |  4.6605 |      0.0127 |
|  4 | Temporal (train 2008–09, test 2010) | LightGBM (defaults)         |    0.6722 |   0.6759 |     0.8294 |  0.2963 |      0.2613 |      0.5214 |  0.4398 |      0.0144 |
|  5 | Temporal (train 2008–09, test 2010) | XGBoost @ TabPFN's N        |    0.6544 |   0.6601 |     0.7761 |  0.2774 |      0.3179 |      0.5214 |  0.2314 |      0.0179 |
|  6 | Temporal (train 2008–09, test 2010) | TabPFN v2 (defaults)        |    0.7079 |   0.7065 |     0.6444 |  0.226  |      0.4682 |      0.5214 |  0.9318 |     23.6168 |

## B2b. Rolling-origin backtest over 2010 (9 monthly test sets)

| Model                                       |   ROC-AUC (mean of 9 months) |   ROC-AUC sd |   Log loss |   Brier |   median n_train |   s per month |
|:--------------------------------------------|-----------------------------:|-------------:|-----------:|--------:|-----------------:|--------------:|
| TabPFN v2, all history (most recent N rows) |                       0.7427 |       0.0603 |     0.6063 |  0.2092 |            40009 |       20.7967 |
| TabPFN v2, last 3 months                    |                       0.7375 |       0.0578 |     0.6111 |  0.211  |              752 |        0.6995 |
| XGBoost, all history                        |                       0.7136 |       0.0709 |     0.6288 |  0.2191 |            40009 |        0.6807 |
| Logistic regression, last 3 months          |                       0.706  |       0.0605 |     0.7957 |  0.2683 |              752 |        0.053  |
| LightGBM, last 3 months                     |                       0.6828 |       0.0748 |     0.7386 |  0.2502 |              752 |        0.0545 |
| XGBoost, last 3 months                      |                       0.6823 |       0.0649 |     0.6806 |  0.2385 |              752 |        0.1305 |
| Prior (base rate), last 3 months            |                       0.5    |       0      |     0.6961 |  0.2515 |              752 |        0.0001 |

## B3. Imbalance-aware metrics

|    | Split                               | Model                       |   Base rate |   Accuracy of always-'no' |   PR-AUC |   Precision@top10% |   Recall@top10% |   Lift@top10% |
|---:|:------------------------------------|:----------------------------|------------:|--------------------------:|---------:|-------------------:|----------------:|--------------:|
|  0 | Random stratified                   | Prior (base rate)           |       0.113 |                     0.887 |    0.113 |              0.108 |           0.096 |         0.96  |
|  1 | Random stratified                   | Logistic regression         |       0.113 |                     0.887 |    0.465 |              0.505 |           0.448 |         4.487 |
|  2 | Random stratified                   | XGBoost (sensible defaults) |       0.113 |                     0.887 |    0.493 |              0.543 |           0.482 |         4.821 |
|  3 | Random stratified                   | TabPFN v2 (defaults)        |       0.113 |                     0.887 |    0.471 |              0.52  |           0.461 |         4.617 |
|  4 | Temporal (train 2008–09, test 2010) | Prior (base rate)           |       0.521 |                     0.479 |    0.521 |              0.595 |           0.114 |         1.141 |
|  5 | Temporal (train 2008–09, test 2010) | Logistic regression         |       0.521 |                     0.479 |    0.62  |              0.741 |           0.142 |         1.422 |
|  6 | Temporal (train 2008–09, test 2010) | XGBoost (sensible defaults) |       0.521 |                     0.479 |    0.675 |              0.756 |           0.144 |         1.45  |
|  7 | Temporal (train 2008–09, test 2010) | TabPFN v2 (defaults)        |       0.521 |                     0.479 |    0.706 |              0.805 |           0.154 |         1.544 |

## B4. Scaling with training size (GPU: Tesla T4)

|    |   n_train | Model                       |   ROC-AUC |   Fit s |   Predict s (2k rows) |   ms per predicted row |
|---:|----------:|:----------------------------|----------:|--------:|----------------------:|-----------------------:|
|  0 |       250 | Logistic regression         |    0.7253 |  0.0171 |                0.0157 |                 0.0078 |
|  1 |       250 | XGBoost (sensible defaults) |    0.7045 |  0.0582 |                0.0218 |                 0.0109 |
|  2 |       250 | TabPFN v2 (defaults)        |    0.7614 |  0.1657 |                0.8611 |                 0.4306 |
|  3 |       500 | Logistic regression         |    0.7522 |  0.0484 |                0.0145 |                 0.0072 |
|  4 |       500 | XGBoost (sensible defaults) |    0.7339 |  0.0817 |                0.0239 |                 0.012  |
|  5 |       500 | TabPFN v2 (defaults)        |    0.7676 |  0.1583 |                1.0677 |                 0.5339 |
|  6 |      1000 | Logistic regression         |    0.7744 |  0.0292 |                0.014  |                 0.007  |
|  7 |      1000 | XGBoost (sensible defaults) |    0.7419 |  0.1069 |                0.0197 |                 0.0098 |
|  8 |      1000 | TabPFN v2 (defaults)        |    0.7743 |  0.221  |                1.5191 |                 0.7595 |
|  9 |      2000 | Logistic regression         |    0.7745 |  0.0448 |                0.0134 |                 0.0067 |
| 10 |      2000 | XGBoost (sensible defaults) |    0.7509 |  0.136  |                0.0168 |                 0.0084 |
| 11 |      2000 | TabPFN v2 (defaults)        |    0.7682 |  0.3011 |                2.653  |                 1.3265 |
| 12 |      5000 | Logistic regression         |    0.7766 |  0.1788 |                0.0152 |                 0.0076 |
| 13 |      5000 | XGBoost (sensible defaults) |    0.7624 |  0.2591 |                0.0253 |                 0.0127 |
| 14 |      5000 | TabPFN v2 (defaults)        |    0.7792 |  1.231  |                7.9096 |                 3.9548 |
| 15 |     10000 | Logistic regression         |    0.7796 |  0.4137 |                0.0334 |                 0.0167 |
| 16 |     10000 | XGBoost (sensible defaults) |    0.7772 |  0.3073 |                0.0205 |                 0.0102 |
| 17 |     10000 | TabPFN v2 (defaults)        |    0.7813 |  1.2121 |               23.2803 |                11.6401 |

## B5a. Expected calibration error (lower is better)

|    | Split                               | Model                       |    ECE |
|---:|:------------------------------------|:----------------------------|-------:|
|  0 | Random stratified                   | Logistic regression         | 0.0131 |
|  1 | Random stratified                   | XGBoost (sensible defaults) | 0.0104 |
|  2 | Random stratified                   | TabPFN v2 (defaults)        | 0.0202 |
|  3 | Temporal (train 2008–09, test 2010) | Logistic regression         | 0.2004 |
|  4 | Temporal (train 2008–09, test 2010) | XGBoost (sensible defaults) | 0.2169 |
|  5 | Temporal (train 2008–09, test 2010) | TabPFN v2 (defaults)        | 0.0893 |

## B5b. Paired bootstrap ΔAUC (TabPFN minus baseline)

|    | Split                               | Comparison                           |    ΔAUC |   95% CI low |   95% CI high |   share of resamples TabPFN ahead |
|---:|:------------------------------------|:-------------------------------------|--------:|-------------:|--------------:|----------------------------------:|
|  0 | Random stratified                   | TabPFN − XGBoost (sensible defaults) | -0.01   |      -0.0168 |       -0.003  |                             0.006 |
|  1 | Random stratified                   | TabPFN − XGBoost @ TabPFN's N        | -0.0009 |      -0.0083 |        0.0071 |                             0.386 |
|  2 | Random stratified                   | TabPFN − Logistic regression         |  0.0053 |      -0.0008 |        0.0111 |                             0.95  |
|  3 | Temporal (train 2008–09, test 2010) | TabPFN − XGBoost (sensible defaults) |  0.0334 |       0.0125 |        0.0544 |                             1     |
|  4 | Temporal (train 2008–09, test 2010) | TabPFN − XGBoost @ TabPFN's N        |  0.053  |       0.0327 |        0.0732 |                             1     |
|  5 | Temporal (train 2008–09, test 2010) | TabPFN − Logistic regression         |  0.1137 |       0.0888 |        0.1352 |                             1     |

## B5c. Seed sensitivity (2,000-row test subset, 3 seeds)

| Model                                 |   mean |    std |    min |    max |
|:--------------------------------------|-------:|-------:|-------:|-------:|
| TabPFN v2 (context sample + seed)     | 0.7825 | 0.001  | 0.7813 | 0.7833 |
| XGBoost (row/column subsampling seed) | 0.7843 | 0.0014 | 0.783  | 0.7858 |

