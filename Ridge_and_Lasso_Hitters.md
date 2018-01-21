Hitters\_Lasso
================
Amit
February 26, 2017

Ridge Regression and Lasso Regression
-------------------------------------

I will use the package `glmnet`.There are some missing values here, so before proceeding I will remove them: Lets make a training and validation set

``` r
library(ISLR)
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loaded glmnet 2.0-5

``` r
library(plotmo)
```

    ## Loading required package: plotrix

    ## Loading required package: TeachingDemos

Hitters dataset-Major League Baseball Data from the 1986 and 1987 seasons For description of dataset, please refer -<https://cran.r-project.org/web/packages/ISLR/ISLR.pdf>

``` r
summary(Hitters)
```

    ##      AtBat            Hits         HmRun            Runs       
    ##  Min.   : 16.0   Min.   :  1   Min.   : 0.00   Min.   :  0.00  
    ##  1st Qu.:255.2   1st Qu.: 64   1st Qu.: 4.00   1st Qu.: 30.25  
    ##  Median :379.5   Median : 96   Median : 8.00   Median : 48.00  
    ##  Mean   :380.9   Mean   :101   Mean   :10.77   Mean   : 50.91  
    ##  3rd Qu.:512.0   3rd Qu.:137   3rd Qu.:16.00   3rd Qu.: 69.00  
    ##  Max.   :687.0   Max.   :238   Max.   :40.00   Max.   :130.00  
    ##                                                                
    ##       RBI             Walks            Years            CAtBat       
    ##  Min.   :  0.00   Min.   :  0.00   Min.   : 1.000   Min.   :   19.0  
    ##  1st Qu.: 28.00   1st Qu.: 22.00   1st Qu.: 4.000   1st Qu.:  816.8  
    ##  Median : 44.00   Median : 35.00   Median : 6.000   Median : 1928.0  
    ##  Mean   : 48.03   Mean   : 38.74   Mean   : 7.444   Mean   : 2648.7  
    ##  3rd Qu.: 64.75   3rd Qu.: 53.00   3rd Qu.:11.000   3rd Qu.: 3924.2  
    ##  Max.   :121.00   Max.   :105.00   Max.   :24.000   Max.   :14053.0  
    ##                                                                      
    ##      CHits            CHmRun           CRuns             CRBI        
    ##  Min.   :   4.0   Min.   :  0.00   Min.   :   1.0   Min.   :   0.00  
    ##  1st Qu.: 209.0   1st Qu.: 14.00   1st Qu.: 100.2   1st Qu.:  88.75  
    ##  Median : 508.0   Median : 37.50   Median : 247.0   Median : 220.50  
    ##  Mean   : 717.6   Mean   : 69.49   Mean   : 358.8   Mean   : 330.12  
    ##  3rd Qu.:1059.2   3rd Qu.: 90.00   3rd Qu.: 526.2   3rd Qu.: 426.25  
    ##  Max.   :4256.0   Max.   :548.00   Max.   :2165.0   Max.   :1659.00  
    ##                                                                      
    ##      CWalks        League  Division    PutOuts          Assists     
    ##  Min.   :   0.00   A:175   E:157    Min.   :   0.0   Min.   :  0.0  
    ##  1st Qu.:  67.25   N:147   W:165    1st Qu.: 109.2   1st Qu.:  7.0  
    ##  Median : 170.50                    Median : 212.0   Median : 39.5  
    ##  Mean   : 260.24                    Mean   : 288.9   Mean   :106.9  
    ##  3rd Qu.: 339.25                    3rd Qu.: 325.0   3rd Qu.:166.0  
    ##  Max.   :1566.00                    Max.   :1378.0   Max.   :492.0  
    ##                                                                     
    ##      Errors          Salary       NewLeague
    ##  Min.   : 0.00   Min.   :  67.5   A:176    
    ##  1st Qu.: 3.00   1st Qu.: 190.0   N:146    
    ##  Median : 6.00   Median : 425.0            
    ##  Mean   : 8.04   Mean   : 535.9            
    ##  3rd Qu.:11.00   3rd Qu.: 750.0            
    ##  Max.   :32.00   Max.   :2460.0            
    ##                  NA's   :59

``` r
dim(Hitters)
```

    ## [1] 322  20

``` r
Hitters=na.omit(Hitters)
with(Hitters,sum(is.na(Salary)))
```

    ## [1] 0

``` r
dim(Hitters)
```

    ## [1] 263  20

``` r
set.seed(1)
train=sample(seq(263),180,replace=FALSE)
train
```

    ##   [1]  70  98 150 237  53 232 243 170 161  16 259  45 173  97 192 124 178
    ##  [18] 245  94 190 228  52 158  31  64  92   4  91 205  80 113 140 115  43
    ##  [35] 244 153 181  25 163  93 184 144 174 122 117 251   6 104 241 149 102
    ##  [52] 183 224 242  15  21  66 107 136  83 186  60 211  67 130 210  95 151
    ##  [69]  17 256 207 162 200 239 236 168 249  73 222 177 234 199 203  59 235
    ##  [86]  37 126  22 230 226  42  11 110 214 132 134  77  69 188 100 206  58
    ## [103]  44 159 101  34 208  75 185 201 261 112  54  65  23   2 106 254 257
    ## [120] 154 142  71 166 221 105  63 143  29 240 212 167 172   5  84 120 133
    ## [137]  72 191 248 138 182  74 179 135  87 196 157 119  13  99 263 125 247
    ## [154]  50  55  20  57   8  30 194 139 238  46  78  88  41   7  33 141  32
    ## [171] 180 164 213  36 215  79 225 229 198  76

``` r
library(glmnet)
x=model.matrix(Salary~.-1,data=Hitters) 
y=Hitters$Salary
```

Fit a ridge-regression model.

This can be done by calling `glmnet` with `alpha=0`

``` r
fit.ridge=glmnet(x,y,alpha=0)
par(mfrow=c(1,2))
plot_glmnet(fit.ridge,xvar="lambda",label=5)

plot_glmnet(fit.ridge,label=5)
```

![](README_figs/README-unnamed-chunk-6-1.png)

``` r
cv.ridge=cv.glmnet(x,y,alpha=0)
plot(cv.ridge)
```

![](README_figs/README-unnamed-chunk-6-2.png)

To fit a lasso model; `alpha=1`

``` r
fit.lasso=glmnet(x,y)
par(mfrow=c(1,2))
plot_glmnet(fit.lasso,xvar="lambda",label=5)
plot_glmnet(fit.lasso,label=5)
```

![](README_figs/README-unnamed-chunk-7-1.png)

``` r
cv.lasso=cv.glmnet(x,y)
plot(cv.lasso, label=5)
```

    ## Warning in plot.window(...): "label" is not a graphical parameter

    ## Warning in plot.xy(xy, type, ...): "label" is not a graphical parameter

    ## Warning in axis(side = side, at = at, labels = labels, ...): "label" is not
    ## a graphical parameter

    ## Warning in axis(side = side, at = at, labels = labels, ...): "label" is not
    ## a graphical parameter

    ## Warning in box(...): "label" is not a graphical parameter

    ## Warning in title(...): "label" is not a graphical parameter

``` r
coef(cv.lasso)
```

    ## 21 x 1 sparse Matrix of class "dgCMatrix"
    ##                       1
    ## (Intercept) 115.3773590
    ## AtBat         .        
    ## Hits          1.4753071
    ## HmRun         .        
    ## Runs          .        
    ## RBI           .        
    ## Walks         1.6566947
    ## Years         .        
    ## CAtBat        .        
    ## CHits         .        
    ## CHmRun        .        
    ## CRuns         0.1660465
    ## CRBI          0.3453397
    ## CWalks        .        
    ## LeagueA       .        
    ## LeagueN       .        
    ## DivisionW   -19.2435216
    ## PutOuts       0.1000068
    ## Assists       .        
    ## Errors        .        
    ## NewLeagueN    .

![](README_figs/README-unnamed-chunk-7-2.png)

Alternative approach- selection of `lambda` using our earlier train/validation division for the lasso.

``` r
lasso.tr=glmnet(x[train,],y[train])
pred=predict(lasso.tr,x[-train,])
dim(pred)
```

    ## [1] 83 89

``` r
rmse= sqrt(apply((y[-train]-pred)^2,2,mean))
plot(log(lasso.tr$lambda),rmse,type="b",xlab="Log(lambda)")
```

![](README_figs/README-unnamed-chunk-8-1.png)

``` r
lam.best=lasso.tr$lambda[order(rmse)[1]]
lam.best
```

    ## [1] 19.98706

``` r
coef(lasso.tr,s=lam.best)
```

    ## 21 x 1 sparse Matrix of class "dgCMatrix"
    ##                        1
    ## (Intercept)  107.9416686
    ## AtBat          .        
    ## Hits           0.1591252
    ## HmRun          .        
    ## Runs           .        
    ## RBI            1.7340039
    ## Walks          3.4657091
    ## Years          .        
    ## CAtBat         .        
    ## CHits          .        
    ## CHmRun         .        
    ## CRuns          0.5386855
    ## CRBI           .        
    ## CWalks         .        
    ## LeagueA      -30.0493021
    ## LeagueN        .        
    ## DivisionW   -113.8317016
    ## PutOuts        0.2915409
    ## Assists        .        
    ## Errors         .        
    ## NewLeagueN     2.0367518
