library(tidyr)
library(surveillance)

data <- read.csv("Influenza.csv")
adjacentMatrix <- read.csv("AdjacentMatrix.csv", check.names=FALSE,
                           row.names='')
df <- pivot_wider(data[c('value', 'date', 'location')], names_from = location, values_from = value)

df_sts <-sts(as.matrix(subset(df, select=-c(date))),
             start = c(2002, 1) , frequency = 52,
             neighbourhood = as.matrix(adjacentMatrix))

plot(fluBYBW, type = observed ~ time)
plot(df_sts, type = observed ~ time)


?hhh4


## Not run: 

### last but not least, a more sophisticated (and time-consuming)

### analysis of weekly counts of influenza from 140 districts in

### Southern Germany (originally analysed by Paul and Held, 2011,

### and revisited by Held and Paul, 2012, and Meyer and Held, 2014)

data("fluBYBW")

class(fluBYBW)

head(fluBYBW@observed)

fluBYBW@neighbourhood

fluBYBW@epoch



?sts

plot(fluBYBW, type = observed ~ time)

plot(fluBYBW, type = observed ~ unit,
     
     ## mean yearly incidence per 100.000 inhabitants (8 years)
     
     population = fluBYBW@map$X31_12_01 / 100000 * 8)

## For the full set of models for data("fluBYBW") as analysed by

## Paul and Held (2011), including predictive model assessement

## using proper scoring rules, see the (computer-intensive)

## demo("fluBYBW") script:

demoscript <- system.file(file.path("demo", "fluBYBW.R"),
                          
                          package = "surveillance")

demoscript

#file.show(demoscript)

## Here we fit the improved power-law model of Meyer and Held (2014)

## - autoregressive component: random intercepts + S = 1 sine/cosine pair

## - neighbour-driven component: random intercepts + S = 1 sine/cosine pair

##   + population gravity with normalized power-law weights

## - endemic component: random intercepts + trend + S = 3 sine/cosine pairs

## - random intercepts are iid but correlated between components

f.S1 <- addSeason2formula(
  
  ~-1 + ri(type="iid", corr="all"), # spatially correlated random effects
  
  S = 1, period = 52) # seasonality with one pair of sine-cosine waves

f.end.S3 <- addSeason2formula(
  
  ~-1 + ri(type="iid", corr="all") + I((t-208)/100), # spatially correlated random effects and a linear time trend
  
  S = 3, period = 52) # seasonality with three sine cosine waves

## for power-law weights, we need adjaceny orders, which can be

## computed from the binary adjacency indicator matrix

nbOrder1 <- neighbourhood(fluBYBW)

neighbourhood(fluBYBW) <- nbOrder(nbOrder1, 15)

plot(fluBYBW, unit = 9)

## full model specification

fluModel <- list(
  
  ar = list(f = f.S1),
  
  ne = list(f = update.formula(f.S1, ~ . + log(pop)), # adding a population offset; this means large regions attract more infections
            
            weights = W_powerlaw(maxlag=max(neighbourhood(fluBYBW)),
                                 
                                 normalize = TRUE, log = TRUE)),
  
  end = list(f = f.end.S3, offset = population(fluBYBW)), # population offset in the intercept; number of "unexplained" cases depends on population size
  
  family = "NegBin1", data = list(pop = population(fluBYBW)),
  
  optimizer = list(variance = list(method = "Nelder-Mead")),
  
  verbose = TRUE,
  
  subset = 2:364) # important for prediction: fit only to first seven years

## CAVE: random effects considerably increase the runtime of model estimation

## (It is usually advantageous to first fit a model with simple intercepts

## to obtain reasonable start values for the other parameters.)

set.seed(1)  # because random intercepts are initialized randomly

fluFit <- hhh4(fluBYBW, fluModel)

summary(fluFit, idx2Exp = TRUE, amplitudeShift = TRUE)

plot(fluFit, type = "fitted", total = TRUE)

plot(fluFit, type = "season")

range(plot(fluFit, type = "maxEV"))

plot(fluFit, type = "maps", prop = TRUE)

gridExtra::grid.arrange(
  
  grobs = lapply(c("ar", "ne", "end"), function (comp)
    
    plot(fluFit, type = "ri", component = comp, main = comp,
         
         exp = TRUE, sub = "multiplicative effect")),
  
  nrow = 1, ncol = 3)

plot(fluFit, type = "neweights", xlab = "adjacency order")

# one step ahead prediction:

# one-step-ahead predictions for time points 365 to 416

osa <- oneStepAhead(fluFit, tp = c(364, 415), # tp is shifted by one, see documentation
                    
                    type = "final") # "final" means the model is not updated after each week.

# if "final" is used this is really quick, otherwise it will take very long.

# plot fan plot:

plot(osa, start = 2009, unit  =2)

# adding this to the original plot is really tedious...

# to get forecasts at longer horizons without the need to simulate you can use the hhh4addon package:

# library(devtools)

# install_github("jbracher/hhh4addon", build_vignettes = TRUE)

library(hhh4addon)

path_forecast <- predictive_moments(fluFit, t_condition = 364, lgt = 5)

fanplot_prediction(path_forecast)

plot(fluFit, type = "fitted", unit = 1)

fanplot_prediction(path_forecast, unit = 1, add  =TRUE)

# predictive means:

mu <- path_forecast$mu_matrix

# predictive variances:

sigma2 <- path_forecast$var_matrix

# to evaluate probabilistics forecasts we can use a negative binomial approcximation

# the size parameter can be obtaine as

size <- pmin(abs(mu / (sigma2 / mu - 1)), 10000)

# example:

plot(dnbinom(0:20, mu = mu[1, 1], size = size[1, 1]))

## End(Not run)

########################################################################

## An endemic-only "hhh4" model can also be estimated using MASS::glm.nb

########################################################################

## weekly counts of measles, Weser-Ems region of Lower Saxony, Germany

data("measlesWeserEms")

## fit an endemic-only "hhh4" model

## with time covariates and a district-specific offset

hhh4fit <- hhh4(measlesWeserEms, control = list(
  
  end = list(f = addSeason2formula(~1 + t, period = measlesWeserEms@freq),
             
             offset = population(measlesWeserEms)),
  
  ar = list(f = ~-1), ne = list(f = ~-1), family = "NegBin1",
  
  subset = 1:nrow(measlesWeserEms)
  
))

summary(hhh4fit)

## fit the same model using MASS::glm.nb

measlesWeserEmsData <- as.data.frame(measlesWeserEms, tidy = TRUE)

measlesWeserEmsData$t <- c(hhh4fit$control$data$t)

glmnbfit <- MASS::glm.nb(
  
  update(formula(hhh4fit)$end, observed ~ . + offset(log(population))),
  
  data = measlesWeserEmsData
  
)

summary(glmnbfit)

## Note that the overdispersion parameter is parametrized inversely.

## The likelihood and point estimates are all the same.

## However, the variance estimates are different: in glm.nb, the parameters

## are estimated conditional on the overdispersion theta.

