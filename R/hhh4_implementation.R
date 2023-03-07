library(tidyr)
library(surveillance)

# Data for all states
setwd("C:/Users/555ka/Coding/GIT-Projects/DeepAR_InfluenzaForecast/DeepAR_InfluenzaForecast")
data <- read.csv("Notebooks/DataProcessing/influenza.csv")
adjacentMatrix <- read.csv("Notebooks/DataProcessing/AdjacentMatrix.csv", check.names=FALSE,
                         row.names='')
population_vector <- read.csv("Notebooks/DataProcessing/PopulationVector.csv",
                          check.names=FALSE, row.names = "Location")

# For Developing purpose select only the BW data (faster computations)

# data <- read.csv("Notebooks/DataProcessing/BWDataset/influenzaBW.csv")
# adjacentMatrix <- read.csv("Notebooks/DataProcessing/BWDataset/AdjacentMatrixBW.csv",
#                           check.names=FALSE,row.names='Index')
# population_vector <- read.csv("Notebooks/DataProcessing/BWDataset/PopulationVectorBW.csv",
#                              check.names=FALSE, row.names = "Location")


df <- pivot_wider(data[c('value', 'date', 'location')], names_from = location, values_from = value)
df[is.na(df)] <- 0
df_sts <-sts(as.matrix(subset(df, select=-c(date))),
             start = c(2002, 1) , frequency = 52.25,# maybe take len(df)/years here
             neighbourhood = as.matrix(adjacentMatrix),
             population = as.vector(t(population_vector)))
plot(df_sts, type = observed ~ time)


?hhh4

?sts

## Here we fit the improved power-law model of Meyer and Held (2014)

## - autoregressive component: random intercepts + S = 1 sine/cosine pair

## - neighbour-driven component: random intercepts + S = 1 sine/cosine pair

##   + population gravity with normalized power-law weights

## - endemic component: random intercepts + trend + S = 3 sine/cosine pairs

## - random intercepts are iid but correlated between components

f.ne <- addSeason2formula(
  
  ~-1 + ri(type="iid", corr="all") + log(pop), # spatially correlated random effects 
  ## JB: added population offset here directly for better readability
  
  S = 1, period = 52) # seasonality with one pair of sine-cosine waves

f.end <- addSeason2formula(
  
  ~-1 + ri(type="iid", corr="all") + I((t-208)/100), # spatially correlated random effects and a linear time trend
  
  S = 1, period = 52) # seasonality with three sine cosine waves
## JB: I set S = 1 here as I don't think there is much of a point in adding 3 waves here
## (the EN component plays hardly any role anyway)

## for power-law weights, we need adjaceny orders, which can be

## computed from the binary adjacency indicator matrix

nbOrder1 <- neighbourhood(df_sts)

neighbourhood(df_sts) <- nbOrder(nbOrder1, 15) + 1
## JB: I added plus one here so we can subsume the AR in the NE component
## i.e., we treat the same region now as if it was a direct neighbour, the direct neighbours
## as if they were second-order neighbours etc. This removes quite a few parameters and in my
## opinion is a more reasonable model anyway as otherwise we have separate terms for seasonality
## in the two components, which does not make sense.

plot(df_sts, unit = 9)

## full model specification

fluModel <- list(
  
  ar = list(f = ~ -1), ## JB: I removed the AR component here as it is now subsumed in the NE component, see above
  
  ne = list(f = f.ne, # adding a population offset; this means large regions attract more infections
            
            weights = W_powerlaw(maxlag=max(neighbourhood(df_sts)),
                                 
                                 normalize = TRUE, log = TRUE)),
  
  end = list(f = f.end, offset = population(df_sts)), # population offset in the intercept; number of "unexplained" cases depends on population size
  
  family = "NegBin1", data = list(pop = population(df_sts)),
  
  optimizer = list(variance = list(method = "Nelder-Mead")),
  
  verbose = TRUE,
  
  subset = 2:364) # important for prediction: fit only to first seven years

#print(population(fluBYBW))
print(population(df_sts))

## CAVE: random effects considerably increase the runtime of model estimation

## (It is usually advantageous to first fit a model with simple intercepts

## to obtain reasonable start values for the other parameters.)

set.seed(1)  # because random intercepts are initialized randomly


fluFit <- hhh4(df_sts, fluModel)

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
confint(osa, level = c(0.1,0.5,0.9))
?confint

# if "final" is used this is really quick, otherwise it will take very long.

# plot fan plot:

plot(osa, start = 2000, unit  =2)

# adding this to the original plot is really tedious...

# to get forecasts at longer horizons without the need to simulate you can use the hhh4addon package:

# library(devtools)

# install_github("jbracher/hhh4addon", build_vignettes = TRUE)

library(hhh4addon)

y.start <- observed(df_sts)[365,]
Simulation <- simulate(fluFit, nsim=100, seed=1, subset = 366:380, y.start = y.start)
summary(colSums(Simulation,dims=2))
plot(Simulation,"fan", means.args=list(),key.args=list())

vignette("hhh4addon")
vignette("hhh4_spacetime")

path_forecast <- predictive_moments(fluFit, t_condition = 364, lgt = 5)

fanplot_prediction(path_forecast)

plot(fluFit, type = "fitted", unit = 16)# unit 16 equals the Düw LKs

fanplot_prediction(path_forecast, unit = 16, add  =TRUE)

# predictive means:

mu <- path_forecast$mu_matrix

# predictive variances:

sigma2 <- path_forecast$var_matrix

# to evaluate probabilistics forecasts we can use a negative binomial approximation

# the size parameter can be obtaine as

size <- pmin(abs(mu / (sigma2 / mu - 1)), 10000)

# example:

plot(dnbinom(0:20, mu = mu[1, 1], size = size[1, 1]))

# JB: you can also extract quantiles, e.g. for unit Bad Dürkheim and time 369 (based on data up to 366):

qnbinom(p = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975), 
        
        mu = mu["t=369", "LK Bad Dürkheim"], size = size["t=369", "LK Bad Dürkheim"])

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

