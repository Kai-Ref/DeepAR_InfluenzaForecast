library(tidyr)
library(surveillance)

setwd("C:/Users/555ka/Coding/GIT-Projects/DeepAR_InfluenzaForecast/DeepAR_InfluenzaForecast")
data <- read.csv("Notebooks/DataProcessing/influenza.csv")
adjacentMatrix <- read.csv("Notebooks/DataProcessing/AdjacentMatrix.csv", check.names=FALSE,
                           row.names='')
population_vector <- read.csv("Notebooks/DataProcessing/PopulationVector.csv",
                              check.names=FALSE, row.names = "Location")

df <- pivot_wider(data[c('value', 'date', 'location')], names_from = location, values_from = value)
print(df[1:4,"date"])
print(df[(train_length):(test_length+4),c("SK München","date")])


# Filter the DataFrame for dates before '30.09.2016'
train_length <- length(unique(df[df$date < as.Date("2016-09-30"), ]$date))
test_length <- length(unique(df[df$date < as.Date("2018-09-30"), ]$date))

df[is.na(df)] <- 0
df_sts <-sts(as.matrix(subset(df, select=-c(date))),
             start = c(2001, 1) , frequency = 52,# maybe take len(df)/years here or 52.25
             neighbourhood = as.matrix(adjacentMatrix),
             population = as.vector(t(population_vector)))
?addSeason2formula

plot(df_sts[1:train_length], type = observed ~ time)

plot(df_sts, unit = 16)

f.ne <- addSeason2formula(~-1 + ri(type="iid", corr="all") + log(pop), S = 1, period = 52) 

f.end <- addSeason2formula(~-1 + ri(type="iid", corr="all") + I((t-208)/100), S = 1, period = 52) 

nbOrder1 <- neighbourhood(df_sts)

neighbourhood(df_sts) <- nbOrder(nbOrder1, 15) + 1

## full model specification

fluModel <- list(ar = list(f = ~ -1), 
                 ne = list(f = f.ne,weights = W_powerlaw(maxlag=max(neighbourhood(df_sts)),
                                 normalize = TRUE, log = TRUE)),
                 end = list(f = f.end, offset = population(df_sts)), 
                 family = "NegBin1", 
                data = list(pop = population(df_sts)),
                optimizer = list(variance = list(method = "Nelder-Mead")),
                verbose = TRUE,
                subset = 1:train_length) 
set.seed(1)  

fluFit <- hhh4(df_sts, fluModel)



# 1. One Step Ahead Predictions 

# -> Problem keine 4-weekAhead Vorhersagen
predictions <- oneStepAhead(fluFit, tp = c(train_length+1,test_length), type = "final")
plot(predictions, unit=16, probs = c(.1,.9),start = NULL, means.args = NULL,)

# 2.Simulate

#ergebnisse machen nicht wirklich Sinn -> Kann man hier auch einen rolling approach wählen?
# simulate from model (generates an "sts" object)
y.start <- observed(df_sts)[train_length+1,]
simData <- simulate(fluFit, nsim = 100, seed = 1,subset = (train_length+1):test_length,
                    y.start = y.start,)
summary(colSums(simData,dims=2))
plot(simData,"fan", means.args=list(),key.args=list())
dim(simData)# nTime x nUnit x nsim
# plot simulated data
plot(simData, unit=411)

# 3. hhh4addon
# ability to add higher order lags
# computation of predictive and marginal first and second moments

# to get forecasts at longer horizons without the need to simulate you can use the hhh4addon package:

library(hhh4addon)

vignette("hhh4")
vignette("hhh4_spacetime")

results <- list()
for (t_condition in (train_length):(test_length)){
  path_forecast <- predictive_moments(fluFit, t_condition = t_condition, lgt = 4)
  results[[t_condition - train_length + 1]] <- path_forecast
  print(t_condition)
}

regions <- unique(data$location)
combined_df <- data.frame()
for (region in regions){
  i<-1
  print(region)
  region_df <- data.frame(matrix(nrow = 0, ncol = 9, dimnames = list(NULL, c("Time",0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, "Location"))), stringsAsFactors = FALSE)
  colnames(region_df) <- sub("^X", "", colnames(region_df))
  for (path_forecast in results){
      # determine the distribution parameters for each region and path_forecast
      mu <- path_forecast$mu_matrix[, region]
      sigma2 <- path_forecast$var_matrix[, region]
      size <- pmin(abs(mu / (sigma2 / mu - 1)), 10000)
      # determine quantiles from the distribution parameters
      for (t in 1:4){
        time_point <-as.numeric(sub("^t=", "", names(mu[t])))
        rowname <- sprintf("%s.%s",  time_point,t)
        quantiles <- qnbinom(p = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975), 
                mu = mu[t], size = size[t])
        region_df[i, ] <- c(rowname, quantiles, region)
        i<-i+1
      }
  }
  # Concatenate the regional data frame with the combined data frame
  combined_df <- rbind(combined_df, region_df)
}
print(combined_df)
write.csv(combined_df, file = "C:/Users/555ka/Coding/GIT-Projects/DeepAR_InfluenzaForecast/DeepAR_InfluenzaForecast/R/results.csv", row.names = FALSE)





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
for (t in (train_length + 2):(test_length)){
  updated_string <- gsub("t=t", paste("t=", t, sep=""), "t=t")
  qnbinom(p = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975), 
          mu = mu[updated_string, "LK Bad Dürkheim"], 
          size = size[updated_string, "LK Bad Dürkheim"])
}







