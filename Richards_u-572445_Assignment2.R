## Byronie Richards
##u-572445

## Load packages ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)

## Load data -------------------------------------------------------------------
#dataDir <- "./input/"
mushrooms <- read.delim ("./input/mushrooms.txt", stringsAsFactors = FALSE)
edibility <- read.delim("./input/edibility.txt", stringsAsFactors = FALSE)
survey <- read.delim("./input/clean_survey.txt", stringsAsFactors = FALSE)

## Exercise 1 ------------------------------------------------------------------
#habitat_number <-function(input_df, habitat) {
  #filter(mushrooms, wood == habitat |
           #urban == habitat   |
           #grasses == habitat |
           #meadows == habitat | 
           #path == habitat    |
           #leaves == habitat)
#}

habitat_number <- function(input_df, habitat) {
  if (habitat == "leaves") {
    return(ncol(input_df))
  } else {
    return(nrow(input_df))
  }
}

answer1 <-habitat_number(mushrooms, "leaves")

## Exercise 2 ------------------------------------------------------------------
white_mushrooms <- function(mushrooms) {}



count_contributing_factor <- function(colum_of_df, cont_factor) {
  sum(row_of_df == cont_factor)
}

## create the new variable
NYPD_2015$ROAD.RAGERS <- apply(NYPD_2015[, grepl("CONTRIBUTING.FACTOR",
                                                 names(NYPD_2015))], 1, count_contributing_factor,
                               "Aggressive Driving/Road Rage")
NYPD_vehicles <- mutate_at(NYPD_2015, vars(contains("VEHICLE.TYPE.CODE")),
                           funs(discretize_vehicles)) %>%
  mutate(NUM.VEHICLES = VEHICLE.TYPE.CODE.1 + VEHICLE.TYPE.CODE.2 +
           VEHICLE.TYPE.CODE.3 + VEHICLE.TYPE.CODE.4 + VEHICLE.TYPE.CODE.5)
```
## add a `CAUSE_DELAY` column
determine_delay <- function(delays) {
  cause_of_delay <- "UNKNOWN"
  delay <- delays[2]
  explained_delay <- sum(delays[3:6], na.rm = TRUE)
  if (delay <= 0) {
    cause_of_delay <- NA
  } else {
    if (explained_delay >= delay) {
      cause_of_delay <- "KNOWN"
    }
  }
  cause_of_delay
}

delays <- flights
delays$DELAY_CAUSE <- apply(flights[, grepl("DELAY", names(flights))], 1,
                            determine_delay)

## Exercise 3 ------------------------------------------------------------------
## Exercise 4 ------------------------------------------------------------------
ggplot(answer4, aes(x = mushroom_part, fill = color_of_part)) + geom_bar() +
  scale_fill_manual(values = c("white" = "white", "yellow" = "goldenrod",
                               "pink" = "lightsalmon2", "buff" = "khaki", "brown" = "brown",
                               "gray" = "gray", "black" = "black", "green" = "darkolivegreen4",
                               "purple" = "mediumorchid4", "red" = "firebrick3", "cinnamon" = "tan",
                               "orange" = "darkorange", "chocolate" = "chocolate"), guide = FALSE)
## Exercise 5 ------------------------------------------------------------------
## Exercise 6 ------------------------------------------------------------------
## Exercise 7 ------------------------------------------------------------------
## Exercise 8 ------------------------------------------------------------------
## Exercise 9 ------------------------------------------------------------------
## Exercise 10 ------------------------------------------------------------------
edible_shrooms <- sample(edibility$species, 4)
backup_shrooms <- mushrooms
backup_shrooms$edibility = "poisonous" backup_shrooms$edibility[backup_shrooms$species %in% edible_shrooms] <- "edible"
