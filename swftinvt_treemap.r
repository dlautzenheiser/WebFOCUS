install.packages("portfolio")
library(portfolio)

data <- read.csv("c:\\ibi\\apps\\r_data\\rprogram_treemap_counts.csv")
png("C:/ibi/apps/r_data/rprogram_treemap.png", width=520,height=520)
map.market(id=data$id, area=data$count, group=data$keyword, color=data$count, main="Purpose")
dev.off()
