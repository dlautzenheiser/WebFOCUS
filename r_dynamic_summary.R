keywords<-read.table("c:/ibi/apps/R_DATA/RPROGRAM_KEYWORD_COUNTS.csv", header=TRUE, sep=",")
png("C:/ibi/apps/r_data/rprogram_summary_counts.png", width=520,height=200)
summary(keywords$Count)
dev.off()
