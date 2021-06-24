#Machine Learning Algorithms in Prostate Cancer data
#EDA
#Bar Plot for 
ggplot(as.data.frame(df), aes(factor(diagnosis_result), fill = factor(diagnosis_result))) + geom_bar()
#Histogram
library(magrittr)
mean_area = round(mean(area), 2)
mean_area
ggplot(df, aes(x = diagnosis_result, y = mean_area)) +
  geom_bar(stat = "identity")
#Supervised Learning

#Support Vector Machine

#Import data
df <- read.table(file.choose(), header = T, sep = )
df <- subset(df, select = -c(id))
attach(df)
names(df)
head(df)

#Normalize data
#define Min-Max normalization function
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#apply Min-Max normalization 
df_norm <- as.data.frame(lapply(df[1:9], min_max_norm))
head(df_norm)

#First we need to divide test and train data
sample.size <- floor(0.70 * nrow(df_norm))
indexes <- sample(seq_len(nrow(df_norm)), size = sample.size)

train <- df_norm[indexes,]
test <- df_norm[-indexes,]

train
test

write.table(train, file="train.csv")
read.table("train.csv", header = TRUE)
#Run the SVM
library(e1071)
svmdata <- svm(formula= diagnosis_result ~ radius + texture+perimeter+area+smoothness+compactness+symmetry+fractal_dimension
               , data = train, type = "C-classification", kernel="linear")
pred <- predict(svmdata,test)
pred
table(pred, test$diagnosis_result)

#ANN
library(nnet)
nnet1 = nnet(factor(diagnosis_result) ~ radius + texture+perimeter+area+smoothness+compactness+symmetry+fractal_dimension, 
             data=train, size=4, decay = 0.05, MaxNWTS = 20)
predict1 <- predict(nnet1, test, type = "class")
table(predict1, test$diagnosis_result)

#Unsupervised Learning
#Hierarhical Clustering
df.clustering <- subset(train, select = -c(diagnosis_result))
sl.out <- hclust(dist(df.clustering, method="euclidian"), method="single")
plot(sl.out)

#Non-Hierarchical Clustering
cl <- kmeans(df.clustering,4)
cl

#To get center and membership of countries
plot(df.clustering, col = cl$cluster)
points(cl$centers, col = 1:2, pch = 8, cex=2)

#PCA
df.pca.train <- subset(train, select = -c(diagnosis_result))
df.pca.train
prin_comp <- prcomp(t(df.pca.train), scale. = TRUE)

plot(prin_comp$x[,1], prin_comp$x[,2])
prin_comp.var <- prin_comp$sdev^2
prin_comp.var.per <- round(prin_comp.var/sum(prin_comp.var)*100, 1)
barplot(prin_comp.var.per, main="Scree Plot", xlab="Principal Component", ylab="Percent Variation")
library(ggplot2)
pca.data <- data.frame(Sample=rownames(prin_comp$x), X=prin_comp$x[,1], Y=prin_comp$x[,2])
pca.data

ggplot(data=pca.data, aes(x=X, y=Y, label=Sample))+
  geom_text()+
  xlab(paste("PC1 - ", prin_comp.var.per[1], "%", sep=""))+
  ylab(paste("PC2 - ", prin_comp.var.per[2], "%", sep=""))+
  theme_bw()+
  ggtitle("PCA Graph")
prin_comp.var.per[3]
prin_comp.var.per[4]
################################
names(prin_comp)
prin_comp$center
prin_comp$scale

prin_comp$rotation
dim(prin_comp$x)

biplot(prin_comp, scale =0)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex

#scree plot
plot(prop_varex, xlab ="Principal Component", ylab="Proportion Variance Explained", type="b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab="Principal Component", ylab = "Cumulative Proportion of Variance Explained", type="b")

train.data <- data.frame()
#Pca 183
K <- eigen(cor(train))
K
print(train%*%K$vec, digits=5)

pca <- princomp(train, center = TRUE, cor=TRUE, scores=TRUE)
pca$scores
train
library(jmv)
data3 <- read.table(file.choose(), header=T, sep= )
pca(train, vars = c('diagnosis_result', 'radius', 'texture','perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension'),
    nFactorMethod = "parallel", nFactors = 1, minEigen = 1,
    rotation = "varimax", hideLoadings = 0.3, screePlot =FALSE, 
    eigen = FALSE, factorCor =FALSE, 
    factorSummary = FALSE, 
    kmo =FALSE, bartlett = FALSE)
#######################################

