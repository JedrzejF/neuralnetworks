library(knitr)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)

train <- read.csv("~/titanic/train/train.tsv", sep = '\t', stringsAsFactors = F, na.strings = c("NA", ""))
test <- read.csv("~/titanic/test-A/in.tsv", sep = '\t', header = FALSE, stringsAsFactors = F, na.strings = c("NA", ""))
dev <- read.csv("~/titanic/dev-0/in.tsv", sep = '\t', header = FALSE, stringsAsFactors = F, na.strings = c("NA", ""))

colnames(test) <- colnames(train)[-1]
colnames(dev) <- colnames(train)[-1]
train$Type <- 'train'
test$Type <- 'test'
dev$Type <- 'dev'
test$Survived <- NA
dev$Survived <- NA
all <- rbind(train, test, dev)

all$Survived <- as.factor(all$Survived)

all$PclassSex[all$Pclass=='1' & all$Sex=='male'] <- 'P1Male'
all$PclassSex[all$Pclass=='2' & all$Sex=='male'] <- 'P2Male'
all$PclassSex[all$Pclass=='3' & all$Sex=='male'] <- 'P3Male'
all$PclassSex[all$Pclass=='1' & all$Sex=='female'] <- 'P1Female'
all$PclassSex[all$Pclass=='2' & all$Sex=='female'] <- 'P2Female'
all$PclassSex[all$Pclass=='3' & all$Sex=='female'] <- 'P3Female'
all$PclassSex <- as.factor(all$PclassSex)

all$Title <- sapply(all$Name, function(x) {strsplit(x, split='[\t.]')[[1]][2]})
all$Title <- sub(' ', '', all$Title)

all$Title[all$Title %in% c("Mlle", "Ms")] <- "Miss"
all$Title[all$Title== "Mme"] <- "Mrs"
all$Title[!(all$Title %in% c('Master', 'Miss', 'Mr', 'Mrs'))] <- "Rare Title"
all$Title <- as.factor(all$Title)

all$Fsize <- all$SibSp+all$Parch +1

TicketGroup <- all %>%
  select(Ticket) %>%
  group_by(Ticket) %>%
  summarise(TicketSize=n())
all <- left_join(all, TicketGroup, by = "Ticket")

all$Group <- all$Fsize
for (i in 1:nrow(all)){
  all$Group[i] <- max(all$Group[i], all$TicketSize[i])
}

all$GroupSize[all$Group==1] <- 'solo'
all$GroupSize[all$Group==2] <- 'duo'
all$GroupSize[all$Group>=3 & all$Group<=4] <- 'group'
all$GroupSize[all$Group>=5] <- 'large group'
all$GroupSize <- as.factor(all$GroupSize)

kable(all[which(is.na(all$Embarked)),c('Title', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Group') ])
all$Embarked[all$Ticket=='113572'] <- 'C'
all$Embarked <- as.factor(all$Embarked)

all$FarePP <- all$Fare/all$TicketSize
tab3 <- all[(!is.na(all$FarePP)),] %>%
  group_by(Pclass) %>%
  summarise(MedianFarePP=median(FarePP))
all <- left_join(all, tab3, by = "Pclass")
all$FarePP[which(all$FarePP==0)] <- all$MedianFarePP[which(all$FarePP==0)]

set.seed(5723)
AgeLM <- lm(Age ~ Pclass + Sex + SibSp + Parch + Embarked + Title + GroupSize, data=all[!is.na(all$Age),])
all$AgeLM <- predict(AgeLM, all)

indexMissingAge <- which(is.na(all$Age))
indexAgeSurvivedNotNA<- which(!is.na(all$Age) & (!is.na(all$Survived)))
all$Age[indexMissingAge] <- all$AgeLM[indexMissingAge]

all$Cabin[is.na(all$Cabin)] <- "unknown"
all$Cabin <- substring(all$Cabin, 1, 1)
all$Cabin <- as.factor(all$Cabin)

TicketSurvivors <- all %>%
  group_by(Ticket) %>%
  summarize(SumSurvived = sum(as.numeric(Survived)-1, na.rm=T))
all <- left_join(all, TicketSurvivors)

all$AnySurvivors[all$TicketSize==1] <- 'other'
all$AnySurvivors[all$TicketSize>=2] <- ifelse(all$SumSurvived[all$TicketSize>=2]>=1, 'survivors in group', 'other')
all$AnySurvivors <- as.factor(all$AnySurvivors)

trainClean <- all %>%
  filter(Type == 'train')
testClean <- all %>%
  filter(Type == 'test')
devClean <- all %>%
  filter(Type == 'dev')

set.seed(7445)
caret_matrix <- train(x=trainClean[,c('PclassSex', 'GroupSize', 'FarePP', 'AnySurvivors', 'Age', 'Cabin', 'Title')], y=trainClean$Survived, data=trainClean, method='rf', trControl=trainControl(method="cv", number=7))
caret_matrix

rf_imp <- varImp(caret_matrix, scale = FALSE)
rf_imp <- rf_imp$importance
rf_gini <- data.frame(Variables = row.names(rf_imp), MeanDecreaseGini = rf_imp$Overall)

ggplot(rf_gini, aes(x=reorder(Variables, MeanDecreaseGini), y=MeanDecreaseGini, fill=MeanDecreaseGini)) +
  geom_bar(stat='identity') + coord_flip() + theme(legend.position="none") + labs(x="") +
  ggtitle('Variable Importance Random Forest') + theme(plot.title = element_text(hjust = 0.5))

test_predicts <- predict(caret_matrix, testClean)
dev_predicts <- predict(caret_matrix, devClean)

write.table(test_predicts, file = "~/titanic/test-A/out.tsv", row.names = FALSE, col.names = FALSE)
write.table(dev_predicts, file = "~/titanic/dev-0/out.tsv", row.names = FALSE, col.names = FALSE)
