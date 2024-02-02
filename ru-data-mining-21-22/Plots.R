rm(list=ls())

library(ggplot2)
library(scales)
library(dplyr)
library(caret)

# Change working directory if needed, Josse.
setwd("C:/Users/Daan/Documents/Projecten/DataMiningResearch")

brfss.df <- read.csv("BRFSS2015Preprocessed.csv", header = T)

summary(brfss.df)


# Bar plots
ggplot(brfss.df, aes(x=HeartDiseaseorAttack)) + 
  labs(title="Has a doctor, nurse, or other health professional\never told you that you had coronary heart disease or a myocardial infarction (heart attack)?") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=HighBP, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you ever been told by a doctor, nurse, or or other health professional\nthat you have high blood pressure?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=HighChol, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you ever been told by a doctor, nurse, or other health professional\nthat your blood cholesterol is high?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=CholCheck, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you had a cholesterol check within the past five years?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

BMI.level.order <- c("Underweight", "Healthy weight", "Overweight", "Class 1 Obese", "Class 2 Obese", "Class 3 Obese")
ggplot(brfss.df, aes(x=factor(BMI, levels=BMI.level.order), fill=HeartDiseaseorAttack)) + 
  labs(title="What is your body mass index (BMI)? (BMI values categorized under CDC standards)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=Smoker, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you smoked at least 100 cigarettes (5 packs) in your entire life?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=Stroke, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you ever had a stroke?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

Diabetes.level.order <- c("No diabetes", "Pre-diabetes", "Diabetes")
ggplot(brfss.df, aes(x=factor(Diabetes, levels=Diabetes.level.order), fill=HeartDiseaseorAttack)) + 
  labs(title="Were you ever told by a doctor, nurse, or other health professional that you had diabetes?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=PhysActivity, fill=HeartDiseaseorAttack)) + 
  labs(title="During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=Fruits, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you consume at least 1 portion of fruit daily?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=Veggies, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you consume at least 1 portion of vegetables daily?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=HvyAlcoholConsump, fill=HeartDiseaseorAttack)) + 
  labs(title="Is your weekly alcohol consumption considered heavy\n(over 14 drinks per week for men, over 7 drinks per week for women)?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=AnyHealthcare, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you have any kind of health care coverage, including health insurance,\nprepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=NoDocbcCost, fill=HeartDiseaseorAttack)) + 
  labs(title="Was there a time in the past 12 months when you needed to see a doctor\nbut could not because of cost?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

GenHlth.level.order <- c("Excellent", "Very good", "Good", "Fair", "Poor")
ggplot(brfss.df, aes(x=factor(GenHlth, level=GenHlth.level.order), fill=HeartDiseaseorAttack)) + 
  labs(title="Would you say that in general your health is:", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=MentHlth, fill=HeartDiseaseorAttack)) + 
  labs(title="Now thinking about your mental health, which includes stress, depression, and problems with emotions,\nfor how many days during the past 30 days was your mental health not good?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=PhysHlth, fill=HeartDiseaseorAttack)) + 
  labs(title="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=DiffWalk, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you have serious difficulty walking or climbing stairs?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=Sex, fill=HeartDiseaseorAttack)) + 
  labs(title="What is your biological gender (sex)?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

ggplot(brfss.df, aes(x=Age, fill=HeartDiseaseorAttack)) + 
  labs(title="What is your age? (Ages categorized into 13 different age categories)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

Education.level.order <- c("College graduate", "Some college or technical school", "High school graduate", "Some high school", "Elementary school", "Never attended school or only kindergarten")
ggplot(brfss.df, aes(x=factor(Education, levels=Education.level.order), fill=HeartDiseaseorAttack)) + 
  labs(title="What is the highest grade or year of school you have completed?\n(Answers categorized into 6 different categories)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  coord_flip() + 
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")

Income.level.order <- c("Income >= $75,000", "$50,000 <= Income < $75,000", "$35,000 <= Income < $50,000", "$25,000 <= Income < $35,000", "$20,000 <= Income < $25,000", "$15,000 <= Income < $20,000", "$10,000 <= Income < $15,000", "Income < $10,000")
ggplot(brfss.df, aes(x=factor(Income, levels=Income.level.order), fill=HeartDiseaseorAttack)) + 
  labs(title="Is your annual household income from all sources:", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  coord_flip() + 
  geom_bar(position="dodge") +
  geom_text(stat='count', aes(label=..count..), color="red", size=5, position=position_dodge(width=.9), vjust="bottom")


# Relative bar plots
brfss.df.relative <- brfss.df %>%
  count(HighBP, HeartDiseaseorAttack) %>%
  group_by(HighBP) %>%
  mutate(HighBPRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=HighBP, y=HighBPRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you ever been told by a doctor, nurse, or other health professional\nthat you have high blood pressure?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=HighBPRel, label=percent(HighBPRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(HighChol, HeartDiseaseorAttack) %>%
  group_by(HighChol) %>%
  mutate(HighCholRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=HighChol, y=HighCholRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you ever been told by a doctor, nurse, or other health professional\nthat your blood cholesterol is high?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=HighCholRel, label=percent(HighCholRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(CholCheck, HeartDiseaseorAttack) %>%
  group_by(CholCheck) %>%
  mutate(CholCheckRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=CholCheck, y=CholCheckRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you had a cholesterol check within the past five years?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=CholCheckRel, label=percent(CholCheckRel)), position=position_stack(), color="red", size=5, vjust="top")

BMI.level.order <- c("Underweight", "Healthy weight", "Overweight", "Class 1 Obese", "Class 2 Obese", "Class 3 Obese")
brfss.df.relative <- brfss.df %>%
  count(BMI, HeartDiseaseorAttack) %>%
  group_by(BMI) %>%
  mutate(BMIRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=factor(BMI, levels=BMI.level.order), y=BMIRel, fill=HeartDiseaseorAttack)) + 
  labs(title="What is your body mass index (BMI)? (BMI values categorized under CDC standards)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=BMIRel, label=percent(BMIRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(Smoker, HeartDiseaseorAttack) %>%
  group_by(Smoker) %>%
  mutate(SmokerRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=Smoker, y=SmokerRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you smoked at least 100 cigarettes (5 packs) in your entire life?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=SmokerRel, label=percent(SmokerRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(Stroke, HeartDiseaseorAttack) %>%
  group_by(Stroke) %>%
  mutate(StrokeRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=Stroke, y=StrokeRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Have you ever had a stroke?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=StrokeRel, label=percent(StrokeRel)), position=position_stack(), color="red", size=5, vjust="top")

Diabetes.level.order <- c("No diabetes", "Pre-diabetes", "Diabetes")
brfss.df.relative <- brfss.df %>%
  count(Diabetes, HeartDiseaseorAttack) %>%
  group_by(Diabetes) %>%
  mutate(DiabetesRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=factor(Diabetes, levels=Diabetes.level.order), y=DiabetesRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Were you ever told by a doctor, nurse, or other health professional that you had diabetes?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=DiabetesRel, label=percent(DiabetesRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(PhysActivity, HeartDiseaseorAttack) %>%
  group_by(PhysActivity) %>%
  mutate(PhysActivityRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=PhysActivity, y=PhysActivityRel, fill=HeartDiseaseorAttack)) + 
  labs(title="During the past month, other than your regular job, did you participate in any physical activities\nor exercises such as running, calisthenics, golf, gardening, or walking for exercise?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=PhysActivityRel, label=percent(PhysActivityRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(Fruits, HeartDiseaseorAttack) %>%
  group_by(Fruits) %>%
  mutate(FruitsRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=Fruits, y=FruitsRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you consume at least 1 portion of fruit daily?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=FruitsRel, label=percent(FruitsRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(Veggies, HeartDiseaseorAttack) %>%
  group_by(Veggies) %>%
  mutate(VeggiesRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=Veggies, y=VeggiesRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you consume at least 1 portion of vegetables daily?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=VeggiesRel, label=percent(VeggiesRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(HvyAlcoholConsump, HeartDiseaseorAttack) %>%
  group_by(HvyAlcoholConsump) %>%
  mutate(HvyAlcoholConsumpRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=HvyAlcoholConsump, y=HvyAlcoholConsumpRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Is your weekly alcohol consumption considered heavy?\n(over 14 drinks per week for men, over 7 drinks per week for women)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=HvyAlcoholConsumpRel, label=percent(HvyAlcoholConsumpRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(AnyHealthcare, HeartDiseaseorAttack) %>%
  group_by(AnyHealthcare) %>%
  mutate(AnyHealthcareRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=AnyHealthcare, y=AnyHealthcareRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs,\nor government plans such as Medicare, or Indian Health Service?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=AnyHealthcareRel, label=percent(AnyHealthcareRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(NoDocbcCost, HeartDiseaseorAttack) %>%
  group_by(NoDocbcCost) %>%
  mutate(NoDocbcCostRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=NoDocbcCost, y=NoDocbcCostRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Was there a time in the past 12 months when you needed to see a doctor,\nbut could not because of cost?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=NoDocbcCostRel, label=percent(NoDocbcCostRel)), position=position_stack(), color="red", size=5, vjust="top")

GenHlth.level.order <- c("Excellent", "Very good", "Good", "Fair", "Poor")
brfss.df.relative <- brfss.df %>%
  count(GenHlth, HeartDiseaseorAttack) %>%
  group_by(GenHlth) %>%
  mutate(GenHlthRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=factor(GenHlth, levels=GenHlth.level.order), y=GenHlthRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Would you say that in general your health is:", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=GenHlthRel, label=percent(GenHlthRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(MentHlth, HeartDiseaseorAttack) %>%
  group_by(MentHlth) %>%
  mutate(MentHlthRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=MentHlth, y=MentHlthRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Now thinking about your mental health, which includes stress, depression, and problems with emotions,\nfor how many days during the past 30 days was your mental health not good?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=MentHlthRel, label=percent(MentHlthRel)), position=position_stack(), color="red", size=2.5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(PhysHlth, HeartDiseaseorAttack) %>%
  group_by(PhysHlth) %>%
  mutate(PhysHlthRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=PhysHlth, y=PhysHlthRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=PhysHlthRel, label=percent(PhysHlthRel)), position=position_stack(), color="red", size=2.5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(DiffWalk, HeartDiseaseorAttack) %>%
  group_by(DiffWalk) %>%
  mutate(DiffWalkRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=DiffWalk, y=DiffWalkRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Do you have serious difficulty walking or climbing stairs?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=DiffWalkRel, label=percent(DiffWalkRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(Sex, HeartDiseaseorAttack) %>%
  group_by(Sex) %>%
  mutate(SexRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=Sex, y=SexRel, fill=HeartDiseaseorAttack)) + 
  labs(title="What is your biological gender (sex)?", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=SexRel, label=percent(SexRel)), position=position_stack(), color="red", size=5, vjust="top")

brfss.df.relative <- brfss.df %>%
  count(Age, HeartDiseaseorAttack) %>%
  group_by(Age) %>%
  mutate(AgeRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=Age, y=AgeRel, fill=HeartDiseaseorAttack)) + 
  labs(title="What is your age? (Ages categorized into 13 different age categories)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  geom_text(aes(y=AgeRel, label=percent(AgeRel)), position=position_stack(), color="red", size=5, vjust="top")

Education.level.order <- c("College graduate", "Some college or technical school", "High school graduate", "Some high school", "Elementary school", "Never attended school or only kindergarten")
brfss.df.relative <- brfss.df %>%
  count(Education, HeartDiseaseorAttack) %>%
  group_by(Education) %>%
  mutate(EducationRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=factor(Education, levels=Education.level.order), y=EducationRel, fill=HeartDiseaseorAttack)) + 
  labs(title="What is the highest grade or year of school you have completed?\n(Answers categorized into 6 different categories)", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  coord_flip() +
  geom_text(aes(y=EducationRel, label=percent(EducationRel)), position=position_stack(), color="red", size=5, hjust="top")

#Income.level.order <- c("Income < $10,000", "$10,000 <= Income < $15,000", "$15,000 <= Income < $20,000", "$20,000 <= Income < $25,000", "$25,000 <= Income < $35,000", "$35,000 <= Income < $50,000", "$50,000 <= Income < $75,000", "Income >= $75,000")
Income.level.order <- c("Income >= $75,000", "$50,000 <= Income < $75,000", "$35,000 <= Income < $50,000", "$25,000 <= Income < $35,000", "$20,000 <= Income < $25,000", "$15,000 <= Income < $20,000", "$10,000 <= Income < $15,000", "Income < $10,000")
brfss.df.relative <- brfss.df %>%
  count(Income, HeartDiseaseorAttack) %>%
  group_by(Income) %>%
  mutate(IncomeRel = n / sum(n))
ggplot(brfss.df.relative, aes(x=factor(Income, Income.level.order), y=IncomeRel, fill=HeartDiseaseorAttack)) + 
  labs(title="Is your annual household income from all sources:", fill="Has had Heart Disease\nor Heart Attack") +
  scale_fill_grey(start=.2, end=.7) +
  scale_y_continuous(labels=percent_format()) +
  geom_col() +
  coord_flip() +
  geom_text(aes(y=IncomeRel, label=percent(IncomeRel)), position=position_stack(), color="red", size=5, hjust="top")

