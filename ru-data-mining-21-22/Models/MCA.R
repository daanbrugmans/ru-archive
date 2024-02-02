library("FactoMineR")
library("factoextra")



create_MCA_plot <- function(dataset, name){
  categories = apply(dataset, 2, function(x) nlevels(as.factor(x)))
  
  res.mca = MCA(dataset, graph = FALSE)
  
  
  # data frame with variable coordinates
  res.mca.vars = data.frame(res.mca$var$coord, Variable = rep(names(categories), categories))
  
  # data frame with observation coordinates
  res.mca.obs = data.frame(res.mca$ind$coord)
  
  # plot of variable categories
  ggplot(data = res.mca.obs, aes(x = Dim.1, y = Dim.2)) +
    geom_hline(yintercept = 0, colour = "gray70") +
    geom_vline(xintercept = 0, colour = "gray70") +
    geom_point(colour = "gray50", alpha = 0.7) +
    geom_density2d(colour = "gray80") +
    geom_text(data = res.mca.vars, 
              aes(x = Dim.1, y = Dim.2, 
                  label = rownames(res.mca.vars), colour = Variable)) +
    ggtitle(paste("MCA plot of", name, sep=" ")) +
    scale_colour_discrete(name = "Variable")
  
}

#Create plots of different groups of participants
create_MCA_plot(brfss.df, "participants with and without cardiovascular diseases")

create_MCA_plot(brfss.df.with.heartDiseaseorAttack, "participants with cardiovascular diseases")

create_MCA_plot(brfss.df.without.heartDiseaseorAttack.filtered, "participants without cardiovascular diseases")

