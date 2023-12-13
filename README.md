# Application of COR_loss (Classification optimizing loss)

This application refers to the research paper: 
Metric Multi-Task Learning for Regression and Ordinal
Classification: A novel loss function avoiding the problem
of imbalanced classes

## Information
Achieving perfect models with flawless predictions in real-world applications is impossible. A statistical model with a single target variable can only generalize a complex problem to a limited extent. To enhance generalization, a model must be trained on multiple, often contradictory objectives. Multi-task learning (MTL) enables the joint training of various supervised learning (SL) targets, as separately trained, independent models neglect target correlations. We introduce a novel loss function that allows for the minimization of the trade-off between point estimation and ordinal classification within a shared value range. Combined through an additional hyperparameter, it considers the codependency of the targets and enables a joint training of regression and multiclass classification on ordinal classes. Due to convexity and fuzzy logic, the function is also applicable with the gradient descent (GD) method and avoids the necessity to use methods for imbalanced classes. In contrast to comparable approaches, our methodology not only classifies but also determines class-optimized metrical predictions. The fuzzy logic-based, pseudo-metric consideration of classification allows for the optimization of the metric estimator through ordinal classes without information loss due to discretization of the point estimation, which would be necessary for classification tasks. To demonstrate the usefulness, we evaluate the method using freely available datasets commonly employed for assessing regression or classification tasks. A shared value range among targets is assumed. Out-of-sample evaluation with a focus on maximizing the classification target ($\alpha=0$), compared with a baseline regression, demonstrates that the applied COR loss function can achieve significant improvements in classification results (F1-Score: $+11.1\%$ for Boston Housing, $+17.1\%$ for Ames Housing) despite the challenge of imbalanced classes. This result is possible due to an average two times higher bias of the regression estimator. Minimizing the compromise between both targets results in a much less biased mean absolute error (MAE) ($+35\%$) with a reasonable improvement in classification accuracy (F1-Score: $+9.3\%$ for Boston Housing, $+10.2\%$ for Ames Housing). The increase in computing time remains within reasonable limits ($~1.25-~2$ times).
