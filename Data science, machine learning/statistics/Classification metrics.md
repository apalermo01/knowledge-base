- Accuracy = (TP + TN) / (TP + FP + FN + TN)
	- use this when the dataset classes are well balanced
- Precision = (TP) / (TP + FP)
	- use this when false positives are unacceptable
	- examples: credit default, crime prediction, etc.
	- If a model is optimized for precision, lots of true positives will fall through the cracks, but those that we do catch we can be more certain that it's not a false positive
- Recall / recall / hit rate / true positive rate / sensitivity
	- Use this when false negatives are unacceptable
	- probability of a positive test, conditioned on truly being positive
	- examples: medical screening / diagnosis
	- If a model is optimized for recall, you may get lots of false alarms, but you'll be very likely to capture all the real emergencies
$$
\text{Recall} = \frac{\text{true positives}}{\text{total \# of positives}} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$
- Specificity
	- Probability of a negative test, conditioned on being truly negative
$$
\text{specificity} = \frac{\text{true negatives}}{\text{total \# of negatives}} = \frac{\text{TN}}{\text{TN} + \text{FP}}
$$
- F1 score
	- balance between precision and recall
$$
\text{F1} = 2 * \frac{\text{precision} * \text{recall}}{\text{precision} + \text{recall}}
			$$  
-  weighted F1 score
	- use $\beta$ paramerter to describe how much more importance to give to recall vs. precision
$$
\text{F}_\beta = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}{(\beta^2 * \text{precision}) + \text{recall}}
$$

## References
- https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
- https://en.wikipedia.org/wiki/Sensitivity_and_specificity