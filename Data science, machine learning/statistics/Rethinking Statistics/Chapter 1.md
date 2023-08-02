
# Summary

- The statistical models that scientists frequently used are like Golems - they do EXACTLY what they are told to do and any mistake in its instructions can lead to terrible results (i.e. wrong conclusions)
- Introductory stats introduces students to a "zoo" of golems, and they need to pick the correct golem for a specific dataset / research question by going through a massive decision tree.
- The current paradigm of picking a hypothesis, then picking a model that can falsify the hypothesis is deeply flawed because:
	- hypotheses are not models. Multiple models and fit the same hypothesis, and multiple hypotheses may fit a single model, so strict falsification is impossible
		- What if one model rejects the null hypothesis and another model does not?
	- You can't always trust the data, especially at the cutting edge, so even IF there was a 1:1 correspondence between a hypothesis and a model, there validity of the data would be questioned.
- Falsification is *Consensual*, not *Logical*. Since data can be messy, a consensus can only be reached after much experiment and debate, something that introductory science textbooks often sweep under the rug.
- The goal of this book is to go from picking out a pre-built golem that best fits the test to *Golem engineering* (aka *Modeling*). We're going to break things eventually, but since we know how it works we'll be able to notice it and figure out how to fix it.
- This book focuses on 4 tools:
	- Bayesian data analysis
	- Model comparison
	- Multilevel models
	- Graphical causal models


## Bayesian Data Analysis

- Counting the number of ways that data could happen, according to our assumptions
- "However, it is important to realize that even when a Bayesian procedure and frequentist procedure give exactly the same answer, our Bayesian golems aren't justifying their inferences with imagined repeat sampling."
- Many people find the Bayesian approach intuitive, because people frequently misinterpret p-values in Bayesian terms.

## Model comparison and prediction

- 2 tools for choosing which model might make the best prediction: cross-validation, and information criteria

## Multilevel models

- Models are composed of parameters, and each parameter in a model may be described by its own model -> in a way its "models all the way down"
- **partial pooling** - a trick that multilevel models exploit to prevent overfitting by pooling information across units in the data to produce better estimates for all units. Examples:
	- adjust estimates for repeat sampling: when more than one observation arises from the same individual, location, or time, then traditional, single-level models may mislead us
	- to adjust estimates for imbalance in sampling: When some individuals, locations, or times are samples more than others, we may also be mislead by single-level models
	- To study variation: If our research question include variation among individuals or other groups within the data, then multilevel models are a big help, because they model variation explicitly. 
	- To avoid averaging: pre-averaging data to construct variables can be dangerous. Averaging removes variation, manufacturing false confidence. Multilevel models preserve the uncertainty the in the original pre-averaged values, while still using the average to make predictions.

## Graphical Causal Models

- second paradox in prediction: *models that are causally incorrect can make better predictions than those that are causally correct*. Focusing on prediction can be misleading. This is called the **Identification Problem**. 
- Most simple heuristic model: **Directed Acyclic Graph (DAG)**
	- Not detailed models, but they allow us to deduce what statistical models can provide valid inferences, assuming the DAG is true. 
	- The DAG comes from domain knowledge / information from outside the dataset.