Stroke Prediction using Deep Learning
===

Predicting incidents of stroke can be very valuable for patients across the world. In the United States alone, someone has a stroke every 40 seconds and someone dies of a stroke every 4 minutes. Strokes can happen at any time and medical professionals already know the characteristics of people who can be most prone to strokes. Doctors cannot monitor all patients at the same time either. This is where machine learning can help save lives and aid doctors in giving people the care they deserve. A machine learning model can keep track of thousands of patients in real-time and give doctors a chance at keeping up with demand. With the dawn of big data and increasing advances in biometric sensors, this is only looking like a better idea by the year. That being said, a machine learning model that can predict strokes should be trained extremely well as to not give false negatives and performance should be optimal for all types of patients. This task is not easy but very useful if done correctly.  

This project is an attempt at recreating the neural network descrribed in the paper by [Cheon et al.](https://pubmed.ncbi.nlm.nih.gov/31141892/) The initial attempt recreates the exact neural network architecture stated in the paper. The second attempt is a custom neural network that I came up with to accommodate my data source since I did not have access to the study's dataset.


Data Preparation
---
The dataset used in this project can be found via the following link: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/download  

I first preprocessed the dataset to make sure it was optimal for feeding into a neural network. To do this this, I first removed the `id` column since it only provides a unique id to each entry and does not offer any actual health data. After that, I filled the missing values in the `bmi` column with the mean value of all other entries. Next, I turned all string data and turned it to numeric form. This was done to the `gender`, `ever_married`, `Residence_type`, and `smoking_status` columns.  

After that, the `age` and `avg_glucose_level` values were placed into bins of value ranges. For the `age` column, values > 18 were placed in the children category, ages greater than 18 and less than 35 were placed in the adults category, ages greater than 35 and less than 65 were placed in the older adults category, and any age values above 65 were placed in the elderly category.  

The `avg_glucose_level` column was split into bins based off of values given by the American Diabetes Association. Values less than or equal to 100 were considered normal, values greater than 100 and below 125 were considered in the pre-diabetes category, and values greater than 125 were considered in the diabetes category.  

The same idea was applied to the `bmi` column. For this column, data was split according to classifications given by the Centers for Disease Control where are underweight, levels between 18.6 and 24.9 are considered normal, levels between 25 and 29.9 are overweight, and levels above 30 are considered obese.  

Finally, to deal with the imbalanced classes, I performed adaptive synthetic sampling. This method generates synthetic data by creating more samples of the imbalanced class. This helps the model learn more efficiently because it reduces bias and helps it learn the more difficult examples in the dataset.  

First Attempt
---
As stated earlier, this attempt recreates the exact architecture stated in Cheon et al. Specifically, they used a simple feed-forward neural network trained using a standard back-propagation algorithm. The architecture was comprised of an input layer with 11 neurons, four hidden layers with 22, 10, 10, and 10 neurons, respectively. The output layer had a single neuron with regression output. Each layer had ReLU activation and dropout probability of 0.2. There was also batch normalization after the first two layers. They also used Adam optimization with a learning rate of 0.001 and L2 regularization. For the loss function, they used binary cross entropy. Training was done over 50 epochs with a batch size of 5. This model resulted in 56.8% loss.  

The results of this neural network did not look like the results stated in the research paper. The accuracy is very sporadic in the first 20 epochs and then skyrockets toward the end of the training session. The validation loss is also very unstable and both the training and validation losses decrease way too early in the training session. Cheon et al. mention that they obtained an accuracy score of 84.03% [2]. The results above are not even close to that.  

I believe these results were obtained because I did not use the dataset used by the authors. They used data collected by the Korea Centers for Disease Control and Prevention (KCDC) via the Korean National Hospital Discharge In-depth Injury Survey (KNDHDS) between the years of 2013 to 2016.  

This dataset included a plethora of data points for stroke patients. To train their model, the study specifically ended up using 11 variables including gender, age, type of insurance, mode of admission, length of hospital stay, hospital region, total number of hospital beds, stroke type, brain surgery status, and Charlson Comorbidity Index (CCI) score. This dataset also seems to be specific to the population of South Korea.  

I did not have access to this dataset and could not find any sample of it online. Because of this, I ended up using the Kaggle dataset I mentioned previously. I suspect that the mismatch in data is what is causing poor performance. The KNDHDS dataset that the authors used might have been more complex than the dataset from Kaggle and the study’s neural network architecture might be overkill for it. For example, the KNDHDS dataset has 15,099 total stroke patients, specific regional data, and even has sub classifications for which type of stroke the patient had. For the Kaggle dataset, there are 5,111 total patient entries and there are no sub categories on most features and those that do have them are very vague.  

Second Attempt
---

Given the results stated above, I decided to create my own neural network architecture that is simpler than the network stated in Cheon et al. Perhaps with a simpler model that’s more fit to the Kaggle dataset, I can achieve better results. After trial and error and experimentation, I ended up going with a neural network composed of an input layer with 64 neurons, relu activation, and uniform kernel initialization, a hidden layer with 32 neurons and relu activation, a second hidden layer with 16 neurons and relu activation, a third hidden layer with 10 neurons and relu activation, and an output layer with one neuron and sigmoid activation. This model was trained for 30 epochs at a batch size of 5 using the Adam optimizer at a learning rate of 0.001. After successful training, this model reported a loss value of 37.3% and accuracy of 83.7%.

This model clearly showed better results. The loss was reduced by 19.5% and the accuracy increased by 9.2%. This is a bigger improvement than I expected. The graphs also show more stable patterns as training went on. The accuracy no longer jumps sporadically and the loss does not plateau so early in the training session. The results from this model are also significantly closer to the results stated in the research paper.  

Because of these results, I think my hypothesis was correct where I assumed that the first model was not appropriate for the Kaggle dataset. This second model is slightly simpler since it doesn’t have dropouts, batch normalization, or as much epoch training time. The second model is also significantly wider at the beginning with a 64 neuron input layer. It is known that having a complex neural network gives the risk of training issues and that not all neural network architectures fit well with all datasets. This exact statement was proven here where better results were obtained by making slight changes to simplify the architecture.


Resources
---
“Stroke Facts,” Centers for Disease Control and Prevention, 17-Mar-2021. [Online]. Available: https://www.cdc.gov/stroke/facts.htm. [Accessed: 03-May-2021].  

 Cheon S, Kim J, Lim J. The Use of Deep Learning to Predict Stroke Patient Mortality. Int J
 Environ Res Public Health. 2019;16(11):1876. Published 2019 May 28.
 doi:10.3390/ijerph16111876  

Aditya Khosla, Yu Cao, Cliff Chiung-Yu Lin, Hsu-Kuang Chiu, Junling Hu, and Honglak Lee. 2010. An integrated machine learning approach to stroke prediction. In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '10). Association for Computing Machinery, New York, NY, USA, 183–192. DOI:https://doi.org/10.1145/1835804.1835830  

"Diagnosis,” Diagnosis | ADA. [Online]. Available: https://www.diabetes.org/a1c/diagnosis. [Accessed: 03-May-2021].  

“About Adult BMI,” Centers for Disease Control and Prevention, 17-Sep-2020. [Online]. Available: https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html. [Accessed: 03-May-2021].
