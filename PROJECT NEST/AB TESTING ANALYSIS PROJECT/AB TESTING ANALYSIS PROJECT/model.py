import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(0)
control_group = np.random.normal(100, 15, 100)  
test_group = np.random.normal(110, 15, 100)     

mean_control = np.mean(control_group)
mean_test = np.mean(test_group)
t_stat, p_value = stats.ttest_ind(control_group, test_group)

alpha = 0.05
print(f"Control Group Mean: {mean_control}")
print(f"Test Group Mean: {mean_test}")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
if p_value < alpha:
    print("The difference between the control group and test group is statistically significant.")
else:
    print("There is no statistically significant difference between the control group and test group.")
plt.hist(control_group, bins=30, alpha=0.5, label='Control Group')
plt.hist(test_group, bins=30, alpha=0.5, label='Test Group')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('A/B Test Results')
plt.show()

control_labels = np.zeros(len(control_group))
test_labels = np.ones(len(test_group))

data = np.concatenate((control_group, test_group))
labels = np.concatenate((control_labels, test_labels))

X_train, X_test, y_train, y_test = train_test_split(data.reshape(-1, 1), labels, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
