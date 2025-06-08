Per-class accuracy explained
For each label (class), the accuracy measures how well our model correctly predicted samples belonging to that class. Formally:

Per-class accuracy for label 
𝑖
=
Number of correctly predicted samples of class 
𝑖
Total number of true samples of class 
𝑖
×
100
%
Per-class accuracy for label i= 
Total number of true samples of class i
Number of correctly predicted samples of class i
​
 ×100%
Breaking down our example:
Label	Meaning	Per-class accuracy	Interpretation
0	Big drop	100.0%	our model predicted all the "big drop" samples correctly (perfect for this class).
1	Small drop	33.3%	our model correctly predicted only about 1/3 of the "small drop" samples; many were misclassified.
2	Neutral	33.3%	Similarly, only about 1/3 of "neutral" samples were correctly predicted.
3	Small rise	16.7%	our model struggled with "small rise" samples, correctly predicting only ~17%.
4	Big rise	80.0%	our model did well on "big rise" samples but missed some.