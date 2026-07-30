5 CLUSTERS:
C-INDEX: 0.236
DUNN INDEX: 0.265

7 CLUSTERS:
C-INDEX: 0.228
DUNN INDEX: 0.299

9 CLUSTERS:
C-INDEX: 0.203
DUNN INDEX: 0.307

10 CLUSTERS:
C-INDEX: 0.246
DUNN INDEX: 0.194

12 CLUSTERS:
C-INDEX: 0.227
DUNN INDEX: 0.163

15 CLUSTERS:
C-INDEX: 0.202
DUNN INDEX: 0.177

--- Report ---
For C-Index it performed better with k=10 clusters, which is also 
the number of unique labels, and the second best was k=5.
For Dunn Index it performed better with k=9 and k=7 clusters,
and scores higher than 9 clusters, show decreasing performance.
In conclusion, despite the two metrics disagreeing on the optimal
number of clusters, we can see that Dunn Index is better at
evaluating the cluster separation quality in this type of dataset.
