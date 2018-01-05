# restaurantfinder
This project creates a visualization of restaurant ratings. 
It uses machine learning on a yelp dataset to create a 
visualization of Berkeley divided into regions. specifically,
a k-means algorithm is used to find clusters with centroids.
These regions are shaded a certain color depending
on the predicted ration of close restaurants. Yellow
is 5 stars, blue is 1. 

This project was created for computer science
sixty-one A at Cal. 


#######USE INSTRUCTIONS##########
you can create a user by going to the USER folder and
creating a new .dat file with your prefs on restaurants

Generate a visualization by running -u to select a user 
from the USER directory:
python3 recommend.py -u one_cluster

You can get finner groupings by increasing the number of
clusters with the -k option:
python3 recommend.py -u likes_everything -k 3

Predict what rating a user would give to a restaurant even
if they havn't visited it by using the -p option:
python3 recommend.py -u likes_southside -k 5 -p

Filter based on categories by using the -q option:
python3 recommend.py -u likes_expensive -k 2 -p -q Sandwiches



