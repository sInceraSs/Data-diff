#Datasets

##Chicago Crime Data

The origin dataset is c15.csv.

c15-m1.csv and c15-m2.csv have renamed columns from c15.csv as follows
* Case Number has become CaseNumber
* Primary Type has become PrimaryType
* Location Description has become LocationDescription
* Community Area has become CommunityArea
* Updated On has become UpdatedOn

c15-m1.csv and c15-m2.csv have removed the following columns
* IUCR
* FBI Code
* X Coordinate
* Y Coordinate
* Latitude
* Longitude
* Location

c15-m1.csv has removed rows where the value of column 'Arrest' is 'false'.

c15-m2.csv has removed rows where the value of column 'Arrest' is false'. It then removed rows where 'Primary Type' is 'Narcotics'.
