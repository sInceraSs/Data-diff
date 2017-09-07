# DataDiff: Synthesis of Succinct SQL Modification Scripts from Dataset Versions
This project is under the Orpheus-DB project at the [University of Illinois at Urbana Champaign][uiuc] led by [Prof. Aditya Parameswaran][prof].

### Premise
Data science leads to the proliferation dataset versions in shared file systems and folders. These versions may have been generated as a result of normalization, adding values, updating values, or adding or deleting rows. Often, the program that was used to generate these dataset versions is not known or hard to access --- data scientists may use multiple types of scripting tools for this purpose.

### Main Purpose
Given a dataset D and a subsequent dataset D' that was derived from D can we succinctly describe the changes that take us from D to D'? 

### Use Cases
Data-Diff helps users understand compactly the changes that have been made to datasets; it allows us to potentially record and recreate the transformation operations so that they can be applied to other datasets. Since we can succintly record the modifications, we only need to store the sequence of operations and the original dataset D, which is often smaller. 


### Challenges
The primary two challenges are in detecting selection queries, and in detecting whether values of a column were changed rather than removed.

For selections, the main challenge comes into play when the selection spans multiple columns. In this case, it is possible that no distinct value from a column was removed, which means that simple logic does not work to determine the query that was run. In these cases, you have to search for predicates that make progress toward a correct solution and minimize false positives, and layer them on top of each other. 

### Major Assumptions
* Currently, the datasets are assumed to be CSVs, with the first row being column names.
* We assume we have access to a primary key.


### Using Data-Diff

#### OSX Installation
```sh
$ brew install python3
$ git clone https://github.com/orpheus-db/data-diff
$ cd data-diff
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
Data-Diff was designed on Python3, and we reccommend using a virtual environment.

#### Running Data-Diff
```sh
$ python runner.py <origin_dataset> <destination_dataset> <optional-schema-file>
```
Data-Diff will prompt you for input about the columns in the origin and destination, their mapping, and the index of the primary key. If you have provided a schema file, Data-Diff will not prompt you for any of the following information; it will be populated from the schema file. See the data/ folder in this repository for a sample schema file.

It will first ask about the columns from the destination that are not in the origin. It will ask whether each column was added or is merely a renaming of a column in the origin.

It will then ask the same for the columns from the origin that are not in the destination, except this time it will ask if they are deletions rather than additions.

Then, it will ask you to map the columns that you indicated are merely renamings to each other. It will display a column from the destination and ask you to map it to the column from the origin. 

Finally, it will ask for the index of the primary key from the destination columns.

Note that all of these prompts merely require the user to enter a number. Column indices are numbered from zero. 

#### Running Data-Diff on the Sample Data
The repository has crime data from the [City of Chicago][chicagodata]. You may want to run Data-Diff on the sample origin and destination datasets, and you can do so by running the command above.
```sh
$ python runner.py data/c15.csv data/c15-m2.csv data/c15-c15-m1-schema.json
```
The above command will use the schema file. 
```sh
$ python runner.py data/c15.csv data/c15-m2.csv < data/stdin
```
Typically, Data-Diff prompts the user for some information about columns and primary keys when a schema file is not provided, but the above command uses a stored response to all of the questions that Data-Diff would ask a user. You will still see everything print to the screen, but you can ignore all the output except the last few lines, which will start with "Removed values". Those last few lines are the diff between the datasets.

You could also simply run:
```
$ python runner.py data/c15.csv data/c15-m2.csv
```
Data-Diff will then prompt you for input, as described above. 

#### Output
Data-Diff will print to stdout of your terminal as it is being used. 

When Data-Diff has finished running, it will tell you whether or not it has found a perfect matching or not. It will then identify the values that were removed, and which column those values belonged to in the origin dataset. 

### Other Projects
Why do previous approaches fail? There has been some work, given dataset D and query V on identifying a view selection query Q such that V is approximately Q(D), with previous works looking at the question of identifying a succinct selection query, XXX. 

While this work is certainly related, we are targeting a much more general problem since we are allowed to modify the dataset D using multiple SQL queries, in addition to applying view selection type operations. The aforementioned problem becomes intractable much sooner. 

In a similar vein, there is work on identifying diffs of git and svn repositories but this is line-by-line as opposed to conceptual changes.

### Completed Work
- Data is read in from a CSV, and processed using Pandas
- Detects column differences between datasets and prompts user to confirm differences
- Utilize a schema file for column differences and finding the primary key
- Can adequately detect single column selection queries and succinctly describe them
- Can adequately detect modifications to a single column where one modification was performed on all the modified rows
- Can adequately detect modifications to a signle column where a split point was used to perform one modification to rows with a value less than or equal to the split point, and another modification was performed on the rest of the rows.

For further details on these cases, see the documentation inside the docs/ folder of this repository.

### Future Work
While these were not cases that were a part of this project, these could be considerations for future work on this topic.
- Improve deletion detection to better detect predicates that span multiple columns
- Operate upon a Database rather than flat csv files
- Support primary keys that are multi column
- Create a front-end of some sort

[//]: # (Need references below)
  [prof]: http://web.engr.illinois.edu/~aditygp/#
  [uiuc]: http://www.illinois.edu
  [chicagodata]: https://data.cityofchicago.org/
