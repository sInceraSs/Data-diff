# Detailed Work Log
The readme for the project contains a brief overview of the work that was completed, as well as recommendations for future projects.
This page is meant to serve as a more detailed work log.

## Completed Work
- Either use a schema file, or get input from the user
  - This portion is used to create the column_mapping and the pk_dest index
  - Note that the column mapping maps the index of destination columns to the index of those same columns in the origin.
    - It is a python dictionary.
- Detect rows that were removed based on a predicate that spans one column
- Detect modifications to rows where the same modification was applied to all the modified rows for any given column
- Detect modifications to rows where two modifications were applied to the modified rows for any given column
  - This is the split point case.
- Report when a modification could not be found, and report the primary keys that were modified in any given column
- Report the rows that were added by returning the primary keys for those rows


## Recommendations for Future Work
- The deleted rows case has some of the earlier code in this codebase. 
  - While the logic is valid, some refactoring may need to be done to ensure its extensibility.
    - It prefers to use the column_mapping whereas other parts of the codebase modifies the columns and column labels of both dataframes to be the same.
  - Future projects could play with the threshold for differences (len(diff) -- see detect_rows.py:26)
    - Having this threshold in the first place isn't something I'm completely happy with, but it works for single column predicates
- Improve deletion detection to better detect predicates that span multiple columns
  - It's likely that you will have to seriously consider 'taking' false positives when going down this route.
  - Finding perfect matches here will be very difficult
- Operate upon a Database rather than flat csv files
- Build other Modification classes
  - column_regression.py currently has one class that detects Linear Modifications, but it would be worthwhile to try other types of models.
- Support primary keys that are multi column
  - This is a 'nice-to-have' feature, but adds significant complexity to the codebase
- Create a front-end of some sort
