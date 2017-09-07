# Data-Diff Strategies

## Program Flow

![Flowchart](/docs/flowchart.jpeg)

The program is broken into two major steps. The first is determining the Schema Mapping from the Origin to the Destination, and the second is determining the Data Changes from the Origin to the Destination.

A view of how the process will progess is provided below.

![TemporalFlowchart](/docs/temporalflowchart.jpeg)

The same view, but with a little more detail, is below.

![DetailTemporal](/docs/detailtemporal.jpeg)


## Schema Mapping
Detecting differences in schema between the origin and the destination is rather straightforward. The program finds all the schema elements that exist in the destination that do not exist in the origin, and prompts the user to identify whether these schema elements are renamings of schema elements from the origin, or whether they are completely new columns. It does a respective operation for columns that exist in the origin but not in the destination, asking if the schema elements are renamed for the destination or whether they were removed in the destination. 

After prompting the user for this information, it asks the user to map the renamed schema elements in the origin to the ones in the destination. 

It also prompts the user for the Primary Key. Right now, the primary key must be one column, but it will be expanded to support multi-column primary keys, as this does occur in many datasets.

Determining the differences in schema is necessary for the next steps, which are determining the differences between the data itself. 

## Data Changes
### Deletion
For every column that exists in both the destination and the origin, we compute the set of values for that column in both datasets. Then, we compute the difference between these two sets, and store it. The difference between the sets is a potential modification to the origin that could yield the destination. 

After doing this for every column, we take the list of differences, and order this in ascending order by the size of the set. 

We then take elements from the front of this list, and attempt to apply the modifications represented by that element to the origin dataset. If that modification makes progress toward the destination, then we keep that modification, and continue iterating over the list. If the modification does not make progress toward the destination, we ignore it. If we have a reached a scenario when we have found a perfect matching, we terminate iteration, and return. 

This methodology proves to be effective for sequential operations of selection or deletion, but proves to not be effective when individual predicates span multiple columns. In order to work towards being effective in that second case, we must derive a different methodology. 

### Changes
Given that we have the primary key, we can find the primary keys of all the rows that have experienced some modifications to their values. Then, given these primary keys, we iterate over every column, and search for modifications. 

## General Strategies
### New Rows
Report that the new rows have been added.

### New Columns
Report which columns have been added, and whether the new column is derived data or new data.. This information is confirmed by the user.

### Deleted Rows
Follow suit with the strategy for deletion as outlined above.
In this case, we are able to handle when the predicate for removing values is on a single column.

### Deleted Columns
Report which columns have been deleted. This information is confirmed by the user, either via a schema file or via console input.

### Modified Rows
Follow suit with the strategy for changes as outlined above.
In this case, we are able to handle when a single modification is performed for any column for all the modified rows.
We are also able to handle when two modifications are performed for any column on the modified rows. That is, one modification is performed for all the values less than or equal to the split point, and another modification is performed for values greater than the split point. We consider the modifications to be atomic and both to occur at the same time; if a value is below the split point before any modifications, and the modification makes it greater than the split point, it will not be modified again. The only modification that will be applied to that value is the one for values less than or equal to the split point.

### Modified Columns
These are renamings of columns; this information is gathered from the user, either via a schema file or via console input.
