# Steppig through the tableau prep workflow

## Data preparation process

need might arise from data problems during analysis (e.g. strange looking charts, inconsistent values (Hi vs High))

- Cleaning data does NOT change the underlying source

Selecting a value highlights associated values in the other rows

ex: combine spellings of values on a row (e.g. Hi vs High)

- check # rows for each value
- change the spelling of an incorrect field to the correct spelling (replaces the wrong spelling with the correct spelling)- grouped values
- changes arrow -> gives a list of all cleaning operations
- right click any step after loading data -> check data in tableau desktop
- file -> save as -> save as a tableau flow file (.tfl) or packaged tableau flow (.tflx) (this one saves a copy of the original data)

- context menu -> group and replace can do more kinds of cleaning

## Outputting clean data

- -> add output
- save clean data as a file or data source
- default type: .hyper
- run the flow to make the extract

## Using the output file as a data source

- Start -> connect to a file -> more -> navigate to location -> open 
