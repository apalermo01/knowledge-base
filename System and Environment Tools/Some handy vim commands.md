# Normal mode

- Changing text
	- `c#l` - changes current some number of characters to the right
	- `ciw` - put cursor anywhere in word, lets you change the entire word, not just starting where the cursor is like in `cw`

- Macros
	- `q<register name>` - starts recording a macro in the specified register. Press `q` to stop.
	- `@<register name>` - run the macro stored in the specified register
	