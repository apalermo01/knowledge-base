- separate from OS users
- when postgres is first installed, it always comes with a superuser called 'postgres'
- it also makes a  system user called postgres (I think -> need to double check that this is exactly what's happening)

**To create / delete a role**

in SQL: 
```sql
CREATE ROLE <rolename>;
DROP ROLE <rolename>;
```


in terminal:
```bash
createuser <rolename>
dropuser <rolename>
```

Note: on ubuntu, these require you to be acting as the user postgres, so you'll either have to do:
```bash
sudo -u postgres createuser <rolename>
sudo -u postgres dropuser <rolename>
```

or

```bash
su postgres # switch to user postgres, entering a password if needed
createuser <rolename>
dropuser <rolename>
```

**To list users**

```sql
SELECT rolname FROM pg_roles;
```







# References 

- https://www.postgresql.org/docs/9.3/user-manag.html
- https://www.postgresql.org/docs/9.3/database-roles.html