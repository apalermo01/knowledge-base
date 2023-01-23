### Install

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**to start psql**:
```bash
sudo -u postgres psql
```


### commands

**list databases** - `\l` or `SELECT datname FROM pg_database;`

**connect to a database** - `\c <database name>` ex: `\c postgres`

**show tables in a database**: `\dt`



# Sources

- https://www.digitalocean.com/community/tutorials/how-to-install-postgresql-on-ubuntu-22-04-quickstart
- https://www.postgresqltutorial.com/postgresql-administration/postgresql-show-databases/