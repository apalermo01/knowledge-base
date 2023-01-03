
```bash
ssh-keygen -b 4096
```
- select the filename
- select a password

## adding ssh key to agent
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/<name of ssh key>
```

