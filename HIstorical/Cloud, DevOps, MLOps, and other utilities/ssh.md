

**create an ssh key**

`ssh-keygen -t ed25519 -C "email@example.com"`

**add key to ssh agent**
`eval "$(ssh-agent -s)"`
`ssh-add ~/.ssh/<name of ssh key>`

**make ssh key persist**

in ~/.ssh/config, put:

```
Host <hostname>
	IdentityFile <path to ssh key>
```

example for github: 
```
Host github.com
	IdentityFile ~/.ssh/gh_desktop
```


# Sources
- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
- https://stackoverflow.com/questions/64865626/can-i-permanently-add-ssh-private-key-to-my-user-agent