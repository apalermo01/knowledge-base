`su` - "run a command with substitute user and group ID" -> basically, this allows you to change users in the shell

become root with current user's sudo password:
`sudo -i`

become root using root password
`su root`
	- note: on ubuntu, root password is disabled by default, so this won't work unless you explicitly set a root password

# References 

- https://askubuntu.com/questions/34329/su-command-authentication-failure