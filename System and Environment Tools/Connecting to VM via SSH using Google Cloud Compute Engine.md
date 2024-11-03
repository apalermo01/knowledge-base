

- Create the VM instance, make sure "Allow HTTP traffic" and "Allow HTTPS traffic" are checked
- export the vm name as an environment variable (e.g. export VMNAME="whatever the vm name is in gcloud")
- enable connection to ssh
    - run `gcloud init` if needed
    - run `gcloud compute ssh $VMNAME`

