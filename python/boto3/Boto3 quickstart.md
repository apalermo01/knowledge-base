# Boto3 quickstart

Before executing anything, ensure that the aws cli is installed and configured<br> 
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html


```python
import boto3
```

Make a boto3 session.

On my machine, I currently have 2 sets of credentials: one for my work account and another for a personal account so I can keep the ArmadaIQ infastructure separate from any tutorials / documentation I work through which I might make publically available. 

`profile_name` specifies which set of credentials to use (no argument for the profile name uses my work account as defaul)


```python
sess = boto3.Session(profile_name="personal")
```

select which service to use


```python
s3 = sess.resource('s3')
s3
```




    s3.ServiceResource()



print out bucket names


```python
for bucket in s3.buckets.all():
  print(bucket.name)
```

    abucket-20210719-1519

