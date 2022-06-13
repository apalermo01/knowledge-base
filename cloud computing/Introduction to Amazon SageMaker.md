# Intro to SageMaker tutorial

3 components:<br> 
- notebooks
- training service
- hosting service

**Training jobs**: <br> 
specify the compute resources that you need, then SageMaker will make the required EC2 instances, run the training, then tear it down when finished<br> 

bring your own algos in the form of docker images<br> 

![image.png](attachment:image.png)



from example notebook instance:<br> 

using sample-notebooks -> sgboos_direct_marketing_sagemaker

after transformations:<br> 
`boto3.Session().resource('s3').Bucket(bucket).object...`<br> 
these 3 lines are uploading the train, val, and test sets to an S3 bucket<br> 

