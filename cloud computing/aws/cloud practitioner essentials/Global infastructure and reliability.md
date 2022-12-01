**aws has multiple large data centers for redundancy**

# AWS regions

**Fundamental problems with datacenters** - things may happen that cause a lost connection to the building<br> 

data centers exist in large groups- regions<br> 

each region has multiple data centers<br> 

each region is connected through a fiber network<br> 

we choose which region to run out of<br> 

each is isolated from every other region. data is NEVER moved between regions without EXPLICIT permission<br> 

**factors that go into choosing a region**<br> 
- compliance
- proximity - physical distance to customer base is important. **latency** - the time it takes for data to be sent and recieved
- Feature availability - brand new services aren't rolled out in all regions at the same time. e.g. amazon braket (quantum computing platform) - currently not available in all regions
- pricing - some locations are more expensive (each region has a different price sheet)

# Availability zones

each region has multiple data centers<br> 

**availability zone** - one or more redundant data centers<br> 

each region has multiple separate and isolated availability zones<br>

When you scale EC2 instances, new instances are created in separate availability zones for redundnacy. Can get up to 10s of miles apart while still having single digis ms latency<br> 

recommended to run across two availability zones in a region<br> 

many services run at the region level. e.g ELB - regional construct. Runs on all availability zone. 

**Regionally scoped service** - available on all availability zones



# Edge locations


what if you have customers all over the world? (or not close to any availability zones)?

**Content delivery networkk** - **Amazon CloudFront** - delivers data / apps using **edge locations** to accelerate communications

**edge locations** - separate from regions. Run DNS (**Amazon Route 53**) <br> 

**AWS Outposts** - AWS can install mini-region inside your own data center. 


**key points**
- regions are geographically isolated areas
- regions contain availability zones (with 10s of miles of separation)
- edge locations run Amazon CloudFront to get content close to globally distributed customers

**CloudFront** stores cached copies of content


# How to Provision AWS resources

everything is an **API** call<br> 
**API** = application programming interface<br> 

use these to provision, call, and manage AWS services<br> 

can use:
- AWS management console
- AWS command line interface (CLI)
- AWS software development kits (SDKs)
+ other tools

**AWS management console**<br> 
manage visually, browser-based. Good for learning, test environments, non-technical resources.<br> 

**AWS CLI** <br> 
make API calls using terminal<br>
actions become scriptable and repeatable<br> 
can run automatically, or based on triggers<br> 

**AWS SDK**<br> 
interact with AWS resources using various programming languages<br> 

**AWS Elastic Beanstalk**<br> 
provision EC2 environments. give it code and desired configurations, then builds out environment for you. can save env configurations. automatically manages and scales EC2 instances and others

**AWS cloud formation**<br> 
infastructure as code tool. JSON or YAML - cloud formation templates. can define what you want without worrying about details. <br> 
storage, databases, analytics, ML, ect. <br> 
manages all calls to backend APIs for you
