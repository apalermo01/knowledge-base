# Module 2: Compute in the cloud

## Elastic Compute Cloud -> EC2

EC2 is essentially a virtual server (to serve client requests) with on-demand capabilities. 

**multitenancy**: sharing underlying hardware between virtual machines<br>
even though they are using the same resources, EC2 instances living on the same machine are not aware of each other becuase the hypervisor handling virtualization keeps them separated

can create as many EC2 instances as you want with windows or linux with various software packages

they are also resizable

**vertical scaling** - instances can get bigger or smaller as needed

can also control networking aspect (public or private?)

uses **Compute as a Service (CaaS)**


## EC2 Instance Types

each type is grouped under and instance family -> different optimizations<br> 

instance families:
- general - balanced / diverse
- compute optimized - Gaming / HPC / modeling
- memory optimized - useful for processing large datasets all at once. 
- accelerated computing - floading point calcs, graphics processing, data pattern matching
- storage optimized - for locally stored data; distributed file systems, data warehousing, and high-frequency online transaction processing (OLTP)

## EC2 pricing

- on-demand pricing - only pay for duration that instance runs for (per hour or per second) - good for baseline / testing
- savings plans - commit to a certain amount of usage for 1 or 3 year term (72% savings)
- reserved instances - steady-state workloads / predictable usage (~75% discourt) - 1 or 3 year term with installments
- spot instances - ask for spare capacity (can be reclaimed at any time- 2 minute warning) - ~ 90% reduction
- dedicated hosts - reserve a single machine

## Scaling EC2

Capacity can grow or shrink based on needs<br> 

There is no single point of failure for EC2 isntances.<br> 

**EC2 auto scaling**: <br> 
scale up: add more power to existing machines<br> 
scale out: add more instances<br> 

can add the right amount of power for each task<br>. 

adds instances based on demand and decomissions when they're not needed<br>

## Elastic Load Balancing

tells the client which EC2 instance to use based on load<br> 

ensures an even distribution of workload<br> 

regional construct<br> - automatically highly available<br> 
scales with EC2 isntances - no disruption to traffic handling<br>

also handles front-end to back-end communications<br> 

front end doesn't know or care what backend instances are running<br> 

many ways to do it, many different services

## Messaging and queuing

if cashier and barista are out of sync- issues appear<br> 
fix: use a buffer / queue (e.g. order board)<br> 
applications send messages to each other<br> 
**tightly coupled** - apps talk directly to each other. if they go out of sync, then everyone is in trouble.<br> 
**loosely coupled** - if one component fails, it a isolated and there is no cascade of errors. Messages go through a queue. If reciever goes down, the line in the queue just gets bigger

**Amazon Simple Queue Service (SQS)**<br> 
send, store, and recieve messages between components at any volume<br> 
Where messages are placed until they are processed<br>
scale automatically<br> 


**Amazon Simple Notification Service**<br>
sends messages to sevices and end users<br> 
**Amazon SNS topic:** - a channel for messages to be delivered<br>
can send one message to a topic which will fan out to subscriber - s (which can be anything - end users, other AWS services, etc.)

**Monolithic application** - application whose components are tightly coupled (if one component fails, the whole things goes down)

**Microservice approach** - components of an application are loosely coupled

## Additional compute services

**EC2** - good for a bunch of different services<br> 
requires that you set up and manage instances over time<br>
you are responsible for patching, architecting solutions, and other management processes ect.<br>

**Serverless** - you cannot see or access underlying infastructure / instances<br> 
all management processes are taken care of for you

**AWS lambda** - upload code into lambda function, config trigger, then service waits for trigger. Code is run in a managed environment when the trigger goes off. Designed to run code < 15 min. Better for quick processing. 

**AWS Continer Services**
- Amazon Elastic Container Services (ECS) - run containerized applications at scale.
- Amazon Elastic Kubernetes Services (EKS) - similar, but with different tooling and features

These are container orchestration tools<br> 
**container** = docker container<br> 
**container** = package for code<br> 

containers run in EC2 instances<br> 

Both of these can run on EC2<br> 

If you don't want to bother with EC2 instances, you can use **AWS Fargate**

**Fargate** - serverless comput platform for ECS or EKS<br> 

traditional applications, full access to OS, use EC2<br> 
short runnign functions, service-oriented, event driven, no underlying environment, Lambda<br> 
docker container based: ECS or EKS, then choose platform (EC2 or serverless tool like Fargate)<br> 


**EC2**<br> 
- porvision instances
- upload code
- manage instances while app is running

**Lambda**<br> 
- don't provision or manage servers
- code runs only when triggered

**Containerized applications**<br>
- package code and dependencies into a single object
- uses container orchestration services to manage containers on many different servers* 

**ECS**, **EKS**, **Fargate**
