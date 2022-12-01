**Monitoring** - Observing systems, collecting metrics, and then using data to make decisions<br> 

## AWS CloudWatch

- access to all metrics from a center location
- gain visibility into applications, infastructure, and services
- reduce mean time to resolution (MTTR) and improve TCO (total cost of ownership)
- drive insights to optimize applications and operational resources
- **CloudWatch alarms** - automatically perform actions if value of a metric has gone above or below a predefined threshold
    - ex: can automatically stop an EC2 instance when CPU utilization has remained below a certain percentage for some time

## AWS CloudTrail

comprehensive API auditing tool<br> 

Every request made to AWS is logged in the cloud trail engine.<br> 

logs who did the request, where it came from, result, ect. <br> 

can save logs from CloudTrail in tamper-proof S3 vaults<br> 

## AWS TrustedAdvisor

evaluates resources against 5 pillars:<br> 
- cost optimization
- performance
- security
- fault tolerance
- service limits

some are free, some available based on tier<br> 
