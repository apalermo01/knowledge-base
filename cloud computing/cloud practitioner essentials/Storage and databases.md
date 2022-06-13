## Instance stores and Amazon Elastic Block Store (Amazon EBS)

**block level storage** - place to store files<br> 
good for databases, enterprise software, or filesystems<br> 
this is the storage type for hard drives<br> 

EC2 instances may come with **instance store volumes** - hard drive for EC2 instances.<br> 
if the EC2 instance is stopped, all data written to instance store volume is deleted<br> 

DO NOT WRITE IMPORTANT DATA TO THE DRIVES THAT COME WITH EC2 INSTANCES<br> 

**Elastic Block Store**<br>
create virtual hard drives not tied directly to host that EC2 is running on- so the data persists between stops / starts of EC2 instances. Define size, type, & configurations, provision volume, then attach to EC2 instance.<br> 

can take incremental backups (**snapshots**)

## Amazon Simple Storage Service (S3)

store and retrieve an unlimited amount of data at any scale<br> 

- store data as buckets
- store objects in buckets
- upload a max object size of 5 TB
- version objects
- create mutliple buckets


**Amazon S3 Standard**<br> 
Insane durability<br> 
Stored in at least 3 facilities<br> 
good for **static website hosting** - upload all files into a bucket then host the bucket<br>

**Amazon S3 Standard-infrequent Access (S3 Standard-IA)**<br> 
less frequent, but rapid access. good for backups<br> 

**Amazon S3 Glacier**<br> 
data archive- good for long-term storage for data retention and archives<br> 
vault lock policy to lock from future edits<br> 
multiple options for retrieval<br> 

**S3 Lifecycle management** - move data automatically between tiers<br> 

![image.png](attachment:image.png)


**S3 storage classes**<br> 
- S3 standard: high availability and redundancy. Higher cost. 
- S3 standard-Infrequent Access (STandard-IA): For infrequently accessed data by high availability when needed. lower storage price, but higher retrieval price.
- S3 One Zone-Infrequent Access (One Zone-IA): stores data in one availability zone. cheaper, but single point of failure. 
- S3 Intelligent-Tiering: unknown / changing access patterns. Monitoring / automation fee per object. 
- S3 Glacier: low cost storage for archiving. Minutes to hours for retrieval. 
- S3 Glacier Deep Archive: lowest cost storage. Retrieve within 12 hours.  

**Comparing EBS and S3**<br> 

![image-2.png](attachment:image-2.png)

## Amazon Elastic File System (EFS)

- managed file system
- can handle multiple instances accessing the data at the same time
- difference between EBS: 
    - EBS atttached to EC2 instances, and are an avilability zone resource
    - EFS can have multiple instances reading and writing simultaneously. 
    - linux file system
    - regional
    - automatically scales

## Amazon Relational Database Service (Amazon RDS)

- Store data in a way such that it can relate to other bits of data
- supports: 
    - MySQL
    - PostgreSQL
    - Oracle
    - Microsoft SQL server
    - Amazon Aurora
    - MariaDB

**Lift and shift** - migrate entire db to EC2 instance<br> 

**Amazon RDS offers: **<br> 
- automated patching
- Backups
- Redundancy
- Failover
- Disaster Recovery

how do we make it easier to migrate and manage databases?<br> 

**Amazon Aurora**<br> 
- compatible with MySQL or PostgreSQL
- data replicated across facilities- 6 copies
- up to 15 read replicas
- continuous backups to S3
- point-in-time recovery

## DynamoDB

- serverless database
- create tables - Item -> attributes
- automatically scales and redundant
- ms response time
- noSQL (non-relational)
- no schema needed
- can add / remove attributes at any time and items do not need to have all attributes
- write queries based on small subset of attributes (called keys)
- queries tend to be simpler and based on one table




**Comparing DynamoDB and RDS**

![image.png](attachment:image.png)

![image-2.png](attachment:image-2.png)

## Amazon Redshift

**Data Warehouses** - big data solutions for historical analytics instead of operational analytics for when data becomes really complex<br> 

solution for when we're doing analytics in the past<br> 

- Data warehousing as a service
- can directly run SQL query across exabytes of data across datalakes
- 10x performance compared to traditional databases

## AWS Database migration service (DMS)

What if you already have a database set up?<br> 
- migrates existing databases
- source remains operational during migration
- source and target databases don't have to be of the same type
- homogeneous: source and target are the same type
- second type: heterogeneous migration
- source and target are incompatable, so use **AWS schema conversion tool**
- Then use DMS to migrate
- can also do dev and test, consolidation, and continuous replication
    - e.g. run tests on live database without affecting it
    - consolidate several databases into one

## Additional database services

- no one-size-fits-all database solution
- other types of databases:
    - Amazon DocumentDB - content management
    - Amazon Neptune - graph database (engineered for social networking / recommendation & fraud detection)
    - Amazon Managed Blockchain
    - Amazon Quantum Ledger Database (QLDB) - any entry can NEVER be removed from audits- ledger database
    - Database accelerators
        - Amazon ElastiCache - ms responses to us
        - DynamoDB accelerator


```python

```
