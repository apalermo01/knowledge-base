# Networking

**Virtual Private Cloud (VPC)**<br> 
provision logically isolated section of cloud in a virtual network that we define.<br>

# Connectivity to AWS

**Virtual Private Cloud (VPC)** - define private IP range for resources. place resources inside VPC. Place them into different subnets<br> 
**subnets** - subsets of IP addresses that let you chunk resources together. can control whether they are publically or privately available<br> 

**Public resources**<br> 
**Public gateway**- doorway that is open to the public

![image.png](attachment:image.png)

This is like the front door to a coffee shop<br> 

**VPC with all internal private resources** - Use a private gateway (virtual private gateway). VPN connection between trusted network to VPC. This is like a private bus route going from a building to coffee shop (you need to buzz into the building first).<br> 

These connections are private and are encrypted, but they still use the open internet. Can make a magic doorway from building to coffee shop that no one else can use. We want a private, dedicated connection.<br> 

**AWS direct connect** - dedicated private connection from your data center to AWS. Provides physical line from your data center to AWS. 

![image-2.png](attachment:image-2.png)

![image-3.png](attachment:image-3.png)

## Subnets and access control lists

VPC = fortress. Gateway = access point for public traffic<br> 

**Network hardening**<br> 
Only techical reason to use subnets is to controll accesss to the gateway. Public subnets have access to the gateway, private subnets do not. <br> 

packets = messages from the internet. Gets checked against **Network access control list (Network ACL)** - check if it has permissions to enter or leave the subnet (like passport control officers)<br> 

Network ACL only evaluates packets when they cross subnet boundaries<br> 


**Security group** - instance level control for permissions within subnets<br> 
by default, security groups do not allow anything in. This is like the doorman at a building. Checks whats going in, but not what's going out.<br> 

key difference: Security group is **stateful** (remembers what has gone in)<br> 
network ACL is **stateless**<br>


## Global networking

**Amazon Route 53**<br> 
Amazon's domain name service<br> 
translates website names into IP addressses<br> 
can direct traffic to different endpoints<br> 

Routing policies: 
- Latency-based routing
- Geolocation DNS
- Geoproximity routing
- Weighted round robin

can also use it to buy and manage your own domain name<br> 

**Amazon CloudFront**<br> 
caches and serves content as close as possible<br> 
**Content Delivery Network (CDN)** - A network that delivers edge content to useres based on their geographic location<br> 

**Domain name system** - phonebook of the internet

![image.png](attachment:image.png)

