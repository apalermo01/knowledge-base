![image.png](attachment:image.png)

## Shared responsibility model

EC2 example:<br> 
- EC2 lives in a datacenter
    - AWS: responsible for securing the physical building, network, and hypervisor
        - has been audited by 3rd parties
    - Customer: operating system (we are 100% in charge of the OS, AWS has no back door, AWS does NOT have an encryption key), also responsible for keeping OS patched (AWS cannot deploy a patch in OS), applications (We own and maintain applications), data (ALWAYS our domain to control, it is up to us to ensure that we make use of the appropriate toolset that AWS provides). 
    
    
AWS: security of the cloud<br> 
customer: security in the cloud<br> 

analogy: AWS is the home builder, we are the home buyer<br> 



## User permissions and access


**AWS account root user** - owns account, can do anything they want.<br> 

**MFA** - multi-factor authentication<br> 

don't use root user for everything<br> 

**AWS Identity and Acccess Management (IAM)**<br> 
create and manage useres and permissions<br>  
by default, ALL actions are denied.<br> 

**Principle of least privlege**: A user is granted access only to what they need.<br> 
**IAM Policy**: JSON doc describing which API calls a user may or may not make<br> 
**IAM Groups**: Groupings of users. Attach policy to group and all users in that group will have those permissions<br> 
**Role**: associated permissions allowing or denying actions. Assumed for temporary amounts of time. Temporarily grant access to Useres, external identities, applications, or other AWS services.<br> 


## AWS Organizations

A central location to manage multiple AWS accounts<br> 

- cetnralized management
- consolidated billing
- hierarchical groupings of accounts
- AWS service and API actions control
- **Service Control Policies (SCPs)** - place restrictions on services, resources, and individual API actions
    - applies to root, individual member account, or OU
- **Organizational Unit** - groups to manage accounts with similar business or security requirements



## Compliance

AWS already has infastructure following best practices for a laundry list of insurance and regulatory standards<br> 

The region you choose might help mees complaince<br> 

Be aware of the fact that WE own the data. We have multiple different encryption methdos to meet requirements.<br>

**AWS artifact** - get access to complaince reports from 3rd parties <br> 
**AWS compliance center**<br> 
**AWS risk and security whitepaper**<br> 

**AWS artifact agreements** - can get an agreement to sign with AWS regarding use of information to ensure that data handling conform to specific regulations (e.g. HIPAA)<br> 


**AWS artifact reports** - compliance reports from 3rd party auditors<br> 

## Denial-of-service attacks (DDOS)

objective: shut down app's ability to function by overwhelming it<br> 

Hacker leverages other machines to spam the platform with requests<br>

attack examples:<br> 
- UDP Flood
- HTTP level attacks
- slowloris attack

**How do we stop the attacks?**<br> 
- security groups (only allow proper request traffic using the expected protocols) - UDP attacks get just get shrugged off. 
- Elastic load balancer - can use the queue to wait for slowlorises to finish their requests. 
- **AWS shield with AWS WAF** - ML based firewall
- many of these attacks would require overwhelming the entire AWS region, which would be too expensive for most bad actors

takeaway: well architected system is already protected from most attacks.<br> 

**AWS Shield standard** - automatically protects customers at no cost<br> 
**AWS shield advanced** - gives detailed attack diagnostics and detect and mitigated more sophisticated attacks. Integrates with other services. 

## Additional Security Services

- Encryption: securing a message or data in a way that can only be accessed by authorized parties (lock and key)
    - Encryption at rest (in storage). Integrates with **AWS key management service**
    - Encryption in transit: protecting data when it's moving between services (e.g Secure Socket Layers (SSL))
- Amazon Inspector: Runs an automated security assessment against infastructure. Checks for violations of best practices
    - network configuration reachability
    - Amazon agent
    - Security assessment service
- Amazon GuardDuty: intelligent threat detection

