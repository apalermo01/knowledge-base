## Cloud adoption framework (AWS CAF)

guides the process of moving to AWS

Six core perspecitves of the cloud adoption framework:<br> 
- business: ensures that IT aligns with business needs
    - business managers
    - finance managers
    - budget owners
    - strategy stakeholders
- people: training, staffing, and organizational changes
    - HR
    - staffing
    - people managers
- governance: maximize business value and minimize risks
    - Chief Information Officer (CIO)
    - Program managers
    - Enterprise architects
    - Business Analysts
    - Protfolio managers
- platform: cloud architecture
    - Chief Technology Officer (CTO)
    - IT managers
    - solutions architect
- security: ensures security needs
    - Chief information security officer (CISO)
    - IT security managers
    - IT security analysts
- operations: day-to-day operations
    - IT managers
    - IT support managers

AWS CAF Action plan: helps guide organization for migration<br> 



## Migration strategies

The 6 R's (strategies) of migration:<br> 
- Rehosting (lift and shift) - no actual changes. Pick up applications and move as-is. Without optimization, can still get ~30% savings, then optimize once in the cloud
- Replatforming (lift, tinker, and shift) - make a few optimizations, but no new development. 
- Retire - ~10-20% of application protfolios include things that are no longer being used or already replaced. Use migration as an opportunity to kill applications. 
- Retain - keep applications that are about to be depreciated, so keep them where they live
- Repurchasing - fresh start, end contract or licenses with legacy vendors in favor of cloud solutions. More expensive up front
- Refactoring - add features or performance that are not available on-premesis. Highest initial cost. 

## AWS Snow family

Network with 1 GPS moves 1 PB in ~100 days<br> 

Use physical devices to move data to AWS datacenters. Order the device, copy data onto it, ship back to Amazon, then they will upload it to their hardware. 

- AWS Snowcone
    - 8 TB
- AWS Snowball Edge
    - compute optimized
    - storage optimized
    - fit into existing server racks
    - can run AWS services on premesis
    - usually ship to remote locations
    - use cases: streams from IOT, compression, ect. 
- AWS Snowmobile
    - 45 ft shipping container
    - 100 PB
    - largest migrations & data center shutdowns

## Innovation with AWS


