# TCP

TCP - Transmission Control Protocol operates at layer 4 (transport) of the 
[[OSI model]]. It provides a way of sending data between different computers
with error checking. 

One primary feature is the three-way handshake - the three steps needed to
establish a TCP connection:

- SYN
    - Client sends initial response to server.
- SYN/ACK
    - Server responds to original request.
- ACK
    - Client confirms that it got the confirmation. Data can now be transferred.

The main alternative to TCP is [[UDP]] (User Datagram Protocol)

# References
- https://en.wikipedia.org/wiki/Transmission_Control_Protocol
- https://www.geeksforgeeks.org/tcp-3-way-handshake-process/
