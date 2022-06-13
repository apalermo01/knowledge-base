# Boto3 sample tutorial

https://boto3.amazonaws.com/v1/documentation/api/latest/guide/sqs.html

Use boto3 with SQS.<br>
Generate a new queue, get and use existing queue, push messages onto the queue, and process messages<br> 


```python
import boto3
sess = boto3.Session(profile_name="personal")
```


```python
# get the service resource
sqs = sess.resource("sqs")

# create a queue
queue = sqs.create_queue(QueueName="test", Attributes={'DelaySeconds': '5'})
```


```python
# access identifiers and attributes
print(queue.url)
print(queue.attributes.get('DelaySeconds'))
```

    https://queue.amazonaws.com/710610697119/test
    5


As expected, the queue shows up in the dashboard

![image.png](attachment:image.png)

## Using an existing Queue


```python
# get the service resource
sqs = boto3.Session(profile_name="personal").resource("sqs")

queue = sqs.get_queue_by_name(QueueName="test")

print(queue.url)
print(queue.attributes.get("DelaySeconds"))
```

    https://queue.amazonaws.com/710610697119/test
    5


list all existing queues


```python
for queue in sqs.queues.all():
  print(queue.url)
```

    https://queue.amazonaws.com/710610697119/test



```python
for queue in sqs.queues.all():
  print(queue.attributes)
```

    {'QueueArn': 'arn:aws:sqs:us-east-1:710610697119:test', 'ApproximateNumberOfMessages': '0', 'ApproximateNumberOfMessagesNotVisible': '0', 'ApproximateNumberOfMessagesDelayed': '0', 'CreatedTimestamp': '1626722686', 'LastModifiedTimestamp': '1626722686', 'VisibilityTimeout': '30', 'MaximumMessageSize': '262144', 'MessageRetentionPeriod': '345600', 'DelaySeconds': '5', 'ReceiveMessageWaitTimeSeconds': '0'}


## Sending messages


```python
response = queue.send_message(MessageBody="test")

print(response)
```

    {'MD5OfMessageBody': '098f6bcd4621d373cade4e832627b4f6', 'MessageId': 'f511b02f-d3ce-459d-9498-021ff98394f7', 'ResponseMetadata': {'RequestId': '870ffa1d-89ae-5790-bed0-a372508b49d3', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': '870ffa1d-89ae-5790-bed0-a372508b49d3', 'date': 'Mon, 19 Jul 2021 19:47:49 GMT', 'content-type': 'text/xml', 'content-length': '378'}, 'RetryAttempts': 0}}


response is NOT a resource


```python
print(response.get("MessageId"))
print(response.get("MD5OfMessageBody"))
```

    f511b02f-d3ce-459d-9498-021ff98394f7
    098f6bcd4621d373cade4e832627b4f6


There is now 1 message in the queue


```python
list(sqs.queues.all())[0].attributes
```




    {'QueueArn': 'arn:aws:sqs:us-east-1:710610697119:test',
     'ApproximateNumberOfMessages': '0',
     'ApproximateNumberOfMessagesNotVisible': '0',
     'ApproximateNumberOfMessagesDelayed': '0',
     'CreatedTimestamp': '1626722686',
     'LastModifiedTimestamp': '1626722686',
     'VisibilityTimeout': '30',
     'MaximumMessageSize': '262144',
     'MessageRetentionPeriod': '345600',
     'DelaySeconds': '5',
     'ReceiveMessageWaitTimeSeconds': '0'}



Message with custom attributes


```python
queue.send_message(MessageBody="HI SQS!", MessageAttributes={
  'Author': {
    'StringValue': "Alex",
    "DataType": "String"
  }
})
```




    {'MD5OfMessageBody': '0cde403675f9988b245fbf3b6eff3e84',
     'MD5OfMessageAttributes': 'd78d8039a37dbc92fbda6efca45149db',
     'MessageId': 'e3418ad3-d004-4924-8e70-07ed327065c7',
     'ResponseMetadata': {'RequestId': 'c2692cdf-730b-5284-abde-75b0aa21acc7',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'x-amzn-requestid': 'c2692cdf-730b-5284-abde-75b0aa21acc7',
       'date': 'Mon, 19 Jul 2021 19:47:54 GMT',
       'content-type': 'text/xml',
       'content-length': '459'},
      'RetryAttempts': 0}}



Messages can also be sent in batches


```python
response = queue.send_messages(Entries=[
  {
    'Id': '1',
    'MessageBody': 'First message in a group of two'
  },
  {
    'Id': '2',
    'MessageBody': 'The second message',
    'MessageAttributes': {
      'Author': {
        'StringValue': 'Me',
        'DataType': 'String'
      }
    }
  }
])

# print out any failures
print(response.get('Failed'))
```

    None


Now there should be several messages in the queue


```python
list(sqs.queues.all())[0].attributes
```




    {'QueueArn': 'arn:aws:sqs:us-east-1:710610697119:test',
     'ApproximateNumberOfMessages': '4',
     'ApproximateNumberOfMessagesNotVisible': '0',
     'ApproximateNumberOfMessagesDelayed': '0',
     'CreatedTimestamp': '1626722686',
     'LastModifiedTimestamp': '1626722686',
     'VisibilityTimeout': '30',
     'MaximumMessageSize': '262144',
     'MessageRetentionPeriod': '345600',
     'DelaySeconds': '5',
     'ReceiveMessageWaitTimeSeconds': '0'}



## Processing Messages

Messages are processed in batches


```python
# get service
sqs = boto3.Session(profile_name="personal").resource("sqs")

# get the queue
queue = sqs.get_queue_by_name(QueueName="test")

# Process messages by printing out body and optional author name
for message in queue.receive_messages():
  print(message.body)
  
  # let queue know that the message is processed
  message.delete()
```

    First message in a group of two



```python

```
