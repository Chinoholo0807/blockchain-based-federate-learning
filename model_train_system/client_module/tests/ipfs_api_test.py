import ipfshttpclient

client = ipfshttpclient.connect()
print(client.id())
