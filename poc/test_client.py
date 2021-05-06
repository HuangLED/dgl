import grpc

# import the generated classes
import sampling_service_pb2
import sampling_service_pb2_grpc

# open a gRPC channel
channel = grpc.insecure_channel('localhost:50027')

# create a stub (client)
stub = sampling_service_pb2_grpc.SampleServiceStub(channel)

s = set()
# create a valid request message
for i in range(1000):
  request = sampling_service_pb2.Sample(id=2000, count=30)

  # make the call
  response = stub.SampleNeighbor(request)

  # et voilà
  s.add(response.neighbor)

print(s)