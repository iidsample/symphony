grpc:
	mkdir -p grpc_stubs
	python -m grpc_tools.protoc -I./proto --python_out=./grpc_stubs --grpc_python_out=./grpc_stubs ./proto/datagen.proto
	python -m grpc_tools.protoc -I./proto --python_out=./grpc_stubs --grpc_python_out=./grpc_stubs ./proto/scheduler.proto
	python -m grpc_tools.protoc -I./proto --python_out=./grpc_stubs --grpc_python_out=./grpc_stubs ./proto/node_scheduler.proto

clean:
	rm ./grpc_stubs/*_pb2.py
	rm ./grpc_stubs/*_pb2_grpc.py

