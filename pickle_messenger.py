import pickle
import socket

class PickleMessenger:
    def __init__(self, host, port, is_host=True):
        self.host = host
        self.port = port
        self.socket = None
        self.connection = None
        self.is_host = is_host

    def start(self):
        if self.is_host:
            self.host_server()
            print('hositing socket')
        else:
            self.connect()
            print('connecting to socket')

    def host_server(self):
        if not self.is_host:
            raise Exception("Cannot start PickleMessenger if not host.")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print(f"Listening for incoming connections on {self.host}:{self.port}...")

        self.connection, _ = self.socket.accept()
        print("Connection established.")

    def connect(self):
        if self.is_host:
            raise Exception("Cannot connect PickleMessenger if host.")
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}.")

    def send_object(self, send_object, object_name:str):
        pickled_dict = pickle.dumps({"name":object_name, 'data':send_object})
        self.connection.sendall(pickled_dict)
        print("object sent successfully.")

    def receive_object(self):
        pickled_dict = self.connection.recv(4096)
        received_object = pickle.loads(pickled_dict)
        data_object = received_object['data']
        name = received_object['name']
        print("Received object:")
        print(received_object)
        return data_object, name

    def close(self):
        if self.connection:
            self.connection.close()
        if self.socket:
            self.socket.close()
