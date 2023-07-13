from pickle_messenger import PickleMessenger
import time

test_send_dict = {"apple": 5, "cucumber": 2, "banana": 3}
HOST_IP = "172.26.116.22"
PORT = 12345
messenger = PickleMessenger(HOST_IP, PORT, is_host=False)
messenger.start()
# time.sleep(5)
print(messenger.receive_object())