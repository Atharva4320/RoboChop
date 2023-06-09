from UDPComms import Subscriber, timeout
import warning

sub = Subscriber(5500)

while True:
	try:
		message = sub.get()
		print('message: ', message)

	except timeout:
		warning.warn("UDPComms timeout")
		break