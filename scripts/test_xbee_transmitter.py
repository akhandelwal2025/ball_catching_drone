from digi.xbee.devices import XBeeDevice, XBeeException
import serial.tools.list_ports
import time

BAUD_RATE = 115200
HZ = 50
PERIOD_SEC = 1/HZ
def find_xbee():
    ports = list(serial.tools.list_ports.comports())
    print(ports)
    for port in ports:
        print(f"Trying {port.device}")
        device = XBeeDevice(port.device, BAUD_RATE)
        device.open()
        node_id = device.get_node_id()
        print(f"XBee found on {port.device}, Node ID: {node_id}")
        return device

device = find_xbee()
if device is not None:
    try:
        while True:
            start_sec = time.time()
            device.send_data_broadcast("ABCDEFGH")
            print(f"Elapsed: {time.time() - start_sec}")
            # time.sleep(max(0, PERIOD_SEC - (time.time() - start_sec)))
    except Exception as e:
        print(e)
        print("Closing Device")
        device.close()