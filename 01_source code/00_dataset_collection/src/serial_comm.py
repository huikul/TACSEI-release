#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     :
# File Name  : serial_comm.py

"""
$ pip install pyserial

Step 1: Identify the Serial Port
First, you need to identify the name of the serial port you want to access.
Plug in your USB-to-serial adapter (if you’re using one).
List available serial ports by running the following command in the terminal:

$ dmesg | grep tty
This command will display messages related to serial ports. Look for lines like /dev/ttyUSB0 or /dev/ttyS0.

Step 2: Check Current Permissions
Check the current permissions of the serial port by running:

bash
$ ls -l /dev/ttyUSB0
Replace /dev/ttyUSB0 with your specific port name if it’s different.

The output will look something like this:
crw-rw---- 1 root dialout 188, 0 Sep  2 09:47 /dev/ttyUSB0
This indicates that the device is owned by the root user and the dialout group. Regular users typically do not have permission to access this device unless they are a member of the dialout group.

Step 3: Add Your User to the dialout Group
To allow your user to access serial ports, add the user to the dialout group:
Add user to the group:
Replace your_username with your actual username.

$ sudo usermod -aG dialout your_username
Log out and log back in to apply the changes. Alternatively, you can reboot your system.

"""

import numpy as np
import serial
import time
from abc import ABC, abstractmethod


class SerialPort(ABC):
    def __init__(self, port, baudrate=9600, timeout=1, parity='N', stopbits=1, bytesize=8):
        """
        Initializes the SerialPort class with port, baudrate, timeout, parity, stopbits, and bytesize.

        :param port: Serial port to open (e.g., 'COM3' or '/dev/ttyUSB0')
        :param baudrate: Communication speed (default is 9600)
        :param timeout: Read timeout in seconds (default is 1)
        :param parity: Parity checking (default is 'N' for None, can be 'E' for Even, 'O' for Odd)
        :param stopbits: Number of stop bits (default is 1, can be 1, 1.5, or 2)
        :param bytesize: Number of data bits (default is 8, can be 5, 6, 7, or 8)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.parity = parity
        self.stopbits = stopbits
        self.bytesize = bytesize
        self.serial_connection = None

    def open(self):
        """Opens the serial port connection with specified configurations."""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=self._get_parity(self.parity),
                stopbits=self._get_stopbits(self.stopbits),
                bytesize=self._get_bytesize(self.bytesize)
            )
            print(
                f"Serial port {self.port} opened successfully with settings: baudrate={self.baudrate}, parity={self.parity}, stopbits={self.stopbits}, bytesize={self.bytesize}.")
        except serial.SerialException as e:
            print(f"Failed to open serial port {self.port}: {e}")
            self.serial_connection = None

    def _get_parity(self, parity):
        """Returns the parity setting based on user input."""
        parity_dict = {'N': serial.PARITY_NONE, 'E': serial.PARITY_EVEN, 'O': serial.PARITY_ODD}
        return parity_dict.get(parity, serial.PARITY_NONE)

    def _get_stopbits(self, stopbits):
        """Returns the stop bits setting based on user input."""
        stopbits_dict = {1: serial.STOPBITS_ONE, 1.5: serial.STOPBITS_ONE_POINT_FIVE, 2: serial.STOPBITS_TWO}
        return stopbits_dict.get(stopbits, serial.STOPBITS_ONE)

    def _get_bytesize(self, bytesize):
        """Returns the byte size setting based on user input."""
        bytesize_dict = {5: serial.FIVEBITS, 6: serial.SIXBITS, 7: serial.SEVENBITS, 8: serial.EIGHTBITS}
        return bytesize_dict.get(bytesize, serial.EIGHTBITS)

    def close(self):
        """Closes the serial port connection."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print(f"Serial port {self.port} closed.")
        else:
            print("Serial port already closed or not opened.")

    def send_message(self, message):
        """
        Sends a message through the serial port.

        :param message: The message to send. Must be a string.
        """
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(message.encode('utf-8'))
                print(f"Sent message: {message}")
            except serial.SerialTimeoutException as e:
                print(f"Failed to send message due to timeout: {e}")
            except Exception as e:
                print(f"Failed to send message: {e}")
        else:
            print("Serial port is not open. Cannot send message.")

    def receive_message(self):
        """
        Receives a message from the serial port.

        :return: The received message as a string, or None if there is an error.
        """
        if self.serial_connection and self.serial_connection.is_open:
            try:
                message = self.serial_connection.readline().decode('utf-8').strip()
                # print(f"Received message: {message}")
                return message
            except serial.SerialException as e:
                print(f"Failed to read from serial port: {e}")
                return None
        else:
            print("Serial port is not open. Cannot receive message.")
            return None

    def receive_tactile_data(self):
        """
        Receives tactile data.   int, 3 fingers, each finger 4*8(tactile)+4(between chambers)
        tac_data  [0, 0:8, :] first finger, tactile data        [0, 8, :]   forces between chambers
        tac_data  [1, 0:8, :] second finger, tactile data        [0, 8, :]   forces between chambers
        tac_data  [2, 0:8, :] third finger, tactile data        [0, 8, :]   forces between chambers
        """
        tac_data = np.zeros([3, 9, 4], dtype=np.float32)
        if self.serial_connection and self.serial_connection.is_open:
            try:
                # start_time = time.time()
                # str_tac = self.serial_connection.readline().decode('utf-8').strip()     # empty
                str_tac = self.serial_connection.readline().decode('utf-8').strip()     # the sent command
                for cnt in range(0, 27):
                    str_tac = self.serial_connection.readline().decode('utf-8').strip()
                    numbers = self.extract_with_find(str_tac)
                    numbers = np.array(numbers, dtype=np.int32)
                    tac_data[numbers[0]-1, numbers[1], :] = numbers[2::]
                    # time.sleep(0.002)
                    # print("TTTT: ", time.time()-start_time)
                tac_data[1, 0:4, :] = tac_data[1, 0:4, :][::-1]
                tac_data[1, 4:8, :] = tac_data[1, 4:8, :][::-1]
                return tac_data
            except serial.SerialException as e:
                print(f"Failed to read from serial port: {e}")
                return None
        else:
            print("Serial port is not open. Cannot receive message.")
            return None

    def receive_tactile_long(self):
        """
        receive all data by a str
        Receives tactile data.   int, 3 fingers, each finger 4*8(tactile)+4(between chambers)
        tac_data  [0, 0:8, :] first finger, tactile data        [0, 8, :]   forces between chambers
        tac_data  [1, 0:8, :] second finger, tactile data        [0, 8, :]   forces between chambers
        tac_data  [2, 0:8, :] third finger, tactile data        [0, 8, :]   forces between chambers
        """
        tac_data = np.zeros([3, 9, 4], dtype=np.float32)
        if self.serial_connection and self.serial_connection.is_open:
            try:
                start_time = time.time()
                # str_tac = self.serial_connection.readline().decode('utf-8').strip()     # empty
                # str_tac = self.serial_connection.readline().decode('utf-8').strip()     # the sent command
                str_tac = self.serial_connection.readline().decode('utf-8').strip()
                numbers = self.extract_with_find_long(str_tac)
                # numbers = np.array(numbers, dtype=np.int32)
                # tac_data[numbers[0]-1, numbers[1], :] = numbers[2::]
                print("TTTT: ", time.time()-start_time)
                return tac_data
            except serial.SerialException as e:
                print(f"Failed to read from serial port: {e}")
                return None
        else:
            print("Serial port is not open. Cannot receive message.")
            return None

    def extract_with_find_long(self, str_data):
        indices = [
            ('F1_', str_data.find('F1_')),
            ('F2_', str_data.find('F2_')),
            ('F3_', str_data.find('F3_')),
            ('R0', str_data.find('R0')),
            ('R1', str_data.find('R1')),
            ('R2', str_data.find('R2')),
            ('R3', str_data.find('R3')),
            ('R4', str_data.find('R4')),
            ('R5', str_data.find('R5')),
            ('R6', str_data.find('R6')),
            ('R7', str_data.find('R7')),
            ('R8', str_data.find('R8')),
            (':', str_data.find(':')),
            ('a', str_data.find('a')),
            ('b', str_data.find('b')),
            ('c', str_data.find('c'))
        ]
        numbers = []
        for key, index in indices:
            if index != -1:
                start = index + len(key)
                end = start
                while end < len(str_data) and str_data[end].isdigit():
                    end += 1
                numbers.append(str_data[start:end])
        return numbers

    def extract_with_find(self, str_data):
        indices = [
            ('F', str_data.find('F')),
            ('_R', str_data.find('_R')),
            (':', str_data.find(':')),
            ('a', str_data.find('a')),
            ('b', str_data.find('b')),
            ('c', str_data.find('c'))
        ]
        numbers = []
        for key, index in indices:
            if index != -1:
                start = index + len(key)
                end = start
                while end < len(str_data) and str_data[end].isdigit():
                    end += 1
                numbers.append(str_data[start:end])
        return numbers




if __name__ == '__main__':
    import os, yaml
    import datetime

    home_dir = os.environ['HOME']
    fildir_yaml = home_dir +"/J_06_SR/TAMSER_python/03_control/cfg/01_test_serial_comm.yaml"
    with open(fildir_yaml, 'r') as f:
        cfg_info = yaml.load(f.read(), Loader=yaml.FullLoader)

    serial_port = SerialPort(cfg_info['serial_port']['port'], cfg_info['serial_port']['baudrate'])
    serial_port.open()

    try:
        while True:
            # Sending a message
            str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

            serial_port.send_message("Hello, Serial!,{}\r\n".format(str_time))

            # Receiving a message
            response = serial_port.receive_message()
            if response:
                print(f"Received: {response}")

            # Wait for 1 second before next iteration
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping communication.")

    finally:
        # Ensure the serial port is closed before exiting
        serial_port.close()



    print("Done.")









