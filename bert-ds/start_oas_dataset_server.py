import os, logging
import sys, glob, pickle
import gzip
import numpy as np
import scipy as sp
import scipy.misc
import time
import zmq
import struct

from utils import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)



class ServerTask():
    """ServerTask"""
    def __init__(self, oas_dataset_file):
        self.oasFile = oas_dataset_file
        self.oasGenerator = self.read_oas()


    def read_oas(self):
        fh = open(self.oasFile, 'r')
        for line in fh:
            seq = line.rstrip()
            yield seq
        fh.close()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:5570')

        logging.info("Started server.")

        while True:
            message = socket.recv()
            max_len = struct.unpack('i', message)[0]

            try:
                while True:
                    seq = next(self.oasGenerator)
                    if len(seq) <= max_len:
                        socket.send_string(seq)
                        break
            except:
                logging.info("Resetting generator.")
                self.oasGenerator = self.read_oas() ## Reset the generator.
                while True:
                    seq = next(self.oasGenerator)
                    if len(seq) <= max_len:
                        socket.send_string(seq)
                        break

        socket.close()
        context.term()



if __name__ == "__main__":
    server = ServerTask("oas_processed_heavy_shuffled")
    server.run()