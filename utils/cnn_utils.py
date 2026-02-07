from pyuvm import *
from pooling_utils import PoolingBFM
from convolution_utils import ConvolutionBFM
from cocotb.triggers import RisingEdge

class CNN_BFM(metaclass=utility_classes.Singleton):
    def __init__(self):
        self.pooling_bfm = PoolingBFM()
        self.convolution_bfm = ConvolutionBFM()

    async def send_config(self, seq_item):
        if seq_item.layer_type == 'pooling':
            await self.pooling_bfm.send_config(seq_item)
        elif seq_item.layer_type == 'convolution':
            await self.convolution_bfm.send_config(seq_item)

    async def get_result(self):
        while self.pooling_bfm.mtr_queue.empty() and self.convolution_bfm.mtr_queue.empty():
            await RisingEdge(cocotb.top.clk)

        if not self.pooling_bfm.mtr_queue.empty():
            return await self.pooling_bfm.get_result()
        elif not self.convolution_bfm.mtr_queue.empty():
            return await self.convolution_bfm.get_result()
        
    async def wait_for_result(self, seq_item):
        if seq_item.layer_type == 'pooling':
            return await self.pooling_bfm.wait_for_result()
        elif seq_item.layer_type == 'convolution':
            return await self.convolution_bfm.wait_for_result()

        
    async def reset(self):
        await self.pooling_bfm.reset()
        await self.convolution_bfm.reset()

    def start_bfm(self):
        self.pooling_bfm.start_bfm()
        self.convolution_bfm.start_bfm()