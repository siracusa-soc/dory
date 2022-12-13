import numpy as np
import random
from ..Accelerator import Accelerator
from ..Util import div_and_ceil, divisible


class Neureka(Accelerator):
    TP_IN = 32
    TP_IN_3x3 = 28  # specific for Neureka to fit bitwidth of 256bits - 28 ki * 9 spatial = 252 bits
    TP_OUT = 32
    H_OUT = 6
    W_OUT = 6
    KI_PER_WORD = 4
    BANDWIDTH = 256  # bits
    WORD_LEN = 256 // 32  # n 32bit words in bandwidth
    WORD_SIZE = 32 // 8
    __name = 'neureka'

    def __init__(self):
        pass

    @property
    def name(self):
        return self.__name

    def weights_ko_len(self, ko, dw):
        return div_and_ceil(ko, self.TP_IN_3x3) if dw else ko

    def weights_ki_size(self, ki, ks, qw, dw):
        data_size = self.BANDWIDTH // 8  # in bytes
        if dw:
            data_len = qw if ks[0] == 3 else 1
        else:
            data_len = div_and_ceil(ki, self.TP_IN_3x3) * qw if ks[0] == 3 \
                else div_and_ceil(ki, self.TP_IN)
        return data_len * data_size

    def weights_size(self, ko, ki, ks, qw, dw):
        return self.weights_ko_len(ko, dw) * self.weights_ki_size(ki, ks, qw, dw)

    def conv_unroll(self, w, qw, layout='CoutCinK', dw=False):
        if layout == "CoutCinK":
            if dw:
                w = w.transpose(1, 0, 2, 3)  # Swap Cout and Cin
        elif layout == "CoutKCin":
            if dw:
                w = w.transpose(3, 0, 1, 2)
            else:
                w = w.transpose(0, 3, 1, 2)
        else:
            raise Exception(f'Format {layout} not implemented.')

        fs = w.shape[2]

        if dw:
            assert fs == 3, "Only support filter size of 3 with depthwise convolution"
            assert w.shape[0] == 1, "Assumes that the Cout is equal to 1 in case of depthwise convolution"

        if fs == 1:
            return self.conv1x1_unroll(w, qw)
        elif fs == 3:
            return self.conv3x3_unroll(w, qw)

    def conv_roll(self, wbytes, qw, shape, layout='CoutCinK'):
        w = np.zeros(shape, dtype=np.int8)
        if layout == 'CoutCinK':
            wv = w
        elif layout == 'CoutKCin':
            wv = w.transpose(0, 3, 1, 2)
        else:
            raise Exception(f'Format {layout} not implemented.')

        fs = shape[2]
        if fs == 1:
            self.conv1x1_roll(wv, wbytes, qw, shape)
        elif fs == 3:
            self.conv3x3_roll(wv, wbytes, qw, shape)

        return w

    def conv1x1_unroll(self, w, qw):
        Ko, Ki, _, _ = w.shape
        w = w.reshape(Ko, Ki)
        nb_ki = div_and_ceil(Ki, self.TP_IN)
        wbytes = np.zeros((Ko, nb_ki, self.WORD_LEN, self.WORD_SIZE), dtype=np.uint8)
        for i in range(Ko):
            for j in range(nb_ki):
                tile = w[i, j*self.TP_IN:(j+1)*self.TP_IN]
                for k, ki_start in enumerate(range(0, len(tile), self.KI_PER_WORD)):
                    subtile_word = self.__subtile_bit_unroll_1x1(tile[ki_start:ki_start+self.KI_PER_WORD], qw)
                    wbytes[i, j, k] = self.__subtile_bits_unpack_1x1(subtile_word)
        wbytes = wbytes.reshape(-1)
        return wbytes

    def conv1x1_roll(self, wv, wb, qw, shape):
        Ko, Ki, _, _ = shape
        wv = wv.reshape(Ko, Ki)
        nb_ki = div_and_ceil(Ki, self.TP_IN)
        wb = wb.reshape(Ko, nb_ki, self.WORD_LEN, self.WORD_SIZE)
        for i in range(Ko):
            for j in range(nb_ki):
                subtile = wv[i, j*self.TP_IN:(j+1)*self.TP_IN]
                for k, ki_start in enumerate(range(0, subtile.size, self.KI_PER_WORD)):
                    subtile_word = self.__subtile_bits_pack_1x1(wb[i, j, k])
                    self.__subtile_bit_roll_1x1(subtile_word, subtile[ki_start:ki_start+self.KI_PER_WORD], qw)

        wv[wv & (1 << (qw-1)) != 0] |= -(1 << qw)

    def conv3x3_unroll(self, w, qw):
        Ko, Ki, H, W = w.shape
        TP_IN = self.TP_IN_3x3
        nb_ki = div_and_ceil(Ki, TP_IN)
        wbytes = np.zeros((Ko, nb_ki, qw, self.WORD_LEN * self.WORD_SIZE), dtype=np.uint8)
        for i in range(Ko):
            for j in range(nb_ki):
                tile = w[i, j*TP_IN:(j+1)*TP_IN].transpose(1, 2, 0).reshape(H*W, -1)
                for bit in range(qw):
                    subtile_bits = [self.__subtile_bit_unroll_3x3(subtile, bit) for subtile in tile]
                    self.__subtile_bits_pack_3x3(subtile_bits, wbytes[i, j, bit])
        wbytes = wbytes.reshape(-1)
        return wbytes


    def conv3x3_roll(self, wv, wb, qw, shape):
        Ko, Ki, H, W = shape
        TP_IN = self.TP_IN_3x3
        nb_ki = div_and_ceil(Ki, TP_IN)
        wb = wb.reshape(Ko, nb_ki, qw, self.WORD_LEN * self.WORD_SIZE)
        for i in range(Ko):
            for j, ki in enumerate(range(0, Ki, TP_IN)):
                subtile = wv[i, ki:ki+TP_IN].transpose(1, 2, 0).reshape(H*W, -1)
                for bit in range(qw):
                    subtile_bit = self.__subtile_bits_unpack_3x3(wb[i, j, bit])
                    for k in range(H*W):
                        self.__subtile_bit_roll_3x3(subtile[k], subtile_bit[k], bit)

        #qw = np.int8(qw)
        wv[wv & (1 << (qw-1)) != 0] |= -(1 << qw)

    def __subtile_bits_unpack_1x1(self, x):
        return [(x >> i * 8) & 0xff for i in range(4)]

    def __subtile_bits_pack_1x1(self, bytes):
        x = 0
        for i, byte in enumerate(bytes):
            x += byte << i * 8
        return x

    def __subtile_bit_unroll_1x1(self, tile, qw):
        retval = 0
        for i in range(qw):
            for j, el in enumerate(tile):

                assert (el.item()-int(el.item())) == 0, "Found discrepancy!"
                if int(el.item()) & (1 << i):
                    retval |= 1 << (i * self.KI_PER_WORD + j)
        return retval

    def __subtile_bit_roll_1x1(self, ki_bits, w, qw):
        for bit in range(qw):
            for i in range(w.size):
                if ki_bits & (1 << (bit * self.KI_PER_WORD + i)):
                    w[i] |= 1 << bit

    def __subtile_bits_pack_3x3(self, l, wb):
        """
        Packs 9x28bits into 256bits (32 bytes)
        """
        for i, subtile_bit in enumerate(l):
            byte_start = i * self.TP_IN_3x3 // 8
            if i % 2 == 1:
                wb[byte_start] |= (subtile_bit & 0xf) << 4
                byte_start += 1
                subtile_bit = subtile_bit >> 4
            for j in range(4):
                wb[byte_start] = subtile_bit & 0xff
                byte_start += 1
                subtile_bit = subtile_bit >> 8

    def __subtile_bits_unpack_3x3(self, wb):
        """
        Unpacks 9x28bits from 256bits (32 bytes)
        """
        subtile_bit = np.zeros(9, dtype=np.int32)
        for i in range(256 // 28):
            byte_start = i * 28 // 8
            bit_start = 0
            if i % 2 == 1:
                subtile_bit[i] |= ((wb[byte_start] & 0xf0) >> 4) << bit_start
                bit_start += 4
                byte_start += 1
            for j in range(3):
                subtile_bit[i] |= wb[byte_start] << bit_start
                bit_start += 8
                byte_start += 1
            if i % 2 == 0:
                subtile_bit[i] |= (wb[byte_start] & 0xf) << bit_start
        return subtile_bit

    def __subtile_bit_unroll_3x3(self, subtile, bit_idx):
        retval = 0
        for i, el in enumerate(subtile):
            assert (el.item()-int(el.item())) == 0, "Found discrepancy!"
            if int(el.item()) & (1 << bit_idx):
                retval |= 1 << i
        return retval

    def __subtile_bit_roll_3x3(self, w, subtile_bit, bit):
        for i in range(w.size):
            if subtile_bit.item() & (1 << i):
                w[i] |= 1 << bit

    def heuristic_l2(self, tile_n_out, tile_n_in, tile_h_out,
                     constr_total_size, ks, modifier=1000000):
        TP_IN = self.TP_IN_3x3 if ks[0] == 3 else self.TP_IN
        heuristics_l2 = [
            # Geometrical shape of tiles
            {
                "value": divisible(tile_n_in, TP_IN),
                "prio": 3
            },
            {
                "value": divisible(tile_n_out, self.TP_OUT),
                "prio": 1
            },
            {
                "value": divisible(tile_h_out, self.H_OUT),
                "prio": 1.5
            },
            # Total dimension of tile
            {
                "value": constr_total_size,
                "prio": 0.000001
            }
        ]

        sum_heuristics = 0
        for h in heuristics_l2:
            sum_heuristics += int(modifier * h["prio"]) * h["value"]

        return sum_heuristics

    def heuristic_l1(self, n_out, n_in, h_out, w_out,
                     tile_n_out, tile_n_in, tile_h_out, tile_w_out,
                     constr_total_size, zero, ks, modifier=1000000):
        TP_IN = self.TP_IN_3x3 if ks[0] == 3 else self.TP_IN
        heuristics = [
            # Geometrical shape of tiles
            {
                "value": divisible(tile_n_in, TP_IN),
                "prio": 3
            },
            {
                "value": divisible(tile_n_out, self.TP_OUT),
                "prio": 1
            },
            {
                "value": divisible(tile_w_out, self.W_OUT),
                "prio": 2
            },
            {
                "value": divisible(w_out-tile_w_out, tile_w_out),
                "prio": 2
            },
            {
                "value": divisible(tile_h_out, self.H_OUT),
                "prio": 1.5
            },
            # Geometrical shape of border tiles
            {
                "value": divisible(n_out - zero, tile_n_out),
                "prio": 0.01
            },
            {
                "value": divisible(n_in - zero, tile_n_in) % TP_IN,
                "prio": 0.03
            },
            {
                "value": divisible(w_out - zero, tile_w_out) % self.W_OUT,
                "prio": 0.02
            },
            {
                "value": divisible(h_out - zero, tile_h_out) % self.H_OUT,
                "prio": 0.01
            },
            # Total dimension of tile
            {
                "value": constr_total_size,
                "prio": 0.000001
            }
        ]

        sum_heuristics = 0
        for h in heuristics:
            sum_heuristics += int(modifier * h["prio"]) * h["value"]

        return sum_heuristics


if __name__ == "__main__":

    def test(name, ko, ki, fs, qw):
        print(f'Test {name} shape=({ko:3}, {ki:3}, {fs}, {fs}) qw={qw}: ', end='', flush=True)
        shape = (ko, ki, fs, fs)
        test_in = np.random.randint(low=-1<<(qw-1), high=(1<<(qw-1))-1, size=shape, dtype=np.int8)
        acc = Neureka()
        test_out = acc.conv_roll(acc.conv_unroll(test_in, qw), qw, shape)

        if not np.array_equal(test_in, test_out):
            print(f'Fail!')
            print('Test in:')
            print(test_in.reshape(-1))
            print('Test out:')
            print(test_out)
            print(test_in[np.equal(test_in, test_out)])
            return False
        else:
            print(f'Success!')
            return True

    def test_generator(fs, test_count):
        print(f'Testing {fs}x{fs} convolution:')
        pass_count = 0
        for i in range(test_count):
            Ko = random.randint(1, 128)
            Ki = random.randint(1, 128)
            qw = random.randint(2, 8)
            is_pass = test(f'[{i}]', Ko, Ki, fs, qw)
            if is_pass:
                pass_count += 1
        print(f'Passed {pass_count} out of {test_count} tests.')

    TEST_COUNT = 30

    test_generator(1, TEST_COUNT)
    test_generator(3, TEST_COUNT)
