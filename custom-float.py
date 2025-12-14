import math


class CustomFloat:
    def __init__(self, bits=8, exp_bits=4):
        self.bits = bits
        self.exp_bits = exp_bits
        self.mant_bits = bits - 1 - exp_bits
        self.bias = (1 << (exp_bits - 1)) - 1

    def encode(self, x):
        if x == 0:
            return 0

        sign = 0 if x > 0 else 1
        x = abs(x)

        e = int(math.floor(math.log2(x)))
        m = x / (2**e) - 1

        e_enc = e + self.bias
        m_enc = int(round(m * (2**self.mant_bits)))

        return (sign << (self.bits - 1)) | (e_enc << self.mant_bits) | m_enc

    def decode(self, bits):
        sign = (bits >> (self.bits - 1)) & 1
        e_enc = (bits >> self.mant_bits) & ((1 << self.exp_bits) - 1)
        m_enc = bits & ((1 << self.mant_bits) - 1)

        e = e_enc - self.bias
        m = 1 + m_enc / (2**self.mant_bits)

        return ((-1) ** sign) * m * (2**e)


if __name__ == "__main__":
    cf = CustomFloat(bits=16, exp_bits=8)  ## More exp_bits -> more range, more mantissa bits -> more precision
    print(cf.encode(15.234))
    print(cf.decode(cf.encode(15.234)))
