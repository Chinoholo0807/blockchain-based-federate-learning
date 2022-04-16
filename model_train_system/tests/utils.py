def hex2bytes(hex_str):
    '''
    transfer hex_str(type HexStr) to bytes
    '''
    return bytes.fromhex(str(hex_str)[2:])